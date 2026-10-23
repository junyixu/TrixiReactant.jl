# 2026-10-22 E26：整个 rhs!（和 RK 阶段）写成一个 Pallas kernel

E23 把 FD 体积分写成 Pallas kernel 之后，26 万单元的 rhs! 还要 32.3 ms（体积分只占 2.2 ms），剩下的是面积分、源项和布局转换
（XLA 的逐元素 kernel 20.1 ms / 12.2 GB、转置 5.8 ms / 4.2 GB）；A100 上 Trixi 原生 6.5 ms。理论下限：读 u、Ja、J⁻¹、坐标，
写 du，一共 23 个状态大小的分量数组（1.5 GB），按 570 GB/s 约 3 ms。这里把整个 rhs! 放进一个 kernel 验证这个估计。

- 脚本：`experiments/e26_pallas_fused.py`（kernel、纯 JAX 的对照和计时、lower 成 StableHLO 文本），
  `experiments/e12_p4est_array.jl` 的 `E12_OPTS=faces,fused`（rhs!）和 `faces,fusedrk`（RK 阶段），文件由
  `E26_FUSED_MLIR` / `E26_FUSEDRK_MLIR` 给出
- 算例：`elixir_euler_source_terms_nonperiodic.jl`（3D Euler，Dirichlet 边界 = 收敛测试的精确解，源项，
  `FluxLaxFriedrichs(max_abs_speed_naive)`，Kennedy–Gruber FD，n = 4），砖形网格 `cart128x4x64` = 256 × 8 × 128 = 262,144 单元
- 机器：GCE TPU v5e `ct5lp-hightpu-1t`（机器镜像 `tpu-ubuntu-base`，jax 0.11.0），Float32
- 日志：Google Drive `GDrive:mpcdf-s3/trixi-reactant/2026-10-22-e26-fused/`（本地 `logs/` 同名目录；MPCDF S3 账号被停用）

## kernel

- 数组就是砖形路径的状态：每个分量一个 (n, n, n, Nx, Ny, Nz) 数组（节点 i, j, k，再是单元网格），不用转换布局。
  TPU 按最后两维 (Ny, Nz) = (8, 128) 分块，正好一个 vreg；节点维度在前面，取一层节点就是选 vreg
- 一个 grid step = 一个 x 层（Ny × Nz = 1024 个单元）。y、z 方向的邻居在同一个块里，用 `pltpu.roll` 平移一个单元；
  x 方向邻居贴着的那层节点作为两个额外的块传进来（同一个 u，BlockSpec 只取一层节点，下标错开一个 x 层）
- 顺序：体积分（E23 的 slab 写法）→ 6 个面：自己的面值、邻居的面值（域边界上用精确解）、自己的外法向 ±Ja^d，
  LLF 通量乘 `inverse_weights[1]` 加到面节点 → 源项 → 乘 −J⁻¹。每个界面在两边各算一次（不用分散写回）
- `stage` 变体：输入再加 K 和标量 (t, dt, a, b)，输出 K' = a K + dt du、U' = U + b K'，du 不写回 HBM
- 每块 1024 个单元比 2048 个快（4.27 vs 4.89 ms，纯 JAX）

## 正确性

| 检查 | 结果 |
|---|---|
| kernel 对同样数学的 jnp（XLA），26 万单元 | 4.0e-7（rhs!），4.5e-7（RK 阶段） |
| rhs! 对 Trixi，`cart2`（4×4×4 单元） | 5.6e-5（XLA 砖形路径 Float32 也是 5.6e-5） |
| rhs! 对 Trixi，`cart4x2x8`（8×4×16，各方向不同，查轴有没有弄错） | 2.0e-4（XLA 砖形路径 Float32 也是 2.0e-4，Float64 3.2e-13） |
| 完整模拟 `cart2`，317 步（Trixi 317），`fused` 和 `fusedrk` 结果完全相同 | 对 Trixi Float64 1.6e-4 / 1.0e-3，对 GPU 测试参考值 7.5e-5 / 1.3e-3，都 PASS |

## 性能（TPU v5e，Float32，262,144 单元，FD）

同一个网格上：

| | rhs! | RK 步，每个阶段一个程序 | RK 步，每步一个程序 | RK 步，整个循环一个程序 |
|---|---|---|---|---|
| XLA（`dim,fdloop,faces`） | 55.6 ms | 283.8 ms | | |
| E23 Pallas 体积分（`dim,faces,pallas`） | 32.7 ms | 165.4 ms | | |
| `fused`（rhs! 一个 kernel，更新在 XLA） | 7.97 ms（kernel 本身 3.11 ms） | 61.1 ms | 35.1 ms | 28.7 ms |
| **`fusedrk`（每个 RK 阶段一个 kernel）** | | 51.8 ms | 24.4 ms | **18.2 ms** |
| 参考：A100 上 Trixi 原生（E17） | 6.5 ms | | | ≈ 33 ms（5 × rhs!，不算更新） |

- **rhs! 的 kernel 本身 3.11 ms**（profile），和估计的 3 ms 一致；纯 JAX 计时 4.27 ms 里还包括把 5 个输出拼起来
- **RK 步 18.2 ms：比 XLA 快 15.6 倍、比 E23 快 9 倍，比 A100 上 Trixi 原生的 rhs!（5 × 6.5 = 33 ms）还快 1.8 倍**。
  每个阶段 3.6 ms，按最少读写（u、K、x 方向两层邻居面、Ja、J⁻¹、坐标，写 U'、K'，2.4 GB）是 650 GB/s，约为 v5e 峰值的 80%
- 程序边界上的布局转换：Reactant 把 Julia 数组作为维度反过来的 `f32[128,8,256,4,4,4]` 交给 XLA，XLA 给入口参数选的布局
  是 Nx 在最内层（`{2,5,4,3,1,0:T(4,128)}`），kernel 要 Nz 在最内层，所以**每个程序的入口要把 18 个输入、出口把 5 个输出各拷一遍**
  （每个 0.21 ms，共约 4.8 ms，rhs! 的 7.97 ms 里有一大半是这个）。整个循环一个程序时只在入口拷一次，所以最快；
  每个阶段一个程序时每次都拷，最慢。这和 E14 的结论（XLA 写法每个阶段一个程序最好）相反：kernel 对 XLA 是不透明的，
  放在一个大程序里不会让融合变差
- 换一个单元网格（128 × 8 × 256，最长的维度放在最后）也一样：rhs! 8.27 ms，每阶段一个程序 51.8 ms，每步 24.4 ms，
  整个循环 18.2 ms —— 拷贝还在，XLA 给入口参数选的布局和 kernel 要的不一样；能避开的办法只有把循环放进一个程序

## 结论

- 布局转换和面积分确实可以优化掉：整个 RK 阶段一个 kernel，HBM 上只剩最少的读写，单个 v5e 芯片上 3D FD 的时间步比
  A100 上 Trixi 原生快
- 代价：kernel 只能在 TPU 上用；方程（通量、边界值、源项）要在 JAX 里再写一遍，现在只有 Euler 3D + 这个算例；
  只做了砖形网格（非结构网格要 gather，见 E6、E12）
- 用的时候时间循环应该整个放进一个程序（`@trace`，或者每步一个程序 + 主机判断 CFL），避免程序边界上的布局转换
