# 2026-10-14 E11：让 Trixi 的 3D flux differencing kernel 过 Reactant（raise）

导师 2026-10-13 的要求：`HalfSweep()`、`FullSweep()`、`FullSweepGlobal()` 中任意一个过 Reactant。
HalfSweep / FullSweep 用 shared memory + `@synchronize`，Reactant 还不能 raise 屏障，所以选
**`FullSweepGlobal`**（不用 shared memory，每个节点一个线程，ndrange = (n, n, n, 单元数)）。

- 机器：RunPod Secure L4（EUR-IS-1，$0.59/h，约 1.5 h）；`:latest` 镜像（Reactant 0.2.291、KA 0.9.x、Trixi 0.17.14）
- 算例：E4 的 `3d_euler_fd_fullsweepglobal`（P4est 3D `elixir_euler_source_terms_nonperiodic.jl`，
  Kennedy–Gruber，Float32，64 个单元）；参考值 = Trixi 自己的 kernel 在 KA CPU 后端上的结果
- 脚本：
  - `experiments/e4_kernel_inventory.jl` 新增环境变量：`E4_KERNEL`（只测一个 kernel）、`E4_WGS` /
    `E4_WGS_MODE`（重放时换 workgroup 大小）、`E4_NDRANGE=static`（ndrange 作为静态参数）、
    `E4_UNROLL_IN=1`（见下面第 1 条）；主循环加了 `if abspath(PROGRAM_FILE) == @__FILE__`，可以被 `include`
  - `experiments/e11_fsg_variants.jl`：用 E4 录下的参数重放 FullSweepGlobal 的副本（原样 / 节点循环展开 / while）
  - `experiments/e11_remui_repro.jl`：不依赖 Trixi 的最小 4D KA kernel
- 日志：S3 `2026-10-14-fd-kernel-raise/`（本地 `logs/` 同名目录）

## 两个卡点

**1. KA 的越界检查在 4D 时变成 Reactant 不能 raise 的循环。** KA 的 CUDA 后端（Reactant 也走它）总是
`DynamicCheck`，每个线程先算 `__validindex`，也就是 `I in ndrange`。Base 把它算成
`all(map(in, Tuple(I), indices))`，4D 时留下一个遍历栈上 `NTuple{4,Bool}` 的循环
（MLIR 里是 `memref.alloca : memref<4xi8>` + `scf.while` + 越界 `llvm.intr.trap`），报
"cannot raise op to stablehlo"。

- **任何 4D ndrange 的 KA kernel 都中招**：最小复现 `y[i,j,k,e] = 2x[i,j,k,e]` 就 raise 不了
  （`repro2_*`）。E4 里能 raise 的 kernel 都是 ≤ 3D，FullSweepGlobal 是唯一 4D 的。
- 换静态 workgroup、静态 ndrange 都没用（检查还在，循环还在）。
- **修法**：给 `in(::CartesianIndex{N}, ::CartesianIndices{N, NTuple{N, OneTo{Int}}})` 加一个展开的、
  无分支的方法 `(i₁ ∈ r₁) & (i₂ ∈ r₂) & …`（`E4_UNROLL_IN=1`，不改 Trixi、不改 KA，只加方法）。
  之后最小复现在 CPU 和 GPU 后端都 raise 成功、结果精确（`repro3_*`）。上游该改的是 KA 的
  `__validindex`（或 Base 的这个 `in`）。

**2. Reactant 编译器段错误**：`AffineCFGPass` 里的 `OptimizeRem::matchAndRewrite(arith::RemUIOp)`。
`remui` 来自把线性线程号拆成 (i, j, k, element) 的除法/取余（除数是运行时的 ndrange、workgroup 大小）。

- CPU 后端：静态 workgroup (4,4,4,1) 时不崩；但第 1 条修好之后又崩（编译走得更远了）
- GPU 后端：只换 workgroup 还崩；**静态 ndrange + 静态 workgroup** 不崩
- 把 kernel 里三个节点循环展开（`unrolled_sum`，NNODES 是 `Val`）也不崩，CPU、GPU 都是
- 最小复现 kernel 不崩，触发条件和 Trixi kernel 里带 `SVector` 的节点循环有关，还没找到更小的复现

## 结果

| FullSweepGlobal，raise | 静态 ndrange | 静态 workgroup | 展开的 `in` | 结果 |
|---|---|---|---|---|
| Trixi 原 kernel | — | — | — | 段错误（CPU、GPU） |
| Trixi 原 kernel | — | ✓ | — | CPU：scf.while；GPU：段错误 |
| Trixi 原 kernel | ✓ | ✓ | — | scf.while（CPU、GPU） |
| Trixi 原 kernel | — | ✓ | ✓ | 段错误（CPU、GPU） |
| **Trixi 原 kernel** | ✓ | ✓ | ✓ | **raise 成功**，误差 2.2e-7（GPU）/ 1.1e-7（CPU），**244 个 gather、10 个 scatter** |
| 节点循环展开的副本 | — | ✓ | ✓ | **raise 成功**，误差 2.2e-7 / 8.6e-8，**没有 gather/scatter** |
| `while` 循环的副本 | — | ✓ | — | scf.while（没和展开的 `in` 一起测） |

| 64 个单元，一次 kernel（L4） | 时间 |
|---|---|
| Trixi 原 kernel，不 raise（Reactant 调 CUDA kernel） | 0.191 ms |
| Trixi 原 kernel，raise（三项修改） | 0.400 ms（244 个 gather） |
| 节点循环展开的副本，raise | **0.153 ms** |

- **Trixi 源码不改也能 raise**：只在启动方式（静态 ndrange、静态 workgroup）和一个 `in` 方法上动手。
  但原来的 `for ii in 1:NNODES` 循环 raise 后，循环变量变成下标，`u[v, ii, j, k, e]` 这类读就成了 gather
  （244 个）；TPU 上每个下标约 8 ns（E6），会很慢。
- **节点循环展开后没有 gather**，GPU 上还比不 raise 的原 kernel 快。对 Trixi 的改动很小：
  FullSweepGlobal 的三个循环展开（或 KA 的 `@unroll`），并像 HalfSweep 一样给静态 workgroup 大小。
- 还没在 TPU 上跑这个 kernel；也只测了 64 个单元的正确性和单次时间。

## 2026-10-15：`unsafe_indices` 和 TPU

**KA 的 `@kernel unsafe_indices = true`**（KA 0.9 源码 `src/macros.jl` 的 `transform_gpu!`）：只是不插入
`__active_lane__ = __validindex(__ctx__)`，也就是去掉第 1 条的越界检查。文档说"users must avoid
`@index(Global)`"，但代码里并不禁止：CUDA 后端的 `@index(Global)` 是
`expand(__iterspace(ctx), blockIdx().x, threadIdx().x)`（CUDA.jl `CUDAKernels.jl`），和越界检查无关，照样能算；
文档的意思是没有检查以后，ndrange 不能被 workgroup 整除时多出来的线程会越界读写。FullSweepGlobal 的
ndrange (4, 4, 4, 单元数) 正好被 (4, 4, 4, 1) 整除，所以可以用。代价是要改 Trixi 的 kernel 定义
（加 `unsafe_indices = true`），而展开的 `in` 方法只是追加一个方法。

TPU v5e（`e11_fsg_variants.jl` 新增 `<variant>_unsafe`、`E11_TREES`）：

| 64 个单元，raise，静态 workgroup (4,4,4,1) | 结果 | 一次 kernel |
|---|---|---|
| 节点循环展开 + 展开的 `in`（`E4_UNROLL_IN=1`） | 正确，误差 1.3e-7，没有 gather | 0.212 ms |
| 节点循环展开 + `unsafe_indices = true`（不加 `in` 方法） | 正确，误差 1.3e-7，没有 gather | 0.186 ms |
| Trixi 原循环 + `unsafe_indices` + 静态 ndrange（244 个 gather） | 为 TPU 编译时内存超过 33 GB，VM 卡死，只好重启 | — |

- 所以 FullSweepGlobal 节点循环展开后，在真 TPU 上也能 raise 并且结果正确；`unsafe_indices = true`
  可以代替展开的 `in`
- 带 gather 的原 kernel 在 TPU 上编译不出来（L4 上可以），没有再试

更大的网格（`E11_TREES`，节点循环展开 + `unsafe_indices`，TPU v5e，一次 kernel）：

| 单元数 | 节点数 | 误差 | 时间 | 每个节点 |
|---|---|---|---|---|
| 64 | 4096 | 1.3e-7 | 0.186 ms | — |
| 4096 | 26 万 | 3.5e-7 | 0.389 ms | 1.5 ns |
| 32768 | 210 万 | 2.6e-7 | 4.34 ms | 2.1 ns |
| 262144 | 1677 万 | 3.5e-7 | 47.8 ms | 2.8 ns |

- 1677 万节点时只这一个体积分 kernel 就要 47.8 ms，和 E9 整个 flux differencing rhs!（49.2 ms，
  换成 128 友好的网格后 22.7 ms）差不多
- **profile（2026-10-15，`E4_PROFILE=1`）**：raise 出来的 kernel 在 XLA 里变成约 240 个 kernel，两种网格大小都要读写
  **约 100 个状态大小的数组**。4096 个单元时状态只有 5 MB，数据留在片上，等效 3 TB/s、0.18 ms；26 万单元时状态 335 MB，
  全走 HBM（~720 GB/s），每次 34 GB、47.5 ms。所以每个节点的时间变长是从片上掉到 HBM，根本问题是那 ~100 遍读写
  （布局没问题，单元维度在最内层）。E12 优化后的数组写法也是 78–110 遍，同一个量级

## 要报的问题

- **KernelAbstractions**：4D ndrange 的越界检查 raise 不了（最小复现 + 上面的修法）
- **Reactant**：`OptimizeRem`（`arith.remui`）段错误；附 Trixi 的 FullSweepGlobal 重放脚本（还没有最小复现）
- **Trixi**：FullSweepGlobal 用静态 workgroup、展开节点循环，就能在 Reactant 上 raise 且没有 gather
