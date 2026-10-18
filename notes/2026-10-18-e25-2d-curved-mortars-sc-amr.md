# 2026-10-18 E25：2D 弯曲非结构网格、mortar、shock capturing、AMR

E16/E20 的 2D 只验证过笛卡尔网格（GPU 测试的算例），mortar（E18）、AMR（E21）、shock capturing（E22）只有 3D。
这里把这些都补到 2D，让 2D 和 3D 覆盖的东西一样。

- 脚本：`experiments/e12_p4est_array.jl`（2D gather 路径：`mesh_arrays2d(semi; mortars)`、`setup2d_core`、
  `mortar_arrays2d`、`mortar_small_block2d`、`mortar_large_block2d`、`setup_padded2d`；shock capturing 改成和维数无关）
- 机器：GCE TPU v5e `ct5lp-hightpu-1t`（机器镜像 `tpu-ubuntu-base`），CPU 后端的检查和 TPU 的任务同时跑；
  有 mortar 的参考值用 Trixi 默认（polyester）CPU 路径（`~/env-poly`，E18）
- 日志：S3 `2026-10-18-e25-2d/`（本地 `logs/` 同名目录）

## 做法

- **弯曲非结构网格**：原来的 2D gather 路径就能处理（两种面朝向：同向、反向），只是没测过。
  算例用 Trixi 的 "flag" 网格（Abaqus 文件里的非结构四边形网格，用 transfinite mapping 弯曲，Dirichlet 边界）
- **mortar**（照 Trixi 2D 的 `prolong2mortars!`、`calc_mortar_flux!`、`mortar_fluxes_to_elements!`）：每个 mortar 2 个小面
  （位置 1、2），小面的邻居值 = 大面换到 mortar 的节点顺序（正向或反向）再乘 `forward_lower`/`forward_upper`；
  大面 = $-2 (R_u f_2 + R_l f_1)$，换回大面的节点顺序。和 3D 一样作为 gather 源的额外一块、再用一次 gather 替换大面的通量
- **shock capturing**：指示器（模态能量：每个方向乘一次逆 Vandermonde）、邻居平滑（mortar 的小单元数从邻居表的行数推出）、
  子单元有限体积（Trixi 的 `normal_vectors_1/2`）都按维数写
- **AMR**：`setup_padded2d` 和 3D 一样按容量补齐（多余单元复制单元 1 的面），反向的面总是处理（掩码是数据）
- 另外碰到一个 Reactant bug：Trixi 2D 对流的精确解（Dirichlet 边界值）里 `@muladd` 把 `x - v t` 变成 SVector 的 `muladd`，
  StaticArrays 用 `map` 实现，Reactant 的 `map` overlay 把 `SVector` 当成 XLA 数组（和 `dot`/`norm` 同一类，issue 草稿 6）；
  给 `overloaded_map` 加了 `StaticVector` 方法

## 结果：rhs! 对 Trixi

max|du − du_Trixi| / max|du_Trixi|，t = 0.3 的精确解（Sedov：初值）：

| 算例 | 网格 | 体积分 | Float64（CPU 后端） | Float32 | Float64（TPU） |
|---|---|---|---|---|---|
| `elixir_advection_unstructured_flag.jl` | 弯曲非结构，384 单元，736 个界面里 32 个节点顺序反向 | 弱形式 | 1.7e-13 | 6.7e-5 | 3.5e-12 |
| `elixir_euler_source_terms_nonconforming_unstructured_flag.jl` 去掉角上的加密 | 96 单元 | 弱形式 / FD | 1.2e-13 / 2.2e-13 | 5.6e-5 / 1.3e-4 | 3.5e-12 / 4.9e-12 |
| 同上，加密一次 | 384 单元 | 弱形式 / FD | 3.5e-13 / 8.7e-13 | 1.8e-4 / 3.6e-4 | |
| `elixir_advection_nonconforming_flag.jl` | 弯曲，周期，168 单元，**72 个 mortar** | 弱形式 | 2.0e-13 | 1.2e-4 | 5.8e-12 |
| `elixir_euler_source_terms_nonconforming_unstructured_flag.jl` | 弯曲非结构，角上加密，168 单元，**80 个 mortar** | 弱形式 / FD | 1.6e-13 / 2.9e-13 | 6.4e-5 / 9.4e-5 | 5.0e-12 / 5.1e-12 |
| `elixir_euler_sedov.jl`（shock capturing） | 4×4 树加密两次，256 单元，n = 5，周期 | sc | 1.9e-16 | 3.6e-8 | 3.9e-15 |

## 结果：完整模拟

l2 / linf 的最大相对差（对所有变量取最大）：

| 算例 | 步数 | Float64 / CPU 后端 对 Trixi | Float32 / TPU 对 Trixi | 对 Trixi 自己的测试值（Float32 / TPU） |
|---|---|---|---|---|
| 对流，flag 网格（协调） | 11 / 11 | | 5.2e-6 / 1.4e-5 | 5.2e-6 / 1.4e-5 |
| Euler，flag 网格（协调，96 单元），弱形式；FD | 149 / 149；150 / 150 | | 2.4e-5 / 1.4e-4；3.0e-5 / 2.3e-4 | （去掉了加密，没有） |
| 对流，72 个 mortar | 11 / 11 | 4.3e-13 / 7.2e-13 | 8.0e-5 / 1.8e-4 | 8.0e-5 / 1.8e-4 |
| Euler，弯曲非结构，80 个 mortar | 221 / 221 | 4.7e-14 / 9.8e-13 | 3.7e-5 / 1.3e-4 | 3.7e-5 / 1.3e-4 |
| Sedov（shock capturing） | 103 / 103 | 2.0e-15 / 3.0e-15 | 6.8e-5 / 7.9e-5 | 6.8e-5 / 7.9e-5（Float64 对测试值 1.8e-9，测试值只有 9 位） |

- Float64 和 Trixi 一样到舍入误差，Float32 在 TrixiTest 的 Float32 容差内，步数都相同

## 结果：AMR

`amr` task（E21 的做法），l2 / linf 最大相对差：

| elixir | 精度 / 后端 | 步数 | 自适应 | 编译 | 结束时单元数（Trixi） | 对 Trixi CPU Float64 | 对 Trixi 自己的测试值 |
|---|---|---|---|---|---|---|---|
| `elixir_euler_weak_blast_wave_amr.jl`（shock capturing，每 5 步自适应），到 t = 0.1 | Float64 / CPU 后端 | 79 / 79 | 15 | 2 | 193（193），中间 64–196 | 1.5e-15 / 1.0e-14 | 1.5e-15 / 4.7e-15 |
| 同上 | Float32 / TPU | 79 / 79 | 15 | 2 | 193（193） | 4.9e-5 / 1.1e-4 | 4.9e-5 / 1.1e-4 |
| `elixir_advection_amr_unstructured_flag.jl`（弯曲非结构，Dirichlet，每 5 步），到 t = 10 | Float64 / CPU 后端 | 184 / 184 | 36 | 1 | 192（192），中间 177–222 | 1.0e-15 / 7.9e-16 | 2.0e-13 / 1.7e-14 |
| 同上 | Float32 / TPU | 184 / 184 | 36 | 1 | 192（192） | 1.6e-5 / 7.3e-7 | 1.6e-5 / 7.3e-7 |

- blast wave 初始网格只有 64 个单元（初始自适应没有加密），容量 128 在第 10 步不够了，按 320 单元 / 76 个 mortar 重新编译一次
- 每次自适应在主机上：blast wave 1.0–1.3 s，弯曲对流 32 ms；每步 TPU 上 15.7 / 5.3 ms
- 修了一个 E21 里没碰到的问题：网格从没有 mortar 变成有 mortar 时参数 NamedTuple 的字段顺序变了（类型变了），
  调用编译好的函数报 `MisMatchedThunkTypeError`；现在 `setup_padded`/`setup_padded2d` 先去掉 `p0.mort` 再 merge
- 还有一个 Reactant bug：traced `Bool` 相减（Trixi 的 `x_trans_periodic_2d`，2D 对流 Gaussian 的边界值）编译失败
  （"raise_triton_custom_call"），Float64 转换也没有方法；按整数相减绕过（issue 草稿 10）

## 性能

弯曲非结构 flag 网格加密 8 次（157 万单元，n = 4，2517 万节点），TPU v5e，Float32，gather 路径：rhs! 弱形式 388.6 ms、
FD 392.0 ms（每个节点 15 ns；E16 笛卡尔 100 万单元 227 ms，13.5 ns）。整个模拟一个程序（`@trace`）放不进 16 GB HBM
（17.8 / 19.3 GB），按 E14 应该每个阶段一个程序。

## 结论

- 2D 和 3D 现在覆盖一样的东西：弯曲非结构网格、mortar、AMR（不用重新编译）、shock capturing，rhs! 和完整模拟
  Float64 都和 Trixi 一样到舍入误差，Float32 在 TPU 上都在容差内，单元数和步数都和 Trixi 一样
- 2D 的 mortar 代码一次就对了：3D 的写法（小面作为 gather 源的一块、大面的通量再 gather 替换）直接降一维
