# 2026-10-17 E22：shock capturing（Hennemann–Gassner）

Trixi 的实际算例大多用 `VolumeIntegralShockCapturingHG`：每个单元一个混合系数 α，体积分 = (1 − α) flux differencing +
α 一阶子单元有限体积。GPU 测试里没有它，这里把它加进数组原型。

- 脚本：`experiments/e12_p4est_array.jl`（体积分 `sc`，gather 路径），算例 `sedov`（`elixir_euler_sedov.jl`，
  t = 0.3 和 Trixi 自己的测试一样）、`blastamr`（E21）
- 机器：同 E21；参考值用 Trixi 默认（polyester）CPU 路径和 Trixi 自己的测试值（`test_p4est_3d.jl`，Float64）
- 日志：S3 `2026-10-17-e20-e24/`

## 做法（照 Trixi 0.17.14 的 `IndicatorHennemannGassner` 和 `fv_kernel!`）

- **α**：指示变量（Sedov 是 ρ p）逐点算（Trixi 的函数 broadcast）→ 用逆 Legendre Vandermonde 矩阵三个方向变到模态 →
  全部 / 去掉最高阶 / 去掉两个最高阶的能量 → `1 / (1 + exp(-s / threshold (energy − threshold)))`，再按 `alpha_min`、
  `alpha_max` 截断；平滑：每个单元至少取邻居（界面和 mortar）最大 α 的一半（邻居表在 CPU 上建好，一次 gather）；
  α 接近 0 时当成 0（纯 DG），和 Trixi 一样
- **子单元有限体积**：每个方向 n − 1 个子单元界面，用 Trixi 预先算好的子单元法向量（`cache.normal_vectors`），
  面通量（这里是 LLF）在界面上算一次，权重倒数乘差分（单元边界处为零）
- 两点体通量是 `flux_ranocha`（用到 `ln_mean`，E15 的无分支版本）
- 全是数组运算，没有按单元的分支

## 结果

rhs! 对 Trixi（Sedov，64 单元，n = 6，周期），max|du − du_Trixi| / max|du_Trixi|：Float64 **2.4e-16**，Float32 1.0e-7。

完整模拟到 t = 0.3（28 步）：

| | 步数 | 对 Trixi CPU Float64 | 对 Trixi 自己的测试值 |
|---|---|---|---|
| Float64 / CPU 后端 | 28 / 28 | 1.6e-16 / 1.7e-16 | 1.3e-9 / 7.7e-10（测试值只有 9 位） |
| Float32 / TPU | 28 / 28 | 3.2e-7 / 9.1e-8 | 3.2e-7 / 9.2e-8 |

- Float64 和 Trixi 逐位几乎一样：数组写法的运算顺序和 Trixi 的循环一致
- shock capturing + AMR：见 E21

性能（TPU v5e，Float32，gather 路径，Sedov 网格加大，n = 6）：shock capturing 对比只用 flux differencing（同一个两点通量
`flux_ranocha`；只用 FD 的 Sedov 会算出 NaN，这里只看时间）：

| 网格 | rhs! shock capturing | rhs! 只用 FD | 多出来的 | 一个 RK 步（`hoststage`，shock capturing） |
|---|---|---|---|---|
| 16³ 树，4096 单元（88 万 DOF） | 7.31 ms | 6.65 ms | +10% | 35.5 ms |
| 32³ 树，32768 单元（708 万 DOF） | 85.5 ms | 75.7 ms | +13% | 432 ms |

- 指示器、子单元有限体积和混合只比纯 FD 多 10–13%（数组写法里每个单元都同时算 FD 和 FV，没有分支）
