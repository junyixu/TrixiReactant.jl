# 2026-10-12 E9：3D Euler 数组写法（3D flux differencing 的运行和编译时间）

E7 的 3D 版（`experiments/e9_euler3d_array.jl`）：5 个变量，每个 (n, n, n, nx, nx, nx)，polydeg 3，
弱形式或 flux differencing（Kennedy–Gruber，所有节点对的两点通量形状 n×n×n×n×单元），邻居用切片，
3D 收敛测试 + 源项，[0,2]³ 周期。和 Trixi 3D（笛卡尔 `P4estMesh`，CPU，Float64）逐点比对：
L=1 4e-13，L=2 2.9e-11（CPU、TPU、A100 都一样），Float32 1e-5 – 7e-5。

`experiments/e8_trixi_gpu_baseline.jl ... 3`：上游 Trixi 原生 CUDA 跑 P4est 3D
`elixir_euler_source_terms.jl`（`initial_refinement_level = 0`，trees (2^L)³，Float32，
flux differencing 用 Trixi 默认的 3D kernel）。

## 性能（Float32，每次 rhs!）

| L=6（64³ 单元，1677 万 DOF） | 弱形式 | flux differencing |
|---|---|---|
| TPU v5e，数组写法 | 37.0 ms（2.20 ns/DOF） | 49.2 ms（2.93） |
| A100，同一份数组代码（Reactant） | 12.9 ms（0.77 ns/DOF） | 9.2 ms（0.55） |
| A100，上游 Trixi 原生 CUDA | 34.8 ms（2.07） | **5.2 ms（0.31）** |

- 3D 里 Trixi 有专门的 shared-memory flux differencing kernel（HalfSweep），比数组写法快 1.8 倍；
  弱形式用的是通用 kernel，数组写法快 2.7 倍。
- 编译（A100）：首次 rhs! 62 s（含预热），之后 rhs! 5–6 s、整个时间循环 21–33 s。

- **3D 里 TPU 掉队更多**：比同一份代码在 A100 上慢 3–5 倍（2D 是 2–3.6 倍），flux differencing
  比 Trixi 的专用 kernel 慢 9.4 倍；按 Spot 价格每个 DOF 也贵约 3.6 倍。弱形式和 Trixi 的通用 kernel
  差不多（37 vs 35 ms）。可能的原因：7 维的节点对数组、好几个长度 4 的维度，TPU 的 (8,128) 分块
  要补齐，转置也多（没 profile）。

## TPU 上的细节

| L | DOF | 弱形式 rhs! | flux differencing rhs! | 编译 rhs! / 时间循环 |
|---|---|---|---|---|
| 3 | 3.3 万 | 0.65 ms | 0.55 ms | 65 s（含预热）/ 47 s |
| 4 | 26 万 | 2.07 ms | 1.90 ms | 7 s / 31–40 s |
| 5 | 210 万 | 9.15 ms | 10.5 ms | 8–10 s / 37–47 s |
| 6 | 1677 万 | 37.0 ms | 49.2 ms | 8–17 s / 36–68 s |

- **编译时间不是问题**：3D flux differencing 最大的程序 rhs! 17 s、整个时间循环 68 s。
- **Float32 收敛**（flux differencing，t = 0.5）：L=1→4 是 6.8e-3、1.4e-3、5.9e-5、5.1e-6（阶 2.3、4.5、3.5），
  L=5 4.7e-6 —— 和 2D 一样停在 ~5e-6 的 Float32 地板。
