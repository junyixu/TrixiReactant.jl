# 2026-10-12 E7/E8：2D Euler 写成数组运算，在 TPU 和 A100 上和 Trixi 比

- TPU：Cloud TPU v5e 1 芯片，按需（europe-west4-b），`:tpu` 容器；日志 S3 `2026-10-12-tpu-v5e-e6e7/`
- GPU：RunPod Community A100 SXM 80 GB（$1.39/h，驱动 590），`:latest` 镜像；日志 S3 `2026-10-12-a100-e7e8/`

## 方法 `experiments/e7_euler2d_array.jl`

和 Trixi TreeMesh 2D 完全相同的离散：polydeg 3 LGL，LLF（`max_abs_speed_naive`），体积分用弱形式
（`derivative_hat`）或 flux differencing（`derivative_split` + `flux_kennedy_gruber`），
`initial_condition_convergence_test` + `source_terms_convergence_test`，[0,2]² 周期，
2^L × 2^L 个单元，5 级低存储 RK（CarpenterKennedy2N54）。整个 `rhs!` 只用数组运算：

- 弱形式体积分：每个方向一次矩阵乘（`dot_general`，(n×n) · (n × 其余)）
- flux differencing：用广播一次算出所有节点对的两点通量（形状 n×n×n×单元），再与
  `D_split` 相乘求和（没有 `dot_general`）
- 邻居：`slice` = 切片 + 拼接（结构网格）；`gather` = 4 个变量的整个面（4n = 16 个数）
  通过下标数组一次取（非结构网格需要的形式）
- 状态是 4 个数组的元组，每个 (n, n, nx, ny)；时间循环用 `@trace for` 编进一个 XLA 程序

**和 Trixi 逐点比对**（`check`，t = 0.3 的 rhs!，参考是 Trixi 在 CPU 上、Float64、笛卡尔
`P4estMesh`）：四种变体在 CPU、TPU、A100 上 Float64 的相对差都是 5e-13 – 2e-11（随网格加密变大，
是 rhs 相消造成的），Float32 是 2e-5 – 2e-4。

`experiments/e8_trixi_gpu_baseline.jl`：上游 Trixi 原生 CUDA（KA kernel）跑同一个问题
（P4est 2D `elixir_euler_source_terms.jl`，trees_per_dimension = (2^L, 2^L)，Float32），
计时一次 `ode.f`。

## 性能（Float32，每次 rhs!）

| L=10（1677 万 DOF） | 弱形式 | flux differencing |
|---|---|---|
| TPU v5e，数组，切片 | 11.1 ms（0.66 ns/DOF） | 13.7 ms（0.82） |
| TPU v5e，数组，gather 面 | 41.4 ms（2.47） | 43.0 ms（2.56） |
| A100，同一份数组代码（Reactant） | 5.8 ms（0.34） | 3.8 ms（0.23） |
| A100，同一份数组代码，gather 面 | 5.1 ms（0.30） | 4.0 ms（0.24） |
| A100，**上游 Trixi 原生 CUDA**（P4est） | 16.4 ms（0.98） | 13.7 ms（0.82） |

| L=8（105 万 DOF） | 弱形式 | flux differencing |
|---|---|---|
| TPU v5e，数组，切片 / gather | 0.80 / 1.42 ms | 0.72 / 1.28 ms |
| A100，数组，切片 / gather | 0.52 / 0.46 ms | 0.31 / 0.33 ms |
| A100，Trixi 原生 CUDA | 1.54 ms | 1.40 ms |

- **TPU 上的数组写法和 A100 上的 Trixi 原生 CUDA 一样快甚至更快**（弱形式 11.1 vs 16.4 ms，
  flux differencing 13.7 vs 13.7 ms）；但同一份数组代码在 A100 上还要快 2–3.6 倍。
  Trixi 的 P4est 代码处理的是一般曲线网格（要算度量项），比较不完全对等。
- **gather 在 TPU 上贵 3.7 倍，在 A100 上不贵**（A100 上 gather 版甚至略快，可能是切片拼接影响了融合，未查）。
  这和 E6 一致：TPU 每个下标约 8 ns，16 个数一块的面 gather 只有 ~19 GB/s。
- 小网格（L ≤ 7）单次 `rhs!` 被约 0.3 ms 的调用开销主导；时间循环编译进 XLA 后每个 RK 级在
  L=7 只要 0.46 ns/DOF。TPU 上时间循环里每级（1.1 ns/DOF @ L=10）比单独 rhs!（0.66）慢，
  RK 更新和循环携带的状态在 TPU 上有额外拷贝；A100 上两者相同（0.34–0.35）。
- **编译时间**：第一次编译含 Reactant 预热约 75 s，之后每个配置 rhs! 3–11 s、整个时间循环
  10–13 s（L=10）。不是障碍。
- **dot_general 精度**（弱形式，TPU，L=8 / L=10）：DEFAULT 0.66 ms、HIGH 0.70 / 10.3 ms、
  HIGHEST 0.80 / 11.1 ms。n=4 的小收缩不是瓶颈，HIGHEST 只贵 8–17%。

### 每 10⁹ 个 DOF·rhs 的费用（L=10，用上表时间和每小时价格）

| | 弱形式 | flux differencing |
|---|---|---|
| TPU v5e Spot €0.49/h（≈$0.53） | $0.00010 | $0.00012 |
| TPU v5e 按需 €1.37/h（≈$1.48） | $0.00027 | $0.00034 |
| A100 Community $1.39/h，数组代码 | $0.00013 | $0.00009 |
| A100 Community $1.39/h，Trixi 原生 | $0.00038 | $0.00032 |

按 Spot 价格，TPU 和 A100 跑数组代码的性价比相当，都比现在的 Trixi kernel 便宜约 3–4 倍。

## Float64

| 每次 rhs!，L=9（419 万 DOF） | 弱形式 | flux differencing |
|---|---|---|
| TPU v5e Float32 | 2.69 ms | 2.63 ms |
| TPU v5e Float64（软件模拟） | 128 ms（**48×**） | 270 ms（**103×**） |
| A100 Float32 / Float64（L=10） | 5.8 / 9.2 ms（1.6×） | 3.8 / 8.6 ms（2.3×） |

TPU 上的 Float64 只能用来验证，不能用来算：flux differencing 计算量大，模拟的代价更高
（E1 的 sin/cos 归约是 170×）。A100 上 Float64 基本只是带宽翻倍。

## 收敛阶（`eoc`，t = 0.5，CFL 0.5，ρ 的 L2 误差，期望阶数 4）

| L（2^L × 2^L 单元） | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| TPU F32 弱形式 DEFAULT（bf16 矩阵乘） | 5.5e-3 | 1.0e-2 | 1.4e-2 | 1.9e-2 | 2.2e-2 | 2.6e-2 | 3.1e-2 |
| TPU F32 弱形式 HIGH（bf16×3） | 8.4e-4 | 5.6e-5 | 3.3e-5 | 6.2e-5 | 2.1e-4 | 3.9e-4 | 3.0e-4 |
| TPU F32 弱形式 HIGHEST | 8.4e-4 | 5.1e-5 | **2.6e-6** | 2.8e-6 | 3.7e-6 | 6.1e-6 | 1.1e-5 |
| TPU F32 flux differencing | 8.2e-4 | 5.3e-5 | **2.8e-6** | 2.7e-6 | 3.4e-6 | 5.2e-6 | 9.4e-6 |
| TPU F64 弱形式（模拟） | 8.4e-4 | 5.1e-5 | 1.5e-6 | 1.4e-7 | 5.7e-9 | 4.2e-10 | **2.8e-11** |
| TPU F64 flux differencing（模拟） | 8.2e-4 | 5.3e-5 | 1.3e-6 | 1.3e-7 | 6.2e-9 | 4.4e-10 | **2.9e-11** |

- **`dot_general` 必须用 HIGHEST**：DEFAULT（bf16）完全不收敛，HIGH 在 L=3 之后停住，
  比 HIGHEST 差 10–100 倍；而 HIGHEST 只慢 8–17%（见上）。flux differencing 没有 `dot_general`，
  不受这个设置影响。
- **Float32 的精度地板约 3e-6**（L=4，16×16 单元，4096 DOF 就到了），之后误差随加密**变大**
  （步数加倍、`J⁻¹(…)` 的相消更严重），到 L=8 是 1e-5。两种体积分一样。
- **Float64 一直保持 4 阶**，L=8 到 2.8e-11；但在 TPU 上慢 49 倍（L=8：291 s vs 5.9 s）。
- **CPU 上的 Float32 一模一样**（L=4 2.6e-6，L=7 6.0e-6），所以地板是 Float32 本身的舍入，
  不是 TPU 的问题；CPU 上的 Float64 和 TPU 模拟的 Float64 在打印的位数上完全相同。

## 自适应时间步（`adapt`）

`@trace while t < tend`，每步在图里算 CFL 步长（`maximum` 归约），最后一步截到 `tend`：

| | 自适应 | 固定 dt（同样步数） |
|---|---|---|
| L=7 flux differencing | 914 步，0.563 s，L2 ρ 5.30e-6 | 0.547 s，5.30e-6 |
| L=7 弱形式 | 914 步，0.389 s，6.04e-6 | 0.520 s |
| L=9 flux differencing | 3656 步，77.1 s，2.06e-5 | 73.7 s，1.91e-5 |

开销 0–5%，结果和固定步长一致。编译：自适应 98–115 s（含约 75 s 的首次预热），固定 16–26 s。
坑：被 traced 循环重新绑定的标量参数（`t = t + dt`）会写回传进去的 `ConcreteRNumber`，
同一个编译好的函数第二次调用前要换新的标量（第一次跑时因此得到了错误结果，已修正重跑）。
