# 2026-10-11 E4：Trixi 的 KA kernel 哪些能被 Reactant raise

RunPod **Secure** RTX A4500（20 GB，Ampere sm_86，驱动 580.159.04，EU-RO-1，$0.25/h）。
镜像 `ghcr.io/junyixu/trixidocker:latest`（`2f72102`）：Julia 1.12.7，Trixi 0.17.14，Reactant 0.2.291，
CUDA.jl 6.2，KernelAbstractions 0.9.44。日志在 S3 `trixi-reactant/2026-10-11-e4-a4500/`
（`summary.md`；`static/` 是第一轮、基函数矩阵用 StaticArray 的结果）。

先在 Community 3070/3080 Ti 上连续碰到 4 台坏主机（`cuInit` 999），换 Secure 才跑起来。

## 方法 `experiments/e4_kernel_inventory.jl`

1. 用上游 P4est elixir 建 `semi`（`trixi_include(...; real_type = Float32)`，case 和 kernel 变体照搬
   `test/test_cuda_2d.jl`、`test_cuda_3d.jl`），然后在 CPU 上算一次 `compute_coefficients` 和 `rhs!`。
2. `Trixi.trixi_backend(::Array)` 返回一个**录制后端**：每次 kernel launch 记下 kernel、`ndrange`、
   参数（launch 前后各深拷贝一份），同时用 KA 的 CPU 后端执行它作为参考结果。
   CPU 后端跑不了的 kernel（见下文 HalfSweep）改用**原生 CUDA launch** 做参考（表中标 ref CUDA）。
3. 每个不同的 kernel 单独重放：参数经 `Adapt` 转成 Reactant 数组，`@compile raise=...` 编译，
   运行后与参考结果比较所有被写过的数组（相对误差，NaN 只在一边出现记为 MISMATCH）。
   HLO 里还有 `custom_call`/`kernel_call` 记为 kept（kernel 没被 raise，原样作为 PTX 调用）。
4. 三种模式：`gpu noraise`（Reactant + CUDA，kernel 保持为 PTX）、`gpu raise`、`cpu raise`
   （raise 是 TPU 的必要条件，CPU 后端 raise 是 TPU 的代理）。每个 (模式, case) 一个进程；进程崩溃时
   `run_e4.sh` 把崩溃的 kernel 记下、跳过它重跑该 case。

这样不需要把 Trixi 的 cache 整体搬到 Reactant（E3 的卡点），只转换 kernel 真正收到的参数。

## 为了让 kernel 能被单独重放而做的处理（Trixi kernel 本身一行没改）

| 问题 | 归属 | 处理 |
|---|---|---|
| `node_indices` 是 `Symbol` 元组数组，kernel 里用 `===` 比较 | Trixi（E3 已知） | 编码成 `Int8`，给 5 个 index helper 加方法 |
| `LobattoLegendreBasis`/`LobattoLegendreMortarL2` 的类型参数要求 `<: AbstractMatrix{RealT}`，而 `TracedRArray{T}` 的 eltype 是 `TracedRNumber{T}`，Reactant 追踪时报 "Failed to find a valid type for typevar" | Trixi × Reactant | 用一个薄 `DeviceMatrix{T} <: AbstractMatrix{T}` 包住 Reactant 数组（矩阵仍在设备内存，同 CUDA 路径） |
| 每个 kernel 都收一个 `typeof(mesh)`；Reactant 的 CUDA launcher 检查 `typeof(arg)`（= `DataType`），当作非 isbits 拒绝，降参时又用 `sizeof(arg)`（mesh 实例的大小）；CUDA.jl 把它当零大小的 ghost | **Reactant bug** | 给 `_check_no_traced_in_kernel_arg(::Type{DataType})` 加方法，并让这些类型的 `sizeof` 为 0 |
| 3D HalfSweep（shared memory）kernel 在 KA **CPU** 后端报 `UndefVarError: Ja1_node`：跨 `@synchronize` 的局部变量没声明 `@private`，只在 GPU 上碰巧能用 | **Trixi bug** | 参考结果改用原生 CUDA launch |

第一轮把基函数矩阵当 `SMatrix` 传（常量聚合）：kernel 用运行时下标访问它，LLVM 要先拷到栈上
（`llvm.alloca`），raise 一律拒绝 —— 这是测试框架造成的假象，第二轮改成 `DeviceMatrix` 重跑。

## 结果

20 个 `@kernel` 定义（`compute_coefficients`、通用 `volume_integral`、2D 4 个、3D flux differencing
6 个方法 + turbo 4 个 + 其余 4 个）都被 16 个 case 覆盖到。完整的逐 case 表在 S3 的 `summary.md`
（本地 `logs/2026-10-11-e4-a4500/summary.md`）；GPU raise 和 CPU raise 的结论完全一致。

| kernel（定义数） | 不 raise（GPU） | raise | raise 时的卡点 |
|---|---|---|---|
| `prolong2boundaries`（2D、3D） | ✓ | ✓ | — |
| `prolong2interfaces_and_calc_interface_flux`（2D、3D） | ✓ | ✓（2D 有一例错） | `2d_advection`：4096 个输出中 80 个 NaN，其余 case 正确 |
| `calc_boundary_flux`（2D、3D） | ✓ | 3D ✓，2D ✗ | 2D MHD 的边界数据用 `sinpi` → `llvm.call @__nv_sinpif` |
| `calc_surface_integral_and_apply_jacobian_and_calc_sources`（2D、3D） | ✓ | 只有 2D 无源项时 ✓ | 有源项：`llvm.alloca`（i64）；3D advection：`scf.while` |
| `volume_integral`（弱形式，也是 2D flux differencing） | ✓ | ✗ | `scf.while`；3D advection 时编译器段错误 |
| `flux_differencing` ×6、`flux_differencing_turbo` ×4（3D） | ✓ | ✗ | HalfSweep/FullSweep（shared memory + `@synchronize`）："failed to raise func"；FullSweepGlobal、turbo FullSweep：段错误 |
| `compute_coefficients`（初值） | ✓ | ✗ | `scf.while`（`CartesianIndices` 循环），MHD 还有 `sinpi` |

- **不 raise**：20 个全部能经 Reactant 在 GPU 上跑，和参考结果的相对差 ≤ 4e-7（很多逐位一致）。
- **raise**：4 个定义在所有 case 都对（2D/3D `prolong2boundaries`、3D `prolong2interfaces`、3D
  `calc_boundary_flux`），2 个部分 case 能（2D `prolong2interfaces`、2D `calc_surface_integral`），
  14 个从来不能 —— 包括所有体积分 kernel，也就是大部分计算量。
- raise 成功的面 kernel 每个有 20–42 个 gather、最多 36 个 scatter（3D MHD 界面 kernel 42/36），
  在 TPU 上正是慢的那种模式（见 E5）。
- 失败原因多是 Reactant 的 raise 能力缺口（`scf.while` 循环、小的栈数组、`sinpi` 没有 lowering、
  段错误、错误编译），不是 Trixi 的设计问题；shared memory 的屏障 Reactant 自己的测试里写着
  "We don't yet auto lower this via polygeist"（`ref/Reactant.jl/test/integration/cuda.jl:67`）。
- 第一轮（基函数矩阵用 `SMatrix`）`volume_integral` 报的是 `llvm.alloca` of struct，换成设备数组后
  变成 `scf.while`：前者是测试框架的假象，`static/summary.md` 留作对照。

## 发现的问题（待整理成 issue，附最小复现）

- Reactant：kernel 的 `Type` 参数被拒（见上表）；2D `prolong2interfaces_and_calc_interface_flux_KAkernel!`
  在 `2d_advection` 里 raise 后 4096 个输出有 80 个是 NaN（GPU 和 CPU 后端一样，不 raise 时逐位正确，
  同一 kernel 在 Euler/MHD case 里 raise 正确）；`FullSweepGlobal` 的 flux-differencing kernel raise 时
  **段错误**（`AffineCFGPass` / `OptimizeRem` / `valueCmp`，处理 `arith.remui` 时），3D advection 的
  `volume_integral` 和 turbo FullSweep 也会，GPU 和 CPU 后端都一样；libdevice 的
  `__nv_sinpif` 调用不能 raise（`sin` 可以）。
- Trixi：KA CPU 后端跑不了 HalfSweep kernel（缺 `@private`）；`node_indices` 用 `Symbol`；
  基函数结构体的 `<: AbstractMatrix{RealT}` 约束。
