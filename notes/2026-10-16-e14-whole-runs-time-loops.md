# 2026-10-16 E14：完整模拟对 GPU 测试的参考值；时间循环的写法

2026-10-16 计划的第 1、2 项。E12 只比过一次 rhs!，这里把 Trixi GPU 测试（`test/test_cuda_3d.jl`）的 Float32 算例从头跑到尾，
和测试里的 l2/linf 比；再查 E12 里"时间循环比 5 次 rhs! 慢、FD 的循环编译 12–32 分钟"的原因。

- 脚本：`experiments/e12_p4est_array.jl` 新增 `run`（完整模拟）和 `loops`（时间循环）两个 task
- 机器：GCP TPU v5e-1（us-west4-a，按需），`:tpu` 镜像；同一台 VM 上用 CPU 后端（容器不加 `--privileged`）做检查
- 日志：S3 `2026-10-16-e14-e19-tpu/`（本地 `logs/` 同名目录）

## 做法（`run`）

- 时间积分和 elixir 一样：Carpenter–Kennedy 2N54；**时间步按 CFL 每步算**，照搬 Trixi `max_dt`（`stepsize_dg3d.jl`）：
  每个单元在节点上取各方向 $|Ja^d \cdot \lambda| / J$ 的最大值、对方向求和，$\Delta t = \mathrm{cfl} \cdot 2 / (n \max)$；
  常速度方程（对流）取每个节点上的和的最大值。最后一步截到 `tend`（和 ODE 积分器的 tstop 一样）
- 误差用 **Trixi 自己的 `calc_error_norms`**（`AnalysisCallback` 的分析网格，Float64）算我们最终的状态
- 参考：GPU 测试里的 l2/linf；同一个 elixir 在 CPU 上用 Trixi 跑 Float32 和 Float64（`trixi_include`，关掉输出文件）
- 判据：TrixiTest 对 `RealT_for_test_tolerances = Float32` 的容差（`atol = 500 eps(Float32)` ≈ 6e-5，`rtol = sqrt(eps(Float32))` ≈ 3.5e-4）
- 两种循环：`host`（编译一个 RK 步，Julia 里循环调用，每步读回 t 判断结束）和 `trace`（整个模拟 `@trace while` 编译成一个程序）

## 结果：完整模拟（TPU v5e，Float32）

表里是 l2 / linf 的最大相对差（对所有变量取最大）。"步数"是我们的 / Trixi 的。

| 算例 | 步数 | 对 Trixi CPU Float32 | 对 Trixi CPU Float64 | 对 GPU 测试参考值 | 参考：Trixi Float32 对 GPU 参考 |
|---|---|---|---|---|---|
| 3D 对流 `elixir_advection_basic.jl`（周期，512 单元） | 19 / 19 | 1.1e-5 / 6.6e-4 | 8.3e-6 / 5.2e-4 | 8.3e-6 / 5.2e-4 | 2.7e-6 / 1.2e-3 |
| 3D Euler `..._nonperiodic.jl` 弱形式（64 单元） | 317 / 317 | 2.4e-4 / 5.0e-4 | 1.8e-4 / 2.4e-4 | （测试里没有） | — |
| 同上 FD（Kennedy–Gruber），`host` | 317 / 317 | 1.5e-4 / 1.4e-3 | 1.3e-4 / 1.3e-3 | 2.0e-4 / 1.5e-3 | 5.2e-5 / 6.4e-4 |
| 同上 FD，`trace` | 317 / 317 | 1.2e-4 / 1.1e-3 | 1.5e-4 / 1.1e-3 | 6.8e-5 / 8.9e-4 | |
| 同上，`dim,faces,fdloop,step`（砖形路径） | 317 / 317 | 弱 9.5e-5 / 7.4e-4，FD 3.9e-5 / 7.2e-4 | | FD 6.6e-5 / 3.9e-4 | |
| 3D MHD（E15） | 84 / 84 | 5.2e-5 / 2.5e-4 | 1.1e-4 / 3.3e-4 | 1.1e-4 / 3.3e-4 | 1.2e-4 / 2.4e-4 |
| 2D（E16）、mortar（E18） | 见各自的笔记 | | | | |

- **全部在 GPU 测试的 Float32 容差内**，步数和 Trixi 一样；`host` 和 `trace` 两种循环都对
- 我们和 Trixi Float32 的差别与 Trixi 自己 Float32 和 Float64 的差别同一个量级，就是 Float32 的舍入
- 小算例的编译：`host` 18–130 s（一个 RK 步），`trace` 18–20 s；时间循环本身 0.4–5 s

## 时间循环（`loops`，26 万单元，TPU v5e）

E12 的结论是"时间循环比 5 次 rhs! 慢、FD 的循环编译 12–32 分钟"。这里同样的步数（固定 dt，10 步）用 4 种写法，
最后的状态逐位相同：

- `trace`：E12 的 `@trace for`，所有步数一个程序
- `hoststep`：一个程序 = 一个 RK 步（5 个阶段），Julia 里循环调用（异步派发，最后才同步）
- `hoststage`：一个程序 = 一个阶段（rhs! + 更新），每步调用 5 次
- `hostrhs`：rhs! 和更新分开编译，每步调用 10 次

v4 = `dim,fdloop,faces`，单元 128×64×32（单独 rhs! 最快的布局）；v7 = 再加 `an,step`，单元 32×64×128（128 在最后，E12 时间循环最快的布局）：

| 每个 RK 步，ms（括号：相对 5 次 rhs!） | v4 弱形式 | v4 FD | v7 弱形式 | v7 FD |
|---|---|---|---|---|
| 单独一次 rhs! | 39.0 | 55.8 | 55.1 | 75.1 |
| `trace` | 300.5（1.54） | 编译超过 1 h（E12） | 230.4（0.84） | 编译 12 min 以上（E12） |
| `hoststep` | 198.9（1.02），编译 57 s | 显存不够（RESOURCE_EXHAUSTED） | 249.9（0.91） | **1225.7（3.26），编译 1778 s** |
| `hoststage` | **197.1（1.01）**，编译 13 s | **282.6（1.01）**，编译 27 s | 277.7（1.01） | 379.1（1.01） |
| `hostrhs` | 208.7（1.07） | 292.5（1.05） | 290.2（1.05） | 389.0（1.04） |

- **FD 慢的原因是"一个程序里放 5 次 FD rhs!"，不是 while 循环**：v7 FD 的 `hoststep`（没有循环，只是 5 个阶段在一个程序里）
  也慢 3.3 倍、编译 30 分钟；每个阶段一个程序（`hoststage`）就正好是 5 次 rhs!，编译 27–39 s。v4 的整步程序连 16 GB HBM 都放不下
- 从 Julia 派发的开销可以忽略（`hoststage` 1.01×）；`hostrhs` 多 4–7%（更新不能和 rhs! 融合，多一遍读写）
- 弱形式在 v7 布局下 `trace` 反而比单独 rhs! 快（XLA 在循环里用默认布局，128 正好在最后），但 v4 布局的单独 rhs! 更快，
  `hoststage` 让每个程序用自己的最好布局
- **现在 26 万单元最好的 RK 步：弱形式 197 ms，FD 283 ms**（E12：弱形式 230 ms，FD 的循环编译不出来）。
  E9 笛卡尔同网格 5 × 18.9 / 22.7 ms ≈ 95 / 114 ms，还差 2–2.5 倍（rhs! 本身的差距，见 E12）
- 2D 100 万单元（1677 万 DOF，E16 的 gather 路径）：弱形式 `hoststep` 0.98×、`trace` 1.02×，FD `hoststep` 1.04× ——
  2D 没有这个问题

## 结论

- 数组原型把 GPU 测试的 3D 算例（对流、Euler 弱形式/FD、MHD）完整跑完，和 Trixi 在 Float32 容差内一致，步数相同
- 时间循环应该**每个 RK 阶段编译成一个程序、在 Julia 里循环**：编译快（半分钟以内）、每步正好 5 次 rhs!；
  `@trace` 整个循环和"整个 RK 步一个程序"在 FD 上都会让 XLA 的融合变差
