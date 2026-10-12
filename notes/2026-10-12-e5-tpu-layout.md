# 2026-10-12 E5：raise 后的 kernel 在 TPU 上为什么慢

Cloud TPU v5e 1 芯片（`v5litepod-1`，europe-west4-b）。Spot 的两个区夜里都没容量，好不容易开出来
10 多分钟就被抢占，最后用按需（€1.37/h，约 45 分钟）。在 `ghcr.io/junyixu/trixidocker:tpu` 容器里跑
（`sudo docker run --privileged --net=host`），GCP 上拉 4 GB 镜像只要 184 s。
日志：S3 `trixi-reactant/2026-10-12-tpu-v5e-e5/`。

`experiments/e5_tpu_layout.jl`：E2 的 1D DGSEM（Burgers，LLF，周期，polydeg 3，5 级低存储 RK），
100 步整个时间循环编进一个 XLA 程序，Float32，`dot_general_precision = HIGHEST`。

| 变体 | 16384 单元 | 1048576 单元 | gather |
|---|---|---|---|
| kernel，节点在前 `(n, nelem)`，邻居查表（= E2） | 143.9 ms | 10515 ms | 4 |
| kernel，节点在前，邻居现算 | 58.5 ms | 3753 ms | 2 |
| kernel，单元在前 `(nelem, n)`，邻居查表 | 145.0 ms | 10571 ms | 4 |
| kernel，单元在前，邻居现算 | 145.6 ms | 10583 ms | 4 |
| 数组写法（矩阵乘 + 切片拼接），节点在前 | 4.46 ms | 201 ms | 0 |
| 数组写法，单元在前 | 3.27 ms | 186 ms | 0 |
| （E2）3070 原生 CUDA.jl / 3070 raise | 17.2 / 3.1 ms | 256 / 413 ms | — |

- **不是布局**：XLA 在 TPU 上自己选了单元维在最内层的布局（`f32[16384,4]{0,1:T(4,128)}`），
  所以两种存储方式一样快。之前"最内维 4 被补齐到 128"的猜测被否定。
- **是 gather**：raise 出来的 gather 每次只取一个标量（`slice_sizes={1,1}`），时间基本和 gather
  个数成正比；邻居现算只在节点在前时把 gather 从 4 个减到 2 个，速度就快了 2.5–2.8 倍。
- **数组写法**没有 gather、只有一个 `dot_general`：比 E2 快 44–57 倍，1M 单元时比 3070 上的原生
  CUDA 还快（186 vs 256 ms）。
- 精度：kernel 变体和 CPU Float32 参考逐位一致；数组写法 100 步后误差 4e-6（求和顺序不同）。
  单次 rhs 的误差大（1e-2 / 0.5）是 E2 已知的相消问题，不是 bug。
- 结论：TPU 要的是数组运算，不是逐线程 kernel。Trixi 的界面 kernel raise 后有 20–42 个标量 gather
  （E4），所以即使 raise 成功在 TPU 上也会很慢。

XLA dump（13 MB）留在已删除的 VM 上没有拉回；需要时用
`XLA_FLAGS=--xla_dump_to=<dir> julia experiments/e5_tpu_layout.jl tpu 16384 2` 重现。
