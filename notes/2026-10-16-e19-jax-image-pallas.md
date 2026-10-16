# 2026-10-16 E19：带 JAX 的镜像；Pallas kernel 在 Reactant 里跑在 TPU 上

为手写 TPU kernel（Pallas）做准备：镜像里装 Python + JAX + PythonCall，并验证 JAX lower 出来的 Pallas kernel
能嵌进 Reactant 程序、在真 TPU 上跑对。

## 镜像 `:jax` / `:jax-tpu`

- TrixiDocker 分支 `feat/docker-jax`（`043ff03`、`87adff1`），CI run 38840569266 两个变体都成功；不合入 `main`
- **uv 一条龙**：uv 装它自己的 CPython 3.14.8（`/opt/uv-python`）和 venv `/opt/jax-env`，里面 `jax==0.11.0`（tpu 变体是
  `jax[tpu]`，带 jax 自己的 libtpu）和 numpy；主环境加 PythonCall 0.9.37（`PRESERVE_ALL`，其他包版本不动）
- PythonCall 按它文档 "If you already have Python and required Python packages installed" 配置：
  `JULIA_CONDAPKG_BACKEND=Null`（CondaPkg 什么都不装）+ `JULIA_PYTHONCALL_EXE=/opt/jax-env/bin/python`
  （不设的话会用 PATH 里系统的 python，没有 jax）。两个变量写进镜像 ENV 和 `/etc/profile.d/julia.sh`
- **不用 Conda**：CondaPkg 会合并环境里所有包的 `CondaPkg.toml`，Reactant 的要求 `python <= 3.12` 并 pip 装 tensorflow
  （扩展只在导出 SavedModel 时用）
- **Python 3.14 可以**：PythonCall 支持 ≥ 3.10（3.14.0/3.14.1 除外），jaxlib 0.11.x 和 libtpu 有 cp314 wheel
- **jax 选 0.11.0**：Pallas kernel 是 jax 序列化（Mosaic）后交给 libtpu 的，新 jax 写的旧 libtpu 可能读不了。
  Reactant 0.2.291 用 libtpu 0.0.45.dev20260728；jax 的 `[tpu]` extra 固定的 libtpu：0.11.0 → 0.0.44.\*，0.11.1 → 0.0.46.\*，
  0.11.2 → 0.0.48.\*，所以 0.11.0 是比 Reactant 的 libtpu 旧的最新版本
- `JAX_PLATFORMS=cpu` 默认：同一个进程里 jax 只做 lowering，不打开 TPU（否则会和 Reactant 抢 TPU、加载两份 libtpu）；
  纯 JAX 跑 TPU 时设 `JAX_PLATFORMS=tpu`
- 每次加载都会有一条 warning：Reactant 的 PythonCall 扩展 `import tensorflow` 失败（不影响）
- 构建时检查（`docker/jax_smoke.jl`）：Reactant 追踪一个 jax 函数（`jax.numpy.sum` → 6.0）；Pallas kernel 在没有 TPU 的
  构建机上 lower 成 TPU（`jit(...).trace(...).lower(lowering_platforms=("tpu",))`），`tpu_custom_call` 嵌进 Reactant 模块后还在

## TPU 上的测试（`experiments/e19_pallas_smoke.jl`，TPU v5e，`:jax-tpu`）

一个逐元素 Pallas kernel（`o = 2x + 1`，256×128 Float32）：

| | 结果 |
|---|---|
| JAX 自己在 TPU 上跑（`JAX_PLATFORMS=tpu`，jax 的 libtpu） | 最大误差 0.0 |
| jax lower → `Reactant.Ops.hlo_call` 嵌进 Reactant 函数 → `@jit` 在 TPU 上跑（Reactant 的 libtpu） | **最大误差 0.0** |

- 所以 jax 0.11.0 序列化的 Mosaic kernel，Reactant 0.2.291 的 libtpu 能读能跑：手写 TPU kernel 可以放进 Reactant 程序
- 坑：Reactant 保持 Julia 的维度顺序（Julia 的 (256, 128) 数组就是 `tensor<256x128>`，不是反过来）；`@jit` 写在
  `using Reactant` 之前的同一个顶层 `if` 块里会在展开宏时报错（顶层语句是逐条展开的，要分开写）

## 下一步

- 用 Pallas 写体积分（E12 的 rhs! 每次读写 78–110 遍状态大小的数组，理论上约 2.5 遍）
