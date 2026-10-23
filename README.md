# TrixiReactant.jl

Feasibility study: can [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) run on TPUs through
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)? Scope: upstream Trixi's KernelAbstractions
path and `P4estMesh` (2D and 3D); Float32 on TPU.

Two routes were tried:

- **Raising**: Trixi's per-thread KernelAbstractions kernels are raised by Reactant to StableHLO
  (E2, E4, E11).
- **Array code**: the DG `rhs!` written as whole-array operations that reuse Trixi's pointwise
  physics, compiled by XLA, with optional hand-written Pallas kernels spliced in (E5–E26).

## Where to look

| Path | Content |
|---|---|
| `notes/` | One result note per experiment (E1–E25), plus `notes/issues/` with the drafts of the upstream issues |
| `experiments/` | Experiment scripts, `reactant_overrides.jl` (methods that let upstream Trixi accept Reactant arrays) |
| `tools/` | Helper scripts (result upload, VM setup) |
| [Issues](../../issues) | Discussion and upstream tracking: raising vs. array code, volume integral kernels, `P4estMesh` as array operations, ... |
| [TrixiReactantSlides](https://github.com/junyixu/TrixiReactantSlides) | Progress reports (Typst slides), published at <https://junyixu.github.io/TrixiReactantSlides/> |

## Reproducing the experiments

All experiments run in the prebuilt images of
[junyixu/TrixiDocker](https://github.com/junyixu/TrixiDocker), which pin the whole environment
(Julia 1.12, Trixi 0.17.14, Reactant 0.2.297, CUDA.jl, KernelAbstractions, ODE solvers; the image
README lists the exact versions). The overrides depend on Trixi internals, so keep the versions.

| Tag | Use |
|---|---|
| `ghcr.io/junyixu/trixidocker:latest` | NVIDIA GPU (CUDA 12.9 or newer on the host) |
| `ghcr.io/junyixu/trixidocker:tpu` | Cloud TPU VM (v5e) |
| `:enum`, `:enum-tpu` | Trixi PR #3332 (`NodeIndex` enum), E13 |
| `:jax`, `:jax-tpu` | Adds JAX 0.11 and Pallas, E19, E23, E26 |

```sh
git clone https://github.com/junyixu/TrixiReactant.jl && cd TrixiReactant.jl

# GPU
docker run --rm --gpus all -v $PWD:/work -w /work ghcr.io/junyixu/trixidocker:latest \
    julia -t 8 experiments/e7_euler2d_array.jl gpu check 3,4

# TPU VM (the container needs --privileged and --net=host to see the TPU)
sudo docker run --rm --privileged --net=host -v $PWD:/work -w /work ghcr.io/junyixu/trixidocker:tpu \
    julia -t 8 experiments/e7_euler2d_array.jl tpu check 3,4
```

`JULIA_PROJECT` inside the images already points at the pinned environment. The first argument of
most scripts is the backend (`cpu`, `gpu` or `tpu`); the header of each script and the matching
note in `notes/` document the remaining arguments and the measured results.

- `experiments/run_all.sh` runs the early experiments (E1–E3) and writes one log per run to `logs/`;
  `experiments/run_e4.sh` runs the kernel inventory (E4).
- The `P4estMesh` array prototype is `experiments/e12_p4est_array.jl`, driven by environment
  variables such as `E12_OPTS`; see `notes/2026-10-15-e12-p4est-array.md`.
- On a TPU VM, run Julia inside the container, or set `LD_PRELOAD=` if you run it on the host: the
  VM image preloads tcmalloc, which crashes Julia.
- `tools/s3_upload.sh` uploads results with `rclone`; it is skipped with `RCLONE_UPLOAD=0`.
