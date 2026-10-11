# TrixiReactant.jl

Feasibility study: running [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) through
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), with TPUs as the target.

- `experiments/` — scripts (E1–E3) and `reactant_overrides.jl`, the method overrides that let
  upstream Trixi's KernelAbstractions path take Reactant arrays
- `notes/` — results per run
- `docs/` — background notes
- `ref/` (ignored) — local clones of Reactant.jl and TrixiCUDA.jl

## Environment

Run on a GPU pod with the precompiled image `ghcr.io/junyixu/trixidocker:latest`
([junyixu/TrixiDocker](https://github.com/junyixu/TrixiDocker)): `JULIA_PROJECT` there points at
the pinned environment (Trixi 0.17.14, Reactant 0.2.291, CUDA.jl 6.2). The overrides depend on
Trixi internals, so keep that version.

```sh
git clone https://github.com/junyixu/TrixiReactant.jl && cd TrixiReactant.jl
experiments/run_all.sh        # logs in logs/
```
