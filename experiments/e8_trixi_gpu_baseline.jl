# E8: GPU baseline for E7 — upstream Trixi's native CUDA (KernelAbstractions) path on the
# same problem: P4est `elixir_euler_source_terms.jl` in 2D or 3D (Cartesian, periodic,
# polydeg 3, LLF with max_abs_speed_naive), Float32, weak form or flux differencing
# (Kennedy-Gruber; Trixi's default 3D kernel), (2^L)^dim elements. Times one rhs! (`ode.f`).
# usage (TrixiDocker image, NVIDIA GPU): julia experiments/e8_trixi_gpu_baseline.jl [levels] [volumes] [2|3]
using Trixi, CUDA, OrdinaryDiffEqLowStorageRK, Printf

const LEVELS = parse.(Int, split(get(ARGS, 1, "6,7,8,9"), ','))
const VOLUMES = split(get(ARGS, 2, "weak,fd"), ',')
const DIM = parse(Int, get(ARGS, 3, "2"))
println("device: ", CUDA.name(CUDA.device()), "  Trixi ", pkgversion(Trixi))

function bench_rhs(level, vol)
    volume_integral = vol == "weak" ? VolumeIntegralWeakForm() :
                      VolumeIntegralFluxDifferencing(flux_kennedy_gruber)
    solver = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
                   volume_integral = volume_integral)
    mod = Module()
    Base.eval(mod, :(using Trixi, CUDA, OrdinaryDiffEqLowStorageRK))
    trixi_include(mod, joinpath(examples_dir(), "p4est_$(DIM)d_dgsem", "elixir_euler_source_terms.jl");
                  trees_per_dimension = ntuple(_ -> 2^level, DIM), initial_refinement_level = 0,
                  solver = solver, sol = nothing,
                  real_type = Float32, storage_type = CuArray)
    # the `ode` binding is newer than this function's world (Julia 1.12): look it up late
    return Base.invokelatest(m -> time_rhs(getglobal(m, :ode)), mod)
end

function time_rhs(ode)
    u, du = ode.u0, similar(ode.u0)
    t = 0.0f0
    ode.f(du, u, ode.p, t)
    CUDA.synchronize()
    ts = map(1:20) do _
        CUDA.@elapsed ode.f(du, u, ode.p, t)
    end
    ndofs = length(u) ÷ (DIM + 2)
    return minimum(ts) * 1e3, ndofs, u isa CuArray
end

println("CSV,task,backend,RealT,level,dofs,volume,neighbours,precision,compile_rhs_s,rhs_ms,ns_per_dof_rhs,compile_sim_s,step_ms,ns_per_dof_stage")
for level in LEVELS, vol in VOLUMES
    try
        ms, ndofs, ongpu = bench_rhs(level, vol)
        @printf("Trixi CUDA %dD level=%d dofs=%9d %-4s rhs %9.3f ms (%6.2f ns/DOF)  on GPU: %s\n",
                DIM, level, ndofs, vol, ms, ms * 1e6 / ndofs, ongpu)
        println(join(("CSV", "trixi_cuda_$(DIM)d", "gpu", "Float32", level, ndofs, vol, "p4est", "-", "",
                      round(ms, digits = 4), round(ms * 1e6 / ndofs, digits = 3), "", "", ""), ","))
    catch err
        println("level=$level $vol FAIL ", first(first(split(sprint(showerror, err), '\n')), 160))
    end
    flush(stdout)
end
