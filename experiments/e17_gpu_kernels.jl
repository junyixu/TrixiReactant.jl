# E17: native GPU baseline for the array prototype (E12) and the advisor's question whether a
# modified KernelAbstractions kernel alone (no Reactant) gets the same speed. On the E12 case
# (P4est 3D elixir_euler_source_terms_nonperiodic.jl, T×T×T trees refined once, polydeg 3,
# LLF with max_abs_speed_naive, Kennedy-Gruber, Float32), with CuArrays:
#   rhs!     Trixi's whole rhs! (`ode.f`), weak form and flux differencing (default kernel)
#   volume   Trixi's 3D flux-differencing kernels HalfSweep / FullSweep (shared memory) and
#            FullSweepGlobal, and copies of FullSweepGlobal: `orig` (the same body, static
#            workgroup) and `unroll` (node loops unrolled, E11), launched with KA directly
# usage (TrixiDocker image, NVIDIA GPU): julia experiments/e17_gpu_kernels.jl [T=32] [reps=20]
using Trixi, CUDA, KernelAbstractions, OrdinaryDiffEqLowStorageRK, Printf, StaticArrays
using Trixi: get_node_vars, get_contravariant_vector, set_node_vars!, nnodes, nvariables, nelements

const T = parse(Int, get(ARGS, 1, "32"))
const REPS = parse(Int, get(ARGS, 2, "20"))
println("device: ", CUDA.name(CUDA.device()), "  Trixi ", pkgversion(Trixi), "  KA ", pkgversion(KernelAbstractions),
        "  trees ", T, "^3 refined once")

@inline unrolled_sum(f, ::Val{1}) = f(1)
@inline unrolled_sum(f, ::Val{N}) where {N} = unrolled_sum(f, Val(N - 1)) + f(N)

# Trixi 0.17.14's FullSweepGlobal body (dg_3d_gpu.jl)
@inline function fsg_orig_body!(du, u, equations, dg, volume_flux, ::Val{NNODES}, ::Val{NVARIABLES},
                                derivative_split, contravariant_vectors, i, j, k, element) where {NNODES, NVARIABLES}
    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    du_local = zero(SVector{NVARIABLES, eltype(du)})
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    for ii in 1:NNODES
        Ja1_avg = 0.5f0 * (Ja1_node + get_contravariant_vector(1, contravariant_vectors, ii, j, k, element))
        fluxtilde1 = volume_flux(u_node, get_node_vars(u, equations, dg, ii, j, k, element), Ja1_avg, equations)
        du_local = du_local + derivative_split[i, ii] * fluxtilde1
    end
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    for jj in 1:NNODES
        Ja2_avg = 0.5f0 * (Ja2_node + get_contravariant_vector(2, contravariant_vectors, i, jj, k, element))
        fluxtilde2 = volume_flux(u_node, get_node_vars(u, equations, dg, i, jj, k, element), Ja2_avg, equations)
        du_local = du_local + derivative_split[j, jj] * fluxtilde2
    end
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    for kk in 1:NNODES
        Ja3_avg = 0.5f0 * (Ja3_node + get_contravariant_vector(3, contravariant_vectors, i, j, kk, element))
        fluxtilde3 = volume_flux(u_node, get_node_vars(u, equations, dg, i, j, kk, element), Ja3_avg, equations)
        du_local = du_local + derivative_split[k, kk] * fluxtilde3
    end
    set_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

# E11's copy with the three node loops unrolled at compile time
@inline function fsg_unroll_body!(du, u, equations, dg, volume_flux, ::Val{NNODES}, ::Val{NVARIABLES},
                                  derivative_split, contravariant_vectors, i, j, k, element) where {NNODES, NVARIABLES}
    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    du1 = unrolled_sum(Val(NNODES)) do ii
        Ja1_avg = 0.5f0 * (Ja1_node + get_contravariant_vector(1, contravariant_vectors, ii, j, k, element))
        derivative_split[i, ii] * volume_flux(u_node, get_node_vars(u, equations, dg, ii, j, k, element),
                                              Ja1_avg, equations)
    end
    du2 = unrolled_sum(Val(NNODES)) do jj
        Ja2_avg = 0.5f0 * (Ja2_node + get_contravariant_vector(2, contravariant_vectors, i, jj, k, element))
        derivative_split[j, jj] * volume_flux(u_node, get_node_vars(u, equations, dg, i, jj, k, element),
                                              Ja2_avg, equations)
    end
    du3 = unrolled_sum(Val(NNODES)) do kk
        Ja3_avg = 0.5f0 * (Ja3_node + get_contravariant_vector(3, contravariant_vectors, i, j, kk, element))
        derivative_split[k, kk] * volume_flux(u_node, get_node_vars(u, equations, dg, i, j, kk, element),
                                              Ja3_avg, equations)
    end
    set_node_vars!(du, du1 + du2 + du3, equations, dg, i, j, k, element)
end

for (body, name, unsafe_name) in ((:fsg_orig_body!, :fsg_orig!, :fsg_orig_unsafe!),
                                  (:fsg_unroll_body!, :fsg_unroll!, :fsg_unroll_unsafe!))
    @eval begin
        @kernel function $name(du, u, equations, dg, volume_flux, nnodes, nvariables, derivative_split,
                               contravariant_vectors)
            i, j, k, element = @index(Global, NTuple)
            $body(du, u, equations, dg, volume_flux, nnodes, nvariables, derivative_split, contravariant_vectors,
                  i, j, k, element)
        end
        # no bounds check (the workgroup divides the ndrange)
        @kernel unsafe_indices = true function $unsafe_name(du, u, equations, dg, volume_flux, nnodes, nvariables,
                                                            derivative_split, contravariant_vectors)
            i, j, k, element = @index(Global, NTuple)
            $body(du, u, equations, dg, volume_flux, nnodes, nvariables, derivative_split, contravariant_vectors,
                  i, j, k, element)
        end
    end
end

# time of one call in ms: CUDA events around REPS calls after a warm-up
function gputime(f)
    f()
    CUDA.synchronize()
    return CUDA.@elapsed(for _ in 1:REPS
                             f()
                         end) / REPS * 1e3
end

function gpu_ode(vol)
    volume_integral = vol == "weak" ? VolumeIntegralWeakForm() : VolumeIntegralFluxDifferencing(flux_kennedy_gruber)
    solver = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive), volume_integral = volume_integral)
    mod = Module()
    Base.eval(mod, :(using Trixi, CUDA, OrdinaryDiffEqLowStorageRK))
    redirect_stdout(devnull) do
        trixi_include(mod, joinpath(examples_dir(), "p4est_3d_dgsem", "elixir_euler_source_terms_nonperiodic.jl");
                      trees_per_dimension = (T, T, T), initial_refinement_level = 1, solver = solver,
                      sol = nothing, real_type = Float32, storage_type = CuArray)
    end
    return Base.invokelatest(getglobal, mod, :ode)
end

function main()
    println("CSV,what,variant,elements,ms,maxdiff")
    flush(stdout)
    for vol in ("weak", "fd")
        ode = gpu_ode(vol)
        semi = ode.p
        u, du = copy(ode.u0), similar(ode.u0)
        E = nelements(semi.solver, semi.cache)
        t = gputime(() -> ode.f(du, u, semi, 0.0f0))
        @printf("rhs!    %-4s %d elements: %8.3f ms  (Trixi native CUDA, default kernels)\n", vol, E, t)
        println("CSV,rhs,$vol,$E,", round(t, digits = 4), ",")
        flush(stdout)
        vol == "fd" || continue
        # `wrap_array` is an unsafe wrapper: keep u and du alive while it is used
        GC.@preserve u du volume_kernels(semi, u, du, E)
    end
end

function volume_kernels(semi, u, du, E)
    mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    U, DU = Trixi.wrap_array(u, semi), Trixi.wrap_array(du, semi)
    backend = KernelAbstractions.get_backend(U)
    vi = dg.volume_integral
    ref = nothing
    for kt in (Trixi.HalfSweep(), Trixi.FullSweep(), Trixi.FullSweepGlobal())
        c = merge(cache, (; flux_differencing_kernel = kt))
        t = gputime(() -> Trixi.calc_volume_integral!(backend, DU, U, mesh, Trixi.have_nonconservative_terms(equations),
                                                      equations, vi, dg, c))
        r = Array(DU)
        ref === nothing && (ref = r)
        d = maximum(abs, r .- ref) / maximum(abs, ref)
        @printf("volume  %-28s %8.3f ms  max|du - du_HalfSweep| %.1e\n", "Trixi " * string(nameof(typeof(kt))), t, d)
        println("CSV,volume,Trixi_", nameof(typeof(kt)), ",$E,", round(t, digits = 4), ",", d)
        flush(stdout)
    end
    n = nnodes(dg)
    args = (DU, U, equations, dg, vi.volume_flux, Val(n), Val(nvariables(equations)), dg.basis.derivative_split,
            cache.elements.contravariant_vectors)
    for (name, kern, wgs) in (("orig, dynamic workgroup", fsg_orig!, nothing),
                              ("orig, (4,4,4,1)", fsg_orig!, (n, n, n, 1)),
                              ("unroll, dynamic workgroup", fsg_unroll!, nothing),
                              ("unroll, (4,4,4,1)", fsg_unroll!, (n, n, n, 1)),
                              ("unroll, (4,4,4,4)", fsg_unroll!, (n, n, n, 4)),
                              ("unroll, unsafe_indices (4,4,4,1)", fsg_unroll_unsafe!, (n, n, n, 1)),
                              ("unroll, unsafe_indices (4,4,4,4)", fsg_unroll_unsafe!, (n, n, n, 4)))
        k = wgs === nothing ? kern(backend) : kern(backend, wgs)
        t = gputime(() -> k(args...; ndrange = (n, n, n, E)))
        d = maximum(abs, Array(DU) .- ref) / maximum(abs, ref)
        @printf("volume  %-28s %8.3f ms  max|du - du_HalfSweep| %.1e\n", "FSG " * name, t, d)
        println("CSV,volume,FSG ", name, ",$E,", round(t, digits = 4), ",", d)
        flush(stdout)
    end
end

main()
