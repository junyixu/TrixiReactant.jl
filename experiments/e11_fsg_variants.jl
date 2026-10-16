# E11: can Trixi's 3D flux-differencing kernel `FullSweepGlobal` (no shared memory) go through
# Reactant with raising? E4 found two blockers:
#   1. a segfault in Reactant's `AffineCFGPass` (`OptimizeRem` on `arith.remui`) — it goes away
#      when the kernel is launched with a static workgroup size (E4_WGS=4,4,4,1);
#   2. then "cannot raise op to stablehlo: scf.while" — the node loops `for ii in 1:NNODES`.
# This script records the launch with the E4 harness (CPU reference from Trixi's own kernel) and
# replays copies of the kernel that differ only in how the three node loops are written:
#   orig    the loops as in Trixi 0.17.14 (sanity check: should fail like Trixi's kernel)
#   unroll  the loops unrolled at compile time (NNODES is a `Val`)
#   while   `while ii <= NNODES` with an integer counter
#   <variant>_unsafe   the same body, but `@kernel unsafe_indices = true`: KA then emits
#           no bounds check at all (instead of E4_UNROLL_IN=1). `@index(Global)` is still computed
#           from block and thread index, so this is only correct when the workgroup size divides
#           the ndrange (here (4,4,4,1) and (4,4,4,nelements)).
# E11_TREES=n replays on an n×n×n-tree mesh (8n³ elements) instead of the elixir's 2×2×2 (64).
# usage (TrixiDocker image): E4_WGS=4,4,4,1 julia experiments/e11_fsg_variants.jl [cpu|gpu|tpu] raise <variant>
include(joinpath(@__DIR__, "e4_kernel_inventory.jl"))   # harness only; its cases do not run
using Trixi: get_node_vars, get_contravariant_vector, set_node_vars!

const VARIANT = get(ARGS, 3, "unroll")

# sum of f(1) + ... + f(N), unrolled when the method is compiled
@inline unrolled_sum(f, ::Val{1}) = f(1)
@inline unrolled_sum(f, ::Val{N}) where {N} = unrolled_sum(f, Val(N - 1)) + f(N)

# The kernel bodies, as in Trixi 0.17.14 apart from the node loops. Each variant below is a
# kernel with the same signature as Trixi's, so the recorded arguments replay unchanged.
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

@inline function fsg_while_body!(du, u, equations, dg, volume_flux, ::Val{NNODES}, ::Val{NVARIABLES},
                                 derivative_split, contravariant_vectors, i, j, k, element) where {NNODES, NVARIABLES}
    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    du_local = zero(SVector{NVARIABLES, eltype(du)})
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    m = 1
    while m <= NNODES
        Ja1_avg = 0.5f0 * (Ja1_node + get_contravariant_vector(1, contravariant_vectors, m, j, k, element))
        du_local = du_local + derivative_split[i, m] *
                              volume_flux(u_node, get_node_vars(u, equations, dg, m, j, k, element), Ja1_avg, equations)
        Ja2_avg = 0.5f0 * (Ja2_node + get_contravariant_vector(2, contravariant_vectors, i, m, k, element))
        du_local = du_local + derivative_split[j, m] *
                              volume_flux(u_node, get_node_vars(u, equations, dg, i, m, k, element), Ja2_avg, equations)
        Ja3_avg = 0.5f0 * (Ja3_node + get_contravariant_vector(3, contravariant_vectors, i, j, m, element))
        du_local = du_local + derivative_split[k, m] *
                              volume_flux(u_node, get_node_vars(u, equations, dg, i, j, m, element), Ja3_avg, equations)
        m += 1
    end
    set_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

# one kernel with KA's bounds check and one without (`unsafe_indices = true`) per body
for (body, name, unsafe_name) in ((:fsg_orig_body!, :fsg_orig!, :fsg_orig_unsafe!),
                                  (:fsg_unroll_body!, :fsg_unroll!, :fsg_unroll_unsafe!),
                                  (:fsg_while_body!, :fsg_while!, :fsg_while_unsafe!))
    @eval begin
        @kernel function $name(du, u, equations, MeshT, kernel_type, have_nonconservative_terms, combine_fluxes,
                               dg, volume_flux, nnodes, nvariables, derivative_split, contravariant_vectors)
            i, j, k, element = @index(Global, NTuple)
            $body(du, u, equations, dg, volume_flux, nnodes, nvariables, derivative_split, contravariant_vectors,
                  i, j, k, element)
        end
        @kernel unsafe_indices = true function $unsafe_name(du, u, equations, MeshT, kernel_type,
                                                            have_nonconservative_terms, combine_fluxes, dg,
                                                            volume_flux, nnodes, nvariables, derivative_split,
                                                            contravariant_vectors)
            i, j, k, element = @index(Global, NTuple)
            $body(du, u, equations, dg, volume_flux, nnodes, nvariables, derivative_split, contravariant_vectors,
                  i, j, k, element)
        end
    end
end

const VARIANTS = Dict(v => getglobal(@__MODULE__, Symbol("gpu_fsg_", v, "!"))
                      for v in ("orig", "unroll", "while", "orig_unsafe", "unroll_unsafe", "while_unsafe"))
const TREES = parse(Int, get(ENV, "E11_TREES", "2"))

function record_fsg()
    name, path, kw = CASES[findfirst(c -> c[1] == "3d_euler_fd_fullsweepglobal", CASES)]
    mod = Module()
    Base.eval(mod, :(using Trixi, OrdinaryDiffEqLowStorageRK, OrdinaryDiffEqSSPRK))
    trixi_include(mod, path; sol = nothing, real_type = Float32, kw..., trees_per_dimension = (TREES, TREES, TREES))
    Base.invokelatest(record_rhs!, mod)
    return only(filter(L -> kernel_name(L.gpu_f) == "flux_differencing_KAkernel!", LAUNCHES))
end

println("backend=$BACKEND raise=$RAISE variant=$VARIANT trees=$(TREES)^3 workgroupsize=",
        isempty(WGS) ? "recorded (dynamic)" : "($WGS) as $WGS_MODE", " ndrange=$NDRANGE_MODE")
L = record_fsg()
println("recorded: ndrange=", L.ndrange, " reference from ", L.ref)
L2 = merge(L, (; gpu_f = VARIANTS[VARIANT]))
ghost_type_args!(L2.before)
r = try
    Base.invokelatest(check_launch, L2)
catch err
    println("FAIL ", firstline(err))
    foreach(l -> println("    | ", first(l, 200)), split(sprint(showerror, err), '\n')[2:min(end, 12)])
    exit(1)
end
status = !(r.err <= 1e-4) ? "MISMATCH" : (r.leftover > 0 ? "ok (kernel kept)" : "ok (raised)")
@printf("RESULT %s %s %s: %s err=%.1e compile=%.0fs run=%.3fms gather=%d scatter=%d\n", VARIANT,
        isempty(WGS) ? "dynamic" : WGS, NDRANGE_MODE, status, r.err, r.compile, r.run_ms, r.gathers, r.scatters)
foreach(d -> println("    ", d), r.details)
