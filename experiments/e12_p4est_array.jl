# E12: DGSEM on a 3D P4estMesh as array operations (the array route approved on 2026-10-13).
# Unlike E7/E9 (Cartesian, neighbours by slicing), this takes any conforming P4estMesh from a
# Trixi semidiscretization: curved elements (contravariant vectors, inverse Jacobian from
# Trixi's cache), unstructured connectivity, Dirichlet boundaries. The physics is Trixi's own:
# fluxes, two-point volume fluxes, boundary values and source terms are Trixi's pointwise
# functions, applied node by node by broadcasting over arrays (Reactant traces the scalar
# function once). The DG operators are array operations:
#   volume   weak form (derivative_hat along each node dimension of the contravariant fluxes) or
#            flux differencing (all node pairs along each dimension, averaged metric terms)
#   surface  per element face ("slot", 6 per element): own face values by slicing; the
#            neighbour's face values by ONE gather of whole faces (all variables, n×n nodes)
#            from [all faces; Dirichlet boundary values], then one of the 8 face orientations;
#            the numerical flux per slot with the outward normal of Trixi's primary element.
#            Each interface flux is thus evaluated from both sides (no scatter needed).
# No mortars (no AMR / nonconforming meshes) yet. State: one (n, n, n, nelements) array per
# variable; Trixi's element order.
# Cases (as in test/test_cuda_3d.jl, plus a curved unstructured mesh):
#   cart<T>   p4est_3d_dgsem/elixir_euler_source_terms_nonperiodic.jl with T×T×T trees (8T³ elements)
#   hohq      p4est_3d_dgsem/elixir_euler_source_terms_nonperiodic_hohqmesh.jl (curved, polydeg 4)
# Tasks:
#   check [cases] [volumes]                rhs! vs Trixi's rhs! (Float64 reference on the CPU)
#   bench [cases] [volumes] [RealT]        time per rhs! / RK step, compile times
#   profile [case] [volumes]               `Reactant.@profile` of one rhs! + optimized XLA HLO in logs/
# E12_ABLATE=nogather replaces the neighbour gather by the own faces (wrong results; measures
# what the gather costs).
# usage (TrixiDocker image): julia experiments/e12_p4est_array.jl <tpu|gpu|cpu> <task> [args...]
using Reactant, Trixi, Printf, LinearAlgebra, StaticArrays
using Trixi: nnodes, nvariables, nelements, indices2direction, get_normal_direction,
             index_to_start_step_3d, surface_indices

const BACKEND = get(ARGS, 1, "tpu")
const TASK = get(ARGS, 2, "check")
Reactant.set_default_backend(BACKEND)
arg(i, default) = get(ARGS, i + 2, default)
parse_list(s) = split(s, ',')
const ABLATE = get(ENV, "E12_ABLATE", "")
# optimizations (cumulative steps, see notes/2026-10-15-e12-p4est-array.md):
#   dim     derivative along a node dimension as n slice terms instead of transpose + matmul
#   fdloop  flux differencing as a sum over ii of D_split[:, ii] .* f(u, u_ii) (no node-pair arrays)
#   an      Trixi's pointwise functions run ONCE on whole arrays (array-valued numbers `AN`) and
#           return all components, instead of one broadcast per component
#   step    RK stages use the rhs! result directly (no `dU` buffer carried through the time loop),
#           so its last operations can fuse with the stage update
#   trees   uniformly refined meshes: every tree is an (m, m, m) block of elements, state
#           (n, n, n, m, m, m, ntrees); faces inside a tree from slices as with `faces`, faces between
#           trees by ONE gather of whole tree faces ((n m)^2 nodes x all variables per index) plus the
#           tree face's orientation; works for unstructured meshes (overrides `faces`)
#   faces   brick-shaped meshes (all face orientations the identity, no periodicity): elements in
#           a global (Nx, Ny, Nz) order found from the face connectivity, state (n, n, n, Nx, Ny, Nz),
#           face fluxes straight from slices, boundary faces added by padding (no gather, no concatenation)
const OPTS = Set(split(get(ENV, "E12_OPTS", ""), ','; keepempty = false))
has(o) = o in OPTS
optstr() = isempty(OPTS) ? "base" : join(sort(collect(OPTS)), "+")

# Trixi's wave speed estimates call `norm(normal_direction)` on an SVector of traced scalars;
# Reactant's overlay of `norm` sends any array holding traced values to an XLA dot product,
# which fails for an SVector, so give its fallback a method for static vectors
Reactant.TracedLinearAlgebra.overloaded_norm(v::StaticVector, p::Real = 2) =
    p == 2 ? sqrt(sum(abs2, v)) : sum(x -> abs(x)^p, v)^(1 / p)
# Trixi's own `sqrt`/`log` (NaN for negative arguments, which XLA's give anyway) are defined
# for `Real`, and a traced scalar is not one
Trixi.sqrt(x::Reactant.TracedRNumber) = Base.sqrt(x)
Trixi.log(x::Reactant.TracedRNumber) = Base.log(x)

# ---------------------------------------------------------------- cases
ex(e) = joinpath(examples_dir(), "p4est_3d_dgsem", e)
volume_integral(vol) = vol == "weak" ? VolumeIntegralWeakForm() :
                       VolumeIntegralFluxDifferencing(flux_kennedy_gruber)
function case_elixir(case, vol)
    if startswith(case, "cart")
        # cart<T> (T×T×T trees), cart<A>x<B>x<C>, optionally _<L>: initial refinement level L (default 1)
        spec, level = occursin('_', case) ? (split(case[5:end], '_')[1], parse(Int, split(case, '_')[2])) :
                      (case[5:end], 1)
        T = occursin('x', spec) ? Tuple(parse.(Int, split(spec, 'x'))) : ntuple(_ -> parse(Int, spec), 3)
        return ex("elixir_euler_source_terms_nonperiodic.jl"),
               (; trees_per_dimension = T, initial_refinement_level = level, volume_integral = volume_integral(vol))
    elseif startswith(case, "hohq")   # hohq or hohq_<L> (refined L times)
        level = occursin('_', case) ? parse(Int, split(case, '_')[2]) : 0
        file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/11461efbfb02c42e06aca338b3d0b645/raw/81deeb1ebc4945952c30af5bb75fe222a18d975c/abaqus_half_circle_3d.inp",
                              joinpath(tempdir(), "abaqus_half_circle_3d.inp"))
        return ex("elixir_euler_source_terms_nonperiodic_hohqmesh.jl"),
               (; mesh = P4estMesh{3}(file; initial_refinement_level = level),
                solver = DGSEM(polydeg = 4, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
                               volume_integral = volume_integral(vol)))
    end
    error("unknown case $case")
end

# Float64 semidiscretization on the CPU, as the elixir sets it up (no time integration)
function trixi_ode(case, vol)
    path, kw = case_elixir(case, vol)
    mod = Module()
    Base.eval(mod, :(using Trixi, OrdinaryDiffEqLowStorageRK))
    redirect_stdout(devnull) do
        trixi_include(mod, path; sol = nothing, kw...)
    end
    return Base.invokelatest(getglobal, mod, :ode)
end

# Trixi's rhs! (`ode.f`) at the exact solution at time t; arrays (NV, n, n, n, nelements)
function trixi_rhs(ode, t)
    semi = ode.p
    u = Trixi.compute_coefficients(t, semi)
    du = similar(u)
    ode.f(du, u, semi, t)
    # `wrap_array` is an `unsafe_wrap`: keep the vectors alive while copying
    return GC.@preserve du u (copy(Trixi.wrap_array(du, semi)), copy(Trixi.wrap_array(u, semi)))
end

# the same equations with parameters of type RealT, so Float32 runs stay in Float32
# (`similar(equations, RealT)` is the rule `trixi_adapt` uses)
real_equations(eq::CompressibleEulerEquations3D, RealT) = CompressibleEulerEquations3D(RealT(eq.gamma))
real_equations(eq, RealT) = similar(eq, RealT)

# ---------------------------------------------------------------- mesh → arrays (CPU, once)
# Volume node of face `d` (Trixi's direction 1..6 = -x, +x, -y, +y, -z, +z) at the surface
# indices (l, m) used by Trixi's `surface_flux_values` and `calc_surface_integral!`.
face_node(d, l, m, n) = d == 1 ? (1, l, m) : d == 2 ? (n, l, m) : d == 3 ? (l, 1, m) :
                        d == 4 ? (l, n, m) : d == 5 ? (l, m, 1) : (l, m, n)

# the 8 ways the n×n nodes of two faces can be matched: neighbour value at (l, m) is the
# neighbour face's node ORIENTS[o](l, m, n)
const ORIENTS = ((l, m, n) -> (l, m), (l, m, n) -> (m, l), (l, m, n) -> (n + 1 - l, m),
                 (l, m, n) -> (l, n + 1 - m), (l, m, n) -> (n + 1 - l, n + 1 - m),
                 (l, m, n) -> (m, n + 1 - l), (l, m, n) -> (n + 1 - m, l),
                 (l, m, n) -> (n + 1 - m, n + 1 - l))

rev(G::Array, dims) = reverse(G; dims)
rev(G, dims) = Reactant.Ops.reverse(Reactant.TracedUtils.materialize_traced_array(G); dimensions = collect(dims))
tr12(G) = permutedims(G, (2, 1, 3, 4))
# out[l, m, :, :] = G[ORIENTS[o](l, m, n)..., :, :]
orient(G, o) = o == 1 ? G : o == 2 ? tr12(G) : o == 3 ? rev(G, (1,)) : o == 4 ? rev(G, (2,)) :
               o == 5 ? rev(G, (1, 2)) : o == 6 ? rev(tr12(G), (1,)) : o == 7 ? rev(tr12(G), (2,)) :
               rev(tr12(G), (1, 2))

function orient_code(map_lm, n)
    for (o, f) in enumerate(ORIENTS)
        all(map_lm[l, m] == f(l, m, n) for l in 1:n, m in 1:n) && return o
    end
    error("face matching is not one of the 8 orientations")
end

# node positions visited by Trixi's "delayed indexing" loops for an index tuple, in loop order
function visited_nodes(indices, n)
    s = map(ix -> index_to_start_step_3d(ix, 1:n), indices)
    pos = [s[1][1], s[2][1], s[3][1]]
    out = Matrix{NTuple{3, Int}}(undef, n, n)
    for j in 1:n
        for i in 1:n
            out[i, j] = Tuple(pos)
            pos .+= (s[1][2], s[2][2], s[3][2])
        end
        pos .+= (s[1][3], s[2][3], s[3][3])
    end
    return out
end

function visited_surface(surface_idx, n)
    s = map(ix -> index_to_start_step_3d(ix, 1:n), surface_idx)
    pos = [s[1][1], s[2][1]]
    out = Matrix{NTuple{2, Int}}(undef, n, n)
    for j in 1:n
        for i in 1:n
            out[i, j] = Tuple(pos)
            pos .+= (s[1][2], s[2][2])
        end
        pos .+= (s[1][3], s[2][3])
    end
    return out
end

function mesh_arrays(semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    @assert size(cache.mortars.neighbor_ids, 2) == 0 "mortars (nonconforming meshes) are not supported yet"
    n, E = nnodes(solver), nelements(solver, cache)
    Ja = cache.elements.contravariant_vectors           # Ja[j, i, ...]: component j of Ja^i
    X = cache.elements.node_coordinates
    S = 6 * E
    slot(d, e) = d + 6 * (e - 1)
    src = zeros(Int, S)          # gather source of each slot: a slot, or S + boundary block
    code = zeros(Int, S)         # orientation of the neighbour face
    nrm = zeros(3, n, n, S)      # outward normal (Trixi's primary element) at the slot nodes
    @unpack neighbor_ids, node_indices = cache.interfaces
    for f in axes(neighbor_ids, 2)
        ep, es = neighbor_ids[1, f], neighbor_ids[2, f]
        pidx, sidx = node_indices[1, f], node_indices[2, f]
        dp, ds = indices2direction(pidx), indices2direction(sidx)
        pvol, svol = visited_nodes(pidx, n), visited_nodes(sidx, n)
        ssurf = visited_surface(surface_indices(sidx), n)
        inv_map = Matrix{NTuple{2, Int}}(undef, n, n)
        for j in 1:n, i in 1:n
            # interface node (i, j) = primary surface node (i, j) = secondary surface node ssurf[i, j]
            @assert pvol[i, j] == face_node(dp, i, j, n)
            @assert svol[i, j] == face_node(ds, ssurf[i, j]..., n)
            normal = get_normal_direction(dp, Ja, pvol[i, j]..., ep)
            nrm[:, i, j, slot(dp, ep)] .= normal
            nrm[:, ssurf[i, j]..., slot(ds, es)] .= .-normal
            inv_map[ssurf[i, j]...] = (i, j)
        end
        src[slot(dp, ep)], code[slot(dp, ep)] = slot(ds, es), orient_code(ssurf, n)
        src[slot(ds, es)], code[slot(ds, es)] = slot(dp, ep), orient_code(inv_map, n)
    end
    # boundaries, grouped by boundary condition (Dirichlet only)
    bgroups = []
    nb = 0
    bcs = semi.boundary_conditions
    groups = bcs isa Trixi.UnstructuredSortedBoundaryTypes ?
             zip(bcs.boundary_condition_types, bcs.boundary_indices) :
             bcs isa BoundaryConditionDirichlet ? ((bcs, 1:length(cache.boundaries.neighbor_ids)),) : ()
    begin
        for (bc, ids) in groups
            @assert bc isa BoundaryConditionDirichlet "only Dirichlet boundaries are supported, got $(typeof(bc))"
            xb = zeros(3, n, n, length(ids))
            for (k, b) in enumerate(ids)
                e, idx = cache.boundaries.neighbor_ids[b], cache.boundaries.node_indices[b]
                d = indices2direction(idx)
                vol = visited_nodes(idx, n)
                for j in 1:n, i in 1:n
                    @assert vol[i, j] == face_node(d, i, j, n)
                    nrm[:, i, j, slot(d, e)] .= get_normal_direction(d, Ja, vol[i, j]..., e)
                    xb[:, i, j, k] .= X[:, vol[i, j]..., e]
                end
                src[slot(d, e)], code[slot(d, e)] = S + nb + k, 1
            end
            push!(bgroups, (; fn = bc.boundary_value_function, xb))
            nb += length(ids)
        end
    end
    @assert all(>(0), src) "some element faces are neither interfaces nor boundaries"
    return (; n, E, S, nb, src, code, nrm, bgroups, mesh, equations, solver, cache,
            ninterfaces = size(neighbor_ids, 2))
end

# ---------------------------------------------------------------- arrays for the rhs!
# Static settings of the rhs!. The equations and orientation codes are type parameters: a
# traced loop (`@trace for`) would otherwise trace the numbers in them.
struct Config{EQ, OR, VOL, SF, VF, ST, BF}
    surface_flux::SF
    volume_flux::VF
    source_terms::ST
    boundary_fns::BF      # one boundary value function per Dirichlet group
end
equations(::Config{EQ}) where {EQ} = EQ
orients(::Config{EQ, OR}) where {EQ, OR} = OR   # orientation codes other than the identity that occur

function setup(semi, vol, RealT)
    m = mesh_arrays(semi)
    @unpack n, E, S, solver, cache = m
    basis = solver.basis
    Ja = cache.elements.contravariant_vectors
    unit(d, i) = (v = zeros(RealT, ntuple(k -> k == d ? n : 1, 4)); v[ntuple(k -> k == d ? i : 1, 4)...] = 1; v)
    ds(d) = reshape(RealT.(basis.derivative_split), ntuple(k -> k == d || k == d + 1 ? n : 1, 5))
    orients = Tuple(o for o in 2:8 if o in m.code)
    p = (; Ja = ntuple(i -> ntuple(j -> RealT.(Ja[j, i, :, :, :, :]), 3), 3),
         invJ = RealT.(cache.elements.inverse_jacobian),
         X = ntuple(j -> RealT.(cache.elements.node_coordinates[j, :, :, :, :]), 3),
         Xb = Tuple(ntuple(j -> RealT.(g.xb[j, :, :, :]), 3) for g in m.bgroups),
         N = ntuple(j -> RealT.(m.nrm[j, :, :, :]), 3),
         src = Int32.(m.src),
         masks = Tuple(reshape(m.code .== o, 1, 1, 1, S) for o in orients),
         dhat = RealT.(basis.derivative_hat), ds = ntuple(ds, 3), dsplit = RealT.(basis.derivative_split),
         e1 = ntuple(d -> unit(d, 1), 3), en = ntuple(d -> unit(d, n), 3),
         factor = RealT(basis.inverse_weights[1]))
    surface_flux = solver.surface_integral.surface_flux
    volume_flux = vol == "weak" ? nothing : solver.volume_integral.volume_flux
    bfns = Tuple(g.fn for g in m.bgroups)
    cfg = Config{real_equations(m.equations, RealT), orients, Symbol(vol), typeof(surface_flux),
                 typeof(volume_flux), typeof(semi.source_terms), typeof(bfns)}(surface_flux, volume_flux,
                                                                               semi.source_terms, bfns)
    return p, cfg, m
end

# ---------------------------------------------------------------- pointwise physics
# Component C of a Trixi pointwise function, called node by node on scalars. C is a type
# parameter: Reactant traces what a broadcast function captures, and a traced integer could
# not index the SVector.
struct Component{C, F}
    f::F
end
component(f, c) = Component{c, typeof(f)}(f)
(p::Component{C})(xs...) where {C} = p.f(xs...)[C]
pointwise(f, nout, arrays...) = has("an") ? pointwise_an(f, nout, arrays...) :
                                ntuple(c -> broadcast(component(f, c), arrays...), nout)

# Array-valued numbers (E12_OPTS=an): an `AN{T}` wraps an array (or a constant) and makes scalar
# arithmetic elementwise, so a Trixi pointwise function called with `AN`s returns an SVector of
# `AN`s holding all components at once (with shared subexpressions computed once). Real constants
# are converted to T, so Float32 stays Float32.
struct AN{T} <: Number
    a::Any
end
Base.promote_rule(::Type{AN{T}}, ::Type{<:Real}) where {T} = AN{T}
Base.convert(::Type{AN{T}}, x::Real) where {T} = AN{T}(T(x))
Base.convert(::Type{AN{T}}, x::AN{T}) where {T} = x
Base.zero(::Type{AN{T}}) where {T} = AN{T}(zero(T))
Base.one(::Type{AN{T}}) where {T} = AN{T}(one(T))
Base.zero(::AN{T}) where {T} = zero(AN{T})
Base.one(::AN{T}) where {T} = one(AN{T})
for op in (:+, :-, :*, :/, :max, :min)
    @eval Base.$op(x::AN{T}, y::AN{T}) where {T} = AN{T}(broadcast($op, x.a, y.a))
    @eval Base.$op(x::AN{T}, y::Real) where {T} = AN{T}(broadcast($op, x.a, T(y)))
    @eval Base.$op(x::Real, y::AN{T}) where {T} = AN{T}(broadcast($op, T(x), y.a))
end
Base.:^(x::AN{T}, y::Integer) where {T} = AN{T}(broadcast(^, x.a, y))
Base.:^(x::AN{T}, y::Real) where {T} = AN{T}(broadcast(^, x.a, T(y)))
Base.literal_pow(::typeof(^), x::AN, ::Val{2}) = x * x
for f in (:-, :sqrt, :abs, :abs2, :inv, :sin, :cos, :exp, :log)
    @eval Base.$f(x::AN{T}) where {T} = AN{T}(broadcast($f, x.a))
end
Base.sincos(x::AN) = (sin(x), cos(x))
Base.muladd(x::AN, y::AN, z::AN) = x * y + z
Trixi.sqrt(x::AN) = sqrt(x)
Trixi.log(x::AN) = log(x)
LinearAlgebra.norm(v::StaticVector{N, <:AN}) where {N} = sqrt(sum(abs2, v))

function pointwise_an(f, nout, arrays...)
    T = Reactant.unwrapped_eltype(first(arrays))
    r = f(map(a -> AN{T}(a), arrays)...)
    return ntuple(c -> r[c].a, nout)
end

@inline sv(xs, ::Val{a}, ::Val{b}) where {a, b} = SVector(ntuple(i -> xs[a + i], Val(b)))

struct CartesianFlux{O, EQ}; eq::EQ; end              # flux(u, orientation)
(f::CartesianFlux{O, EQ})(u...) where {O, EQ} = flux(SVector(u), O, f.eq)
struct NormalFlux{NV, F, EQ}; f::F; eq::EQ; end       # f(u_ll, u_rr, normal)
(f::NormalFlux{NV})(xs...) where {NV} = f.f(sv(xs, Val(0), Val(NV)), sv(xs, Val(NV), Val(NV)),
                                            sv(xs, Val(2NV), Val(3)), f.eq)
struct AvgNormalFlux{NV, F, EQ}; f::F; eq::EQ; end    # f(u_ll, u_rr, (Ja_ll + Ja_rr) / 2)
(f::AvgNormalFlux{NV})(xs...) where {NV} = f.f(sv(xs, Val(0), Val(NV)), sv(xs, Val(NV), Val(NV)),
                                               0.5f0 * (sv(xs, Val(2NV), Val(3)) + sv(xs, Val(2NV + 3), Val(3))),
                                               f.eq)
struct Source{NV, F, EQ}; f::F; eq::EQ; end           # source_terms(u, x, t)
(f::Source{NV})(xs...) where {NV} = f.f(sv(xs, Val(0), Val(NV)), sv(xs, Val(NV), Val(3)), xs[NV + 4], f.eq)
struct BoundaryValue{F, EQ}; f::F; eq::EQ; end        # boundary_value_function(x, t)
(f::BoundaryValue)(x1, x2, x3, t) = f.f(SVector(x1, x2, x3), t, f.eq)

# ---------------------------------------------------------------- operators
# contraction with an n×n matrix along node dimension d ∈ 1:3 of a 4D array
function apply_dim(M, F, d)
    d == 1 && return reshape(M * reshape(F, size(F, 1), :), size(F))
    perm = d == 2 ? (2, 1, 3, 4) : (3, 2, 1, 4)
    G = permutedims(F, perm)
    return permutedims(reshape(M * reshape(G, size(G, 1), :), size(G)), perm)
end

addt(a, b) = a === nothing ? b : map(+, a, b)

# slice k of node dimension d, kept with size 1 there (broadcasts against the full array)
nodeslice(F, d, k) = F[ntuple(m -> m == d ? (k:k) : Colon(), ndims(F))...]
# a vector placed along dimension d of an N-dimensional array
along(v, d, N) = reshape(v, ntuple(m -> m == d ? length(v) : 1, N))

# the same contraction as `apply_dim`, as n broadcast slice terms: elementwise, so XLA can fuse
# it with the flux computation, and no transposes
function apply_dim_unrolled(M, F, d)
    acc = along(M[:, 1], d, ndims(F)) .* nodeslice(F, d, 1)
    for k in 2:size(F, d)
        acc = acc .+ along(M[:, k], d, ndims(F)) .* nodeslice(F, d, k)
    end
    return acc
end

function volume_weak(p, cfg, U)
    NV = length(U)
    eq = equations(cfg)
    f = ntuple(o -> pointwise(CartesianFlux{o, typeof(eq)}(eq), NV, U...), 3)
    acc = nothing
    for d in 1:3
        Jd = p.Ja[d]
        ap = has("dim") ? apply_dim_unrolled : apply_dim
        term = ntuple(c -> ap(p.dhat, Jd[1] .* f[1][c] .+ Jd[2] .* f[2][c] .+ Jd[3] .* f[3][c], d), NV)
        acc = addt(acc, term)
    end
    return acc
end

# node pairs along node dimension d: left keeps its index at d, right moves it to d + 1
function volume_fd(p, cfg, U)
    NV, sz = length(U), size(U[1])
    eq = equations(cfg)
    vf = AvgNormalFlux{NV, typeof(cfg.volume_flux), typeof(eq)}(cfg.volume_flux, eq)
    acc = nothing
    for d in 1:3
        lsz = (sz[1:d]..., 1, sz[(d + 1):end]...)
        rsz = (sz[1:(d - 1)]..., 1, sz[d:end]...)
        L(q) = reshape(q, lsz)
        R(q) = reshape(q, rsz)
        F = pointwise(vf, NV, map(L, U)..., map(R, U)..., map(L, p.Ja[d])..., map(R, p.Ja[d])...)
        term = ntuple(c -> reshape(sum(p.ds[d] .* F[c]; dims = d + 1), sz), NV)
        acc = addt(acc, term)
    end
    return acc
end

# flux differencing as a sum over ii of D_split[:, ii] .* f(u, u_ii, (Ja + Ja_ii) / 2) along each
# node dimension: every term is elementwise on state-sized arrays, no node-pair arrays
function volume_fd_unrolled(p, cfg, U)
    NV = length(U)
    eq = equations(cfg)
    vf = AvgNormalFlux{NV, typeof(cfg.volume_flux), typeof(eq)}(cfg.volume_flux, eq)
    acc = nothing
    for d in 1:3, ii in 1:size(U[1], d)
        Uii = map(q -> nodeslice(q, d, ii), U)
        Jii = map(q -> nodeslice(q, d, ii), p.Ja[d])
        F = pointwise(vf, NV, U..., Uii..., p.Ja[d]..., Jii...)
        col = along(p.dsplit[:, ii], d, 4)
        acc = addt(acc, ntuple(c -> col .* F[c], NV))
    end
    return acc
end

# faces of all elements as one (n, n, NV, 6E) array: slot d + 6(e - 1)
function all_faces(U)
    n, E = size(U[1], 1), size(U[1], 4)
    fc(q, d) = reshape(d == 1 ? q[1, :, :, :] : d == 2 ? q[n, :, :, :] : d == 3 ? q[:, 1, :, :] :
                       d == 4 ? q[:, n, :, :] : d == 5 ? q[:, :, 1, :] : q[:, :, n, :], n, n, 1, 1, E)
    A = cat((cat((fc(q, d) for d in 1:6)...; dims = 4) for q in U)...; dims = 3)
    return reshape(A, n, n, length(U), 6E)
end

function rhs!(dU, U, p, t, cfg::Config{EQ, OR, vol, SF, VF, ST, BF}) where {EQ, OR, vol, SF, VF, ST, BF}
    NV, n, E = length(U), size(U[1], 1), size(U[1], 4)
    S = 6E
    V = vol === :weak ? volume_weak(p, cfg, U) : has("fdloop") ? volume_fd_unrolled(p, cfg, U) : volume_fd(p, cfg, U)
    # neighbour face values: gather from [faces; boundary values], then orient
    A = all_faces(U)
    blocks = Any[A]
    for (fn, Xb) in zip(cfg.boundary_fns, p.Xb)
        T = zero(Xb[1]) .+ t
        Ub = pointwise(BoundaryValue(fn, EQ), NV, Xb..., T)
        push!(blocks, reshape(cat(map(q -> reshape(q, n, n, 1, :), Ub)...; dims = 3), n, n, NV, :))
    end
    if ABLATE == "nogather"
        G = A
    else
        src = cat(blocks...; dims = 4)
        G = reshape(reshape(src, n * n * NV, :)[:, p.src], n, n, NV, S)
    end
    for (o, mask) in zip(OR, p.masks)
        G = ifelse.(mask, orient(G, o), G)
    end
    own = ntuple(c -> A[:, :, c, :], NV)
    nbr = ntuple(c -> G[:, :, c, :], NV)
    sf = NormalFlux{NV, SF, typeof(EQ)}(cfg.surface_flux, EQ)
    Fs = pointwise(sf, NV, own..., nbr..., p.N...)
    # surface integral: flux of face d onto its face nodes (outward normals, so all with +)
    surf = ntuple(NV) do c
        F = reshape(Fs[c], n, n, 6, E)
        fd(d) = F[:, :, d, :]
        (p.e1[1] .* reshape(fd(1), 1, n, n, E) .+ p.en[1] .* reshape(fd(2), 1, n, n, E) .+
         p.e1[2] .* reshape(fd(3), n, 1, n, E) .+ p.en[2] .* reshape(fd(4), n, 1, n, E) .+
         p.e1[3] .* reshape(fd(5), n, n, 1, E) .+ p.en[3] .* reshape(fd(6), n, n, 1, E))
    end
    T = zero(p.X[1]) .+ t
    src_terms = ST === Nothing ? nothing :
                pointwise(Source{NV, ST, typeof(EQ)}(cfg.source_terms, EQ), NV, U..., p.X..., T)
    du = ntuple(c -> src_terms === nothing ? .-p.invJ .* (V[c] .+ p.factor .* surf[c]) :
                     .-p.invJ .* (V[c] .+ p.factor .* surf[c]) .+ src_terms[c], NV)
    dU === nothing && return du      # E12_OPTS=step: the caller uses du directly
    foreach((d, x) -> d .= x, dU, du)
    return nothing
end


# ---------------------------------------------------------------- brick meshes (E12_OPTS=faces)
# Global element coordinates of a brick-shaped mesh from the face connectivity: the +x neighbour
# of an element is at ix + 1, etc. Works for curved bricks too. `nothing` if the mesh is not a
# brick (other face orientations, periodic wrap-around, holes).
function brick_order(m)
    all(==(1), m.code) || return nothing
    step = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    coord = fill((0, 0, 0), m.E)
    seen = falses(m.E)
    seen[1] = true
    queue = [1]
    while !isempty(queue)
        e = popfirst!(queue)
        for d in 1:6
            s = m.src[d + 6 * (e - 1)]
            s > m.S && continue                                    # boundary face
            d2, e2 = mod1(s, 6), (s - 1) ÷ 6 + 1
            d2 == (isodd(d) ? d + 1 : d - 1) || return nothing     # must meet the opposite face
            c = coord[e] .+ step[d]
            if seen[e2]
                coord[e2] == c || return nothing
            else
                coord[e2], seen[e2] = c, true
                push!(queue, e2)
            end
        end
    end
    all(seen) || return nothing
    lo = reduce((a, b) -> min.(a, b), coord)
    coord = [c .- lo .+ 1 for c in coord]
    Ns = reduce((a, b) -> max.(a, b), coord)
    prod(Ns) == m.E || return nothing
    perm = zeros(Int, Ns...)
    for (e, c) in enumerate(coord)
        perm[c...] = e
    end
    return perm
end

function setup_structured(semi, vol, RealT)
    m = mesh_arrays(semi)
    perm = brick_order(m)
    perm === nothing && error("E12_OPTS=faces needs a brick-shaped, non-periodic mesh")
    length(m.bgroups) == 1 || error("E12_OPTS=faces supports one Dirichlet boundary function")
    @unpack n, E, solver, cache = m
    basis = solver.basis
    Ns = size(perm)
    lex = vec(perm)                                   # Trixi element id of each lexicographic position
    six(A) = RealT.(reshape(A[:, :, :, lex], n, n, n, Ns...))
    Ja = cache.elements.contravariant_vectors
    # outward normals of the -a / +a faces, in the shape of a face slice of the state
    function face_normals(d)
        a = (d + 1) >> 1
        fsz = ntuple(k -> k == a ? 1 : n, 3)
        return ntuple(j -> RealT.(reshape(m.nrm[j, :, :, [d + 6 * (e - 1) for e in lex]], fsz..., Ns...)), 3)
    end
    unit(d, i) = (v = zeros(RealT, ntuple(k -> k == d ? n : 1, 4)); v[ntuple(k -> k == d ? i : 1, 4)...] = 1; v)
    ds(d) = reshape(RealT.(basis.derivative_split), ntuple(k -> k == d || k == d + 1 ? n : 1, 5))
    p = (; Ja = ntuple(i -> ntuple(j -> six(Ja[j, i, :, :, :, :]), 3), 3),
         invJ = six(cache.elements.inverse_jacobian),
         X = ntuple(j -> six(cache.elements.node_coordinates[j, :, :, :, :]), 3),
         Nminus = ntuple(a -> face_normals(2a - 1), 3), Nplus = ntuple(a -> face_normals(2a), 3),
         dhat = RealT.(basis.derivative_hat), ds = ntuple(ds, 3), dsplit = RealT.(basis.derivative_split),
         e1 = ntuple(d -> unit(d, 1), 3), en = ntuple(d -> unit(d, n), 3),
         factor = RealT(basis.inverse_weights[1]))
    surface_flux = solver.surface_integral.surface_flux
    volume_flux = vol == "weak" ? nothing : solver.volume_integral.volume_flux
    bfns = (m.bgroups[1].fn,)
    cfg = Config{real_equations(m.equations, RealT), (), Symbol(vol), typeof(surface_flux),
                 typeof(volume_flux), typeof(semi.source_terms), typeof(bfns)}(surface_flux, volume_flux,
                                                                               semi.source_terms, bfns)
    return p, cfg, merge(m, (; perm))
end

# zero padding of `x` along dimension `dim`: `lo` entries before, `hi` after
function padel(x, dim, lo, hi)
    xm = Reactant.TracedUtils.materialize_traced_array(x)
    T = Reactant.unwrapped_eltype(xm)
    return Reactant.Ops.pad(xm, Reactant.promote_to(Reactant.TracedRNumber{T}, zero(T));
                            low = [k == dim ? lo : 0 for k in 1:ndims(xm)],
                            high = [k == dim ? hi : 0 for k in 1:ndims(xm)])
end

function rhs_structured!(dU, U, p, t, cfg::Config{EQ, OR, vol, SF, VF, ST, BF}) where {EQ, OR, vol, SF, VF, ST, BF}
    NV, sz = length(U), size(U[1])
    n, Ns = sz[1], sz[4:6]
    sz4 = (n, n, n, prod(Ns))
    flat(q) = reshape(q, sz4)
    p4 = merge(p, (; Ja = map(J -> map(flat, J), p.Ja)))
    U4 = map(flat, U)
    V4 = vol === :weak ? volume_weak(p4, cfg, U4) : has("fdloop") ? volume_fd_unrolled(p4, cfg, U4) :
         volume_fd(p4, cfg, U4)
    V = map(v -> reshape(v, sz), V4)
    sf = NormalFlux{NV, SF, typeof(EQ)}(cfg.surface_flux, EQ)
    bv = BoundaryValue(cfg.boundary_fns[1], EQ)
    surf = nothing
    for a in 1:3
        ed, N = 3 + a, Ns[a]
        el(A, r) = A[ntuple(k -> k == ed ? r : Colon(), 6)...]
        uR = map(q -> nodeslice(q, a, n), U)                 # +a faces of all elements
        uL = map(q -> nodeslice(q, a, 1), U)                 # -a faces
        # interfaces between elements k and k + 1 along a, with the +a normal of element k
        Fint = pointwise(sf, NV, map(q -> el(q, 1:(N - 1)), uR)..., map(q -> el(q, 2:N), uL)...,
                         map(q -> el(q, 1:(N - 1)), p.Nplus[a])...)
        # Dirichlet values on the two boundary layers
        xR = map(q -> el(nodeslice(q, a, n), N:N), p.X)
        xL = map(q -> el(nodeslice(q, a, 1), 1:1), p.X)
        FR = pointwise(sf, NV, map(q -> el(q, N:N), uR)..., pointwise(bv, NV, xR..., zero(xR[1]) .+ t)...,
                       map(q -> el(q, N:N), p.Nplus[a])...)
        FL = pointwise(sf, NV, map(q -> el(q, 1:1), uL)..., pointwise(bv, NV, xL..., zero(xL[1]) .+ t)...,
                       map(q -> el(q, 1:1), p.Nminus[a])...)
        term = ntuple(NV) do c
            Gplus = padel(Fint[c], ed, 0, 1) .+ padel(FR[c], ed, N - 1, 0)      # flux out of the +a face
            Gminus = padel(FL[c], ed, 0, N - 1) .- padel(Fint[c], ed, 1, 0)     # flux out of the -a face
            p.en[a] .* Gplus .+ p.e1[a] .* Gminus
        end
        surf = addt(surf, term)
    end
    T = zero(p.X[1]) .+ t
    src_terms = ST === Nothing ? nothing :
                pointwise(Source{NV, ST, typeof(EQ)}(cfg.source_terms, EQ), NV, U..., p.X..., T)
    du = ntuple(c -> src_terms === nothing ? .-p.invJ .* (V[c] .+ p.factor .* surf[c]) :
                     .-p.invJ .* (V[c] .+ p.factor .* surf[c]) .+ src_terms[c], NV)
    dU === nothing && return du      # E12_OPTS=step: the caller uses du directly
    foreach((d, x) -> d .= x, dU, du)
    return nothing
end


# ---------------------------------------------------------------- tree blocks (E12_OPTS=trees)
opposite(d) = isodd(d) ? d + 1 : d - 1
# the two element (or node) axes spanning a face of direction d, in surface-index order
face_axes(d) = ((1, 2, 3)[[k for k in 1:3 if k != (d + 1) >> 1]]...,)

# Tree-local element coordinates of a uniformly refined mesh: tree t holds elements
# (t - 1) m^3 + 1 : t m^3 (Trixi's order), arranged by their neighbours inside the tree.
function tree_order(mm)
    T = size(mm.mesh.tree_node_coordinates)[end]
    m = round(Int, cbrt(mm.E / T))
    m^3 * T == mm.E || error("E12_OPTS=trees needs a uniformly refined mesh")
    tree(e) = (e - 1) ÷ m^3 + 1
    step = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    coord = fill((0, 0, 0), mm.E)
    for t in 1:T
        first_e = (t - 1) * m^3 + 1
        seen = Set([first_e])
        queue = [first_e]
        while !isempty(queue)
            e = popfirst!(queue)
            for d in 1:6
                s = mm.src[d + 6 * (e - 1)]
                s > mm.S && continue
                d2, e2 = mod1(s, 6), (s - 1) ÷ 6 + 1
                tree(e2) == t || continue
                @assert d2 == opposite(d) && mm.code[d + 6 * (e - 1)] == 1 "elements of one tree share its frame"
                c = coord[e] .+ step[d]
                if e2 in seen
                    @assert coord[e2] == c
                else
                    coord[e2] = c
                    push!(seen, e2)
                    push!(queue, e2)
                end
            end
        end
        @assert length(seen) == m^3
        es = first_e:(t * m^3)
        lo = reduce((a, b) -> min.(a, b), coord[es])
        foreach(e -> coord[e] = coord[e] .- lo .+ 1, es)
    end
    tperm = zeros(Int, m, m, m, T)
    for e in 1:mm.E
        tperm[coord[e]..., tree(e)] = e
    end
    @assert all(>(0), tperm)
    return tperm, coord, m, T
end

function setup_trees(semi, vol, RealT)
    mm = mesh_arrays(semi)
    length(mm.bgroups) <= 1 || error("E12_OPTS=trees supports one Dirichlet boundary function")
    tperm, coord, m, T = tree_order(mm)
    @unpack n, E, solver, cache = mm
    basis = solver.basis
    lex = vec(tperm)
    seven(A) = RealT.(reshape(A[:, :, :, lex], n, n, n, m, m, m, T))
    Ja = cache.elements.contravariant_vectors
    function face_normals(d)
        a = (d + 1) >> 1
        fsz = ntuple(k -> k == a ? 1 : n, 3)
        return ntuple(j -> RealT.(reshape(mm.nrm[j, :, :, [d + 6 * (e - 1) for e in lex]], fsz..., m, m, m, T)), 3)
    end
    # tree faces: slot d + 6(t - 1); neighbour tree face, orientation on the (n m)^2 grid, boundary flag
    nm = n * m
    tsrc, tcode, tbnd = collect(1:(6T)), ones(Int, 6T), falses(6T)
    Ntf = ntuple(_ -> ntuple(_ -> zeros(RealT, nm, nm, T), 3), 6)
    Xtf = ntuple(_ -> ntuple(_ -> zeros(RealT, nm, nm, T), 3), 6)
    X = cache.elements.node_coordinates
    for t in 1:T, d in 1:6
        a = (d + 1) >> 1
        b1, b2 = face_axes(d)
        layer = isodd(d) ? 1 : m
        nbr = nothing
        cmap = Matrix{NTuple{2, Int}}(undef, nm, nm)
        for A in 1:m, B in 1:m
            c = ntuple(k -> k == a ? layer : k == b1 ? A : B, 3)
            e = tperm[c..., t]
            sl = d + 6 * (e - 1)
            for q in 1:n, l in 1:n
                P = (l + n * (A - 1), q + n * (B - 1))
                for j in 1:3
                    Ntf[d][j][P..., t] = mm.nrm[j, l, q, sl]
                    Xtf[d][j][P..., t] = X[j, face_node(d, l, q, n)..., e]
                end
            end
            s = mm.src[sl]
            if s > mm.S
                tbnd[d + 6 * (t - 1)] = true
                continue
            end
            d2, e2 = mod1(s, 6), (s - 1) ÷ 6 + 1
            t2 = (e2 - 1) ÷ m^3 + 1
            @assert t2 != t "a tree face cannot meet its own tree"
            nbr === nothing && (nbr = (d2, t2))
            @assert nbr == (d2, t2) "a tree face meets exactly one tree face"
            c1, c2 = face_axes(d2)
            A2, B2 = coord[e2][c1], coord[e2][c2]
            for q in 1:n, l in 1:n
                l2, q2 = ORIENTS[mm.code[sl]](l, q, n)
                cmap[l + n * (A - 1), q + n * (B - 1)] = (l2 + n * (A2 - 1), q2 + n * (B2 - 1))
            end
        end
        if !tbnd[d + 6 * (t - 1)]
            tsrc[d + 6 * (t - 1)] = nbr[1] + 6 * (nbr[2] - 1)
            tcode[d + 6 * (t - 1)] = orient_code(cmap, nm)
        end
    end
    unit(d, i) = (v = zeros(RealT, ntuple(k -> k == d ? n : 1, 4)); v[ntuple(k -> k == d ? i : 1, 4)...] = 1; v)
    ds(d) = reshape(RealT.(basis.derivative_split), ntuple(k -> k == d || k == d + 1 ? n : 1, 5))
    orients = Tuple(o for o in 2:8 if o in tcode)
    p = (; Ja = ntuple(i -> ntuple(j -> seven(Ja[j, i, :, :, :, :]), 3), 3),
         invJ = seven(cache.elements.inverse_jacobian),
         X = ntuple(j -> seven(X[j, :, :, :, :]), 3),
         Nminus = ntuple(a -> face_normals(2a - 1), 3), Nplus = ntuple(a -> face_normals(2a), 3),
         Ntf, Xtf, tsrc = Int32.(tsrc),
         tmasks = Tuple(reshape(tcode .== o, 1, 1, 1, 6T) for o in orients),
         tbnd = ntuple(d -> reshape(tbnd[d:6:end], 1, 1, T), 6),
         dhat = RealT.(basis.derivative_hat), ds = ntuple(ds, 3), dsplit = RealT.(basis.derivative_split),
         e1 = ntuple(d -> unit(d, 1), 3), en = ntuple(d -> unit(d, n), 3),
         factor = RealT(basis.inverse_weights[1]))
    surface_flux = solver.surface_integral.surface_flux
    volume_flux = vol == "weak" ? nothing : solver.volume_integral.volume_flux
    bfns = isempty(mm.bgroups) ? () : (mm.bgroups[1].fn,)
    cfg = Config{real_equations(mm.equations, RealT), orients, Symbol(vol), typeof(surface_flux),
                 typeof(volume_flux), typeof(semi.source_terms), typeof(bfns)}(surface_flux, volume_flux,
                                                                               semi.source_terms, bfns)
    hist = [count(==(o), tcode[.!tbnd]) for o in 1:8]
    println("  trees: ", T, " trees of ", m, "^3 elements, ", count(tbnd), " boundary tree faces, tree-face orientations ",
            join(("$(o):$(h)" for (o, h) in enumerate(hist) if h > 0), " "))
    return p, cfg, merge(mm, (; perm = tperm))
end

# face of direction d of every tree (the element layer at the tree boundary), as (n m, n m, ntrees);
# and back
function tree_face(q, d)
    sz = size(q)
    n, m, T = sz[1], sz[4], sz[7]
    a = (d + 1) >> 1
    f = q[ntuple(k -> k == a ? (isodd(d) ? (1:1) : (n:n)) : k == 3 + a ? (isodd(d) ? (1:1) : (m:m)) : Colon(), 7)...]
    g = permutedims(reshape(f, n, n, m, m, T), (1, 3, 2, 4, 5))
    return reshape(g, n * m, n * m, T)
end
function from_tree_face(F, d, n, m)
    T = size(F, 3)
    a = (d + 1) >> 1
    g = permutedims(reshape(F, n, m, n, m, T), (1, 3, 2, 4, 5))
    return reshape(g, ntuple(k -> k == a ? 1 : n, 3)..., ntuple(k -> k == a ? 1 : m, 3)..., T)
end

function rhs_trees!(dU, U, p, t, cfg::Config{EQ, OR, vol, SF, VF, ST, BF}) where {EQ, OR, vol, SF, VF, ST, BF}
    NV, sz = length(U), size(U[1])
    n, m, T = sz[1], sz[4], sz[7]
    nm = n * m
    sz4 = (n, n, n, m^3 * T)
    flat(q) = reshape(q, sz4)
    p4 = merge(p, (; Ja = map(J -> map(flat, J), p.Ja)))
    U4 = map(flat, U)
    V4 = vol === :weak ? volume_weak(p4, cfg, U4) : has("fdloop") ? volume_fd_unrolled(p4, cfg, U4) :
         volume_fd(p4, cfg, U4)
    V = map(v -> reshape(v, sz), V4)
    sf = NormalFlux{NV, SF, typeof(EQ)}(cfg.surface_flux, EQ)
    # neighbour values on all tree faces: one gather of whole tree faces, then orient
    own = ntuple(d -> map(q -> tree_face(q, d), U), 6)                       # [d][c] (nm, nm, T)
    src = cat((cat((reshape(own[d][c], nm, nm, 1, 1, T) for d in 1:6)...; dims = 4) for c in 1:NV)...; dims = 3)
    G = reshape(reshape(src, nm * nm * NV, 6T)[:, p.tsrc], nm, nm, NV, 6T)
    for (o, mask) in zip(OR, p.tmasks)
        G = ifelse.(mask, orient(G, o), G)
    end
    G5 = reshape(G, nm, nm, NV, 6, T)
    surf = nothing
    for a in 1:3
        ed = 3 + a
        el(A, r) = A[ntuple(k -> k == ed ? r : Colon(), 7)...]
        Ftf = ntuple(2) do side                                               # -a and +a tree faces
            d = 2a - 2 + side
            nbr = ntuple(c -> G5[:, :, c, d, :], NV)
            if !isempty(cfg.boundary_fns)
                ub = pointwise(BoundaryValue(cfg.boundary_fns[1], EQ), NV, p.Xtf[d]..., zero(p.Xtf[d][1]) .+ t)
                nbr = ntuple(c -> ifelse.(p.tbnd[d], ub[c], nbr[c]), NV)
            end
            F = pointwise(sf, NV, own[d]..., nbr..., p.Ntf[d]...)
            ntuple(c -> from_tree_face(F[c], d, n, m), NV)
        end
        if m > 1
            uR = map(q -> nodeslice(q, a, n), U)
            uL = map(q -> nodeslice(q, a, 1), U)
            Fint = pointwise(sf, NV, map(q -> el(q, 1:(m - 1)), uR)..., map(q -> el(q, 2:m), uL)...,
                             map(q -> el(q, 1:(m - 1)), p.Nplus[a])...)
        end
        term = ntuple(NV) do c
            Gplus = padel(Ftf[2][c], ed, m - 1, 0)
            Gminus = padel(Ftf[1][c], ed, 0, m - 1)
            if m > 1
                Gplus = Gplus .+ padel(Fint[c], ed, 0, 1)
                Gminus = Gminus .- padel(Fint[c], ed, 1, 0)
            end
            p.en[a] .* Gplus .+ p.e1[a] .* Gminus
        end
        surf = addt(surf, term)
    end
    Tt = zero(p.X[1]) .+ t
    src_terms = ST === Nothing ? nothing :
                pointwise(Source{NV, ST, typeof(EQ)}(cfg.source_terms, EQ), NV, U..., p.X..., Tt)
    du = ntuple(c -> src_terms === nothing ? .-p.invJ .* (V[c] .+ p.factor .* surf[c]) :
                     .-p.invJ .* (V[c] .+ p.factor .* surf[c]) .+ src_terms[c], NV)
    dU === nothing && return du
    foreach((d, x) -> d .= x, dU, du)
    return nothing
end

treeblocks() = has("trees")
structured() = has("faces") || treeblocks()     # state not in Trixi's (n, n, n, E) layout
setup_any(semi, vol, RealT) = treeblocks() ? setup_trees(semi, vol, RealT) :
                              has("faces") ? setup_structured(semi, vol, RealT) : setup(semi, vol, RealT)
rhs_any!(dU, U, p, t, cfg) = treeblocks() ? rhs_trees!(dU, U, p, t, cfg) :
                             has("faces") ? rhs_structured!(dU, U, p, t, cfg) : rhs!(dU, U, p, t, cfg)
# the state in the layout of the chosen path: per variable (n, n, n, E) or (n, n, n, Nx, Ny, Nz)
function state_vars(u4, m, RealT)
    U = split_vars(u4, RealT)
    structured() || return U
    return map(q -> reshape(q[:, :, :, vec(m.perm)], size(q)[1:3]..., size(m.perm)...), U)
end
# back to Trixi's element order, (n, n, n, E)
function trixi_order(q, m)
    structured() || return q
    out = similar(q, size(q)[1:3]..., m.E)
    out[:, :, :, vec(m.perm)] .= reshape(q, size(q)[1:3]..., :)
    return out
end

function rhs_only!(dU, U, p, t, cfg)
    rhs_any!(dU, U, p, t, cfg)
    return dU
end

# Carpenter–Kennedy 2N-storage RK (as the elixirs), traced time loop
const CK_A = (0.0, -567301805773 / 1357537059087, -2404267990393 / 2016746695238,
              -3550918686646 / 2091501179385, -1275806237668 / 842570457699)
const CK_B = (1432997174477 / 9575080441755, 5161836677717 / 13612068292357,
              1720146321549 / 2090206949498, 3134564353537 / 4481467310338,
              2277821191437 / 14882151754819)
const CK_C = (0.0, 1432997174477 / 9575080441755, 2526269341429 / 6820363962896,
              2006345519317 / 3224310063776, 2802321613138 / 2924317926251)

function step!(U, dU, K, p, t, dt, cfg)
    RealT = eltype(U[1])
    for s in 1:5
        rhs_only!(dU, U, p, t + RealT(CK_C[s]) * dt, cfg)
        for c in eachindex(U)
            K[c] .= RealT(CK_A[s]) .* K[c] .+ dt .* dU[c]
            U[c] .= U[c] .+ RealT(CK_B[s]) .* K[c]
        end
    end
    return nothing
end

function step_fused!(U, K, p, t, dt, cfg)
    RealT = eltype(U[1])
    for s in 1:5
        du = rhs_any!(nothing, U, p, t + RealT(CK_C[s]) * dt, cfg)
        for c in eachindex(U)
            K[c] .= RealT(CK_A[s]) .* K[c] .+ dt .* du[c]
            U[c] .= U[c] .+ RealT(CK_B[s]) .* K[c]
        end
    end
    return nothing
end

function simulate!(U, dU, K, p, t, dt, nsteps, cfg)
    @trace for _ in 1:nsteps
        has("step") ? step_fused!(U, K, p, t, dt, cfg) : step!(U, dU, K, p, t, dt, cfg)
        t = t + dt
    end
    return t
end

mintime(f; reps = 10) = minimum(_ -> (t0 = time_ns(); f(); (time_ns() - t0) / 1e6), 1:reps)
config(f) = with_config(f; dot_general_precision = PrecisionConfig.HIGHEST)
split_vars(u4, RealT) = ntuple(c -> RealT.(u4[c, :, :, :, :]), size(u4, 1))

function describe(m)
    hist = [count(==(o), m.code) for o in 1:8]
    @printf("  mesh: %d elements, n = %d, %d interfaces, %d boundary faces, face orientations %s\n",
            m.E, m.n, m.ninterfaces, m.nb, join(("$(o):$(h)" for (o, h) in enumerate(hist) if h > 0), " "))
end

function task_check()
    cases, vols = parse_list(arg(1, "cart2,hohq")), parse_list(arg(2, "weak,fd"))
    for case in cases, vol in vols
        ode = trixi_ode(case, vol)
        semi = ode.p
        t = 0.3
        du_t, u4 = trixi_rhs(ode, t)
        for RealT in (Float64, Float32)
            p, cfg, m = setup_any(semi, vol, RealT)
            RealT === Float64 && describe(m)
            U = map(Reactant.to_rarray, state_vars(u4, m, RealT))
            dU = map(q -> Reactant.to_rarray(zero(q)), state_vars(u4, m, RealT))
            rp = Reactant.to_rarray(p)
            fc = config(() -> @compile(sync = true, rhs_only!(dU, U, rp, RealT(t), cfg)))
            fc(dU, U, rp, RealT(t), cfg)
            err = maximum(maximum(abs, trixi_order(Float64.(Array(dU[c])), m) .- du_t[c, :, :, :, :]) for c in eachindex(dU))
            @printf("check %-6s %-4s %-7s %s max|du - du_Trixi| / max|du_Trixi| = %.2e\n", case, vol, RealT,
                    optstr(), err / maximum(abs, du_t))
            flush(stdout)
        end
    end
end

function task_bench()
    cases, vols = parse_list(arg(1, "cart8,cart16")), parse_list(arg(2, "weak,fd"))
    RealT = arg(3, "Float32") == "Float64" ? Float64 : Float32
    println("CSV,task,backend,RealT,case,elements,dofs,volume,ablate,compile_rhs_s,rhs_ms,ns_per_dof_rhs,compile_sim_s,step_ms")
    for case in cases, vol in vols
        try
            semi = trixi_ode(case, vol).p
            p, cfg, m = setup_any(semi, vol, RealT)
            describe(m)
            u0 = Trixi.compute_coefficients(0.0, semi)
            u4 = GC.@preserve u0 copy(Trixi.wrap_array(u0, semi))
            U = map(Reactant.to_rarray, state_vars(u4, m, RealT))
            dU, K = (map(q -> Reactant.to_rarray(zero(q)), state_vars(u4, m, RealT)) for _ in 1:2)
            rp = Reactant.to_rarray(p)
            tc_rhs = @elapsed fc = config(() -> @compile(sync = true, rhs_only!(dU, U, rp, RealT(0), cfg)))
            fc(dU, U, rp, RealT(0), cfg)
            t_rhs = mintime(() -> fc(dU, U, rp, RealT(0), cfg))
            nd = m.E * m.n^3
            @printf("%-6s E=%7d dofs=%9d %-4s %-7s %s %s compile rhs %5.1f s  rhs %9.3f ms (%6.2f ns/DOF)\n",
                    case, m.E, nd, vol, RealT, optstr(), ABLATE, tc_rhs, t_rhs, t_rhs * 1e6 / nd)
            flush(stdout)
            # the traced RK loop (5 rhs! per step) needs more device memory than one rhs!
            nsteps, dt = 10, RealT(1e-4)
            t0() = Reactant.ConcreteRNumber(RealT(0))
            # E12_FD_LOOP_MAX_E: skip the flux differencing loop above this many elements (its
            # compile takes 30+ min on the TPU)
            skip_loop = vol == "fd" && m.E > parse(Int, get(ENV, "E12_FD_LOOP_MAX_E", string(typemax(Int))))
            tc_sim, t_sim = skip_loop ? (NaN, NaN) : try
                tc = @elapsed fs = config(() -> @compile(sync = true, simulate!(U, dU, K, rp, t0(), dt, nsteps, cfg)))
                fs(U, dU, K, rp, t0(), dt, nsteps, cfg)
                tc, mintime(() -> fs(U, dU, K, rp, t0(), dt, nsteps, cfg); reps = 3) / nsteps
            catch err
                lines = filter(!isempty, strip.(split(sprint(showerror, err), '\n')))
                i = findfirst(l -> occursin(r"RESOURCE|memory|rror", l), lines)
                println("  loop FAIL ", first(i === nothing ? lines[end] : lines[i], 300))
                NaN, NaN
            end
            @printf("%-6s %-4s step %9.3f ms (compile loop %5.1f s)\n", case, vol, t_sim, tc_sim)
            println(join(("CSV", "bench_p4est3d", BACKEND, RealT, case, m.E, nd, vol, optstr() * ABLATE, round(tc_rhs, digits = 2),
                          round(t_rhs, digits = 4), round(t_rhs * 1e6 / nd, digits = 3), round(tc_sim, digits = 2),
                          round(t_sim, digits = 4)), ","))
        catch err
            println("$case $vol FAIL ", first(first(split(sprint(showerror, err), '\n')), 300))
            Base.show_backtrace(stdout, catch_backtrace())
            println()
        end
        flush(stdout)
    end
end

function task_profile()
    case, vols = arg(1, "cart16"), parse_list(arg(2, "weak,fd"))
    RealT = Float32
    for vol in vols
        semi = trixi_ode(case, vol).p
        p, cfg, m = setup_any(semi, vol, RealT)
        describe(m)
        u0 = Trixi.compute_coefficients(0.0, semi)
        u4 = GC.@preserve u0 copy(Trixi.wrap_array(u0, semi))
        U = map(Reactant.to_rarray, state_vars(u4, m, RealT))
        dU = map(q -> Reactant.to_rarray(zero(q)), state_vars(u4, m, RealT))
        rp = Reactant.to_rarray(p)
        fc = config(() -> @compile(sync = true, rhs_only!(dU, U, rp, RealT(0), cfg)))
        fc(dU, U, rp, RealT(0), cfg)
        @printf("\n== profile case=%s elements=%d %s %s %s: rhs %.3f ms\n", case, m.E, vol, optstr(), ABLATE,
                mintime(() -> fc(dU, U, rp, RealT(0), cfg)))
        try
            Reactant.@profile nrepeat = 5 fc(dU, U, rp, RealT(0), cfg)
        catch err
            println("profile FAIL ", first(first(split(sprint(showerror, err), '\n')), 300))
        end
        hlo = config(() -> @code_xla(rhs_only!(dU, U, rp, RealT(0), cfg)))
        open(io -> print(io, hlo), "logs/e12_xla_$(BACKEND)_$(case)_$(vol)_$(optstr())$(isempty(ABLATE) ? "" : "_" * ABLATE).txt", "w")
        if get(ENV, "E12_PROFILE_LOOP", "0") == "1"   # the traced RK loop (2 steps) as well
            K = map(q -> Reactant.to_rarray(zero(q)), state_vars(u4, m, RealT))
            t0() = Reactant.ConcreteRNumber(RealT(0))
            dt = RealT(1e-4)
            fs = config(() -> @compile(sync = true, simulate!(U, dU, K, rp, t0(), dt, 2, cfg)))
            fs(U, dU, K, rp, t0(), dt, 2, cfg)
            @printf("\n== profile case=%s elements=%d %s %s loop: 2 steps %.3f ms\n", case, m.E, vol, optstr(),
                    mintime(() -> fs(U, dU, K, rp, t0(), dt, 2, cfg); reps = 3))
            try
                Reactant.@profile nrepeat = 3 fs(U, dU, K, rp, t0(), dt, 2, cfg)
            catch err
                println("profile FAIL ", first(first(split(sprint(showerror, err), '\n')), 300))
            end
            hlo = config(() -> @code_xla(simulate!(U, dU, K, rp, t0(), dt, 2, cfg)))
            open(io -> print(io, hlo), "logs/e12_xla_$(BACKEND)_$(case)_$(vol)_$(optstr())_loop.txt", "w")
        end
        flush(stdout)
    end
end

# self-test of the orientation helpers on plain arrays
let n = 4, G = rand(n, n, 2, 3)
    for o in 1:8
        @assert all(orient(G, o)[l, m, :, :] == G[ORIENTS[o](l, m, n)..., :, :] for l in 1:n, m in 1:n) "orient $o"
    end
end

println("backend=$BACKEND  devices=", Reactant.devices(), "  task=$TASK  args=", ARGS[3:end])
TASK == "check" ? task_check() : TASK == "bench" ? task_bench() : TASK == "profile" ? task_profile() :
error("unknown task $TASK")
