# E7: 2D compressible Euler DGSEM written as array operations (no kernels), for TPUs.
# Same discretization as Trixi's TreeMesh 2D (LGL nodes, LLF surface flux with
# `max_abs_speed_naive`, weak form or flux differencing with `flux_kennedy_gruber`,
# `initial_condition_convergence_test` + `source_terms_convergence_test` on [0, 2]^2 periodic),
# so `check` can compare rhs! value by value with Trixi (a Cartesian P4estMesh).
#   volume: weak = one `dot_general` per direction; fd = two-point fluxes of all node pairs by
#           broadcasting, contracted with D_split
#   neighbours: slice = shifts by slicing (structured mesh); gather = whole faces of all four
#           variables fetched through index arrays (what an unstructured mesh needs)
# Tasks:
#   check  [levels]                 rhs! vs Trixi (Float64, CPU) for all four variants
#   bench  [levels] [volumes] [nbs] [precisions] [RealT]   time per rhs! and per RK step
#   eoc    [levels] [volume] [precision] [RealT] [tend]   errors vs the exact solution
#   adapt  [level] [volume]         in-graph CFL time step (`@trace while`) vs fixed dt
# usage (TrixiDocker image): julia experiments/e7_euler2d_array.jl <tpu|gpu|cpu> <task> [args...]
using Reactant, Trixi, Printf, LinearAlgebra

const BACKEND = get(ARGS, 1, "tpu")
const TASK = get(ARGS, 2, "check")
Reactant.set_default_backend(BACKEND)
arg(i, default) = get(ARGS, i + 2, default)
parse_list(s, T) = T === String ? split(s, ',') : parse.(T, split(s, ','))
const POLYDEG = 3
const GAMMA = 1.4

# ---------------------------------------------------------------- basis (as Trixi)
function lgl_nodes_weights(n)
    N = n - 1
    x = -cospi.((0:N) ./ N)
    P = zeros(n, n)
    xold = fill(2.0, n)
    while maximum(abs, x .- xold) > eps()
        xold = copy(x)
        P[:, 1] .= 1
        P[:, 2] .= x
        for k in 2:N
            P[:, k + 1] .= ((2k - 1) .* x .* P[:, k] .- (k - 1) .* P[:, k - 1]) ./ k
        end
        x .= xold .- (x .* P[:, n] .- P[:, N]) ./ (n .* P[:, n])
    end
    return x, 2 ./ (N * n .* P[:, n] .^ 2)
end

function derivative_matrix(x)
    n = length(x)
    λ = [1 / prod(x[j] - x[k] for k in 1:n if k != j) for j in 1:n]
    D = [i == j ? 0.0 : λ[j] / λ[i] / (x[i] - x[j]) for i in 1:n, j in 1:n]
    for i in 1:n
        D[i, i] = -sum(D[i, :])
    end
    return D
end

const CK_A = (0.0, -567301805773 / 1357537059087, -2404267990393 / 2016746695238,
              -3550918686646 / 2091501179385, -1275806237668 / 842570457699)
const CK_B = (1432997174477 / 9575080441755, 5161836677717 / 13612068292357,
              1720146321549 / 2090206949498, 3134564353537 / 4481467310338,
              2277821191437 / 14882151754819)
const CK_C = (0.0, 1432997174477 / 9575080441755, 2526269341429 / 6820363962896,
              2006345519317 / 3224310063776, 2802321613138 / 2924317926251)

# mesh: nx × nx elements on [0, 2]^2, element (ex, ey) has flat index ex + (ey - 1) nx
function setup(level, RealT)
    n, nx = POLYDEG + 1, 2^level
    ξ, w = lgl_nodes_weights(n)
    D = derivative_matrix(ξ)
    dhat = [-D[k, j] * w[k] / w[j] for j in 1:n, k in 1:n]       # Trixi's calc_Dhat
    dsplit = 2 .* D
    dsplit[1, 1] += 1 / w[1]
    dsplit[n, n] -= 1 / w[n]                                       # Trixi's calc_Dsplit
    dx = 2 / nx
    x = [(ex - 1) * dx + (ξ[i] + 1) / 2 * dx for i in 1:n, j in 1:n, ex in 1:nx, ey in 1:nx]
    y = [(ey - 1) * dx + (ξ[j] + 1) / 2 * dx for i in 1:n, j in 1:n, ex in 1:nx, ey in 1:nx]
    flat(ex, ey) = mod1(ex, nx) + (mod1(ey, nx) - 1) * nx
    nb(dx_, dy_) = Int[flat(ex + dx_, ey + dy_) for ex in 1:nx, ey in 1:nx][:]
    e1 = zeros(RealT, n, 1, 1, 1); e1[1] = 1
    en = zeros(RealT, n, 1, 1, 1); en[n] = 1
    p = (; dhat = RealT.(dhat), dsplit = RealT.(dsplit), x = RealT.(x), y = RealT.(y),
         right = nb(1, 0), left = nb(-1, 0), top = nb(0, 1), bottom = nb(0, -1),
         e1, en, e1y = reshape(e1, 1, n, 1, 1), eny = reshape(en, 1, n, 1, 1),
         dsx = reshape(RealT.(dsplit), n, n, 1, 1, 1), dsy = reshape(RealT.(dsplit), 1, n, n, 1, 1),
         factor = RealT(1 / w[1]), invJ = RealT(2 / dx), gm1 = RealT(GAMMA - 1), gamma = RealT(GAMMA),
         half = RealT(0.5), omega = RealT(π), amp = RealT(0.1), c0 = RealT(2),
         rk_a = RealT.(CK_A), rk_b = RealT.(CK_B), rk_c = RealT.(CK_C), n, nx)
    return p, w, x, y
end

function exact(x, y, t)
    ini = 2 + 0.1 * sin(π * (x + y - t))
    return (ini, ini, ini, ini^2)
end
initial_state(x, y, RealT) = Tuple(RealT.(getindex.(exact.(x, y, 0.0), c)) for c in 1:4)

# ---------------------------------------------------------------- operators
# contraction with a node×node matrix along the first / second node index of (n, n, nx, ny)
apply_x(M, F) = reshape(M * reshape(F, size(F, 1), :), size(F))
function apply_y(M, F)
    G = permutedims(F, (2, 1, 3, 4))
    return permutedims(reshape(M * reshape(G, size(G, 1), :), size(G)), (2, 1, 3, 4))
end

pressure(p, rho, m1, m2, E) = p.gm1 .* (E .- p.half .* (m1 .* m1 .+ m2 .* m2) ./ rho)

function euler_flux(p, rho, m1, m2, E, orientation)
    v1, v2 = m1 ./ rho, m2 ./ rho
    pr = pressure(p, rho, m1, m2, E)
    if orientation == 1
        return (m1, m1 .* v1 .+ pr, m1 .* v2, (E .+ pr) .* v1)
    else
        return (m2, m2 .* v1, m2 .* v2 .+ pr, (E .+ pr) .* v2)
    end
end

# Kennedy-Gruber two-point flux, broadcast over (left, right) arrays
function flux_kg(p, rL, m1L, m2L, EL, rR, m1R, m2R, ER, orientation)
    v1L, v2L, v1R, v2R = m1L ./ rL, m2L ./ rL, m1R ./ rR, m2R ./ rR
    pL, pR = pressure(p, rL, m1L, m2L, EL), pressure(p, rR, m1R, m2R, ER)
    rho_avg = p.half .* (rL .+ rR)
    v1_avg, v2_avg = p.half .* (v1L .+ v1R), p.half .* (v2L .+ v2R)
    p_avg = p.half .* (pL .+ pR)
    e_avg = p.half .* (EL ./ rL .+ ER ./ rR)
    vn = orientation == 1 ? v1_avg : v2_avg
    f1 = rho_avg .* vn
    return (f1, f1 .* v1_avg .+ (orientation == 1 ? p_avg : 0), f1 .* v2_avg .+ (orientation == 2 ? p_avg : 0),
            (rho_avg .* e_avg .+ p_avg) .* vn)
end

function volume_weak(p, U)
    f = euler_flux(p, U..., 1)
    g = euler_flux(p, U..., 2)
    return ntuple(c -> apply_x(p.dhat, f[c]) .+ apply_y(p.dhat, g[c]), 4)
end

# sizes come from the array types: inside `@trace for` Reactant traces the scalars in `p`
function volume_fd(p, U)
    n, nx = size(U[1], 1), size(U[1], 3)
    # x: pairs (i, k) of nodes along x, indexed [i, k, j, ex, ey]
    L = map(q -> reshape(q, n, 1, n, nx, nx), U)
    R = map(q -> reshape(q, 1, n, n, nx, nx), U)
    F = flux_kg(p, L..., R..., 1)
    # y: pairs (j, k), indexed [i, j, k, ex, ey]
    Ly = map(q -> reshape(q, n, n, 1, nx, nx), U)
    Ry = map(q -> reshape(q, n, 1, n, nx, nx), U)
    G = flux_kg(p, Ly..., Ry..., 2)
    return ntuple(c -> reshape(sum(p.dsx .* F[c]; dims = 2), n, n, nx, nx) .+
                       reshape(sum(p.dsy .* G[c]; dims = 3), n, n, nx, nx), 4)
end

function llf(p, uL, uR, orientation)
    fL, fR = euler_flux(p, uL..., orientation), euler_flux(p, uR..., orientation)
    vL = (orientation == 1 ? uL[2] : uL[3]) ./ uL[1]
    vR = (orientation == 1 ? uR[2] : uR[3]) ./ uR[1]
    cL = sqrt.(p.gamma .* pressure(p, uL...) ./ uL[1])
    cR = sqrt.(p.gamma .* pressure(p, uR...) ./ uR[1])
    λ = max.(abs.(vL), abs.(vR)) .+ max.(cL, cR)
    return ntuple(c -> p.half .* (fL[c] .+ fR[c]) .- p.half .* λ .* (uR[c] .- uL[c]), 4)
end

# face values (n, nx, ny) of the four variables
face_xl(U) = map(q -> reshape(q[1:1, :, :, :], size(q, 2), size(q, 3), size(q, 4)), U)
face_xr(U) = map(q -> reshape(q[end:end, :, :, :], size(q, 2), size(q, 3), size(q, 4)), U)
face_yl(U) = map(q -> reshape(q[:, 1:1, :, :], size(q, 1), size(q, 3), size(q, 4)), U)
face_yr(U) = map(q -> reshape(q[:, end:end, :, :], size(q, 1), size(q, 3), size(q, 4)), U)

# neighbour values of face arrays: slicing (structured) ...
shift(A, dim, s) = s == 1 ? (dim == 2 ? cat(A[:, 2:end, :], A[:, 1:1, :]; dims = 2) :
                                         cat(A[:, :, 2:end], A[:, :, 1:1]; dims = 3)) :
                            (dim == 2 ? cat(A[:, end:end, :], A[:, 1:(end - 1), :]; dims = 2) :
                                        cat(A[:, :, end:end], A[:, :, 1:(end - 1)]; dims = 3))
nbr_slice(F, idx, dim, s) = map(A -> shift(A, dim, s), F)
# ... or one block gather of all four variables through an index array (unstructured)
function nbr_gather(F, idx, dim, s)
    n, nx, ny = size(F[1])
    S = reshape(cat(map(A -> reshape(A, n, 1, nx * ny), F)...; dims = 2), 4n, nx * ny)
    G = reshape(S[:, idx], n, 4, nx, ny)
    return ntuple(c -> G[:, c, :, :], 4)
end

function rhs!(dU, U, p, t, ::Val{vol}, ::Val{nb}) where {vol, nb}
    n, nx = size(U[1], 1), size(U[1], 3)
    V = vol === :weak ? volume_weak(p, U) : volume_fd(p, U)
    nbr = nb === :slice ? nbr_slice : nbr_gather
    # x interfaces: right face of e and left face of its right neighbour
    Fx = llf(p, face_xr(U), nbr(face_xl(U), p.right, 2, 1), 1)
    FxL = nbr(Fx, p.left, 2, -1)                 # flux through the left face of e
    Fy = llf(p, face_yr(U), nbr(face_yl(U), p.top, 3, 1), 2)
    FyB = nbr(Fy, p.bottom, 3, -1)
    # source terms (exact density, as Trixi's source_terms_convergence_test)
    si, co = sin.(p.omega .* (p.x .+ p.y .- t)), cos.(p.omega .* (p.x .+ p.y .- t))
    rho_ex = p.c0 .+ p.amp .* si
    rho_x = p.omega .* p.amp .* co
    tmp = (2 .* rho_ex .- 1) .* p.gm1
    S = (rho_x, rho_x .* (1 .+ tmp), rho_x .* (1 .+ tmp), 2 .* rho_x .* (rho_ex .+ tmp))
    for c in 1:4
        surf = p.en .* reshape(Fx[c], 1, n, nx, nx) .- p.e1 .* reshape(FxL[c], 1, n, nx, nx) .+
               p.eny .* reshape(Fy[c], n, 1, nx, nx) .- p.e1y .* reshape(FyB[c], n, 1, nx, nx)
        dU[c] .= .-p.invJ .* (V[c] .+ p.factor .* surf) .+ S[c]
    end
    return nothing
end

const VARIANTS = Dict((v, b) => Symbol("rhs_$(v)_$(b)!") for v in (:weak, :fd), b in (:slice, :gather))
rhs_weak_slice!(dU, U, p, t) = rhs!(dU, U, p, t, Val(:weak), Val(:slice))
rhs_weak_gather!(dU, U, p, t) = rhs!(dU, U, p, t, Val(:weak), Val(:gather))
rhs_fd_slice!(dU, U, p, t) = rhs!(dU, U, p, t, Val(:fd), Val(:slice))
rhs_fd_gather!(dU, U, p, t) = rhs!(dU, U, p, t, Val(:fd), Val(:gather))
rhs_fn(vol, nb) = getfield(@__MODULE__, VARIANTS[(Symbol(vol), Symbol(nb))])

function rhs_only!(rhsf!, dU, U, p, t)
    rhsf!(dU, U, p, t)
    return dU
end

function step!(rhsf!, U, dU, K, p, t, dt)
    for s in 1:5
        rhsf!(dU, U, p, t + p.rk_c[s] * dt)
        for c in 1:4
            K[c] .= p.rk_a[s] .* K[c] .+ dt .* dU[c]
            U[c] .= U[c] .+ p.rk_b[s] .* K[c]
        end
    end
    return nothing
end

function simulate!(rhsf!, U, dU, K, p, t, dt, nsteps)
    @trace for _ in 1:nsteps
        step!(rhsf!, U, dU, K, p, t, dt)
        t = t + dt
    end
    return t
end

function max_dt(p, U, cfl)
    rho, m1, m2, E = U
    c = sqrt.(p.gamma .* pressure(p, rho, m1, m2, E) ./ rho)
    λ = maximum(abs.(m1 ./ rho) .+ abs.(m2 ./ rho) .+ 2 .* c)
    return cfl * 2 / (size(rho, 1) * p.invJ * λ)
end

function simulate_adaptive!(rhsf!, U, dU, K, p, t, tend, cfl, nsteps)
    @trace while t < tend
        dt = min(max_dt(p, U, cfl), tend - t)
        step!(rhsf!, U, dU, K, p, t, dt)
        t = t + dt
        nsteps = nsteps + 1
    end
    return t, nsteps
end

# ---------------------------------------------------------------- helpers
mintime(f; reps = 10) = minimum(_ -> (t0 = time_ns(); f(); (time_ns() - t0) / 1e6), 1:reps)
config(f, prec) = with_config(f; dot_general_precision = getfield(PrecisionConfig, Symbol(prec)))
to_r(U) = map(Reactant.to_rarray, U)
host(U) = map(q -> Float64.(Array(q)), U)
dofs(p) = p.n^2 * p.nx^2

function l2_linf(U, w, x, y, t)
    W = reshape(w * w', length(w), length(w), 1, 1) ./ 4      # reference element weights / |Ω̂|
    ex = exact.(x, y, t)
    return [(sqrt(sum(W .* (U[c] .- getindex.(ex, c)) .^ 2) / size(x, 3)^2),
             maximum(abs, U[c] .- getindex.(ex, c))) for c in 1:4]
end

function initial_dt(p, U0, cfl)
    rho, m1, m2, E = map(q -> Float64.(q), U0)
    pr = (GAMMA - 1) .* (E .- 0.5 .* (m1 .^ 2 .+ m2 .^ 2) ./ rho)
    λ = maximum(abs.(m1 ./ rho) .+ abs.(m2 ./ rho) .+ 2 .* sqrt.(GAMMA .* pr ./ rho))
    return cfl * 2 / (p.n * 2 / (2 / p.nx) * λ)
end

function trixi_rhs(level, vol)
    equations = CompressibleEulerEquations2D(GAMMA)
    volume_integral = vol == "weak" ? VolumeIntegralWeakForm() :
                      VolumeIntegralFluxDifferencing(flux_kennedy_gruber)
    solver = DGSEM(polydeg = POLYDEG, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
                   volume_integral = volume_integral)
    # the image sets Trixi's `backend = "kernelabstractions"`, which TreeMesh does not support
    # on the CPU; a Cartesian P4estMesh is the same operator (straight, axis-aligned elements)
    mesh = P4estMesh((2^level, 2^level), polydeg = POLYDEG, initial_refinement_level = 0,
                     coordinates_min = (0.0, 0.0), coordinates_max = (2.0, 2.0), periodicity = true)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                        source_terms = source_terms_convergence_test,
                                        boundary_conditions = boundary_condition_periodic)
    ode = semidiscretize(semi, (0.0, 1.0))
    semi = ode.p  # `semidiscretize` adapts the semidiscretization
    du = similar(ode.u0)
    t = 0.3
    u = Trixi.compute_coefficients(t, semi)
    ode.f(du, u, semi, t)
    du4 = copy(Trixi.wrap_array(du, semi))  # wrap_array is an unsafe_wrap of `du`
    coords = Array(semi.cache.elements.node_coordinates)
    any(isnan, coords) && error("NaN in Trixi's node coordinates")
    return du4, coords, t
end

# ---------------------------------------------------------------- tasks
function task_check()
    for level in parse_list(arg(1, "2,3"), Int)
        for vol in ("weak", "fd")
            du_t, coords, t = trixi_rhs(level, vol)
            for RealT in (Float64, Float32), nb in ("slice", "gather")
                p, w, x, y = setup(level, RealT)
                # Trixi's state at time t, on our nodes
                U = to_r(Tuple(RealT.(getindex.(exact.(x, y, t), c)) for c in 1:4))
                dU = to_r(ntuple(_ -> zeros(RealT, size(x)), 4))
                rp = Reactant.to_rarray(p)
                f = rhs_fn(vol, nb)
                fc = config(() -> @compile(sync = true, rhs_only!(f, dU, U, rp, RealT(t))), "HIGHEST")
                fc(f, dU, U, rp, RealT(t))
                du = host(dU)
                dx = 2 / p.nx
                err, scale = 0.0, maximum(abs, du_t)
                for e in axes(du_t, 4)
                    ex = floor(Int, sum(coords[1, :, :, e]) / p.n^2 / dx) + 1
                    ey = floor(Int, sum(coords[2, :, :, e]) / p.n^2 / dx) + 1
                    for c in 1:4, j in 1:p.n, i in 1:p.n
                        err = max(err, abs(du[c][i, j, ex, ey] - du_t[c, i, j, e]))
                    end
                end
                @printf("check level=%d %-4s %-6s %-7s max|du - du_Trixi| / max|du_Trixi| = %.2e\n",
                        level, vol, nb, RealT, err / scale)
            end
        end
    end
end

function task_bench()
    levels = parse_list(arg(1, "6,7,8,9"), Int)
    vols, nbs = parse_list(arg(2, "weak,fd"), String), parse_list(arg(3, "slice,gather"), String)
    precs = parse_list(arg(4, "HIGHEST"), String)
    RealT = arg(5, "Float32") == "Float64" ? Float64 : Float32
    println("CSV,task,backend,RealT,level,dofs,volume,neighbours,precision,compile_rhs_s,rhs_ms,ns_per_dof_rhs,compile_sim_s,step_ms,ns_per_dof_stage")
    for level in levels, vol in vols, nb in nbs, prec in precs
        vol == "fd" && prec != first(precs) && continue    # no dot_general in fd
        p, w, x, y = setup(level, RealT)
        try
            U, dU, K = to_r(initial_state(x, y, RealT)), to_r(ntuple(_ -> zeros(RealT, size(x)), 4)),
                       to_r(ntuple(_ -> zeros(RealT, size(x)), 4))
            rp = Reactant.to_rarray(p)
            f = rhs_fn(vol, nb)
            t0 = Reactant.ConcreteRNumber(RealT(0))
            tc_rhs = @elapsed fc = config(() -> @compile(sync = true, rhs_only!(f, dU, U, rp, RealT(0))), prec)
            fc(f, dU, U, rp, RealT(0))
            t_rhs = mintime(() -> fc(f, dU, U, rp, RealT(0)))
            nsteps = 10
            dt = RealT(initial_dt(p, initial_state(x, y, Float64), 0.5))
            tc_sim = @elapsed fs = config(() -> @compile(sync = true, simulate!(f, U, dU, K, rp, t0, dt, nsteps)), prec)
            fs(f, U, dU, K, rp, t0, dt, nsteps)
            t_sim = mintime(() -> fs(f, U, dU, K, rp, t0, dt, nsteps); reps = 3) / nsteps
            nd = dofs(p)
            @printf("level=%d dofs=%9d %-4s %-6s %-7s %-7s compile %5.1f s  rhs %9.3f ms (%6.2f ns/DOF)  step %9.3f ms (%6.2f ns/DOF/stage)\n",
                    level, nd, vol, nb, prec, RealT, tc_rhs, t_rhs, t_rhs * 1e6 / nd, t_sim, t_sim * 1e6 / nd / 5)
            println(join(("CSV", "bench", BACKEND, RealT, level, nd, vol, nb, prec, round(tc_rhs, digits = 2),
                          round(t_rhs, digits = 4), round(t_rhs * 1e6 / nd, digits = 3), round(tc_sim, digits = 2),
                          round(t_sim, digits = 4), round(t_sim * 1e6 / nd / 5, digits = 3)), ","))
        catch err
            println("level=$level $vol $nb $prec FAIL ", first(first(split(sprint(showerror, err), '\n')), 160))
        end
        flush(stdout)
    end
end

function task_eoc()
    levels = parse_list(arg(1, "2,3,4,5,6,7"), Int)
    vol, prec = arg(2, "weak"), arg(3, "HIGHEST")
    RealT = arg(4, "Float32") == "Float64" ? Float64 : Float32
    tend = parse(Float64, arg(5, "0.5"))
    nb = "slice"
    println("CSV,task,backend,RealT,precision,volume,level,dofs,nsteps,l2_rho,l2_rhov1,l2_rhov2,l2_rhoe,linf_rho,time_s")
    prev = nothing
    for level in levels
        p, w, x, y = setup(level, RealT)
        U0 = initial_state(x, y, RealT)
        dt0 = initial_dt(p, U0, 0.5)
        nsteps = ceil(Int, tend / dt0)
        dt = RealT(tend / nsteps)
        U, dU, K = to_r(U0), to_r(ntuple(_ -> zeros(RealT, size(x)), 4)), to_r(ntuple(_ -> zeros(RealT, size(x)), 4))
        rp = Reactant.to_rarray(p)
        f = rhs_fn(vol, nb)
        t0 = Reactant.ConcreteRNumber(RealT(0))
        fs = config(() -> @compile(sync = true, simulate!(f, U, dU, K, rp, t0, dt, nsteps)), prec)
        trun = @elapsed fs(f, U, dU, K, rp, t0, dt, nsteps)
        errs = l2_linf(host(U), w, x, y, tend)
        l2 = first.(errs)
        rate = prev === nothing ? fill(NaN, 4) : log2.(prev ./ l2)
        @printf("level=%d dofs=%8d steps=%6d  L2 rho %.3e (EOC %5.2f)  rho_v1 %.3e  rho_e %.3e (EOC %5.2f)  Linf rho %.3e  %.1f s\n",
                level, dofs(p), nsteps, l2[1], rate[1], l2[2], l2[4], rate[4], errs[1][2], trun)
        println(join(("CSV", "eoc", BACKEND, RealT, prec, vol, level, dofs(p), nsteps, l2..., errs[1][2], round(trun, digits = 3)), ","))
        prev = l2
        flush(stdout)
    end
end

function task_adapt()
    level, vol = parse(Int, arg(1, "7")), arg(2, "fd")
    RealT, nb, tend, cfl = Float32, "slice", 0.5, 0.5
    p, w, x, y = setup(level, RealT)
    f = rhs_fn(vol, nb)
    rp = Reactant.to_rarray(p)
    fresh() = (to_r(initial_state(x, y, RealT)), to_r(ntuple(_ -> zeros(RealT, size(x)), 4)),
               to_r(ntuple(_ -> zeros(RealT, size(x)), 4)))
    # adaptive: CFL time step computed inside the compiled program. Scalars that a traced
    # loop rebinds (`t = t + dt`) are written back into the ConcreteRNumber argument, so
    # every call gets fresh ones.
    zero_t() = Reactant.ConcreteRNumber(RealT(0))
    U, dU, K = fresh()
    tc_a = @elapsed fa = config(() -> @compile(sync = true, simulate_adaptive!(f, U, dU, K, rp, zero_t(), RealT(tend), RealT(cfl), Reactant.ConcreteRNumber(0))), "HIGHEST")
    fa(f, U, dU, K, rp, zero_t(), RealT(tend), RealT(cfl), Reactant.ConcreteRNumber(0))  # warm-up
    U, dU, K = fresh()
    t_a, n_a = zero_t(), Reactant.ConcreteRNumber(0)
    ta = @elapsed res = fa(f, U, dU, K, rp, t_a, RealT(tend), RealT(cfl), n_a)
    nsteps_a = Int(Reactant.to_number(res[2]))
    err_a = l2_linf(host(U), w, x, y, tend)[1][1]
    # fixed dt with the same number of steps
    dt = RealT(tend / nsteps_a)
    U, dU, K = fresh()
    tc_f = @elapsed ff = config(() -> @compile(sync = true, simulate!(f, U, dU, K, rp, zero_t(), dt, nsteps_a)), "HIGHEST")
    ff(f, U, dU, K, rp, zero_t(), dt, nsteps_a)  # warm-up
    U, dU, K = fresh()
    tf = @elapsed ff(f, U, dU, K, rp, zero_t(), dt, nsteps_a)
    err_f = l2_linf(host(U), w, x, y, tend)[1][1]
    @printf("adapt level=%d %s: adaptive %d steps, t_end=%.6f, %.3f s (compile %.1f s), L2 rho %.3e | fixed dt: %.3f s (compile %.1f s), L2 rho %.3e\n",
            level, vol, nsteps_a, Float64(Reactant.to_number(res[1])), ta, tc_a, err_a, tf, tc_f, err_f)
end

println("backend=$BACKEND  devices=", Reactant.devices(), "  task=$TASK  args=", ARGS[3:end])
TASK == "check" ? task_check() : TASK == "bench" ? task_bench() :
TASK == "eoc" ? task_eoc() : TASK == "adapt" ? task_adapt() : error("unknown task $TASK")
