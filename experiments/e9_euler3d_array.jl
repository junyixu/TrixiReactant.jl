# E9: 3D compressible Euler DGSEM as array operations (the 3D sibling of E7), mainly for run
# and compile times of 3D flux differencing on a TPU. Same discretization as Trixi in 3D
# (LGL, LLF with `max_abs_speed_naive`, weak form or flux differencing with
# `flux_kennedy_gruber`, convergence test with source terms on [0, 2]^3, periodic), neighbours
# by slicing. Variables are arrays of size (n, n, n, nx, ny, nz). Volumes: weak, fd (all node
# pairs), fdsym (each unordered pair i < k once: the flux is symmetric, D_split has a zero diagonal).
# A grid is a level L ((2^L)^3 elements on [0, 2]^3) or element counts like 128x64x32 (same
# element size as the smallest count on [0, 2], so the periodic solution still fits).
# Tasks:
#   check [levels]                      rhs! vs Trixi (Float64, CPU, Cartesian P4estMesh)
#   bench [grids] [volumes] [RealT]     time per rhs! / RK step, compile times
#   eoc   [levels] [volume] [RealT] [tend]
#   profile [grid] [volumes]            `Reactant.@profile` of one rhs! + optimized XLA HLO in logs/
# usage (TrixiDocker image): julia experiments/e9_euler3d_array.jl <tpu|gpu|cpu> <task> [args...]
using Reactant, Trixi, Printf, LinearAlgebra

const BACKEND = get(ARGS, 1, "tpu")
const TASK = get(ARGS, 2, "check")
Reactant.set_default_backend(BACKEND)
arg(i, default) = get(ARGS, i + 2, default)
parse_list(s, T) = T === String ? split(s, ',') : parse.(T, split(s, ','))
const POLYDEG = 3
const GAMMA = 1.4
const NV = 5
const PAIRS = [(i, k) for i in 1:POLYDEG for k in (i + 1):(POLYDEG + 1)]   # node pairs i < k
parse_grid(g) = occursin('x', g) ? Tuple(parse.(Int, split(g, 'x'))) : ntuple(_ -> 2^parse(Int, g), 3)

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

setup(level::Int, RealT) = setup(ntuple(_ -> 2^level, 3), RealT)

function setup(nxs::NTuple{3, Int}, RealT)
    n = POLYDEG + 1
    ξ, w = lgl_nodes_weights(n)
    D = derivative_matrix(ξ)
    dhat = [-D[k, j] * w[k] / w[j] for j in 1:n, k in 1:n]
    dsplit = 2 .* D
    dsplit[1, 1] += 1 / w[1]
    dsplit[n, n] -= 1 / w[n]
    @assert maximum(abs, [dsplit[i, i] for i in 1:n]) < 1e-12        # used by fdsym
    # fdsym: the flux of pair q = (i, k) enters node i with D_split[i, k], node k with D_split[k, i]
    A = zeros(n, length(PAIRS))
    for (q, (i, k)) in enumerate(PAIRS)
        A[i, q], A[k, q] = dsplit[i, k], dsplit[k, i]
    end
    dx = 2 / minimum(nxs)
    coord(i, e) = (e - 1) * dx + (ξ[i] + 1) / 2 * dx
    x = [coord(i, ex) for i in 1:n, j in 1:n, k in 1:n, ex in 1:nxs[1], ey in 1:nxs[2], ez in 1:nxs[3]]
    y = [coord(j, ey) for i in 1:n, j in 1:n, k in 1:n, ex in 1:nxs[1], ey in 1:nxs[2], ez in 1:nxs[3]]
    z = [coord(k, ez) for i in 1:n, j in 1:n, k in 1:n, ex in 1:nxs[1], ey in 1:nxs[2], ez in 1:nxs[3]]
    unit(d, i) = (v = zeros(RealT, ntuple(m -> m == d ? n : 1, 6)); v[ntuple(m -> m == d ? i : 1, 6)...] = 1; v)
    # D_split as (n, n) placed on node dims (d, d + 1) of the 7D pair arrays
    ds(d) = reshape(RealT.(dsplit), ntuple(m -> m == d || m == d + 1 ? n : 1, 7))
    as(d) = reshape(RealT.(A), ntuple(m -> m == d ? n : m == d + 1 ? length(PAIRS) : 1, 7))
    p = (; dhat = RealT.(dhat), x = RealT.(x .+ y .+ z),
         e1 = ntuple(d -> unit(d, 1), 3), en = ntuple(d -> unit(d, n), 3), ds = ntuple(ds, 3),
         as = ntuple(as, 3),
         factor = RealT(1 / w[1]), invJ = RealT(2 / dx), gm1 = RealT(GAMMA - 1), gamma = RealT(GAMMA),
         half = RealT(0.5), omega = RealT(π), amp = RealT(0.1), c0 = RealT(2),
         rk_a = RealT.(CK_A), rk_b = RealT.(CK_B), rk_c = RealT.(CK_C))
    return p, w, x .+ y .+ z
end

function exact(s, t)  # s = x + y + z
    ini = 2 + 0.1 * sin(π * (s - t))
    return (ini, ini, ini, ini, ini^2)
end
initial_state(s, RealT) = Tuple(RealT.(getindex.(exact.(s, 0.0), c)) for c in 1:NV)

# ---------------------------------------------------------------- operators
# contraction with an n×n matrix along node dimension d ∈ 1:3 of a 6D array
function apply_dim(M, F, d)
    d == 1 && return reshape(M * reshape(F, size(F, 1), :), size(F))
    perm = d == 2 ? (2, 1, 3, 4, 5, 6) : (3, 2, 1, 4, 5, 6)
    G = permutedims(F, perm)
    return permutedims(reshape(M * reshape(G, size(G, 1), :), size(G)), perm)
end

pressure(p, rho, m1, m2, m3, E) = p.gm1 .* (E .- p.half .* (m1 .* m1 .+ m2 .* m2 .+ m3 .* m3) ./ rho)

function euler_flux(p, U, d)
    rho, m1, m2, m3, E = U
    pr = pressure(p, U...)
    m = (m1, m2, m3)
    vd = m[d] ./ rho
    return (m[d], m1 .* vd .+ (d == 1 ? pr : 0), m2 .* vd .+ (d == 2 ? pr : 0),
            m3 .* vd .+ (d == 3 ? pr : 0), (E .+ pr) .* vd)
end

function flux_kg(p, UL, UR, d)
    rL, rR = UL[1], UR[1]
    vL = (UL[2] ./ rL, UL[3] ./ rL, UL[4] ./ rL)
    vR = (UR[2] ./ rR, UR[3] ./ rR, UR[4] ./ rR)
    pL, pR = pressure(p, UL...), pressure(p, UR...)
    rho_avg = p.half .* (rL .+ rR)
    v_avg = ntuple(k -> p.half .* (vL[k] .+ vR[k]), 3)
    p_avg = p.half .* (pL .+ pR)
    e_avg = p.half .* (UL[5] ./ rL .+ UR[5] ./ rR)
    f1 = rho_avg .* v_avg[d]
    return (f1, f1 .* v_avg[1] .+ (d == 1 ? p_avg : 0), f1 .* v_avg[2] .+ (d == 2 ? p_avg : 0),
            f1 .* v_avg[3] .+ (d == 3 ? p_avg : 0), (rho_avg .* e_avg .+ p_avg) .* v_avg[d])
end

function volume_weak(p, U)
    acc = nothing
    for d in 1:3
        f = euler_flux(p, U, d)
        term = ntuple(c -> apply_dim(p.dhat, f[c], d), NV)
        acc = acc === nothing ? term : ntuple(c -> acc[c] .+ term[c], NV)
    end
    return acc
end

# pairs of nodes along node dimension d: left keeps index at d, right moves it to d + 1
function volume_fd(p, U)
    sz = size(U[1])
    acc = nothing
    for d in 1:3
        lsz = (sz[1:d]..., 1, sz[(d + 1):end]...)
        rsz = (sz[1:(d - 1)]..., 1, sz[d:end]...)
        L = map(q -> reshape(q, lsz), U)
        R = map(q -> reshape(q, rsz), U)
        F = flux_kg(p, L, R, d)
        term = ntuple(c -> reshape(sum(p.ds[d] .* F[c]; dims = d + 1), sz), NV)
        acc = acc === nothing ? term : ntuple(c -> acc[c] .+ term[c], NV)
    end
    return acc
end

# node values of the left (j = 1) or right (j = 2) member of every pair, stacked along node dim d
function pair_side(q, d, j)
    sl(r) = ntuple(m -> m == d ? r : Colon(), 6)
    return cat((q[sl(pr[j]:pr[j])...] for pr in PAIRS)...; dims = d)
end

function volume_fdsym(p, U)
    sz, P = size(U[1]), length(PAIRS)
    acc = nothing
    for d in 1:3
        F = flux_kg(p, map(q -> pair_side(q, d, 1), U), map(q -> pair_side(q, d, 2), U), d)
        fsz = (sz[1:(d - 1)]..., 1, P, sz[(d + 1):end]...)
        term = ntuple(c -> reshape(sum(p.as[d] .* reshape(F[c], fsz); dims = d + 1), sz), NV)
        acc = acc === nothing ? term : ntuple(c -> acc[c] .+ term[c], NV)
    end
    return acc
end

function llf(p, uL, uR, d)
    fL, fR = euler_flux(p, uL, d), euler_flux(p, uR, d)
    vL, vR = uL[1 + d] ./ uL[1], uR[1 + d] ./ uR[1]
    cL = sqrt.(p.gamma .* pressure(p, uL...) ./ uL[1])
    cR = sqrt.(p.gamma .* pressure(p, uR...) ./ uR[1])
    λ = max.(abs.(vL), abs.(vR)) .+ max.(cL, cR)
    return ntuple(c -> p.half .* (fL[c] .+ fR[c]) .- p.half .* λ .* (uR[c] .- uL[c]), NV)
end

# face of node dim d at node index i, kept as a 6D array with size 1 in dim d
face(q, d, i) = d == 1 ? q[i:i, :, :, :, :, :] : d == 2 ? q[:, i:i, :, :, :, :] : q[:, :, i:i, :, :, :]
# periodic shift along element dimension 3 + d: s = 1 gives the neighbour at +1
function shift(A, d, s)
    e = 3 + d
    m = size(A, e)
    idx(r) = ntuple(k -> k == e ? r : Colon(), 6)
    return s == 1 ? cat(A[idx(2:m)...], A[idx(1:1)...]; dims = e) :
                    cat(A[idx(m:m)...], A[idx(1:(m - 1))...]; dims = e)
end

function rhs!(dU, U, p, t, ::Val{vol}) where {vol}
    n = size(U[1], 1)
    V = vol === :weak ? volume_weak(p, U) : vol === :fd ? volume_fd(p, U) : volume_fdsym(p, U)
    surf = ntuple(_ -> zero(U[1]), NV)
    for d in 1:3
        uL = map(q -> face(q, d, n), U)                 # right face of each element
        uR = map(q -> shift(face(q, d, 1), d, 1), U)    # left face of the neighbour
        F = llf(p, uL, uR, d)                            # flux through the right face
        FL = map(A -> shift(A, d, -1), F)                # flux through the left face
        surf = ntuple(c -> surf[c] .+ p.en[d] .* F[c] .- p.e1[d] .* FL[c], NV)
    end
    si, co = sin.(p.omega .* (p.x .- t)), cos.(p.omega .* (p.x .- t))
    rho_ex = p.c0 .+ p.amp .* si
    rho_x = p.omega .* p.amp .* co
    tmp = (2 .* rho_ex .- 1.5f0) .* p.gm1
    du2 = rho_x .* (2 .+ tmp)
    S = (2 .* rho_x, du2, du2, du2, rho_x .* (4 .* rho_ex .+ 3 .* tmp))
    for c in 1:NV
        dU[c] .= .-p.invJ .* (V[c] .+ p.factor .* surf[c]) .+ S[c]
    end
    return nothing
end

rhs_weak!(dU, U, p, t) = rhs!(dU, U, p, t, Val(:weak))
rhs_fd!(dU, U, p, t) = rhs!(dU, U, p, t, Val(:fd))
rhs_fdsym!(dU, U, p, t) = rhs!(dU, U, p, t, Val(:fdsym))
rhs_fn(vol) = vol == "weak" ? rhs_weak! : vol == "fd" ? rhs_fd! : vol == "fdsym" ? rhs_fdsym! : error("volume $vol")

function rhs_only!(rhsf!, dU, U, p, t)
    rhsf!(dU, U, p, t)
    return dU
end

function step!(rhsf!, U, dU, K, p, t, dt)
    for s in 1:5
        rhsf!(dU, U, p, t + p.rk_c[s] * dt)
        for c in 1:NV
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

mintime(f; reps = 10) = minimum(_ -> (t0 = time_ns(); f(); (time_ns() - t0) / 1e6), 1:reps)
config(f) = with_config(f; dot_general_precision = PrecisionConfig.HIGHEST)
to_r(U) = map(Reactant.to_rarray, U)
host(U) = map(q -> Float64.(Array(q)), U)
zeros_like(s, RealT) = ntuple(_ -> zeros(RealT, size(s)), NV)

function initial_dt(level, cfl)
    λ = 3 * (1.0 + 1.0)   # |v| ≈ 1 and c ≈ 1 per direction for this state, with margin
    return cfl * 2 / ((POLYDEG + 1) * 2^level * λ)
end
initial_dt(nxs::NTuple{3, Int}, cfl) = initial_dt(round(Int, log2(minimum(nxs))), cfl)

function l2(U, w, s, t)
    n = length(w)
    W = reshape([w[i] * w[j] * w[k] for i in 1:n, j in 1:n, k in 1:n], n, n, n, 1, 1, 1) ./ 8
    ex = exact.(s, t)
    return [sqrt(sum(W .* (U[c] .- getindex.(ex, c)) .^ 2) / prod(size(s)[4:6])) for c in 1:NV]
end

function trixi_rhs(level, vol)
    equations = CompressibleEulerEquations3D(GAMMA)
    volume_integral = vol == "weak" ? VolumeIntegralWeakForm() :
                      VolumeIntegralFluxDifferencing(flux_kennedy_gruber)
    solver = DGSEM(polydeg = POLYDEG, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
                   volume_integral = volume_integral)
    mesh = P4estMesh((2^level, 2^level, 2^level), polydeg = POLYDEG, initial_refinement_level = 0,
                     coordinates_min = (0.0, 0.0, 0.0), coordinates_max = (2.0, 2.0, 2.0),
                     periodicity = true)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                        source_terms = source_terms_convergence_test,
                                        boundary_conditions = boundary_condition_periodic)
    ode = semidiscretize(semi, (0.0, 1.0))
    semi = ode.p
    du = similar(ode.u0)
    t = 0.3
    u = Trixi.compute_coefficients(t, semi)
    ode.f(du, u, semi, t)
    return copy(Trixi.wrap_array(du, semi)), Array(semi.cache.elements.node_coordinates), t
end

function task_check()
    for level in parse_list(arg(1, "1,2"), Int), vol in ("weak", "fd", "fdsym")
        du_t, coords, t = trixi_rhs(level, vol == "weak" ? "weak" : "fd")
        for RealT in (Float64, Float32)
            p, w, s = setup(level, RealT)
            U = to_r(Tuple(RealT.(getindex.(exact.(s, t), c)) for c in 1:NV))
            dU = to_r(zeros_like(s, RealT))
            rp = Reactant.to_rarray(p)
            f = rhs_fn(vol)
            fc = config(() -> @compile(sync = true, rhs_only!(f, dU, U, rp, RealT(t))))
            fc(f, dU, U, rp, RealT(t))
            du = host(dU)
            n, dx = POLYDEG + 1, 2 / 2^level
            err = 0.0
            for e in axes(du_t, 5)
                ex = ntuple(k -> floor(Int, sum(coords[k, :, :, :, e]) / n^3 / dx) + 1, 3)
                for c in 1:NV, k in 1:n, j in 1:n, i in 1:n
                    err = max(err, abs(du[c][i, j, k, ex...] - du_t[c, i, j, k, e]))
                end
            end
            @printf("check3d level=%d %-4s %-7s max|du - du_Trixi| / max|du_Trixi| = %.2e\n",
                    level, vol, RealT, err / maximum(abs, du_t))
        end
    end
end

function task_bench()
    grids, vols = parse_list(arg(1, "3,4,5"), String), parse_list(arg(2, "weak,fd"), String)
    RealT = arg(3, "Float32") == "Float64" ? Float64 : Float32
    println("CSV,task,backend,RealT,level,dofs,volume,compile_rhs_s,rhs_ms,ns_per_dof_rhs,compile_sim_s,step_ms,ns_per_dof_stage")
    for level in grids, vol in vols
        try
            nxs = parse_grid(level)
            p, w, s = setup(nxs, RealT)
            U, dU, K = to_r(initial_state(s, RealT)), to_r(zeros_like(s, RealT)), to_r(zeros_like(s, RealT))
            rp = Reactant.to_rarray(p)
            f = rhs_fn(vol)
            tc_rhs = @elapsed fc = config(() -> @compile(sync = true, rhs_only!(f, dU, U, rp, RealT(0))))
            fc(f, dU, U, rp, RealT(0))
            t_rhs = mintime(() -> fc(f, dU, U, rp, RealT(0)))
            nsteps, dt = 10, RealT(initial_dt(nxs, 0.5))
            t0() = Reactant.ConcreteRNumber(RealT(0))
            tc_sim = @elapsed fs = config(() -> @compile(sync = true, simulate!(f, U, dU, K, rp, t0(), dt, nsteps)))
            fs(f, U, dU, K, rp, t0(), dt, nsteps)
            t_sim = mintime(() -> fs(f, U, dU, K, rp, t0(), dt, nsteps); reps = 3) / nsteps
            nd = length(s)
            @printf("3d level=%s dofs=%9d %-4s %-7s compile rhs %5.1f s, loop %5.1f s  rhs %9.3f ms (%6.2f ns/DOF)  step %9.3f ms (%6.2f ns/DOF/stage)\n",
                    level, nd, vol, RealT, tc_rhs, tc_sim, t_rhs, t_rhs * 1e6 / nd, t_sim, t_sim * 1e6 / nd / 5)
            println(join(("CSV", "bench3d", BACKEND, RealT, level, nd, vol, round(tc_rhs, digits = 2),
                          round(t_rhs, digits = 4), round(t_rhs * 1e6 / nd, digits = 3), round(tc_sim, digits = 2),
                          round(t_sim, digits = 4), round(t_sim * 1e6 / nd / 5, digits = 3)), ","))
        catch err
            println("3d level=$level $vol FAIL ", first(first(split(sprint(showerror, err), '\n')), 160))
        end
        flush(stdout)
    end
end

function task_eoc()
    levels, vol = parse_list(arg(1, "1,2,3,4"), Int), arg(2, "fd")
    RealT = arg(3, "Float32") == "Float64" ? Float64 : Float32
    tend = parse(Float64, arg(4, "0.5"))
    prev = nothing
    for level in levels
        p, w, s = setup(level, RealT)
        nsteps = ceil(Int, tend / initial_dt(level, 0.5))
        dt = RealT(tend / nsteps)
        U, dU, K = to_r(initial_state(s, RealT)), to_r(zeros_like(s, RealT)), to_r(zeros_like(s, RealT))
        rp = Reactant.to_rarray(p)
        f = rhs_fn(vol)
        t0 = Reactant.ConcreteRNumber(RealT(0))
        fs = config(() -> @compile(sync = true, simulate!(f, U, dU, K, rp, t0, dt, nsteps)))
        trun = @elapsed fs(f, U, dU, K, rp, t0, dt, nsteps)
        e = l2(host(U), w, s, tend)
        rate = prev === nothing ? NaN : log2(prev / e[1])
        @printf("eoc3d level=%d dofs=%9d steps=%5d %-4s %-7s L2 rho %.3e (EOC %5.2f)  rho_e %.3e  %.1f s\n",
                level, length(s), nsteps, vol, RealT, e[1], rate, e[5], trun)
        prev = e[1]
        flush(stdout)
    end
end

function task_profile()
    grid, vols = arg(1, "6"), parse_list(arg(2, "weak,fd,fdsym"), String)
    RealT = Float32
    p, w, s = setup(parse_grid(grid), RealT)
    rp = Reactant.to_rarray(p)
    for vol in vols
        U, dU = to_r(initial_state(s, RealT)), to_r(zeros_like(s, RealT))
        f = rhs_fn(vol)
        fc = config(() -> @compile(sync = true, rhs_only!(f, dU, U, rp, RealT(0))))
        fc(f, dU, U, rp, RealT(0))
        @printf("\n== profile3d grid=%s dofs=%d %s: rhs %.3f ms\n", grid, length(s), vol,
                mintime(() -> fc(f, dU, U, rp, RealT(0))))
        try
            Reactant.@profile nrepeat = 5 fc(f, dU, U, rp, RealT(0))
        catch err
            println("profile FAIL ", first(first(split(sprint(showerror, err), '\n')), 300))
        end
        hlo = config(() -> @code_xla(rhs_only!(f, dU, U, rp, RealT(0))))
        open(io -> print(io, hlo), "logs/e9_xla_$(BACKEND)_$(grid)_$(vol).txt", "w")
        flush(stdout)
    end
end

println("backend=$BACKEND  devices=", Reactant.devices(), "  task=$TASK  args=", ARGS[3:end])
TASK == "check" ? task_check() : TASK == "bench" ? task_bench() :
TASK == "eoc" ? task_eoc() : TASK == "profile" ? task_profile() : error("unknown task $TASK")
