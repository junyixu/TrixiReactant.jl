# E5: why are E2's raised kernels slow on a TPU? Same 1D DGSEM (Burgers, LLF, periodic,
# polydeg 3, 5-stage low-storage RK) as E2, everything raised/compiled by Reactant, in
#   kernel × {node, elem} × {array, arith}
#       E2's CUDA.jl kernels; `node` stores u as (nnodes, nelem) like E2 and Trixi, `elem` as
#       (nelem, nnodes); `array` reads neighbours from index arrays (gathers after raising),
#       `arith` computes them (periodic mesh)
#   array × {node, elem}
#       no kernels: matrix product for the volume term, slices + concatenation for the
#       neighbours (what TPU-native code would look like)
# Hypotheses for the 25× slowdown vs. the 3070: gathers from the neighbour index arrays, or
# TPU tiling padding the minor-most dimension (XLA sees a Julia (4, nelem) array as
# [nelem, 4], so the 4 nodes are the minor dimension, padded to 128 lanes).
# usage (TrixiDocker image): julia experiments/e5_tpu_layout.jl [tpu|gpu|cpu] [nelem,...] [nsteps]
# Set XLA_FLAGS=--xla_dump_to=<dir> to see the layouts XLA picked (after_optimizations).
using CUDA, Reactant, Printf

const BACKEND = get(ARGS, 1, "tpu")
const SIZES = parse.(Int, split(get(ARGS, 2, "16384,1048576"), ','))
const NSTEPS = parse(Int, get(ARGS, 3, "100"))
const RealT = Float32
const POLYDEG = 3
Reactant.set_default_backend(BACKEND)

@inline flux(u) = u * u / 2
@inline llf(uL, uR) = (flux(uL) + flux(uR)) / 2 - max(abs(uL), abs(uR)) * (uR - uL) / 2

# ---------------------------------------------------------------- discretization (as E2)
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

function setup(nelem)
    n = POLYDEG + 1
    nodes, weights = lgl_nodes_weights(n)
    D = derivative_matrix(nodes)
    dhat = [weights[j] / weights[i] * D[j, i] for i in 1:n, j in 1:n]
    dx = 2 / nelem
    x = [-1 + (e - 1 + (nodes[i] + 1) / 2) * dx for i in 1:n, e in 1:nelem]
    p = (; dhat = RealT.(dhat), dhatT = RealT.(permutedims(dhat)),
         left = Int32[mod1(e - 1, nelem) for e in 1:nelem],
         right = Int32[mod1(e + 1, nelem) for e in 1:nelem],
         invJ = fill(RealT(2 / dx), nelem), invJrow = fill(RealT(2 / dx), 1, nelem),
         inv_w1 = RealT(1 / weights[1]), inv_wn = RealT(1 / weights[end]))
    u0 = RealT.(@. 1 + sinpi(x) / 2)
    dt = RealT(0.5 * dx / (1.5 * (2 * POLYDEG + 1)))
    return u0, p, dt
end

# ---------------------------------------------------------------- kernels, layout-generic
@inline ld(::Val{:node}, A, i, e) = @inbounds A[i, e]
@inline ld(::Val{:elem}, A, i, e) = @inbounds A[e, i]
@inline st!(::Val{:node}, A, v, i, e) = (@inbounds A[i, e] = v; nothing)
@inline st!(::Val{:elem}, A, v, i, e) = (@inbounds A[e, i] = v; nothing)
@inline right_of(::Val{:array}, right, k, nelem) = @inbounds right[k]
@inline right_of(::Val{:arith}, right, k, nelem) = ifelse(k == nelem, 1, k + 1)
@inline left_of(::Val{:array}, left, e, nelem) = @inbounds left[e]
@inline left_of(::Val{:arith}, left, e, nelem) = ifelse(e == 1, nelem, e - 1)

function volume_kernel!(du, u, dhat, nelem, L)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    e = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = size(dhat, 1)
    if i <= n && e <= nelem
        acc = zero(eltype(du))
        for j in 1:n
            acc += @inbounds(dhat[i, j]) * flux(ld(L, u, j, e))
        end
        st!(L, du, acc, i, e)
    end
    return nothing
end

function interface_flux_kernel!(fstar, u, right, n, nelem, L, NB)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= nelem
        @inbounds fstar[k] = llf(ld(L, u, n, k), ld(L, u, 1, right_of(NB, right, k, nelem)))
    end
    return nothing
end

function surface_jacobian_kernel!(du, fstar, left, invJ, inv_w1, inv_wn, n, nelem, L, NB)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    e = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && e <= nelem
        val = ld(L, du, i, e)
        i == 1 && (val += @inbounds(fstar[left_of(NB, left, e, nelem)]) * inv_w1)
        i == n && (val -= @inbounds(fstar[e]) * inv_wn)
        st!(L, du, val * @inbounds(invJ[e]), i, e)
    end
    return nothing
end

function rhs_kernels!(du, u, fstar, p, L, NB)
    n, nelem = size(p.dhat, 1), length(fstar)
    @cuda threads=(n, 64) blocks=(1, cld(nelem, 64)) volume_kernel!(du, u, p.dhat, nelem, L)
    @cuda threads=256 blocks=cld(nelem, 256) interface_flux_kernel!(fstar, u, p.right, n, nelem, L, NB)
    @cuda threads=(n, 64) blocks=(1, cld(nelem, 64)) surface_jacobian_kernel!(du, fstar, p.left, p.invJ,
                                                                             p.inv_w1, p.inv_wn, n, nelem, L, NB)
    return nothing
end

# ---------------------------------------------------------------- array formulation
function rhs_array!(du, u, fstar, p, ::Val{:node})
    n = size(u, 1)
    du .= p.dhat * flux.(u)
    uR = vcat(u[1, 2:end], u[1, 1:1])            # node 1 of the right neighbour
    fstar .= llf.(u[n, :], uR)
    fL = vcat(fstar[end:end], fstar[1:(end - 1)])  # flux through the left face
    du[1, :] = du[1, :] .+ fL .* p.inv_w1
    du[n, :] = du[n, :] .- fstar .* p.inv_wn
    du .= du .* p.invJrow
    return nothing
end

function rhs_array!(du, u, fstar, p, ::Val{:elem})
    n = size(u, 2)
    du .= flux.(u) * p.dhatT
    uR = vcat(u[2:end, 1], u[1:1, 1])
    fstar .= llf.(u[:, n], uR)
    fL = vcat(fstar[end:end], fstar[1:(end - 1)])
    du[:, 1] = du[:, 1] .+ fL .* p.inv_w1
    du[:, n] = du[:, n] .- fstar .* p.inv_wn
    du .= du .* p.invJ
    return nothing
end

# named functions, not closures (Reactant traces captured variables)
rhs_k_node_array!(du, u, fs, p) = rhs_kernels!(du, u, fs, p, Val(:node), Val(:array))
rhs_k_node_arith!(du, u, fs, p) = rhs_kernels!(du, u, fs, p, Val(:node), Val(:arith))
rhs_k_elem_array!(du, u, fs, p) = rhs_kernels!(du, u, fs, p, Val(:elem), Val(:array))
rhs_k_elem_arith!(du, u, fs, p) = rhs_kernels!(du, u, fs, p, Val(:elem), Val(:arith))
rhs_a_node!(du, u, fs, p) = rhs_array!(du, u, fs, p, Val(:node))
rhs_a_elem!(du, u, fs, p) = rhs_array!(du, u, fs, p, Val(:elem))

const VARIANTS = [
    ("kernel node array (E2)", rhs_k_node_array!, :node),
    ("kernel node arith", rhs_k_node_arith!, :node),
    ("kernel elem array", rhs_k_elem_array!, :elem),
    ("kernel elem arith", rhs_k_elem_arith!, :elem),
    ("array node", rhs_a_node!, :node),
    ("array elem", rhs_a_elem!, :elem),
]

# ---------------------------------------------------------------- CPU reference (node layout)
function rhs_ref!(du, u, fstar, p)
    n, nelem = size(u)
    for k in 1:nelem
        fstar[k] = llf(u[n, k], u[1, p.right[k]])
    end
    for e in 1:nelem, i in 1:n
        acc = zero(eltype(du))
        for j in 1:n
            acc += p.dhat[i, j] * flux(u[j, e])
        end
        i == 1 && (acc += fstar[p.left[e]] * p.inv_w1)
        i == n && (acc -= fstar[e] * p.inv_wn)
        du[i, e] = acc * p.invJ[e]
    end
    return nothing
end

# ---------------------------------------------------------------- time stepping (as E2)
const CK_A = (0.0, -567301805773 / 1357537059087, -2404267990393 / 2016746695238,
              -3550918686646 / 2091501179385, -1275806237668 / 842570457699)
const CK_B = (1432997174477 / 9575080441755, 5161836677717 / 13612068292357,
              1720146321549 / 2090206949498, 3134564353537 / 4481467310338,
              2277821191437 / 14882151754819)

function step!(rhsf!, u, du, du_tmp, fstar, p, dt, T)
    for s in 1:5
        rhsf!(du, u, fstar, p)
        a, b = T(CK_A[s]), T(CK_B[s])
        @. du_tmp = a * du_tmp + dt * du
        @. u = u + b * du_tmp
    end
    return nothing
end

function simulate!(rhsf!, u, du, du_tmp, fstar, p, dt, nsteps, T)
    @trace for _ in 1:nsteps
        step!(rhsf!, u, du, du_tmp, fstar, p, dt, T)
    end
    return u
end

function rhs_only!(rhsf!, du, u, fstar, p)
    rhsf!(du, u, fstar, p)
    return du
end

mintime(f; reps = 10) = minimum(_ -> (t0 = time_ns(); f(); (time_ns() - t0) / 1e6), 1:reps)
relerr(a, b) = maximum(abs, a .- b) / maximum(abs, b)
node_layout(x, L) = L === :node ? Array(x) : permutedims(Array(x))
hlocount(hlo, op) = count("stablehlo.$op", hlo)
config(f) = with_config(f; dot_general_precision = PrecisionConfig.HIGHEST)

println("backend=$BACKEND  devices=", Reactant.devices(), "  nsteps=$NSTEPS  $RealT  polydeg=$POLYDEG")
results = []
for nelem in SIZES
    u0, p, dt = setup(nelem)
    du_r, u_r = similar(u0), copy(u0)
    rhs_ref!(du_r, u0, zeros(RealT, nelem), p)
    simulate!(rhs_ref!, u_r, similar(u_r), zero(u_r), zeros(RealT, nelem), p, dt, NSTEPS, RealT)
    rp = Reactant.to_rarray(p)
    for (name, f, L) in VARIANTS
        print(rpad("nelem=$nelem  $name", 44))
        flush(stdout)
        try
            uin = L === :node ? u0 : permutedims(u0)
            u, du, dtmp = Reactant.to_rarray(uin), Reactant.to_rarray(zero(uin)), Reactant.to_rarray(zero(uin))
            fs = Reactant.to_rarray(zeros(RealT, nelem))
            hlo = config(() -> string(@code_hlo raise=true rhs_only!(f, du, u, fs, rp)))
            ops = (; total = hlocount(hlo, ""), gather = hlocount(hlo, "gather"),
                   scatter = hlocount(hlo, "scatter"), dot = hlocount(hlo, "dot_general"),
                   kernel = count("kernel_call", hlo) + count("custom_call", hlo))
            f_rhs = config(() -> @compile raise=true sync=true rhs_only!(f, du, u, fs, rp))
            f_rhs(f, du, u, fs, rp)
            e_rhs = relerr(node_layout(du, L), du_r)
            t_rhs = mintime(() -> f_rhs(f, du, u, fs, rp))
            f_sim = config(() -> @compile raise=true sync=true simulate!(f, u, du, dtmp, fs, rp, dt, NSTEPS, RealT))
            f_sim(f, u, du, dtmp, fs, rp, dt, NSTEPS, RealT)
            e_sol = relerr(node_layout(u, L), u_r)
            t_sim = mintime(() -> f_sim(f, u, du, dtmp, fs, rp, dt, NSTEPS, RealT); reps = 3)
            @printf("rhs %9.3f ms  sim %10.2f ms  err rhs %.1e u_T %.1e  ops %d (gather %d scatter %d dot %d kernel %d)\n",
                    t_rhs, t_sim, e_rhs, e_sol, ops.total, ops.gather, ops.scatter, ops.dot, ops.kernel)
            push!(results, (nelem, name, t_rhs, t_sim, e_sol, ops))
        catch err
            println("FAIL  ", first(first(split(sprint(showerror, err), '\n')), 160))
        end
    end
end

println("\nbackend=$BACKEND  nsteps=$NSTEPS (sim = whole time loop in one XLA program)")
@printf("%-10s %-24s %11s %12s %10s %7s %8s\n", "nelem", "variant", "rhs [ms]", "sim [ms]", "err(u_T)", "gather", "scatter")
for (nelem, name, t_rhs, t_sim, e_sol, ops) in results
    @printf("%-10d %-24s %11.3f %12.2f %10.1e %7d %8d\n", nelem, name, t_rhs, t_sim, e_sol, ops.gather, ops.scatter)
end
