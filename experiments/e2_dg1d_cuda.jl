# E2: TrixiCUDA-style 1D DGSEM (Burgers, LLF, periodic) written as hand-written CUDA.jl
# kernels, executed
#   cuda            natively with CUDA.jl (Julia loop launches kernels)
#   noraise         through Reactant, kernels embedded as PTX (only possible on NVIDIA)
#   raise           through Reactant, kernels raised to StableHLO (the TPU path)
#   *_shmem         same, with a shared-memory + sync_threads() volume kernel
# The whole time loop (5-stage low-storage RK) is compiled into one XLA program with @trace.
# usage (TrixiDocker image): julia experiments/e2_dg1d_cuda.jl [gpu|cpu] [nelem] [Float32|Float64] [nsteps]
using CUDA, Reactant, Printf
using Trixi: gauss_lobatto_nodes_weights, polynomial_derivative_matrix

const BACKEND = get(ARGS, 1, "gpu")
const NELEM = parse(Int, get(ARGS, 2, "16384"))
const RealT = get(ARGS, 3, "Float32") == "Float64" ? Float64 : Float32
const NSTEPS = parse(Int, get(ARGS, 4, "100"))
const POLYDEG = 3
Reactant.set_default_backend(BACKEND)

@inline flux(u) = u * u / 2
@inline llf(uL, uR) = (flux(uL) + flux(uR)) / 2 - max(abs(uL), abs(uR)) * (uR - uL) / 2

function setup(nelem)
    n = POLYDEG + 1
    nodes, weights = gauss_lobatto_nodes_weights(n)
    D = polynomial_derivative_matrix(nodes)
    dhat = [weights[j] / weights[i] * D[j, i] for i in 1:n, j in 1:n]
    dx = 2 / nelem
    x = [-1 + (e - 1 + (nodes[i] + 1) / 2) * dx for i in 1:n, e in 1:nelem]
    p = (; dhat = RealT.(dhat),
         left = Int32[mod1(e - 1, nelem) for e in 1:nelem],
         right = Int32[mod1(e + 1, nelem) for e in 1:nelem],
         invJ = fill(RealT(2 / dx), nelem),
         inv_w1 = RealT(1 / weights[1]), inv_wn = RealT(1 / weights[end]))
    u0 = RealT.(@. 1 + sinpi(x) / 2)
    dt = RealT(0.5 * dx / (1.5 * (2 * POLYDEG + 1)))
    return u0, p, dt
end

# ---------------------------------------------------------------- CUDA.jl kernels
function volume_kernel!(du, u, dhat)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    e = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= size(du, 1) && e <= size(du, 2)
        acc = zero(eltype(du))
        for j in axes(du, 1)
            @inbounds acc += dhat[i, j] * flux(u[j, e])
        end
        @inbounds du[i, e] = acc
    end
    return nothing
end

# one element per block, like TrixiCUDA's `flux_weak_form_kernel!`
function volume_kernel_shmem!(du, u, dhat)
    n = size(du, 1)
    shmem_dhat = CuDynamicSharedArray(eltype(du), (n, n))
    shmem_flux = CuDynamicSharedArray(eltype(du), n, sizeof(eltype(du)) * n * n)
    i = threadIdx().x
    e = blockIdx().x
    for j in 1:n
        @inbounds shmem_dhat[i, j] = dhat[i, j]
    end
    @inbounds shmem_flux[i] = flux(u[i, e])
    sync_threads()
    acc = zero(eltype(du))
    for j in 1:n
        @inbounds acc += shmem_dhat[i, j] * shmem_flux[j]
    end
    @inbounds du[i, e] = acc
    return nothing
end

function interface_flux_kernel!(fstar, u, right)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= length(fstar)
        n = size(u, 1)
        @inbounds fstar[k] = llf(u[n, k], u[1, right[k]])
    end
    return nothing
end

function surface_jacobian_kernel!(du, fstar, left, invJ, inv_w1, inv_wn)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    e = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= size(du, 1) && e <= size(du, 2)
        @inbounds val = du[i, e]
        if i == 1
            @inbounds val += fstar[left[e]] * inv_w1
        end
        if i == size(du, 1)
            @inbounds val -= fstar[e] * inv_wn
        end
        @inbounds du[i, e] = val * invJ[e]
    end
    return nothing
end

function rhs_cuda!(du, u, fstar, p, ::Val{shmem}) where {shmem}
    n, nelem = size(u)
    if shmem
        @cuda threads=n blocks=nelem shmem=sizeof(RealT) * (n * n + n) volume_kernel_shmem!(du, u, p.dhat)
    else
        @cuda threads=(n, 64) blocks=(1, cld(nelem, 64)) volume_kernel!(du, u, p.dhat)
    end
    @cuda threads=256 blocks=cld(nelem, 256) interface_flux_kernel!(fstar, u, p.right)
    @cuda threads=(n, 64) blocks=(1, cld(nelem, 64)) surface_jacobian_kernel!(du, fstar, p.left, p.invJ, p.inv_w1, p.inv_wn)
    return nothing
end

# Named functions, not closures: Reactant traces captured variables, and a captured `Bool`
# becomes a `TracedRNumber{Bool}`, which breaks `Val(shmem)`.
rhs_plain!(du, u, fstar, p) = rhs_cuda!(du, u, fstar, p, Val(false))
rhs_shmem!(du, u, fstar, p) = rhs_cuda!(du, u, fstar, p, Val(true))

# ---------------------------------------------------------------- CPU reference
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

# ---------------------------------------------------------------- time stepping
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

# `@trace` is a no-op outside Reactant, so the native CUDA run uses the same function
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
relerr(a, b) = maximum(abs, Array(a) .- b) / maximum(abs, b)

# ---------------------------------------------------------------- main
u0, p, dt = setup(NELEM)
n = POLYDEG + 1
@printf("backend=%s  nelem=%d  polydeg=%d  DOFs=%d  %s  nsteps=%d  dt=%.3e\n",
        BACKEND, NELEM, POLYDEG, length(u0), RealT, NSTEPS, dt)

p64 = map(x -> x isa AbstractArray{<:AbstractFloat} || x isa AbstractFloat ? Float64.(x) : x, p)
u_ref = Float64.(u0)
du_ref = similar(u_ref)
rhs_ref!(du_ref, u_ref, zeros(NELEM), p64)
simulate!(rhs_ref!, u_ref, similar(u_ref), zero(u_ref), zeros(NELEM), p64, Float64(dt), NSTEPS, Float64)
# same arithmetic in RealT on the CPU: separates implementation errors from rounding
u_r32 = copy(u0)
du_r32 = similar(u_r32)
rhs_ref!(du_r32, u_r32, zeros(RealT, NELEM), p)
simulate!(rhs_ref!, u_r32, similar(u_r32), zero(u_r32), zeros(RealT, NELEM), p, dt, NSTEPS, RealT)
@printf("CPU %s vs CPU Float64: err(rhs)=%.2e  err(u_T)=%.2e   max|u_T-u0|=%.2e (solution does evolve)\n", RealT, relerr(du_r32, du_ref), relerr(u_r32, u_ref), maximum(abs, u_r32 .- u0))

results = []

if CUDA.functional() && BACKEND == "gpu"
    for shmem in (false, true)
        name = shmem ? "cuda_shmem" : "cuda"
        rhsf! = shmem ? rhs_shmem! : rhs_plain!
        cp = map(x -> x isa AbstractArray ? CuArray(x) : x, p)
        u, du, dtmp, fs = CuArray(u0), CUDA.zeros(RealT, size(u0)), CUDA.zeros(RealT, size(u0)), CUDA.zeros(RealT, NELEM)
        rhs_only!(rhsf!, du, u, fs, cp)
        e_rhs = relerr(du, du_r32)
        t_rhs = mintime(() -> (rhs_only!(rhsf!, du, u, fs, cp); CUDA.synchronize()))
        simulate!(rhsf!, u, du, dtmp, fs, cp, dt, NSTEPS, RealT)
        e_sol = relerr(u, u_r32)
        u .= CuArray(u0)
        t_sim = mintime(() -> (simulate!(rhsf!, u, du, dtmp, fs, cp, dt, NSTEPS, RealT); CUDA.synchronize()); reps = 3)
        push!(results, (name, "ok", e_rhs, e_sol, t_rhs, t_sim, NaN))
    end
end

modes = BACKEND == "gpu" ? [(false, false), (false, true), (true, false), (true, true)] :
        [(true, false), (true, true), (false, false)]
for (raise, shmem) in modes
    name = (raise ? "reactant_raise" : "reactant_noraise") * (shmem ? "_shmem" : "")
    rhsf! = shmem ? rhs_shmem! : rhs_plain!
    try
        rp = Reactant.to_rarray(p)
        u, du, dtmp = Reactant.to_rarray(u0), Reactant.to_rarray(zero(u0)), Reactant.to_rarray(zero(u0))
        fs = Reactant.to_rarray(zeros(RealT, NELEM))
        hlo = string(@code_hlo raise=raise rhs_only!(rhsf!, du, u, fs, rp))
        kernel_left = any(k -> occursin(k, hlo), ("kernel_call", "jit_call", "custom_call", "llvm.func"))
        println("  [$name] HLO: ", count("stablehlo.", hlo), " stablehlo ops, custom_call=", count("custom_call", hlo), ", kernel_call=", count("kernel_call", hlo), ", while=", count("stablehlo.while", hlo))
        tc = @elapsed f_rhs = @compile raise=raise sync=true rhs_only!(rhsf!, du, u, fs, rp)
        f_rhs(rhsf!, du, u, fs, rp)
        e_rhs = relerr(du, du_r32)
        t_rhs = mintime(() -> f_rhs(rhsf!, du, u, fs, rp))
        tc += @elapsed f_sim = @compile raise=raise sync=true simulate!(rhsf!, u, du, dtmp, fs, rp, dt, NSTEPS, RealT)
        f_sim(rhsf!, u, du, dtmp, fs, rp, dt, NSTEPS, RealT)
        e_sol = relerr(u, u_r32)
        println("  [$name] max|u_T-u0| = ", maximum(abs, Array(u) .- u0))
        t_sim = mintime(() -> f_sim(rhsf!, u, du, dtmp, fs, rp, dt, NSTEPS, RealT); reps = 3)
        push!(results, (name, kernel_left ? "ok (kernel kept as PTX)" : "ok (pure StableHLO)",
                        e_rhs, e_sol, t_rhs, t_sim, tc))
    catch err
        msg = first(split(sprint(showerror, err), '\n'))
        push!(results, (name, "FAIL: " * first(msg, 160), NaN, NaN, NaN, NaN, NaN))
    end
end

println()
@printf("%-24s %-34s %10s %10s %10s %12s %9s\n", "mode", "status", "err(rhs)*", "err(u_T)*",
        "rhs [ms]", "sim [ms]", "compile[s]")
for (name, st, e1, e2, t1, t2, tc) in results
    @printf("%-24s %-34s %10.2e %10.2e %10.3f %12.2f %9.1f\n", name, first(st, 34), e1, e2, t1, t2, tc)
    startswith(st, "FAIL") && println("    ", st)
end
println("* relative to the CPU reference in the same precision ($RealT)")
