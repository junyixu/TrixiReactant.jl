# E1: does Reactant see the device, and what do Float32/Float64 cost on it?
# usage (TrixiDocker image): julia experiments/e1_basics.jl [gpu|cpu]
using Reactant, Printf

const BACKEND = get(ARGS, 1, "gpu")
Reactant.set_default_backend(BACKEND)
println("backend = ", BACKEND, "   devices = ", Reactant.devices())

axpy(a, x, y) = a .* x .+ y
fma_chain(x) = sum(abs2, sin.(x) .* cos.(x) .+ x .* x)

function bench(f, args...; reps = 20)
    fc = @compile sync = true f(args...)
    fc(args...)
    t = minimum(1:reps) do _
        t0 = time_ns()
        fc(args...)
        (time_ns() - t0) / 1e6
    end
    return t
end

for T in (Float32, Float64)
    n = 2^24
    x = Reactant.to_rarray(rand(T, n))
    y = Reactant.to_rarray(rand(T, n))
    t1 = bench(axpy, T(2), x, y)
    t2 = bench(fma_chain, x)
    @printf("%-8s n=%d  axpy %.3f ms (%.1f GB/s)   sin/cos-reduce %.3f ms\n",
            T, n, t1, 3 * n * sizeof(T) / t1 / 1e6, t2)
end
