# E11: minimal reproducer for the Reactant crash that stops Trixi's FullSweepGlobal kernel
# (E4: segfault in `OptimizeRem::matchAndRewrite(arith::RemUIOp)` inside `AffineCFGPass`).
# FullSweepGlobal is launched with a 4D ndrange (nodes, nodes, nodes, elements), a dynamic
# workgroup size and `@index(Global, NTuple)`; the kernels here keep only that structure.
# Each call raises one variant; run every variant in its own process, since a crash kills it.
#   variant: copy  y[i,j,k,e] = 2 x[i,j,k,e]
#            loop  y[i,j,k,e] = sum over ii of x[ii,j,k,e]       (one sweep, as FullSweepGlobal)
#            loop3 sums over ii, jj and kk                        (all three sweeps)
#   wgs:     dynamic (no workgroup size, as Trixi) | static:4,4,4,1 | launch:4,4,4,1
# usage (TrixiDocker image): julia experiments/e11_remui_repro.jl [cpu|gpu|tpu] [variant] [wgs] [nelem]
using Reactant, CUDA, KernelAbstractions
const KA = KernelAbstractions

const BACKEND = get(ARGS, 1, "cpu")
const VARIANT = get(ARGS, 2, "loop3")
const WGS = get(ARGS, 3, "dynamic")
const NELEM = parse(Int, get(ARGS, 4, "64"))
Reactant.set_default_backend(BACKEND)

# UNROLL_IN=1: branch-free, unrolled `I in ndrange` for KA's bounds check (see e4_kernel_inventory.jl)
@inline _allin(::Tuple{}, ::Tuple{}) = true
@inline _allin(i::Tuple, r::Tuple) = (first(i) in first(r)) & _allin(Base.tail(i), Base.tail(r))
if get(ENV, "UNROLL_IN", "0") == "1"
    @inline Base.in(i::CartesianIndex{N}, r::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}}) where {N} =
        _allin(Tuple(i), r.indices)
end

@kernel function copy4!(y, x)
    i, j, k, e = @index(Global, NTuple)
    @inbounds y[i, j, k, e] = 2 * x[i, j, k, e]
end

@kernel function loop4!(y, x)
    i, j, k, e = @index(Global, NTuple)
    acc = zero(eltype(y))
    for ii in 1:size(x, 1)
        @inbounds acc += x[ii, j, k, e]
    end
    @inbounds y[i, j, k, e] = acc
end

@kernel function loop3_4!(y, x)
    i, j, k, e = @index(Global, NTuple)
    acc = zero(eltype(y))
    for ii in 1:size(x, 1)
        @inbounds acc += x[ii, j, k, e]
    end
    for jj in 1:size(x, 2)
        @inbounds acc += x[i, jj, k, e]
    end
    for kk in 1:size(x, 3)
        @inbounds acc += x[i, j, kk, e]
    end
    @inbounds y[i, j, k, e] = acc
end

const KERNELS = Dict("copy" => copy4!, "loop" => loop4!, "loop3" => loop3_4!)

function run!(kernel, y, x)
    backend = KA.get_backend(y)
    if WGS == "dynamic"
        kernel(backend)(y, x; ndrange = size(y))
    else
        mode, w = split(WGS, ':')
        wg = Tuple(parse.(Int, split(w, ',')))
        mode == "static" ? kernel(backend, wg)(y, x; ndrange = size(y)) :
                           kernel(backend)(y, x; ndrange = size(y), workgroupsize = wg)
    end
    return y
end

reference(x) = VARIANT == "copy" ? 2 .* x :
               VARIANT == "loop" ? repeat(sum(x; dims = 1), size(x, 1), 1, 1, 1) :
               repeat(sum(x; dims = 1), size(x, 1), 1, 1, 1) .+ repeat(sum(x; dims = 2), 1, size(x, 2), 1, 1) .+
               repeat(sum(x; dims = 3), 1, 1, size(x, 3), 1)

x0 = rand(Float32, 4, 4, 4, NELEM)
x, y = Reactant.to_rarray(x0), Reactant.to_rarray(zeros(Float32, size(x0)))
k = KERNELS[VARIANT]
println("backend=$BACKEND variant=$VARIANT wgs=$WGS nelem=$NELEM unroll_in=", get(ENV, "UNROLL_IN", "0"))
flush(stdout)
hlo = string(@code_hlo raise = true run!(k, y, x))
println("raised: kernel_call=", count("kernel_call", hlo), " stablehlo ops=", count("stablehlo.", hlo),
        " gather=", count("stablehlo.gather", hlo), " while=", count("stablehlo.while", hlo))
f = @compile raise = true sync = true run!(k, y, x)
f(k, y, x)
err = maximum(abs, Array(y) .- reference(x0)) / maximum(abs, reference(x0))
println("RESULT variant=$VARIANT wgs=$WGS: ok, rel. error ", err)
