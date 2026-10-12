# E6: how fast are gathers on a TPU when whole blocks are fetched instead of scalars?
# E5 showed that raised kernels gather one scalar per element and that this is what makes them
# slow. An unstructured (P4est) mesh needs indirect access to neighbours, but it can fetch a
# whole face or element (block of M contiguous numbers) per index. Here: a 64 MiB Float32 array
# with nelem = 2^24 / M blocks of M values, read through
#   copy          Y = X .+ 1                         (bandwidth reference)
#   shift         neighbour e+1 via slices + concat  (structured mesh, no index array)
#   gather        Y = X[idx, :] / X[:, idx]          (one block per index; idx = shift or random)
#   scalar        Y = X[lin]                         (one scalar per index, as raised kernels do)
#   scatter       Y[idx, :] = X                      (blocks written through an index array)
# for both storage orders: "blockmajor" X is (M, nelem) [the block is contiguous in Julia],
# "elemmajor" X is (nelem, M). Bandwidth counts bytes read + written of X and Y.
# usage (TrixiDocker image): julia experiments/e6_tpu_gather.jl [tpu|gpu|cpu] [log2(total floats)]
using Reactant, Random, Printf

const BACKEND = get(ARGS, 1, "tpu")
const LOGN = parse(Int, get(ARGS, 2, "24"))
Reactant.set_default_backend(BACKEND)

copy_op(X) = X .+ 1.0f0
shift_bm(X) = hcat(X[:, 2:end], X[:, 1:1]) .+ 1.0f0            # (M, nelem): next block
shift_em(X) = vcat(X[2:end, :], X[1:1, :]) .+ 1.0f0            # (nelem, M)
gather_bm(X, idx) = X[:, idx] .+ 1.0f0
gather_em(X, idx) = X[idx, :] .+ 1.0f0
scalar_gather(X, lin) = reshape(X[lin], size(X)) .+ 1.0f0
function scatter_bm!(Y, X, idx)
    Y[:, idx] = X .+ 1.0f0
    return Y
end
function scatter_em!(Y, X, idx)
    Y[idx, :] = X .+ 1.0f0
    return Y
end

mintime(f; reps = 20) = minimum(_ -> (t0 = time_ns(); f(); (time_ns() - t0) / 1e6), 1:reps)
ops_in(hlo) = (gather = count("stablehlo.gather", hlo), scatter = count("stablehlo.scatter", hlo))

println("backend=$BACKEND  devices=", Reactant.devices(), "  total floats=2^$LOGN")
rows = []
for M in (1, 4, 16, 64, 256)
    nelem = 2^LOGN ÷ M
    bytes = 2 * 4 * 2^LOGN  # read X, write Y
    shift = [mod1(e + 1, nelem) for e in 1:nelem]
    rnd = randperm(MersenneTwister(1), nelem)
    for layout in (:blockmajor, :elemmajor)
        Xh = rand(Float32, layout === :blockmajor ? (M, nelem) : (nelem, M))
        X = Reactant.to_rarray(Xh)
        Y = Reactant.to_rarray(zero(Xh))
        cases = Any[("copy", copy_op, (X,)),
                    ("shift (slices)", layout === :blockmajor ? shift_bm : shift_em, (X,))]
        for (iname, idxh) in (("shift", shift), ("random", rnd))
            idx = Reactant.to_rarray(idxh)
            g = layout === :blockmajor ? gather_bm : gather_em
            s = layout === :blockmajor ? scatter_bm! : scatter_em!
            push!(cases, ("gather $iname", g, (X, idx)))
            push!(cases, ("scatter $iname", s, (Y, X, idx)))
            # the same access pattern, one scalar per index
            lin = layout === :blockmajor ?
                  vec([m + (idxh[e] - 1) * M for m in 1:M, e in 1:nelem]) :
                  vec([idxh[e] + (m - 1) * nelem for e in 1:nelem, m in 1:M])
            push!(cases, ("scalar $iname", scalar_gather, (X, Reactant.to_rarray(lin))))
        end
        for (name, f, args) in cases
            try
                hlo = string(@code_hlo f(args...))
                fc = @compile sync = true f(args...)
                fc(args...)
                t = mintime(() -> fc(args...))
                o = ops_in(hlo)
                @printf("M=%-4d %-10s %-16s %9.3f ms  %7.1f GB/s  (gather %d scatter %d)\n",
                        M, layout, name, t, bytes / t / 1e6, o.gather, o.scatter)
                push!(rows, (M, layout, name, t, bytes / t / 1e6))
            catch err
                println("M=$M $layout $name FAIL ", first(first(split(sprint(showerror, err), '\n')), 140))
            end
        end
        flush(stdout)
    end
end

println("\nsummary: effective bandwidth [GB/s] (read + write of 2^$LOGN Float32)")
names = unique(r[3] for r in rows)
for layout in (:blockmajor, :elemmajor)
    @printf("%-10s %-16s", layout, "")
    foreach(M -> @printf(" %8s", "M=$M"), (1, 4, 16, 64, 256))
    println()
    for name in names
        @printf("%-10s %-16s", "", name)
        for M in (1, 4, 16, 64, 256)
            r = findfirst(x -> x[1] == M && x[2] == layout && x[3] == name, rows)
            r === nothing ? @printf(" %8s", "–") : @printf(" %8.1f", rows[r][5])
        end
        println()
    end
end
