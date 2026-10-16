# E4: which of Trixi.jl's KernelAbstractions kernels can Reactant compile and raise?
#
# Each case runs an upstream P4est elixir (setup only), then evaluates rhs! once on the CPU with
# a recording KA backend: every kernel launch is captured (kernel, ndrange, arguments) and also
# executed with KA's CPU backend (natively with CUDA if that fails), which gives reference
# outputs. Each distinct kernel is then
# replayed alone with Reactant arrays under `@compile raise=...` and compared with the CPU launch.
# This sidesteps adapting Trixi's caches to Reactant (see reactant_overrides.jl): only the
# arguments the kernel actually receives are converted.
#
# usage (TrixiDocker image): julia experiments/e4_kernel_inventory.jl [gpu|cpu] [raise|noraise] [case|all]
# Optional: E4_KERNEL=<name> checks only that kernel; E4_WGS=4,4,4,1 replays it with that workgroup
# size instead of the recorded one, as a static kernel parameter (E4_WGS_MODE=static, default) or
# as the `workgroupsize` launch keyword (E4_WGS_MODE=launch). E4_NDRANGE=static makes the recorded
# ndrange a static kernel parameter as well (KA then folds the index arithmetic and bounds check).
using Trixi, Reactant, CUDA, KernelAbstractions, StaticArrays, Printf
using OrdinaryDiffEqLowStorageRK, OrdinaryDiffEqSSPRK
const KA = KernelAbstractions

const BACKEND = get(ARGS, 1, "gpu")
const RAISE = get(ARGS, 2, "raise") == "raise"
const ONLY = get(ARGS, 3, "all")
# set by run_e4.sh: kernels that crashed the process before, and a file naming the kernel
# being compiled, which survives a crash
const SKIP = split(get(ENV, "E4_SKIP", ""), ','; keepempty = false)
const CURRENT = get(ENV, "E4_CURRENT", "")
const KERNEL_FILTER = get(ENV, "E4_KERNEL", "")
const WGS = get(ENV, "E4_WGS", "")
const WGS_MODE = get(ENV, "E4_WGS_MODE", "static")
const NDRANGE_MODE = get(ENV, "E4_NDRANGE", "recorded")
# E4_PROFILE=1: `Reactant.@profile` of each replayed kernel, optimized HLO to logs/e4_xla_*.txt

# KA's bounds check (`__validindex`, always on for the CUDA backend) tests `I in ndrange`, which Base
# evaluates as `all(map(in, ...))`; for 4D indices that stays a loop over a stack tuple of Bools,
# an `scf.while` that Reactant cannot raise (E11). E4_UNROLL_IN=1 adds an unrolled, branch-free
# method for the CartesianIndices that KA uses.
@inline _allin(::Tuple{}, ::Tuple{}) = true
@inline _allin(i::Tuple, r::Tuple) = (first(i) in first(r)) & _allin(Base.tail(i), Base.tail(r))
if get(ENV, "E4_UNROLL_IN", "0") == "1"
    @inline Base.in(i::CartesianIndex{N}, r::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}}) where {N} =
        _allin(Tuple(i), r.indices)
end
Reactant.set_default_backend(BACKEND)

# ---------------------------------------------------------------- symbolic node indices
# Interface/boundary kernels receive arrays of tuples like (:end, :i_forward) and branch on
# them with `===`, which XLA cannot represent. Encode them as Int8 and give the index helpers
# methods for the encoded type; the kernels themselves stay untouched.
struct IdxCode
    c::Int8
end
const INDEX_SYMBOLS = (:begin, :end, :i_forward, :i_backward, :j_forward, :j_backward)
encode(s::Symbol) = IdxCode(Int8(findfirst(==(s), INDEX_SYMBOLS)))
const BEGIN, END, I_FWD, I_BWD, J_FWD = (IdxCode(Int8(k)) for k in 1:5)
Base.:(==)(a::IdxCode, b::IdxCode) = a.c == b.c

# not an AbstractArray, so Reactant traces it as a plain struct holding one array
struct CodedIndices{N, A}
    data::A  # size (N, size(original)...)
end
CodedIndices{N}(data) where {N} = CodedIndices{N, typeof(data)}(data)
@inline Base.getindex(x::CodedIndices{N}, I::Vararg{Integer}) where {N} =
    ntuple(k -> IdxCode(@inbounds x.data[k, I...]), Val(N))
Trixi.Adapt.adapt_structure(to, x::CodedIndices{N}) where {N} =
    CodedIndices{N}(Trixi.Adapt.adapt(to, x.data))
function CodedIndices(x::AbstractArray{NTuple{N, Symbol}}) where {N}
    data = Array{Int8}(undef, N, size(x)...)
    for I in CartesianIndices(x), k in 1:N
        data[k, I] = encode(x[I][k]).c
    end
    return CodedIndices{N}(data)
end

@inline function Trixi.index_to_start_step_2d(index::IdxCode, index_range)
    b, e = first(index_range), last(index_range)
    index == BEGIN && return b, 0
    index == END && return e, 0
    index == I_FWD && return b, 1
    return e, -1
end
@inline function Trixi.index_to_start_step_3d(index::IdxCode, index_range)
    b, e = first(index_range), last(index_range)
    index == BEGIN && return b, 0, 0
    index == END && return e, 0, 0
    index == I_FWD && return b, 1, b - e - 1
    index == I_BWD && return e, -1, e + 1 - b
    index == J_FWD && return b, 0, 1
    return e, 0, -1
end
@inline Trixi.interpolation_side(index::IdxCode) = index == BEGIN ? 1 : 2
@inline function Trixi.surface_indices(indices::NTuple{3, IdxCode})
    i1, i2, i3 = indices
    (i1 == BEGIN || i1 == END) && return (i2, i3)
    (i2 == BEGIN || i2 == END) && return (i1, i3)
    return (i1, i2)
end
@inline function Trixi.indices2direction(indices::NTuple{2, IdxCode})
    indices[1] == BEGIN && return 1
    indices[1] == END && return 2
    indices[2] == BEGIN && return 3
    return 4
end
@inline function Trixi.indices2direction(indices::NTuple{3, IdxCode})
    indices[1] == BEGIN && return 1
    indices[1] == END && return 2
    indices[2] == BEGIN && return 3
    indices[2] == END && return 4
    indices[3] == BEGIN && return 5
    return 6
end

# ---------------------------------------------------------------- recording backend
struct RecordingBackend <: KA.Backend end
KA.allocate(::RecordingBackend, ::Type{T}, dims::Tuple) where {T} = Array{T}(undef, dims)
KA.synchronize(::RecordingBackend) = nothing
# `true` makes `kernel(backend)` store the `gpu_` function, which is what the replay needs
KA.isgpu(::RecordingBackend) = true
KA.construct(backend::RecordingBackend, ::S, ::NDRange, f::F) where {S <: KA._Size, NDRange <: KA._Size, F} =
    KA.Kernel{RecordingBackend, S, NDRange, F}(backend, f)
KA.copyto!(::RecordingBackend, A, B) = copyto!(A, B)

const RECORDING = Ref(false)
const LAUNCHES = []

function Trixi.trixi_backend(x::Array)
    RECORDING[] && return RecordingBackend()
    return invoke(Trixi.trixi_backend, Tuple{Any}, x)
end

kernel_name(f) = replace(String(nameof(f)), r"^(gpu|cpu)_" => "")

function (k::KA.Kernel{RecordingBackend, WS, ND})(args...; ndrange = nothing,
                                                    workgroupsize = nothing) where {WS, ND}
    gpu_f = k.f
    mod = parentmodule(gpu_f)
    cpu_f = getfield(mod, Symbol("cpu_", kernel_name(gpu_f)))
    before = deepcopy(args)
    ref = try
        KA.Kernel{KA.CPU, WS, ND, typeof(cpu_f)}(KA.CPU(), cpu_f)(args...; ndrange, workgroupsize)
        :cpu
    catch
        # Trixi's 3D shared-memory kernels keep locals across `@synchronize` without
        # `@private`, which works on GPUs only: use the native CUDA launch as reference
        CUDA.functional() || rethrow()
        foreach(copyto!, leaves(args), leaves(before))
        run_cuda!(gpu_f, WS, ND, args; ndrange, workgroupsize)
        :cuda
    end
    push!(LAUNCHES, (; gpu_f, WS, ND, ndrange, workgroupsize, before, after = deepcopy(args), ref))
    return nothing
end

function run_cuda!(gpu_f, WS, ND, args; ndrange, workgroupsize)
    cargs = Trixi.Adapt.adapt(CuArray, args)
    backend = CUDA.CUDABackend()
    KA.Kernel{typeof(backend), WS, ND, typeof(gpu_f)}(backend, gpu_f)(cargs...; ndrange,
                                                                     workgroupsize)
    KA.synchronize(backend)
    foreach((a, c) -> copyto!(a, Array(c)), leaves(args), leaves(cargs))
    return nothing
end

# ---------------------------------------------------------------- argument conversion
# Go through Adapt like `trixi_adapt` does, so arrays nested in Trixi structs (e.g. the
# basis matrices inside `dg`) are converted as well; `seen` keeps shared arrays shared.
struct ToReactant
    seen::IdDict{Any, Any}
end
Trixi.Adapt.adapt_storage(to::ToReactant, x::Array{<:Union{AbstractFloat, Integer}}) =
    get!(() -> Reactant.to_rarray(x), to.seen, x)
Trixi.Adapt.adapt_storage(to::ToReactant, x::Array{<:NTuple{N, Symbol}}) where {N} =
    get!(to.seen, x) do
        c = CodedIndices(x)
        CodedIndices{N}(Reactant.to_rarray(c.data))
    end
convert_arg(x, seen) = Trixi.Adapt.adapt(ToReactant(seen), x)

# Reactant traces a struct by rebuilding its type with traced fields, but a
# `TracedRArray{T}` is not an `AbstractMatrix{T}` (its eltype is `TracedRNumber{T}`), so
# `LobattoLegendreBasis` and `LobattoLegendreMortarL2`, whose type parameters require
# `<: AbstractMatrix{RealT}`, cannot hold Reactant arrays ("Failed to find a valid type for
# typevar"). Wrap their matrices in a thin `AbstractMatrix{T}` that Reactant traces as a
# plain struct, so they stay device arrays as in Trixi's CUDA path (`E4_MATRICES=device`).
# `E4_MATRICES=static` passes them as StaticArrays instead; kernels then index a constant
# aggregate at run time, which needs a stack copy (`llvm.alloca`) that raising rejects.
const MATRICES = get(ENV, "E4_MATRICES", "device")
struct DeviceMatrix{T, A} <: AbstractMatrix{T}
    data::A
end
DeviceMatrix{T}(data) where {T} = DeviceMatrix{T, typeof(data)}(data)
Base.size(m::DeviceMatrix) = size(m.data)
Base.@propagate_inbounds Base.getindex(m::DeviceMatrix, i::Int, j::Int) = m.data[i, j]
Trixi.Adapt.adapt_structure(to, m::DeviceMatrix{T}) where {T} =
    DeviceMatrix{T}(Trixi.Adapt.adapt(to, m.data))

struct ToStatic end
Trixi.Adapt.adapt_storage(::ToStatic, x::Matrix) = SMatrix{size(x)...}(x)
Trixi.Adapt.adapt_storage(::ToStatic, x::Vector) = SVector{length(x)}(x)
struct ToDevice
    seen::IdDict{Any, Any}
end
Trixi.Adapt.adapt_storage(to::ToDevice, x::Matrix{T}) where {T} =
    get!(() -> DeviceMatrix{T}(Reactant.to_rarray(x)), to.seen, x)
basis_adaptor(to::ToReactant) = MATRICES == "static" ? ToStatic() : ToDevice(to.seen)
Trixi.Adapt.adapt_structure(to::ToReactant, x::Trixi.LobattoLegendreBasis) =
    Trixi.Adapt.adapt(basis_adaptor(to), x)
Trixi.Adapt.adapt_structure(to::ToReactant, x::Trixi.LobattoLegendreMortarL2) =
    Trixi.Adapt.adapt(basis_adaptor(to), x)

function describe_args(args)
    return join((string(i, ":", nameof(typeof(a)), isbits(a) ? "" : "*")
                 for (i, a) in enumerate(args)), " ")
end

# array leaves in a fixed traversal order, for comparing CPU and Reactant results
leaves!(out, x::AbstractArray{<:Number}) = push!(out, x)
leaves!(out, x::Union{Tuple, NamedTuple}) = (foreach(y -> leaves!(out, y), x); out)
leaves!(out, x) = out
leaves(x) = leaves!([], x)

function replay!(rk, ndrange, workgroupsize, args...)
    rk(args...; ndrange, workgroupsize)
    return nothing
end

# Every Trixi kernel takes `typeof(mesh)`. CUDA.jl passes a `Type` argument as a zero-size
# ghost, but Reactant's CUDA launcher (0.2.291) checks `typeof(arg)`, i.e. `DataType`,
# rejects it as non-isbits, and would lower it by `sizeof(arg)`, the size of a mesh
# instance. Tell it that type arguments carry no data (what an upstream fix would do); the
# kernels stay the compiled entry points, fully specialized, and are not modified.
const ReactantCUDAExt = Base.get_extension(Reactant, :ReactantCUDAExt)
ReactantCUDAExt._check_no_traced_in_kernel_arg(::Type{DataType}) = nothing
function ghost_type_args!(args)
    for a in args
        a isa DataType && sizeof(a) != 0 && @eval Base.sizeof(::Type{$a}) = 0
    end
    return nothing
end

function check_launch(L)
    seen = IdDict()
    rargs = convert_arg(L.before, seen)
    backend = KA.get_backend(Reactant.to_rarray(zeros(Float32, 1)))
    targs = rargs
    WS, wgs = L.WS, L.workgroupsize
    if !isempty(WGS)
        w = Tuple(parse.(Int, split(WGS, ',')))
        WS, wgs = WGS_MODE == "static" ? (KA.NDIteration.StaticSize{w}, nothing) : (L.WS, w)
    end
    ND = NDRANGE_MODE == "static" ? KA.NDIteration.StaticSize{Tuple(L.ndrange)} : L.ND
    rk = KA.Kernel{typeof(backend), WS, ND, typeof(L.gpu_f)}(backend, L.gpu_f)
    hlo = string(@code_hlo raise=RAISE replay!(rk, L.ndrange, wgs, targs...))
    leftover = count("custom_call", hlo) + count("kernel_call", hlo)
    gathers, scatters = count("stablehlo.gather", hlo), count("stablehlo.scatter", hlo)
    t = @elapsed f = @compile raise=RAISE sync=true replay!(rk, L.ndrange, wgs, targs...)
    f(rk, L.ndrange, wgs, targs...)
    err = 0.0
    details = String[]
    for (k, (b, a, r)) in enumerate(zip(leaves(L.before), leaves(L.after), leaves(rargs)))
        isequal(a, b) && continue  # not written by the kernel
        ra = Array(r)
        d = maximum(((x, y),) -> isequal(x, y) ? 0.0 : Float64(abs(x - y)), zip(ra, a))
        scale = maximum(x -> isfinite(x) ? Float64(abs(x)) : 0.0, a; init = 0.0)
        e = d / max(scale, eps(Float32))  # NaN in only one of them stays NaN
        err = max(err, e)
        if !(e <= 1e-4)
            bad = findall(i -> !isequal(ra[i], a[i]) && !(abs(ra[i] - a[i]) <= 1e-4 * scale),
                          eachindex(a))
            i1 = CartesianIndices(a)[first(bad)]
            push!(details, string("leaf ", k, " size ", size(a), ": ", length(bad), " differ, NaN cpu/reactant/before ",
                                  count(isnan, a), "/", count(isnan, ra), "/", count(isnan, b),
                                  ", first at ", Tuple(i1), " cpu=", a[i1], " reactant=", ra[i1], " before=", b[i1]))
        end
    end
    tr = minimum(_ -> (t0 = time_ns(); f(rk, L.ndrange, wgs, targs...); (time_ns() - t0) / 1e6), 1:5)
    if get(ENV, "E4_PROFILE", "0") == "1"   # `Reactant.@profile` of the replay + optimized HLO in logs/
        Reactant.@profile nrepeat = 5 f(rk, L.ndrange, wgs, targs...)
        name = kernel_name(L.gpu_f) * "_" * join(L.ndrange, "x")
        open(io -> print(io, @code_xla(raise = RAISE, replay!(rk, L.ndrange, wgs, targs...))),
             "logs/e4_xla_$(BACKEND)_$(name).txt", "w")
    end
    return (; leftover, err, compile = t, run_ms = tr, details, gathers, scatters)
end

# ---------------------------------------------------------------- cases (as in test_cuda_2d/3d.jl)
ex(d, e) = joinpath(examples_dir(), d, e)
const CASES = [
    ("2d_advection", ex("p4est_2d_dgsem", "elixir_advection_basic.jl"), (;)),
    ("2d_euler_weak", ex("p4est_2d_dgsem", "elixir_euler_source_terms.jl"), (;)),
    ("2d_euler_fd", ex("p4est_2d_dgsem", "elixir_euler_source_terms.jl"),  # as in test_cuda_2d.jl
     (; solver = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
                       volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber)))),
    ("2d_mhd_nonperiodic", ex("p4est_2d_dgsem", "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"), (;)),
    ("3d_advection", ex("p4est_3d_dgsem", "elixir_advection_basic.jl"), (;)),
    ("3d_euler_nonperiodic_weak", ex("p4est_3d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"), (;)),
    ("3d_euler_fd_halfsweep", ex("p4est_3d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"),
     (; volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber),
      flux_differencing_kernel = HalfSweep())),
    ("3d_euler_fd_fullsweep", ex("p4est_3d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"),
     (; volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber),
      flux_differencing_kernel = FullSweep())),
    ("3d_euler_fd_fullsweepglobal", ex("p4est_3d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"),
     (; volume_integral = VolumeIntegralFluxDifferencing(flux_kennedy_gruber),
      flux_differencing_kernel = FullSweepGlobal())),
    ("3d_euler_fd_turbo", ex("p4est_3d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"),
     (; volume_integral = VolumeIntegralFluxDifferencing(FluxTurbo(flux_ranocha)))),
    ("3d_mhd_nonperiodic", ex("p4est_3d_dgsem", "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"), (;)),
    ("3d_mhd_fullsweep", ex("p4est_3d_dgsem", "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
     (; flux_differencing_kernel = FullSweep())),
    # the remaining flux-differencing kernel methods (sweep type x nonconservative terms)
    ("3d_mhd_fullsweepglobal", ex("p4est_3d_dgsem", "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
     (; flux_differencing_kernel = FullSweepGlobal())),
    ("3d_euler_fd_turbo_fullsweep", ex("p4est_3d_dgsem", "elixir_euler_source_terms_nonperiodic.jl"),
     (; volume_integral = VolumeIntegralFluxDifferencing(FluxTurbo(flux_ranocha)),
      flux_differencing_kernel = FullSweep())),
    ("3d_mhd_turbo_halfsweep", ex("p4est_3d_dgsem", "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
     (; volume_integral = VolumeIntegralFluxDifferencing(FluxTurbo(flux_hindenlang_gassner,
                                                                   flux_nonconservative_powell)),
      flux_differencing_kernel = HalfSweep())),
    ("3d_mhd_turbo_fullsweep", ex("p4est_3d_dgsem", "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
     (; volume_integral = VolumeIntegralFluxDifferencing(FluxTurbo(flux_hindenlang_gassner,
                                                                   flux_nonconservative_powell)),
      flux_differencing_kernel = FullSweep())),
]

firstline(err) = first(first(split(sprint(showerror, err), '\n')), 170)

# Bindings and methods that `trixi_include` creates are newer than the world age this
# script's top-level loop was compiled in (Julia 1.12 binding partitions), so everything
# touching them runs through `invokelatest`.
function record_rhs!(mod)
    ode = getglobal(mod, :ode)
    du = similar(ode.u0)
    RECORDING[] = true
    try
        try  # initial condition kernel; must not cost the rhs! kernels if it fails
            Trixi.compute_coefficients(first(ode.tspan), ode.p)
        catch err
            println("  compute_coefficients FAILED: ", firstline(err))
        end
        ode.f(du, ode.u0, ode.p, first(ode.tspan))  # rhs! of the semidiscretization
    finally
        RECORDING[] = false
    end
    return nothing
end

# everything below runs only when this file is the script, not when it is `include`d (E11)
if abspath(PROGRAM_FILE) == @__FILE__

results = []
for (name, path, kw) in CASES
    (ONLY == "all" || ONLY == name) || continue
    println("== ", name)
    empty!(LAUNCHES)
    try
        mod = Module()
        Base.eval(mod, :(using Trixi, OrdinaryDiffEqLowStorageRK, OrdinaryDiffEqSSPRK))
        trixi_include(mod, path; sol = nothing, real_type = Float32, kw...)
        Base.invokelatest(record_rhs!, mod)
    catch err
        println("  setup/record FAILED: ", firstline(err))
        lines = split(sprint(showerror, err), '\n')
        foreach(l -> println("      | ", first(l, 200)), lines[2:min(end, 20)])
        push!(results, (name, "(setup)", "FAIL: " * firstline(err), NaN, NaN))
        continue
    end
    for L in unique(L -> (L.gpu_f, typeof(L.before)), LAUNCHES)
        kname = kernel_name(L.gpu_f)
        isempty(KERNEL_FILTER) || kname == KERNEL_FILTER || continue
        print("  ", rpad(kname, 62))
        flush(stdout)
        if kname in SKIP
            println("FAIL  the process crashed on this kernel in an earlier attempt")
            push!(results, (name, kname, "FAIL: process crashed (see .crash*.txt)", NaN, NaN))
            continue
        end
        isempty(CURRENT) || write(CURRENT, kname)
        r = try
            ghost_type_args!(L.before)  # defines methods: run the check in the newest world
            Base.invokelatest(check_launch, L)
        catch err
            println("FAIL  ", firstline(err))
            lines = split(sprint(showerror, err), '\n')
            foreach(l -> println("      | ", first(l, 200)), lines[2:min(end, 14)])
            rargs = try convert_arg(L.before, IdDict()) catch; L.before end
            println("      args (* = not isbits): ", describe_args(rargs))
            push!(results, (name, kname, "FAIL: " * firstline(err), NaN, NaN))
            continue
        end
        status = !(r.err <= 1e-4) ? "MISMATCH" : (r.leftover > 0 ? "ok (kernel kept)" : "ok (raised)")
        @printf("%-17s err=%.1e  compile=%.0fs  run=%.3fms  gather=%d scatter=%d%s\n", status,
                r.err, r.compile, r.run_ms, r.gathers, r.scatters, L.ref === :cuda ? "  ref=cuda" : "")
        foreach(d -> println("      ", d), r.details)
        push!(results, (name, kname, status, r.err, r.run_ms))
    end
end

isempty(CURRENT) || rm(CURRENT; force = true)
println("\nbackend=$BACKEND raise=$RAISE matrices=$MATRICES",
        isempty(WGS) ? "" : "  workgroupsize=($WGS) as $WGS_MODE", "  ndrange=$NDRANGE_MODE",
        "  unroll_in=", get(ENV, "E4_UNROLL_IN", "0"))
@printf("%-30s %-62s %-20s %9s %9s\n", "case", "kernel", "status", "rel.err", "run[ms]")
for (c, k, s, e, t) in results
    @printf("%-30s %-62s %-20s %9.1e %9.3f\n", c, k, first(s, 20), e, t)
    startswith(s, "FAIL") && println(" "^31, s)
end

end  # if PROGRAM_FILE
