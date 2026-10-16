# Method overrides that let upstream Trixi.jl's KernelAbstractions path take Reactant arrays.
# They only add methods (no edits to Trixi.jl), so they depend on Trixi internals: keep Trixi
# pinned to the version in the TrixiDocker image (0.17.14).
#
# Status: `trixi_adapt(ConcreteRArray, Float32, semi)` still fails. The P4est kernels pass
# `node_indices` elements to helpers that compare symbols with `===`
# (`index_to_start_step_2d`, `indices2direction`), which works on CUDA only because a
# CuArray stores Symbol pointers. Next step: an integer-coded index type plus methods for
# those helpers.
using Trixi, Reactant
using Trixi: nvariables, nnodes, nelements

# Trixi wraps u_ode via unsafe_wrap(pointer(u_ode)); Reactant arrays have no host pointer.
# A ReshapedArray is supported by Reactant, including writes back into the parent vector.
const RArr = Union{Reactant.AnyTracedRArray, Reactant.UnionAnyConcreteRArray}
@inline function Trixi.wrap_array(u_ode::RArr, mesh::Trixi.AbstractMesh, equations,
                                  dg::Trixi.DGSEM, cache)
    dims = (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
            nelements(dg, cache))
    return Base.ReshapedArray(u_ode, dims, ())
end
@inline Trixi.wrap_array_native(u_ode::RArr, mesh::Trixi.AbstractMesh, equations,
                                dg::Trixi.DG, cache) = Trixi.wrap_array(u_ode, mesh, equations, dg, cache)

# Containers keep a flat vector (for resize! under AMR) plus multi-dimensional views made
# with unsafe_wrap(pointer(vec)). Without AMR only the views are used in rhs!, so a reshaped
# device copy is enough here.
function Trixi.unsafe_wrap_or_alloc(::Trixi.TrixiAdaptor, vec::Reactant.AnyConcretePJRTArray, size)
    return Reactant.to_rarray(reshape(Array(vec), size))
end

# Interface node_indices (tuples like (:end, :i_forward)) and boundary names are arrays of
# symbols; those cannot live on an XLA device, so keep them on the host.
function Trixi.Adapt.adapt_storage(::Trixi.TrixiAdaptor,
                                   x::AbstractArray{<:Union{Symbol, Tuple{Vararg{Symbol}}}})
    return x
end

# Trixi PR #3332 (`:enum` image): node indices are tuples of the enum `NodeIndex.T` (UInt8)
# instead of symbols, so the index helpers work in kernels as they are. The containers still
# hold *arrays of tuples*, which cannot be XLA buffers, and their types require a
# `DenseArray{NTuple{N, NodeIndex.T}}`: store the enums' bits as an array of size
# (N, size(original)...) and rebuild the tuples on indexing. Signed storage: a UInt8 array as
# kernel argument fails in Reactant 0.2.291 (see e4_kernel_inventory.jl).
struct EnumTupleArray{N, E, M, A} <: DenseArray{NTuple{N, E}, M}
    data::A
end
EnumTupleArray{N, E}(data) where {N, E} = EnumTupleArray{N, E, ndims(data) - 1, typeof(data)}(data)
Base.size(x::EnumTupleArray) = Base.tail(size(x.data))
@inline Base.getindex(x::EnumTupleArray{N, E}, I::Vararg{Integer}) where {N, E} =
    ntuple(k -> reinterpret(E, @inbounds x.data[k, I...]), Val(N))
Trixi.Adapt.adapt_structure(to, x::EnumTupleArray{N, E}) where {N, E} =
    EnumTupleArray{N, E}(Trixi.Adapt.adapt(to, x.data))
function Trixi.Adapt.adapt_storage(::Trixi.TrixiAdaptor{S}, x::Array{NTuple{N, E}}) where {S <: Reactant.ConcreteRArray,
                                                                                         N, E <: Enum}
    data = Array{signed(Base.Enums.basetype(E))}(undef, N, size(x)...)
    for I in CartesianIndices(x), k in 1:N
        data[k, I] = reinterpret(eltype(data), x[I][k])
    end
    return EnumTupleArray{N, E}(Reactant.to_rarray(data))
end
function Trixi.unsafe_wrap_or_alloc(::Trixi.TrixiAdaptor, vec::EnumTupleArray{N, E}, size) where {N, E}
    return EnumTupleArray{N, E}(Reactant.to_rarray(reshape(Array(vec.data), N, size...)))
end

# `LobattoLegendreBasis`/`LobattoLegendreMortarL2` require `<: AbstractMatrix{RealT}` type
# parameters, which a Reactant array (eltype `TracedRNumber{T}` when traced) does not meet
# ("Failed to find a valid type for typevar"). As in E4: wrap their matrices in a thin
# `AbstractMatrix{T}` that Reactant traces as a plain struct.
struct DeviceMatrix{T, A} <: AbstractMatrix{T}
    data::A
end
DeviceMatrix{T}(data) where {T} = DeviceMatrix{T, typeof(data)}(data)
Base.size(m::DeviceMatrix) = size(m.data)
Base.@propagate_inbounds Base.getindex(m::DeviceMatrix, i::Int, j::Int) = m.data[i, j]
Trixi.Adapt.adapt_structure(to, m::DeviceMatrix{T}) where {T} = DeviceMatrix{T}(Trixi.Adapt.adapt(to, m.data))
struct ToDeviceMatrix{RealT} end
Trixi.Adapt.adapt_storage(::ToDeviceMatrix{RealT}, x::Matrix) where {RealT} =
    DeviceMatrix{RealT}(Reactant.to_rarray(RealT.(x)))
Trixi.Adapt.adapt_storage(::ToDeviceMatrix{RealT}, x::Vector{<:AbstractFloat}) where {RealT} = RealT.(x)
for B in (:LobattoLegendreBasis, :LobattoLegendreMortarL2)
    @eval Trixi.Adapt.adapt_structure(::Trixi.TrixiAdaptor{S, RealT}, x::Trixi.$B) where {S <: Reactant.ConcreteRArray, RealT} =
        Trixi.Adapt.adapt(ToDeviceMatrix{RealT}(), x)
end
