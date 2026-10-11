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
