"""
    tovector!(v::AbstractVector, obj; offset = 0)

Store the parameters in the object `obj` in `v`. This is typically used with `obj` as 
an `AbstractMaterial`, an `AbstractMaterialState`, or an `AbstractTensor`. The `offset` input makes it possible 
to write values starting at an offset location in `v`.

The implementation for `AbstractTensor`s is included in the `MaterialModelsBase.jl` package, and use the Mandel
notation for the conversion.
"""
function tovector! end

"""
    tovector([T::Type{<:AbstractArray}], obj)::T

Out-of place version of `tovector!`. Relies on `get_vector_length` and 
`get_vector_eltype` to be correctly defined, and defaults to `T = Array`.

Experimental support for `T = SArray` is also available, but performance may be suboptimal,
and can be improved by implementing a custom function for the given type.
"""
@inline tovector(obj) = tovector(Array, obj)

function tovector(::Type{<:Array}, obj)
    T = get_vector_eltype(obj)
    return tovector!(zeros(T, get_vector_length(obj)), obj)
end

function tovector(::Type{<:SArray}, obj)
    T = get_vector_eltype(obj)
    N = get_vector_length(obj)
    m = MVector{T, N}()
    @inline tovector!(m, obj)
    return SVector{T, N}(m)
end

"""
    fromvector(v::AbstractVector, ::OT; offset = 0)

Output an object of similar type to `OT`, but with parameters according to `v`. This is typically used with
an `AbstractMaterial`, an `AbstractMaterialState`, or an `AbstractTensor`. The `offset` input makes it possible 
to read values starting at an offset location in `v`. 

The implementation for `AbstractTensor`s is included in the `MaterialModelsBase.jl` package, and use the Mandel
notation for the conversion.
"""
function fromvector end

"""
    get_vector_length(obj)

Return the length of the vector representation of `obj` when using 
`tovector` or `tovector!`. The default implementation of `tovector` relies 
on this function being defined.
"""
function get_vector_length end

"""
    get_vector_eltype(obj)

Return the element type of the vector representation of `obj` when using 
`tovector`, i.e. with `T = get_vector_eltype(obj)` and `N = get_vector_length(obj)`,
we have `typeof(obj) == typeof(fromvector(zeros(T, N), obj))`.
"""
function get_vector_eltype end

"""
    get_tensorbase(m::AbstractMaterial)

Return the type of the primary input (strain-like) and associated output (stress-like) 
to the material model. The default is `SymmetricTensor{2, 3}` for small-strain material 
models in 3D, but it can be 
* Small strain material model: Small strain tensor, `ϵ::SymmetricTensor{2, dim}`
* Large strain material model: Deformation gradient, `F::Tensor{2, dim}`
* Traction-separation law: Displacement jump, `Δu::Vec{dim}`

Where `dim = 3` is most common, but any value, 1, 2, or 3, is valid. 
"""
get_tensorbase(::AbstractMaterial) = SymmetricTensor{2, 3}

"""
    get_num_tensorcomponents(::AbstractMaterial)

Returns the number of independent components for the given material. 

!!! note
    It is not required to implement this function, it is inferred by the 
    implementation of [`get_tensorbase`](@ref)

"""
get_num_tensorcomponents(m::AbstractMaterial) = Tensors.n_components(get_tensorbase(m))

"""
    get_num_statevars(m::AbstractMaterial)

Return the number of scalar values required to store the state of `m`, i.e. `length(tovector(initial_material_state(m)))`.
The default implementation works provided that `s = initial_material_state(m)` and `get_vector_length(s)` is defined.
"""
function get_num_statevars(m::AbstractMaterial)
    return get_vector_length(initial_material_state(m))
end

# Implementations for types defined in MaterialModelsBase
tovector!(v::AbstractVector, ::NoMaterialState; kwargs...) = v
fromvector(::AbstractVector{T}, ::NoMaterialState; kwargs...) where {T} = NoMaterialState{T}()
get_vector_length(::NoMaterialState) = 0
get_vector_eltype(::NoMaterialState{T}) where {T} = T

# Tensors.jl implementation
get_vector_length(::TT) where {TT <: AbstractTensor} = Tensors.n_components(Tensors.get_base(TT))
get_vector_eltype(a::AbstractTensor) = eltype(a)

function tovector!(v::AbstractVector, a::SecondOrderTensor; offset = 0)
    return tomandel!(v, a; offset)
end
function tovector!(v::AbstractVector, a::Union{Tensor{4, <:Any, <:Any, M}, SymmetricTensor{4, <:Any, <:Any, M}}; offset = 0) where {M}
    N = round(Int, sqrt(M))
    m = reshape(view(v, offset .+ (1:M)), (N, N))
    tomandel!(m, a)
    return v
end

function fromvector(v::AbstractVector, ::TT; offset = 0) where {TT <: SecondOrderTensor}
    return frommandel(Tensors.get_base(TT), v; offset)
end

function fromvector(v::AbstractVector, ::TT; offset = 0) where {TT <: FourthOrderTensor}
    TB = Tensors.get_base(TT)
    M = Tensors.n_components(TB)
    N = round(Int, sqrt(M))
    return frommandel(TB, reshape(view(v, offset .+ (1:M)), (N, N)))
end
