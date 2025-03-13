
"""
    get_tensorbase(m::AbstractMaterial)

Return the type of the primary input (strain-like) and associated output (stress-like) 
to the material model. The default is `SymmetricTensor{2, 3}` for small-strain material 
models in 3D, but typically, it can be 
* Small strain material model: `SymmetricTensor{2, dim}`
* Large strain material model: `Tensor{2, dim}`
* Traction-separation law: `Vec{dim}`

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
    get_num_statevars(s::AbstractMaterialState)
    get_num_statevars(m::AbstractMaterial)

Return the number of state variables.
A tensorial state variable should be counted by how many components it has. 
E.g. if a state consists of one scalar and one symmetric 2nd order tensor in 3d,
`get_num_statevars` should return 7.

It suffices to define for the state, `s`, only, but defining for material, `m`, 
directly as well can improve performance. 
"""
function get_num_statevars end

get_num_statevars(::NoMaterialState) = 0
get_num_statevars(m::AbstractMaterial) = get_num_statevars(initial_material_state(m))

"""
    get_statevar_eltype(s::AbstractMaterialState)

Get the type used to store each scalar component of the material state variables,
defaults to `Float64`.
"""
get_statevar_eltype(::AbstractMaterialState) = Float64

"""
    get_num_params(m::AbstractMaterial)

Return the number of material parameters in `m`. No default value implemented. 
"""
function get_num_params end

"""
    get_params_eltype(m::AbstractMaterial)

Return the number type for the scalar material parameters, defaults to `Float64`
"""
get_params_eltype(::AbstractMaterial) = Float64

# Conversion functions
"""
    tovector!(v::AbstractVector, m::AbstractMaterial)
Put the material parameters of `m` into the vector `v`. 
This is typically used when the parameters should be fitted.

    tovector!(v::AbstractVector, s::AbstractMaterialState)
Put the state variables in `s` into the vector `v`.
This is typically used when differentiating the material 
wrt. the the old state variables.
"""
function tovector! end

"""
    fromvector(v::AbstractVector, ::MT) where {MT<:AbstractMaterial}
Create a material of type `MT` with the parameters according to `v`

    fromvector(v::AbstractVector, ::ST) where {ST<:AbstractMaterialState}
Create a material state of type `ST` with the values according to `v`
"""
function fromvector end

"""
    tovector(m::AbstractMaterial)

Out-of place version of `tovector!`. Relies on `get_num_params` and 
`get_params_eltype` to be correctly defined
"""
function tovector(m::AbstractMaterial)
    T = get_params_eltype(m)
    return tovector!(zeros(T, get_num_params(m)), m)
end

"""
    tovector(m::AbstractMaterialState)

Out-of place version of `tovector!`. Relies on `get_num_statevars` and 
`get_statevar_eltype` to be correctly defined
"""
function tovector(s::AbstractMaterialState)
    T = get_statevar_eltype(s)
    return tovector!(zeros(T, get_num_statevars(s)), s)
end

# Backwards compatibility
const get_parameter_type = get_params_eltype
const material2vector! = tovector!
const material2vector = tovector
const vector2material = fromvector

export material2vector!, material2vector, vector2material
