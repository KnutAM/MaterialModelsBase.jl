
"""
    get_num_tensorcomponents(::AbstractMaterial)

Returns the number of independent components for the given material. 
- If the material works with the small strain tensor and Cauchy stress, return 6 (default)
- If the material works with the deformation gradient and the 1st Piola-Kirchhoff stress, return 9
- If the material is a cohesive material working with vectors, return the number of vector components (e.g. 3)

Defaults to 6 if not overloaded
"""
get_num_tensorcomponents(::AbstractMaterial) = 6 # Default to small strain version

"""
    get_num_statevars(m::AbstractMaterial)

Return the number of state variables.
A tensorial state variable should be counted by how many components it has. 
E.g. if a state consists of one scalar and one symmetric 2nd order tensor,
`get_num_statevars` should return 7 (if the space dimension is 3).

Defaults to 0 if not overloaded
"""
get_num_statevars(::AbstractMaterial) = 0

"""
    get_num_params(m::AbstractMaterial)

Return the number of material parameters in `m`. No default value implemented. 
"""
function get_num_params end

# Conversion to vectors of parameters
"""
    material2vector!(v::AbstractVector, m::AbstractMaterial)
Put the material parameters of `m` into the vector `m`. 
This is typically used when the parameters should be fitted.
"""
function material2vector! end

"""
    vector2material(v::AbstractVector, ::MT) where {MT<:AbstractMaterial}
Create a material of type `MT` with the parameters according to `v`
"""
function vector2material end

"""
    material2vector(m::AbstractMaterial)
Out-of place version of `material2vector!`. Given `get_num_params`, this function
does not need to be overloaded unless another datatype than Float64 should be used.
"""
material2vector(m::AbstractMaterial) = material2vector!(zeros(get_num_params(m)), m)

struct MaterialDerivatives{T}
    dσdϵ::Matrix{T}
    dσdⁿs::Matrix{T}
    dσdp::Matrix{T}
    dsdϵ::Matrix{T}
    dsdⁿs::Matrix{T}
    dsdp::Matrix{T}
end

"""
    MaterialDerivatives(m::AbstractMaterial, T=Float64)

A struct that saves all derivative information using a `Matrix{T}` for each derivative.
If `get_num_tensorcomponents`, `get_num_statevars`, and `get_num_params` are implemented for `m`,
`MaterialDerivatives` does not need to be overloaded for `m`. 
"""
function MaterialDerivatives(m::AbstractMaterial, T=Float64)
    n_tensor = get_num_tensorcomponents(m)
    n_state = get_num_statevars(m)
    n_params = get_num_params(m)
    return MaterialDerivatives(
        zeros(T, n_tensor, n_tensor),  # dσdϵ
        zeros(T, n_tensor, n_state),   # dσdⁿs
        zeros(T, n_tensor, n_params),  # dσdp
        zeros(T, n_state, n_tensor),   # dsdϵ
        zeros(T, n_state, n_state),    # dsdⁿs
        zeros(T, n_state, n_params)    # dsdp
        )
end

"""
    allocate_differentiation_output(::AbstractMaterial)

When calculating the derivatives of a material, it can often be advantageous to have additional 
information from the solution procedure inside `material_response`. This can be obtained via an 
`AbstractExtraOutput`, and `allocate_differentiation_output` provides a standard function name 
for what `extra_output::AbstractExtraOutput` that should be allocated in such cases.

Defaults to an `NoExtraOutput` if not overloaded. 
"""
allocate_differentiation_output(::AbstractMaterial) = NoExtraOutput()

"""
    differentiate_material!(
        diff::MaterialDerivatives, 
        m::AbstractMaterial, 
        ϵ::Union{SecondOrderTensor, Vec}, 
        old::AbstractMaterialState, 
        Δt,
        cache::AbstractMaterialCache
        dσdϵ::AbstractTensor, 
        extra::AbstractExtraOutput;
        options=nothing
        )

Calculate the derivatives and save them in `diff`. 
"""
function differentiate_material! end