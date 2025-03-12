struct MaterialDerivatives{T}
    dσdϵ::Matrix{T}
    dσdⁿs::Matrix{T}
    dσdp::Matrix{T}
    dsdϵ::Matrix{T}
    dsdⁿs::Matrix{T}
    dsdp::Matrix{T}
end

"""
    MaterialDerivatives(m::AbstractMaterial)

A struct that saves all derivative information using a `Matrix{T}` for each derivative,
where `T=get_params_eltype(m)`. The dimensions are obtained from `get_num_tensorcomponents`, 
`get_num_statevars`, and `get_num_params`. The values should be updated in `differentiate_material!`
by direct access of the fields, where `σ` is the stress, `ϵ` the strain, `s` and `ⁿs` are the current 
and old state variables, and `p` the material parameter vector.

* `dσdϵ`
* `dσdⁿs`
* `dσdp`
* `dsdϵ`
* `dsdⁿs`
* `dsdp`

"""
function MaterialDerivatives(m::AbstractMaterial)
    T = get_params_eltype(m)
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
        extra::AbstractExtraOutput
        dσdϵ::AbstractTensor, 
        )

Calculate the derivatives and save them in `diff`, see
[`MaterialDerivatives`](@ref) for a description of the fields in `diff`.
"""
function differentiate_material! end