struct MaterialDerivatives{T}
    dσdϵ::Matrix{T}
    #dσdⁿs::Matrix{T}
    dσdp::Matrix{T}
    dsdϵ::Matrix{T}
    #dsdⁿs::Matrix{T}
    dsdp::Matrix{T}
end

function Base.getproperty(d::MaterialDerivatives, key::Symbol)
    if key === :dσdⁿs || key === :dsdⁿs
        error("You are probably assuming MaterialModelsBase v0.2 behavior for differentiation")
    else
        @inline getfield(d, key)
    end
end

"""
    MaterialDerivatives(m::AbstractMaterial)

A struct that saves all derivative information using a `Matrix{T}` for each derivative,
where `T=get_params_eltype(m)`. The dimensions are obtained from `get_num_tensorcomponents`, 
`get_num_statevars`, and `get_num_params`. `m` must support `tovector` and `fromvector`, while 
the output of `initial_material_state` must support `tovector`, and in addition the element type
of `tovector(initial_material_state(m))` must respect the element type in `tovector(m)` for any `m`.

The values should be updated in `differentiate_material!` by direct access of the fields, 
where `σ` is the stress, `ϵ` the strain, `s` and `ⁿs` are the current 
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
    dsdp = ForwardDiff.jacobian(p -> tovector(initial_material_state(fromvector(p, m))), tovector(m))
    return MaterialDerivatives(
        zeros(T, n_tensor, n_tensor),  # dσdϵ
        zeros(T, n_tensor, n_params),  # dσdp
        zeros(T, n_state, n_tensor),   # dsdϵ
        dsdp
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

struct StressStateDerivatives{T}
    mderiv::MaterialDerivatives{T}
    dϵdp::Matrix{T}
    dσdp::Matrix{T}
    ϵindex::SMatrix{3, 3, Int} # To allow indexing by (i, j) into only
    σindex::SMatrix{3, 3, Int} # saved values to avoid storing unused rows.
    # TODO: Reduce the dimensions, for now all entries (even those that are zero) are stored.
end

function StressStateDerivatives(::AbstractStressState, m::AbstractMaterial)
    mderiv = MaterialDerivatives(m)
    np = get_num_params(m)
    nt = get_num_tensorcomponents(m) # Should be changed to only save non-controlled entries
    dϵdp = zeros(nt, np)
    dσdp = zeros(nt, np)
    # Should be changed to only save non-controlled entries
    vo = Tensors.DEFAULT_VOIGT_ORDER[3]
    if nt == 6
        index = SMatrix{3, 3, Int}(min(vo[i, j], vo[j, i]) for i in 1:3, j in 1:3)
    else
        index = SMatrix{3, 3, Int}(vo)
    end
    return StressStateDerivatives(mderiv, dϵdp, dσdp, index, index)
end

"""
    differentiate_material!(ssd::StressStateDerivatives, stress_state, m, args...)

For material models implementing `material_response(m, args...)` and `differentiate_material!(::MaterialDerivatives, m, args...)`,
this method will work automatically by
1) Calling `σ, dσdϵ, state = material_response(stress_state, m, args...)` (except that `dσdϵ::FourthOrderTensor{dim = 3}` is extracted)
2) Calling `differentiate_material!(ssd.mderiv::MaterialDerivatives, m, args..., dσdϵ::FourthOrderTensor{3})`
3) Updating `ssd` according to the constraints imposed by the `stress_state`.

For material models that directly implement `material_response(stress_state, m, args...)`, this function should be overloaded directly
to calculate the derivatives in `ssd`. Here the user has full control and no modifications occur automatically, however, typically the 
(total) derivatives `ssd.dσdp`, `ssd.dϵdp`, and `ssd.mderiv.dsdp` should be updated. 
"""
function differentiate_material!(ssd::StressStateDerivatives, stress_state::AbstractStressState, m::AbstractMaterial, ϵ::AbstractTensor, args::Vararg{Any,N}) where {N}
    σ_full, dσdϵ_full, state, ϵ_full = stress_state_material_response(stress_state, m, ϵ, args...)
    differentiate_material!(ssd.mderiv, m, ϵ_full, args..., dσdϵ_full)
    
    if isa(stress_state, NoIterationState)
        copy!(ssd.dσdp, ssd.mderiv.dσdp)
        fill!(ssd.dϵdp, 0)
    else
        sc = stress_controlled_indices(stress_state, ϵ)
        ec = strain_controlled_indices(stress_state, ϵ)
        dσᶠdϵᶠ_inv = inv(get_unknowns(stress_state, dσdϵ_full)) # f: unknown strain components solved for during stress iterations
        ssd.dϵdp[sc, :] .= .-dσᶠdϵᶠ_inv * ssd.mderiv.dσdp[sc, :]
        ssd.dσdp[ec, :] .= ssd.mderiv.dσdp[ec, :] .+ ssd.mderiv.dσdϵ[ec, sc] * ssd.dϵdp[sc, :]
        ssd.mderiv.dsdp .+= ssd.mderiv.dsdϵ[:, sc] * ssd.dϵdp[sc, :]
    end
    return reduce_tensordim(stress_state, σ_full), reduce_stiffness(stress_state, dσdϵ_full), state, ϵ_full
end

"""
    stress_controlled_indices(stress_state::AbstractStressState, ::AbstractTensor)::SVector{N, Int}

Get the `N` indices that are stress-controlled in `stress_state`. The tensor input is used to 
determine if a symmetric or full tensor is used. 
"""
function stress_controlled_indices end

"""
    strain_controlled_indices(stress_state::AbstractStressState, ::AbstractTensor)::SVector{N, Int}

Get the `N` indices that are strain-controlled in `stress_state`. The tensor input is used to 
determine if a symmetric or full tensor is used. 
"""
function strain_controlled_indices end

# NoIterationState
stress_controlled_indices(::NoIterationState, ::AbstractTensor) = SVector{0,Int}()
strain_controlled_indices(::NoIterationState, ::SymmetricTensor) = @SVector([1, 2, 3, 4, 5, 6])
strain_controlled_indices(::NoIterationState, ::Tensor) = @SVector([1, 2, 3, 4, 5, 6, 7, 8, 9])

# UniaxialStress
stress_controlled_indices(::UniaxialStress, ::SymmetricTensor) = @SVector([2, 3, 4, 5, 6])
stress_controlled_indices(::UniaxialStress, ::Tensor) = @SVector([2, 3, 4, 5, 6, 7, 8, 9])
strain_controlled_indices(::UniaxialStress, ::AbstractTensor) = @SVector([1])

# UniaxialNormalStress
stress_controlled_indices(::UniaxialNormalStress, ::AbstractTensor) = @SVector([2,3])
strain_controlled_indices(::UniaxialNormalStress, ::SymmetricTensor) = @SVector([1, 4, 5, 6])
strain_controlled_indices(::UniaxialNormalStress, ::Tensor) = @SVector([1, 4, 5, 6, 7, 8, 9])

# PlaneStress 12 -> 6, 21 -> 9
stress_controlled_indices(::PlaneStress, ::SymmetricTensor) = @SVector([3, 4, 5])
stress_controlled_indices(::PlaneStress, ::Tensor) = @SVector([3, 4, 5, 7, 8])
strain_controlled_indices(::PlaneStress, ::SymmetricTensor) = @SVector([1, 2, 6])
strain_controlled_indices(::PlaneStress, ::Tensor) = @SVector([1, 2, 6, 9])

# GeneralStressState
stress_controlled_indices(ss::GeneralStressState{Nσ}, ::AbstractTensor) where Nσ = controlled_indices_from_tensor(ss.σ_ctrl, true, Val(Nσ))
function strain_controlled_indices(ss::GeneralStressState{Nσ,TT}, ::AbstractTensor) where {Nσ,TT}
    N = Tensors.n_components(Tensors.get_base(TT)) - Nσ
    return controlled_indices_from_tensor(ss.σ_ctrl, false, Val(N))
end
function controlled_indices_from_tensor(ctrl::AbstractTensor, return_if, ::Val{N}) where N
    return SVector{N}(i for (i, v) in pairs(tovoigt(SVector, ctrl)) if v == return_if)
end