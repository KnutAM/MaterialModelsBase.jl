abstract type AbstractStressState end

"""
    material_response(stress_state::AbstractStressState, m::AbstractMaterial, args...)

To be able to use material models implemented for 3d stress and strain states in lower-dimensional 
simulations, such as 2d plane stress, `MaterialModelsBase.jl` provides a set of stress states. 
For some states, such as plane stress, iterations will be performed to find the correct state.
For other states, such as plane strain, the input is only padded with zeros and the out-of-plane 
components are removed from the output. 

For someone implementing a material model, it is also possible to use dispatch on both the 
stress state and the material to provide an efficient implementation of a reduced stress state.
Note that the interface expects the full strain tensor to be given as a fourth output in this case,
but it is optional to implement this but such a deviation should be documented as it could cause 
problems for users of the material implementation. 

The arguments are the same as for `material_response(::AbstractMaterial)`.
However, both a full and reduced strain input is accepted. For a full strain input, 
the out-of-plane components are used as an initial guess. For all cases, 
the full strain tensor giving the desired reduced response is given as a 4th output.

See also [`ReducedStressState`](@ref).
"""
material_response(::AbstractStressState, ::AbstractMaterial, args...)

@inline function material_response(stress_state::AbstractStressState, m::AbstractMaterial, args::Vararg{Any,N}) where N
    stress_3d, stiff_3d, state, strain_3d = stress_state_material_response(stress_state, m, args...)
    return reduce_tensordim(stress_state, stress_3d), reduce_stiffness(stress_state, stiff_3d, strain_3d, stress_3d), state, strain_3d
end

update_stress_state!(::AbstractStressState, σ) = nothing

"""
    ReducedStressState(s::AbstractStressState, m::AbstractMaterial)
    
Creates a subtype of `AbstractMaterial` that wraps a stress state and a material, such that 
calls to `material_response(w::ReducedStressState, args...)` gives the same result as 
`material_response(s, m, args...)`. 
Calls to `initial_material_state`, `allocate_material_cache`, 
`get_num_tensorcomponents`, `get_num_statevars`, `get_vector_length`, 
`get_vector_eltype`, `tovector!`, `tovector`, 
and `allocate_differentiation_output` are forwarded with `m` as the argument. 
`fromvector` returns `ReducedStressState` and is supported as well.
"""
struct ReducedStressState{S<:AbstractStressState,M<:AbstractMaterial} <: AbstractMaterial
    stress_state::S
    material::M
end
for op in ( :initial_material_state, :allocate_material_cache, :get_tensorbase,
            :get_num_tensorcomponents, :get_num_statevars, :get_vector_length, 
            :get_vector_eltype, :allocate_differentiation_output)
    @eval @inline $op(rss::ReducedStressState) = $op(rss.material)
end
function tovector!(v::AbstractVector, rss::ReducedStressState)
    return tovector!(v, rss.material)
end
function fromvector(v::AbstractVector, rss::ReducedStressState)
    return ReducedStressState(rss.stress_state, fromvector(v, rss.material))
end

function material_response(rss::ReducedStressState, args...)
    return material_response(rss.stress_state, rss.material, args...)
end

# Cases without stress iterations
""" 
    FullStressState()

Return the full stress state, without any constraints. 
Equivalent to not giving any stress state to the 
`material_response` function, except that when given, 
the full strain (given as input) is also an output which 
can be useful if required for consistency with the other 
stress states. 
"""
struct FullStressState <: AbstractStressState end

""" 
    PlaneStrain()

Plane strain such that if only 2d-components (11, 12, 21, and 22) are given,
the remaining strain components are zero. The output is the reduced set, 
with the mentioned components. It is possible to give non-zero values for the
other strain components, and these will be used for the material evaluation. 
"""
struct PlaneStrain <: AbstractStressState end

""" 
    UniaxialStrain()

Uniaxial strain such that if only the 11-strain component is given,
the remaining strain components are zero. The output is the reduced set, i.e. 
only the 11-stress-component. It is possible to give non-zero values for the
other strain components, and these will be used for the material evaluation. 
"""
struct UniaxialStrain <: AbstractStressState end

# Cases with stress iterations
"""
    IterationSettings(;tolerance = 1e-8, maxiter = 10)

Settings for stress iterations. Constructors for iterative stress states forwards
given keyword arguments to this constructor and saves the result.
"""
@kwdef struct IterationSettings{T}
    tolerance::T = 1.e-8
    maxiter::Int = 10
end
get_tolerance(is::IterationSettings) = is.tolerance
get_maxiter(is::IterationSettings) = is.maxiter

""" 
    UniaxialStress(; kwargs...)

Uniaxial stress such that 
``\\sigma_{ij}=0 \\forall (i,j)\\neq (1,1)``
The strain input can be 1d (`SecondOrderTensor{1}`).
A 3d input is also accepted and used as an initial 
guess for the unknown strain components.

For finite strains, a uniaxial Kirchhoff stress, τ = P ⋅ F', is found, such that 
``\\tau_{ij} = 0 \\forall (i,j) \\neq (1, 1)``
Additionally, the deformation gradient is forced to be symmetric to the potential
arbitrary rotations, i.e. ``F_{ij} = F_{ji}``.

The optional keyword arguments are forwarded to [`IterationSettings`](@ref).
"""
struct UniaxialStress{T} <: AbstractStressState 
    settings::IterationSettings{T}
end
UniaxialStress(; kwargs...) = UniaxialStress(IterationSettings(; kwargs...))
get_tolerance(ss::UniaxialStress) = get_tolerance(ss.settings)
get_maxiter(ss::UniaxialStress) = get_maxiter(ss.settings)

""" 
    PlaneStress(; kwargs...)

For small strain, find the plane stress state such that 
``\\sigma_{33}=\\sigma_{23}=\\sigma_{13}=0``

For finite strain, find the plane Kirchhoff stress, τ = P ⋅ F', such that
``\\tau{33}=\\tau{23}=\\tau{13}=0``. 
The out-of-plane shear deformation gradient components are forced to be symmetric,
i.e. ``F_{23} = F_{32}`` and ``F_{13} = F_{31}``.

The strain input should be at least 2d, but a 3d input is also accepted and
used as an initial guess for the unknown out-of-plane strain components.

The optional keyword arguments are forwarded to [`IterationSettings`](@ref).
"""
struct PlaneStress{T} <: AbstractStressState 
    settings::IterationSettings{T}
end
PlaneStress(; kwargs...) = PlaneStress(IterationSettings(; kwargs...))

get_tolerance(ss::PlaneStress) = get_tolerance(ss.settings)
get_maxiter(ss::PlaneStress) = get_maxiter(ss.settings)

""" 
    UniaxialNormalStress(; kwargs...)

This is a variation of the uniaxial stress state, such that only
``\\sigma_{22}=\\sigma_{33}=0`` for small strains. The strain input 
must be 3d, and the components ``\\epsilon_{22}`` and ``\\epsilon_{33}`` 
are used as initial guesses.

For finite strains, the condition is on the Kirchhoff stress, τ = P ⋅ F',
``\\tau_{22}=\\tau_{33}=0``. The deformation gradient should be 3d, and
``F_{22}`` and ``F_{33}`` are used as initial guesses.

This case is useful when simulating strain-controlled axial-shear experiments.
Note that the stress and stiffness outputs are the 3d tensors, and that the 
stiffness is **not** modified to account for the stress constraints.

The optional keyword arguments are forwarded to [`IterationSettings`](@ref).
"""
struct UniaxialNormalStress{T} <: AbstractStressState 
    settings::IterationSettings{T}
end
UniaxialNormalStress(; kwargs...) = UniaxialNormalStress(IterationSettings(; kwargs...))

get_tolerance(ss::UniaxialNormalStress) = get_tolerance(ss.settings)
get_maxiter(ss::UniaxialNormalStress) = get_maxiter(ss.settings)

"""
    GeneralStressState(σ_ctrl::AbstractTensor{2,3,Bool}, σ::AbstractTensor{2,3}; kwargs...)

Construct a general stress state controlled by `σ_ctrl` whose component is `true` if that
component is stress-controlled and `false` if it is strain-controlled. If stress-controlled,
σ gives the value to which it is controlled. The current stress, for stress-controlled components
can be updated by calling `update_stress_state!(s::GeneralStressState, σ)`. Components in
σ that are not stress-controlled are ignored.

For finite strains, this works directly on the first Piola-Kirchhoff stress, P, and
no gauge conditions (e.g. symmetry of deformation gradient) is enforced. Hence, it is the user's
responsibility to choose conditions that result in a valid equation system. Note that this might
change in the future to be consistent with other iteration states, but could also be generalized
further to allow more generic constraints. For finite strains this stress state should thus be
used for testing purposes only.

Note that the stress and stiffness outputs are the 3d tensors, and that the 
stiffness is **not** modified to account for the stress constraints.

The optional keyword arguments are forwarded to [`IterationSettings`](@ref).
"""
mutable struct GeneralStressState{Nσ,TS,TI,TC,T} <: AbstractStressState
    σ::TS
    # Reduced mandel indices
    const σm_inds::NTuple{Nσ,Tuple{Int,Int}}    # tensor -> mandel: m -> (i,j)
    const σ_minds::TI                           # mandel -> tensor: (i,j)->m 
    const σ_ctrl::TC
    settings::IterationSettings{T}
end
function GeneralStressState(σ_ctrl::AbstractTensor{2,3,Bool}, σ; kwargs...)
    Nσ = count(Tensors.get_data(σ_ctrl))
    return GeneralStressState{Nσ}(σ_ctrl, σ)
end
function GeneralStressState{Nσ}(σ_ctrl::TC, σ::TS; kwargs...) where {Nσ,TC,TS}
    @assert Nσ == count(Tensors.get_data(σ_ctrl))
    settings = IterationSettings(; kwargs...)
    TB = Tensors.get_base(TC)
    @assert TB == Tensors.get_base(TS)
    N = length(Tensors.get_data(σ_ctrl))
    
    σ_inds_mandel = zeros(Int, N)
    m = 0
    for (i, v) in enumerate(tovoigt(σ_ctrl))
        if v 
            m += 1
            σ_inds_mandel[i] = m
        end
    end
    σ_minds = fromvoigt(TB,σ_inds_mandel)
    σm_inds = Tuple{Int,Int}[]
    σm_minds_vec = Int[]
    for i in 1:3, j in 1:(isa(σ, SymmetricTensor) ? i : 3)
        if σ_ctrl[i,j]
            push!(σm_inds, (i,j))
            push!(σm_minds_vec, σ_minds[i,j])
        end
    end
    copyto!(σm_inds, σm_inds[sortperm(σm_minds_vec)])
    return GeneralStressState(σ, NTuple{Nσ}(σm_inds), σ_minds, σ_ctrl, settings)
end

update_stress_state!(s::GeneralStressState, σ) = (s.σ = σ)
get_tolerance(ss::GeneralStressState) = get_tolerance(ss.settings)
get_maxiter(ss::GeneralStressState) = get_maxiter(ss.settings)

const NoIterationState = Union{FullStressState,PlaneStrain,UniaxialStrain}
const IterationState = Union{UniaxialStress, PlaneStress, UniaxialNormalStress, GeneralStressState}
const State3D = Union{FullStressState, UniaxialNormalStress, GeneralStressState}
const State2D = Union{PlaneStress, PlaneStrain}
const State1D = Union{UniaxialStrain, UniaxialStress}

# Translate from reduced dim input to full tensors
expand_tensordim(::AbstractStressState, a::Tensors.AllTensors{3}) = a
#expand_tensordim(::AbstractStressState, v::Vec{dim,T}) where {dim,T} = Vec{3,T}(i->i>dim ? zero(T) : v[i])
function expand_tensordim(::AbstractStressState, F::Tensor{2,dim,T}) where {dim,T} 
    return Tensor{2,3}((i,j)-> (i<=dim && j<=dim) ? F[i,j] : (i == j ? one(T) : zero(T)))
end
function expand_tensordim(::AbstractStressState, ϵ::SymmetricTensor{2,dim,T}) where {dim,T} 
    return SymmetricTensor{2,3}((i,j)-> (i<=dim && j<=dim) ? ϵ[i,j] : zero(T))
end

# Translate from full tensors to reduced output
reduce_tensordim(::State3D, a::AbstractTensor) = a
reduce_tensordim(::State2D, a::AbstractTensor) = reduce_tensordim(Val{2}(), a)
reduce_tensordim(::State1D, a::AbstractTensor) = reduce_tensordim(Val{1}(), a)
#reduce_tensordim(::Val{dim}, v::Vec{3}) = Vec{dim}(i->v[i])
reduce_tensordim(::Val{dim}, P::Tensor{2}) where dim = Tensor{2,dim}((i, j) -> P[i, j])
reduce_tensordim(::Val{dim}, dPdF::Tensor{4}) where dim = Tensor{4,dim}((i, j, k, l) -> dPdF[i, j, k, l])
reduce_tensordim(::Val{dim}, σ::SymmetricTensor{2}) where dim = SymmetricTensor{2,dim}((i, j) -> σ[i, j])
reduce_tensordim(::Val{dim}, dσdϵ::SymmetricTensor{4}) where dim = SymmetricTensor{4,dim}((i, j, k, l) -> dσdϵ[i, j, k, l])


function stress_state_material_response(stress_state::NoIterationState, 
        m::AbstractMaterial, strain::AbstractTensor, args::Vararg{Any,N}) where N

    strain_3d = expand_tensordim(stress_state, strain)
    stress_3d, stiff_3d, new_state = material_response(m, strain_3d, args...)
    return stress_3d, stiff_3d, new_state, strain_3d
end

function stress_state_material_response(stress_state::IterationState, 
        m::AbstractMaterial, strain::AbstractTensor, args::Vararg{Any,N}) where N

    # Newton options
    tol = get_tolerance(stress_state)
    maxiter = get_maxiter(stress_state)
    
    strain_3d = expand_tensordim(stress_state, strain)

    for _ in 1:maxiter
        stress_3d, stiff_3d, new_state = material_response(m, strain_3d, args...)
        r = get_residual(stress_state, stress_3d, strain_3d)

        if norm(r) < tol
            return stress_3d, stiff_3d, new_state, strain_3d
        end

        drdx = get_drdx(stress_state, stiff_3d, strain_3d, stress_3d)
        strain_3d -= get_full_tensor(stress_state, strain, drdx\r)
    end
    throw(NoStressConvergence("Stress iterations with the NewtonSolver did not converge"))
end

reduce_stiffness(ss::State3D, stiff_3d::AbstractTensor{4,3}, strain_3d, stress_3d) = reduce_stiffness(Val(3), ss, stiff_3d, strain_3d, stress_3d)
reduce_stiffness(ss::State2D, stiff_3d::AbstractTensor{4,3}, strain_3d, stress_3d) = reduce_stiffness(Val(2), ss, stiff_3d, strain_3d, stress_3d)
reduce_stiffness(ss::State1D, stiff_3d::AbstractTensor{4,3}, strain_3d, stress_3d) = reduce_stiffness(Val(1), ss, stiff_3d, strain_3d, stress_3d)

reduce_stiffness(::Val{3}, ::State3D, stiff_3d::AbstractTensor{4,3}, args...) = stiff_3d

function reduce_stiffness(::Union{Val{1}, Val{2}}, stress_state::IterationState, stiff_3d::AbstractTensor{4,3}, strain_3d, stress_3d)
    # Using σ & ϵ, but for finite strain these are P & F
    # •ᶜ: Strain components known (not part of x), stress components calculated
    ∂r∂x, ∂r∂ϵᶜ, ∂σᶜ∂x, ∂σᶜ∂ϵᶜ = extract_substiffnesses(stress_state, stiff_3d, strain_3d, stress_3d)
    dσᶜdϵᶜ = ∂σᶜ∂ϵᶜ - ∂σᶜ∂x * (∂r∂x \ ∂r∂ϵᶜ)
    return _totensor(dσᶜdϵᶜ, stress_state, stiff_3d) # Convert SMatrix to Tensor{4} or SymmetricTensor{4}
end

function reduce_stiffness(::Union{Val{1}, Val{2}}, stress_state::NoIterationState, stiff_3d::AbstractTensor{4,3}, args...)
    return reduce_tensordim(stress_state, stiff_3d)
end

_totensor(dσᶜdϵᶜ::SMatrix{1,1}, ::State1D, ::SymmetricTensor) = frommandel(SymmetricTensor{4,1}, dσᶜdϵᶜ)
_totensor(dσᶜdϵᶜ::SMatrix{1,1}, ::State1D, ::Tensor) = frommandel(Tensor{4,1}, dσᶜdϵᶜ)
_totensor(dσᶜdϵᶜ::SMatrix{3,3}, ::State2D, ::SymmetricTensor) = frommandel(SymmetricTensor{4,2}, dσᶜdϵᶜ)
_totensor(dσᶜdϵᶜ::SMatrix{4,4}, ::State2D, ::Tensor) = frommandel(Tensor{4,2}, dσᶜdϵᶜ)


# =============================== #
# Small strains, SymmetricTensor =#
# =============================== #
# Internal numbering in Tensors.jl
# SymmetricTensor: 11,21,31,22,23,33

# UniaxialStress: only σ11 != 0
#         i:  1,  2,  3,  4,  5
# v contains 22, 33, 32, 31, 21
function get_full_tensor(::UniaxialStress, ::SymmetricTensor, v::SVector{5,T}) where T
    s = T(1/√2)        # 11,   21,     31,     22,   32,     33
    SymmetricTensor{2,3}((0, v[5]*s, v[4]*s, v[1], v[3]*s, v[2]))
end
function get_residual(::UniaxialStress, σ::SymmetricTensor{2,3}, args...)
    SVector{5}(σ[2,2], σ[3,3], σ[3,2]*√2, σ[3,1]*√2, σ[2,1]*√2)
end
function get_drdx(::UniaxialStress, dσdϵ::SymmetricTensor{4, 3}, args...)
    @SMatrix [     dσdϵ[2,2,2,2]    dσdϵ[2,2,3,3] √2*dσdϵ[2,2,3,2] √2*dσdϵ[2,2,3,1] √2*dσdϵ[2,2,2,1];
                   dσdϵ[3,3,2,2]    dσdϵ[3,3,3,3] √2*dσdϵ[3,3,3,2] √2*dσdϵ[3,3,3,1] √2*dσdϵ[3,3,2,1];
                √2*dσdϵ[3,2,2,2] √2*dσdϵ[3,2,3,3]  2*dσdϵ[3,2,3,2]  2*dσdϵ[3,2,3,1]  2*dσdϵ[3,2,2,1];
                √2*dσdϵ[3,1,2,2] √2*dσdϵ[3,1,3,3]  2*dσdϵ[3,1,3,2]  2*dσdϵ[3,1,3,1]  2*dσdϵ[3,1,2,1];
                √2*dσdϵ[2,1,2,2] √2*dσdϵ[2,1,3,3]  2*dσdϵ[2,1,3,2]  2*dσdϵ[2,1,3,1]  2*dσdϵ[2,1,2,1]]
end
function extract_substiffnesses(stress_state::UniaxialStress, D::SymmetricTensor{4,3}, ϵ, σ)
    # •ᶜ: Strain components known (not part of x), stress components calculated
    ∂r∂x = get_drdx(stress_state, D, ϵ, σ)
    ∂r∂ϵᶜ = SMatrix{5,1}(D[2,2,1,1], D[3,3,1,1], √2*D[2,3,1,1], √2*D[1,3,1,1], √2*D[1,2,1,1])
    ∂σᶜ∂x = SMatrix{1,5}(D[1,1,2,2], D[1,1,3,3], √2*D[1,1,2,3], √2*D[1,1,1,3], √2*D[1,1,1,2])
    ∂σᶜ∂ϵᶜ = SMatrix{1,1}(D[1,1,1,1])
    return ∂r∂x, ∂r∂ϵᶜ, ∂σᶜ∂x, ∂σᶜ∂ϵᶜ
end

# UniaxialNormalStress: σ22 = σ33 = 0
#         i:  1,  2
# v contains 22, 33
function get_full_tensor(::UniaxialNormalStress, ::SymmetricTensor, v::SVector{2})
                       # 11,21,31,   22,32,   33
    SymmetricTensor{2,3}((0, 0, 0, v[1], 0, v[2]))
end
function get_residual(::UniaxialNormalStress, σ::SymmetricTensor{2,3}, args...)
    SVector{2}(σ[2,2], σ[3,3])
end
function get_drdx(::UniaxialNormalStress, dσdϵ::SymmetricTensor{4, 3}, args...)
    @SMatrix [dσdϵ[2,2,2,2] dσdϵ[2,2,3,3];
              dσdϵ[3,3,2,2] dσdϵ[3,3,3,3]]
end


# PlaneStress: σ33 = σ23 = σ13 = 0
#         i:  1,  2,  3
# v contains 33, 23, 13
function get_full_tensor(::PlaneStress, ::SymmetricTensor, v::SVector{3,T}) where T
    s = T(1/√2)        # 11,21,   31,  22,   32,     33
    SymmetricTensor{2,3}((0, 0, v[3]*s, 0, v[2]*s, v[1]))
end
function get_residual(::PlaneStress, σ::SymmetricTensor{2,3}, args...)
    SVector{3}(σ[3,3], σ[2,3]*√2, σ[3,1]*√2)
end
function get_drdx(::PlaneStress, dσdϵ::SymmetricTensor{4, 3}, ϵ::SymmetricTensor{2,3}, σ::SymmetricTensor{2,3})
    @SMatrix [     dσdϵ[3,3,3,3] √2*dσdϵ[3,3,3,2] √2*dσdϵ[3,3,3,1];
                √2*dσdϵ[3,2,3,3]  2*dσdϵ[3,2,3,2]  2*dσdϵ[3,2,3,1];
                √2*dσdϵ[3,1,3,3]  2*dσdϵ[3,1,3,2]  2*dσdϵ[3,1,3,1]]
end
function extract_substiffnesses(stress_state::PlaneStress, D::SymmetricTensor{4,3}, ϵ, σ)
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ∂σᶠ∂ϵᶠ = get_drdx(stress_state, D, ϵ, σ)
    ∂σᶠ∂ϵᶜ = @SMatrix [   D[3,3,1,1]    D[3,3,2,2] √2*D[3,3,2,1];
                       √2*D[3,2,1,1] √2*D[3,2,2,2]  2*D[3,2,2,1];
                       √2*D[3,1,1,1] √2*D[3,1,2,2]  2*D[3,1,2,1]]
    
    ∂σᶜ∂ϵᶠ = @SMatrix [   D[1,1,3,3] √2*D[1,1,3,2] √2*D[1,1,3,1];
                          D[2,2,3,3] √2*D[2,2,3,2] √2*D[2,2,3,1];
                       √2*D[2,1,3,3]  2*D[2,1,3,2]  2*D[2,1,3,1]]

    ∂σᶜ∂ϵᶜ = @SMatrix [   D[1,1,1,1]    D[1,1,2,2] √2*D[1,1,2,1];
                          D[2,2,1,1]    D[2,2,2,2] √2*D[2,2,2,1];
                       √2*D[2,1,1,1] √2*D[2,1,2,2]  2*D[2,1,2,1]]

    return ∂σᶠ∂ϵᶠ, ∂σᶠ∂ϵᶜ, ∂σᶜ∂ϵᶠ, ∂σᶜ∂ϵᶜ
end

# =============================== #
# Finite strains, Tensor         =#
# =============================== #
# Internal numbering in Tensors.jl
# Tensor: 11,21,31,12,22,32,13,23,33

function get_drdx(ss::AbstractStressState, dPdF::Tensor{4, 3, T}, F::Tensor{2, 3}, P::Tensor{2, 3}) where {T}
    drdx, _ = get_drdx_dτdF(ss, dPdF, F, P)
    return drdx
end

# Stiffness and stress conversions
function dPdF_to_dτdF(dPdF::Tensor{4,3}, F::Tensor, P::Tensor)
    # TODO: If important, this can be optimized with Tensors.get_expression
    # dτ_ij/dF_kl = d(P_im F_mj')/dF_kl = F_mj' * dP_im/dF_kl + P_im dF_mj'/dF_kl
    #             = δ_in F_jm dP_nm/dF_kl + P_il δ_jk
    #             : (I ⊗̄ F) : dPdF + P ⊗̲ I
    I2 = one(F)
    return otimesu(I2, F) ⊡ dPdF + otimesl(P, I2) # dτdF
end
convert_P_to_τ(P::Tensor{2,3}, F::Tensor{2,3}) = P ⋅ F'

# UniaxialStress
#    i:  1,   2,   3,   4,   5,   6,   7,   8
# x = [F22, F33, F23, F13, F12, F32, F31, F21]
function get_full_tensor(::UniaxialStress, ϵ::Tensor, v::SVector{8})
                     # 11,   21,   31,   12,   22,   32,   13,   23,   33
    return Tensor{2,3}((0, v[8], v[7], v[5], v[1], v[6], v[4], v[3], v[2]))
end

function get_residual(::UniaxialStress, P::Tensor{2,3}, F::Tensor{2,3})
    τ = convert_P_to_τ(P, F)
    SVector{8}(τ[2,2], τ[3,3], τ[2,3], τ[1,3], τ[1,2], F[2,3] - F[3,2], F[1,3] - F[3,1], F[1,2] - F[2,1])
end

function get_drdx_dτdF(::UniaxialStress, dPdF::Tensor{4, 3, T}, F::Tensor{2, 3}, P::Tensor{2, 3}) where {T}
    dτdF = dPdF_to_dτdF(dPdF, F, P)
    drdx = @SMatrix [  
        dτdF[2,2,2,2] dτdF[2,2,3,3] dτdF[2,2,2,3] dτdF[2,2,1,3] dτdF[2,2,1,2] dτdF[2,2,3,2] dτdF[2,2,3,1] dτdF[2,2,2,1];
        dτdF[3,3,2,2] dτdF[3,3,3,3] dτdF[3,3,2,3] dτdF[3,3,1,3] dτdF[3,3,1,2] dτdF[3,3,3,2] dτdF[3,3,3,1] dτdF[3,3,2,1];
        dτdF[2,3,2,2] dτdF[2,3,3,3] dτdF[2,3,2,3] dτdF[2,3,1,3] dτdF[2,3,1,2] dτdF[2,3,3,2] dτdF[2,3,3,1] dτdF[2,3,2,1];
        dτdF[1,3,2,2] dτdF[1,3,3,3] dτdF[1,3,2,3] dτdF[1,3,1,3] dτdF[1,3,1,2] dτdF[1,3,3,2] dτdF[1,3,3,1] dτdF[1,3,2,1];
        dτdF[1,2,2,2] dτdF[1,2,3,3] dτdF[1,2,2,3] dτdF[1,2,1,3] dτdF[1,2,1,2] dτdF[1,2,3,2] dτdF[1,2,3,1] dτdF[1,2,2,1];
              zero(T)       zero(T)        one(T)       zero(T)       zero(T)       -one(T)       zero(T)       zero(T); # R6 = F[2,3] - F[3,2]
              zero(T)       zero(T)       zero(T)        one(T)       zero(T)       zero(T)       -one(T)       zero(T); # R7 = F[1,3] - F[3,1]
              zero(T)       zero(T)       zero(T)       zero(T)        one(T)       zero(T)       zero(T)       -one(T)] # R8 = F[1,2] - F[2,1]
    return drdx, dτdF
end

function extract_substiffnesses(stress_state::UniaxialStress, dPdF::Tensor{4, 3, T}, F::Tensor{2, 3}, P::Tensor{2, 3}) where {T}
    # •ᶜ: Strain components known (not part of x), stress components calculated
    ∂r∂x, dτdF = get_drdx_dτdF(stress_state, dPdF, F, P)
    ∂r∂Fᶜ = SMatrix{8,1}(dτdF[2,2,1,1], dτdF[3,3,1,1], dτdF[2,3,1,1], dτdF[1,3,1,1], dτdF[1,2,1,1], zero(T), zero(T), zero(T))
    ∂Pᶜ∂x = SMatrix{1,8}(dPdF[1,1,2,2], dPdF[1,1,3,3], dPdF[1,1,2,3], dPdF[1,1,1,3], dPdF[1,1,1,2], dPdF[1,1,3,2], dPdF[1,1,3,1], dPdF[1,1,2,1])
    ∂Pᶜ∂Fᶜ = SMatrix{1,1}(dPdF[1,1,1,1])
    return ∂r∂x, ∂r∂Fᶜ, ∂Pᶜ∂x, ∂Pᶜ∂Fᶜ
end


# PlaneStress
#    i:  1,   2,   3,   4,   5
# x = [F33, F23, F13, F32, F31]
function get_full_tensor(::PlaneStress, ϵ::Tensor, v::SVector{5})
                     # 11,21,   31,12,22,   32,   13,   23,   33
    return Tensor{2,3}((0, 0, v[5], 0, 0, v[4], v[3], v[2], v[1]))
end
function get_residual(::PlaneStress, P::Tensor{2,3}, F::Tensor)
    τ = convert_P_to_τ(P, F)
    SVector{5}(τ[3,3], τ[2,3], τ[1,3], F[2,3] - F[3,2], F[1,3] - F[3,1])
end
function get_drdx_dτdF(::PlaneStress, dPdF::Tensor{4, 3, T}, F::Tensor{2,3}, P::Tensor{2,3}) where {T}
    dτdF = dPdF_to_dτdF(dPdF, F, P)
    drdx = @SMatrix [
        dτdF[3,3,3,3] dτdF[3,3,2,3] dτdF[3,3,1,3] dτdF[3,3,3,2] dτdF[3,3,3,1];
        dτdF[2,3,3,3] dτdF[2,3,2,3] dτdF[2,3,1,3] dτdF[2,3,3,2] dτdF[2,3,3,1];
        dτdF[1,3,3,3] dτdF[1,3,2,3] dτdF[1,3,1,3] dτdF[1,3,3,2] dτdF[1,3,3,1];
              zero(T)        one(T)       zero(T)       -one(T)       zero(T); # R4 = F23 - F32
              zero(T)       zero(T)        one(T)       zero(T)       -one(T)] # R5 = F13 - F31
    return drdx, dτdF
end
function extract_substiffnesses(stress_state::PlaneStress, dPdF::Tensor{4, 3, T}, F::Tensor{2, 3}, P::Tensor{2, 3}) where {T}
    # •ᶜ: Strain components known (not part of x), stress components calculated
    ∂r∂x, dτdF = get_drdx_dτdF(stress_state, dPdF, F, P)
    ∂r∂Fᶜ =  @SMatrix [dτdF[3,3,1,1] dτdF[3,3,2,2] dτdF[3,3,1,2] dτdF[3,3,2,1];
                       dτdF[2,3,1,1] dτdF[2,3,2,2] dτdF[2,3,1,2] dτdF[2,3,2,1];
                       dτdF[1,3,1,1] dτdF[1,3,2,2] dτdF[1,3,1,2] dτdF[1,3,2,1];
                             zero(T)       zero(T)       zero(T)       zero(T);
                             zero(T)       zero(T)       zero(T)       zero(T)]
    
    ∂Pᶜ∂x = @SMatrix  [dPdF[1,1,3,3] dPdF[1,1,2,3] dPdF[1,1,1,3] dPdF[1,1,3,2] dPdF[1,1,3,1];
                       dPdF[2,2,3,3] dPdF[2,2,2,3] dPdF[2,2,1,3] dPdF[2,2,3,2] dPdF[2,2,3,1];
                       dPdF[1,2,3,3] dPdF[1,2,2,3] dPdF[1,2,1,3] dPdF[1,2,3,2] dPdF[1,2,3,1];
                       dPdF[2,1,3,3] dPdF[2,1,2,3] dPdF[2,1,1,3] dPdF[2,1,3,2] dPdF[2,1,3,1]]

    ∂Pᶜ∂Fᶜ = @SMatrix [dPdF[1,1,1,1] dPdF[1,1,2,2] dPdF[1,1,1,2] dPdF[1,1,2,1];
                       dPdF[2,2,1,1] dPdF[2,2,2,2] dPdF[2,2,1,2] dPdF[2,2,2,1];
                       dPdF[1,2,1,1] dPdF[1,2,2,2] dPdF[1,2,1,2] dPdF[1,2,2,1];
                       dPdF[2,1,1,1] dPdF[2,1,2,2] dPdF[2,1,1,2] dPdF[2,1,2,1]]
    return ∂r∂x, ∂r∂Fᶜ, ∂Pᶜ∂x, ∂Pᶜ∂Fᶜ
end

# UniaxialNormalStress: τ22=τ33=0
# -Tensor
#         i:  1,  2
# v contains 22, 33
function get_full_tensor(::UniaxialNormalStress, ::Tensor, v::SVector{2})
                     # 11,21,31,12,   22,32,13,23,  33
    return Tensor{2,3}((0, 0, 0, 0, v[1], 0, 0, 0, v[2]))
end
function get_residual(::UniaxialNormalStress, P::Tensor{2,3}, F::Tensor{2,3})
    τ = convert_P_to_τ(P, F)
    return SVector{2}(τ[2,2], τ[3,3])
end
function get_drdx(::UniaxialNormalStress, dPdF::Tensor{4, 3, T}, F::Tensor{2, 3}, P::Tensor{2, 3}) where {T}
    dτdF = dPdF_to_dτdF(dPdF, F, P)
    @SMatrix [dτdF[2,2,2,2] dτdF[2,2,3,3];
              dτdF[3,3,2,2] dτdF[3,3,3,3]]
end

# GeneralStressState
function get_full_tensor(state::GeneralStressState{Nσ}, ::TT, v::SVector{Nσ,T}) where {Nσ,T,TT}
    TB = Tensors.get_base(TT)
    shear_factor = if TB == SymmetricTensor{2,3}
        1/sqrt(2 * one(T))
    elseif TB == Tensor{2,3}
        one(T)
    else
        error("GeneralStressState expects full dimension of strain input")
    end
    s(i,j) = i==j ? one(T) : shear_factor
    f(i,j) = state.σ_ctrl[i,j] ? v[state.σ_minds[i,j]]*s(i,j) : zero(T)
    return TB(f)
end

function get_residual(state::GeneralStressState{Nσ}, stress_3d::AbstractTensor{2,3,T}, args...) where {Nσ, T}
    shear_factor = stress_3d isa SymmetricTensor ? sqrt(2 * one(T)) : one(T)
    s(i,j) = i==j ? one(T) : shear_factor
    f(c) = ((i,j) = c; s(i,j)*(stress_3d[i,j] - state.σ[i,j]))
    return SVector{Nσ,T}((f(c) for c in state.σm_inds))
end

function get_drdx(state::GeneralStressState{Nσ}, stiff_3d::AbstractTensor{4,3,T}, args...) where {Nσ,T}
    shear_factor = stiff_3d isa SymmetricTensor ? sqrt(2 * one(T)) : one(T)
    s(i,j) = i==j ? one(T) : shear_factor
    f(c1,c2) = ((i,j) = c1; (k,l) = c2; stiff_3d[i,j,k,l]*s(i,j)*s(k,l))
    return SMatrix{Nσ,Nσ,T}((f(c1,c2) for c1 in state.σm_inds, c2 in state.σm_inds))
end

