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
    σ, dσdϵ, state, ϵ_full = stress_state_material_response(stress_state, m, args...)
    return reduce_tensordim(stress_state, σ), reduce_stiffness(stress_state, dσdϵ), state, ϵ_full
end

update_stress_state!(::AbstractStressState, σ) = nothing

"""
    ReducedStressState(s::AbstractStressState, m::AbstractMaterial)
    
Creates a subtype of `AbstractMaterial` that wraps a stress state and a material, such that 
calls to `material_response(w::ReducedStressState, args...)` gives the same result as 
`material_response(s, m, args...)`. 
Calls to `initial_material_state`, `allocate_material_cache`, 
`get_num_tensorcomponents`, `get_num_statevars`, `get_num_params`, 
`get_params_eltype`, `tovector!`, `tovector`, 
and `allocate_differentiation_output` are forwarded with `m` as the argument. 
`fromvector` returns `ReducedStressState` and is supported as well.
"""
struct ReducedStressState{S<:AbstractStressState,M<:AbstractMaterial} <: AbstractMaterial
    stress_state::S
    material::M
end
for op in ( :initial_material_state, :allocate_material_cache, 
            :get_num_tensorcomponents, :get_num_statevars, :get_num_params, 
            :get_params_eltype, :allocate_differentiation_output)
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
    IterationSettings(;tolerance=1e-8, max_iter=10)

Settings for stress iterations. Constructors for iterative stress states forwards
given keyword arguments to this constructor and saves the result.
"""
@kwdef struct IterationSettings{T}
    tolerance::T=1.e-8
    max_iter::Int=10
end
get_tolerance(is::IterationSettings) = is.tolerance
get_max_iter(is::IterationSettings) = is.max_iter

""" 
    UniaxialStress(; kwargs...)

Uniaxial stress such that 
``\\sigma_{ij}=0 \\forall (i,j)\\neq (1,1)``
The strain input can be 1d (`SecondOrderTensor{1}`).
A 3d input is also accepted and used as an initial 
guess for the unknown strain components. 

The optional keyword arguments are forwarded to [`IterationSettings`](@ref).
"""
struct UniaxialStress{T} <: AbstractStressState 
    settings::IterationSettings{T}
end
UniaxialStress(; kwargs...) = UniaxialStress(IterationSettings(; kwargs...))
get_tolerance(ss::UniaxialStress) = get_tolerance(ss.settings)
get_max_iter(ss::UniaxialStress) = get_max_iter(ss.settings)

""" 
    PlaneStress(; kwargs...)

Plane stress such that 
``\\sigma_{33}=\\sigma_{23}=\\sigma_{13}=\\sigma_{32}=\\sigma_{31}=0``
The strain input should be at least 2d (components 11, 12, 21, and 22).
A 3d input is also accepted and used as an initial guess for the unknown 
out-of-plane strain components. 

The optional keyword arguments are forwarded to [`IterationSettings`](@ref).
"""
struct PlaneStress{T} <: AbstractStressState 
    settings::IterationSettings{T}
end
PlaneStress(; kwargs...) = PlaneStress(IterationSettings(; kwargs...))

get_tolerance(ss::PlaneStress) = get_tolerance(ss.settings)
get_max_iter(ss::PlaneStress) = get_max_iter(ss.settings)

""" 
    UniaxialNormalStress(; kwargs...)

This is a variation of the uniaxial stress state, such that only
``\\sigma_{22}=\\sigma_{33}=0``
The strain input must be 3d, and the components 
``\\epsilon_{22}`` and ``\\epsilon_{33}`` are used as initial guesses. 
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
get_max_iter(ss::UniaxialNormalStress) = get_max_iter(ss.settings)

"""
    GeneralStressState(σ_ctrl::AbstractTensor{2,3,Bool}, σ::AbstractTensor{2,3}; kwargs...)

Construct a general stress state controlled by `σ_ctrl` whose component is `true` if that 
component is stress-controlled and `false` if it is strain-controlled. If stress-controlled,
σ gives the value to which it is controlled. The current stress, for stress-controlled components
can be updated by calling `update_stress_state!(s::GeneralStressState, σ)`. Components in 
σ that are not stress-controlled are ignored. 

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
get_max_iter(ss::GeneralStressState) = get_max_iter(ss.settings)

const NoIterationState = Union{FullStressState,PlaneStrain,UniaxialStrain}
const IterationState = Union{UniaxialStress, PlaneStress, UniaxialNormalStress, GeneralStressState}
const State3D = Union{FullStressState, UniaxialNormalStress, GeneralStressState}
const State2D = Union{PlaneStress, PlaneStrain}
const State1D = Union{UniaxialStrain, UniaxialStress}

# Translate from reduced dim input to full tensors
expand_tensordim(::AbstractStressState, a::Tensors.AllTensors{3}) = a
#expand_tensordim(::AbstractStressState, v::Vec{dim,T}) where {dim,T} = Vec{3,T}(i->i>dim ? zero(T) : v[i])
function expand_tensordim(::AbstractStressState, a::Tensor{2,dim,T}) where {dim,T} 
    return Tensor{2,3}((i,j)-> (i<=dim && j<=dim) ? a[i,j] : zero(T))
end
function expand_tensordim(::AbstractStressState, a::SymmetricTensor{2,dim,T}) where {dim,T} 
    return SymmetricTensor{2,3}((i,j)-> (i<=dim && j<=dim) ? a[i,j] : zero(T))
end

# Translate from full tensors to reduced output
reduce_tensordim(::State3D, a::AbstractTensor) = a
reduce_tensordim(::State2D, a::AbstractTensor) = reduce_tensordim(Val{2}(), a)
reduce_tensordim(::State1D, a::AbstractTensor) = reduce_tensordim(Val{1}(), a)
#reduce_tensordim(::Val{dim}, v::Vec{3}) = Vec{dim}(i->v[i])
reduce_tensordim(::Val{dim}, a::Tensor{2}) where dim = Tensor{2,dim}((i,j)->a[i,j])
reduce_tensordim(::Val{dim}, A::Tensor{4}) where dim = Tensor{4,dim}((i,j,k,l)->A[i,j,k,l])
reduce_tensordim(::Val{dim}, a::SymmetricTensor{2}) where dim = SymmetricTensor{2,dim}((i,j)->a[i,j])
reduce_tensordim(::Val{dim}, A::SymmetricTensor{4}) where dim = SymmetricTensor{4,dim}((i,j,k,l)->A[i,j,k,l])


function stress_state_material_response(stress_state::NoIterationState, 
        m::AbstractMaterial, ϵ::AbstractTensor, args::Vararg{Any,N}) where N

    ϵ_full = expand_tensordim(stress_state, ϵ)
    σ, dσdϵ, new_state = material_response(m, ϵ_full, args...)
    return σ, dσdϵ, new_state, ϵ_full
end

function stress_state_material_response(stress_state::IterationState, 
        m::AbstractMaterial, ϵ::AbstractTensor, args::Vararg{Any,N}) where N

    # Newton options
    tol = get_tolerance(stress_state)
    maxiter = get_max_iter(stress_state)
    
    ϵ_full = expand_tensordim(stress_state, ϵ)

    for _ in 1:maxiter
        σ_full, dσdϵ_full, new_state = material_response(m, ϵ_full, args...)
        σ_mandel = get_unknowns(stress_state, σ_full)

        if norm(σ_mandel) < tol
            return σ_full, dσdϵ_full, new_state, ϵ_full
        end

        dσdϵ_mandel = get_unknowns(stress_state, dσdϵ_full)
        ϵ_full -= get_full_tensor(stress_state, ϵ, dσdϵ_mandel\σ_mandel)
    end
    throw(NoStressConvergence("Stress iterations with the NewtonSolver did not converge"))
end

reduce_stiffness(ss::State3D, dσdϵ_full) = reduce_stiffness(Val(3), ss, dσdϵ_full)
reduce_stiffness(ss::State2D, dσdϵ_full) = reduce_stiffness(Val(2), ss, dσdϵ_full)
reduce_stiffness(ss::State1D, dσdϵ_full) = reduce_stiffness(Val(1), ss, dσdϵ_full)

reduce_stiffness(::Val{3}, ::State3D, dσdϵ_full::AbstractTensor{4,3}) = dσdϵ_full

function reduce_stiffness(::Union{Val{1}, Val{2}}, stress_state::IterationState, dσdϵ_full::AbstractTensor{4,3})
    ∂σᶠ∂ϵᶠ, ∂σᶠ∂ϵᶜ, ∂σᶜ∂ϵᶠ, ∂σᶜ∂ϵᶜ = extract_substiffnesses(stress_state, dσdϵ_full)
    dσᶜdϵᶜ = ∂σᶜ∂ϵᶜ - ∂σᶜ∂ϵᶠ * (∂σᶠ∂ϵᶠ \ ∂σᶠ∂ϵᶜ)
    return convert_stiffness(dσᶜdϵᶜ, stress_state, dσdϵ_full)
end

function reduce_stiffness(::Union{Val{1}, Val{2}}, stress_state::NoIterationState, dσdϵ_full::AbstractTensor{4,3})
    return reduce_tensordim(stress_state, dσdϵ_full)
end

convert_stiffness(dσᶜdϵᶜ::SMatrix{1,1}, ::State1D, ::SymmetricTensor) = frommandel(SymmetricTensor{4,1}, dσᶜdϵᶜ)
convert_stiffness(dσᶜdϵᶜ::SMatrix{1,1}, ::State1D, ::Tensor) = frommandel(Tensor{4,1}, dσᶜdϵᶜ)
convert_stiffness(dσᶜdϵᶜ::SMatrix{3,3}, ::State2D, ::SymmetricTensor) = frommandel(SymmetricTensor{4,2}, dσᶜdϵᶜ)
convert_stiffness(dσᶜdϵᶜ::SMatrix{4,4}, ::State2D, ::Tensor) = frommandel(Tensor{4,2}, dσᶜdϵᶜ)


# Conversions to mandel SArray for solving equation system
# Internal numbering in Tensors.jl
# Tensor: 11,21,31,12,22,32,13,23,33
# SymmetricTensor: 11,21,31,22,23,33

# UniaxialStress: only σ11 != 0
# -SymmetricTensor
#         i:  1,  2,  3,  4,  5
# v contains 22, 33, 32, 31, 21
function get_full_tensor(::UniaxialStress, ::SymmetricTensor, v::SVector{5,T}) where T
    s = T(1/√2)        # 11,   21,     31,     22,   32,     33
    SymmetricTensor{2,3}((0, v[5]*s, v[4]*s, v[1], v[3]*s, v[2]))
end
function get_unknowns(::UniaxialStress, a::SymmetricTensor{2,3})
    SVector{5}(a[2,2], a[3,3], a[3,2]*√2, a[3,1]*√2, a[2,1]*√2)
end
function get_unknowns(::UniaxialStress, A::SymmetricTensor{4, 3}) 
    @SMatrix [     A[2,2,2,2]    A[2,2,3,3] √2*A[2,2,3,2] √2*A[2,2,3,1] √2*A[2,2,2,1];
                   A[3,3,2,2]    A[3,3,3,3] √2*A[3,3,3,2] √2*A[3,3,3,1] √2*A[3,3,2,1];
                √2*A[3,2,2,2] √2*A[3,2,3,3]  2*A[3,2,3,2]  2*A[3,2,3,1]  2*A[3,2,2,1];
                √2*A[3,1,2,2] √2*A[3,1,3,3]  2*A[3,1,3,2]  2*A[3,1,3,1]  2*A[3,1,2,1];
                √2*A[2,1,2,2] √2*A[2,1,3,3]  2*A[2,1,3,2]  2*A[2,1,3,1]  2*A[2,1,2,1]]
end

# -Tensor
#         i:  1,  2,  3,  4,  5,  6,  7,  8
# v contains 22, 33, 23, 13, 12, 32, 31, 21
function get_full_tensor(::UniaxialStress, ϵ::Tensor, v::SVector{8})
                     # 11,   21,   31,   12,   22,   32,   13,   23,   33
    return Tensor{2,3}((0, v[8], v[7], v[5], v[1], v[6], v[4], v[3], v[2]))
end

function get_unknowns(::UniaxialStress, a::Tensor{2,3})
    SVector{8}(a[2,2], a[3,3], a[2,3], a[1,3], a[1,2], a[3,2], a[3,1], a[2,1])
end

function get_unknowns(::UniaxialStress, A::Tensor{4, 3}) 
    @SMatrix [  A[2,2,2,2] A[2,2,3,3] A[2,2,2,3] A[2,2,1,3] A[2,2,1,2] A[2,2,3,2] A[2,2,3,1] A[2,2,2,1];
                A[3,3,2,2] A[3,3,3,3] A[3,3,2,3] A[3,3,1,3] A[3,3,1,2] A[3,3,3,2] A[3,3,3,1] A[3,3,2,1];
                A[2,3,2,2] A[2,3,3,3] A[2,3,2,3] A[2,3,1,3] A[2,3,1,2] A[2,3,3,2] A[2,3,3,1] A[2,3,2,1];
                A[1,3,2,2] A[1,3,3,3] A[1,3,2,3] A[1,3,1,3] A[1,3,1,2] A[1,3,3,2] A[1,3,3,1] A[1,3,2,1];
                A[1,2,2,2] A[1,2,3,3] A[1,2,2,3] A[1,2,1,3] A[1,2,1,2] A[1,2,3,2] A[1,2,3,1] A[1,2,2,1];
                A[3,2,2,2] A[3,2,3,3] A[3,2,2,3] A[3,2,1,3] A[3,2,1,2] A[3,2,3,2] A[3,2,3,1] A[3,2,2,1];
                A[3,1,2,2] A[3,1,3,3] A[3,1,2,3] A[3,1,1,3] A[3,1,1,2] A[3,1,3,2] A[3,1,3,1] A[3,1,2,1];
                A[2,1,2,2] A[2,1,3,3] A[2,1,2,3] A[2,1,1,3] A[2,1,1,2] A[2,1,3,2] A[2,1,3,1] A[2,1,2,1]]
end

# PlaneStress: σ33=σ23=σ13=0=σ31=σ32
# -SymmetricTensor
#         i:  1,  2,  3
# v contains 33, 23, 13
function get_full_tensor(::PlaneStress, ::SymmetricTensor, v::SVector{3,T}) where T
    s = T(1/√2)        # 11,21,   31,  22,   32,     33
    SymmetricTensor{2,3}((0, 0, v[3]*s, 0, v[2]*s, v[1]))
end
function get_unknowns(::PlaneStress, a::SymmetricTensor{2,3})
    SVector{3}(a[3,3], a[2,3]*√2, a[3,1]*√2)
end
function get_unknowns(::PlaneStress, A::SymmetricTensor{4, 3}) 
    @SMatrix [     A[3,3,3,3] √2*A[3,3,3,2] √2*A[3,3,3,1];
                √2*A[3,2,3,3]  2*A[3,2,3,2]  2*A[3,2,3,1];
                √2*A[3,1,3,3]  2*A[3,1,3,2]  2*A[3,1,3,1]]
end

# -Tensor
#         i:  1,  2,  3,  4,  5,  6
# v contains 33, 23, 13, 32, 31
function get_full_tensor(::PlaneStress, ϵ::Tensor, v::SVector{5})
                     # 11,21,   31,12,22,   32,   13,   23,   33
    return Tensor{2,3}((0, 0, v[5], 0, 0, v[4], v[3], v[2], v[1]))
end

function get_unknowns(::PlaneStress, a::Tensor{2,3})
    SVector{5}(a[3,3], a[2,3], a[1,3], a[3,2], a[3,1])
end

function get_unknowns(::PlaneStress, A::Tensor{4, 3}) 
    @SMatrix [  A[3,3,3,3] A[3,3,2,3] A[3,3,1,3] A[3,3,3,2] A[3,3,3,1];
                A[2,3,3,3] A[2,3,2,3] A[2,3,1,3] A[2,3,3,2] A[2,3,3,1];
                A[1,3,3,3] A[1,3,2,3] A[1,3,1,3] A[1,3,3,2] A[1,3,3,1];
                A[3,2,3,3] A[3,2,2,3] A[3,2,1,3] A[3,2,3,2] A[3,2,3,1];
                A[3,1,3,3] A[3,1,2,3] A[3,1,1,3] A[3,1,3,2] A[3,1,3,1]]
end

# UniaxialNormalStress: σ22=σ33=0
# -SymmetricTensor and Tensor
#         i:  1,  2
# v contains 22, 33
function get_full_tensor(::UniaxialNormalStress, ::SymmetricTensor, v::SVector{2})
                       # 11,21,31,   22,32,   33
    SymmetricTensor{2,3}((0, 0, 0, v[1], 0, v[2]))
end
function get_full_tensor(::UniaxialNormalStress, ϵ::Tensor, v::SVector{2})
                 # 11,21,31,12,   22,32,13,23,  33
return Tensor{2,3}((0, 0, 0, 0, v[1], 0, 0, 0, v[2]))
end
function get_unknowns(::UniaxialNormalStress, a::AbstractTensor{2,3})
    SVector{2}(a[2,2], a[3,3])
end
function get_unknowns(::UniaxialNormalStress, A::AbstractTensor{4, 3}) 
    @SMatrix [A[2,2,2,2] A[2,2,3,3];
              A[3,3,2,2] A[3,3,3,3]]
end

# GeneralStressState
function get_full_tensor(state::GeneralStressState{Nσ}, ::TT, v::SVector{Nσ,T}) where {Nσ,T,TT}
    TB = Tensors.get_base(TT)
    shear_factor = if TB == SymmetricTensor{2,3}
        T(1/√2)
    elseif TB == Tensor{2,3}
        one(T)
    else
        error("GeneralStressState expects full dimension of strain input")
    end
    s(i,j) = i==j ? one(T) : shear_factor
    f(i,j) = state.σ_ctrl[i,j] ? v[state.σ_minds[i,j]]*s(i,j) : zero(T)
    return TB(f)
end

function get_unknowns(state::GeneralStressState{Nσ}, a::AbstractTensor{2,3,T}) where {Nσ, T}
    shear_factor = a isa SymmetricTensor ? T(√2) : one(T)
    s(i,j) = i==j ? one(T) : shear_factor
    f(c) = ((i,j) = c; s(i,j)*(a[i,j]-state.σ[i,j]))
    return SVector{Nσ,T}((f(c) for c in state.σm_inds))
end

function get_unknowns(state::GeneralStressState{Nσ}, a::AbstractTensor{4,3,T}) where {Nσ,T}
    shear_factor = a isa SymmetricTensor ? T(√2) : one(T)
    s(i,j) = i==j ? one(T) : shear_factor
    f(c1,c2) = ((i,j) = c1; (k,l) = c2; a[i,j,k,l]*s(i,j)*s(k,l))
    return SMatrix{Nσ,Nσ,T}((f(c1,c2) for c1 in state.σm_inds, c2 in state.σm_inds))
end

# Extract each part of the stiffness tensor as SMatrix
# UniaxialStress
function extract_substiffnesses(stress_state::UniaxialStress, D::SymmetricTensor{4,3})
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ∂σᶠ∂ϵᶠ = get_unknowns(stress_state, D)
    ∂σᶠ∂ϵᶜ = SMatrix{5,1}(D[2,2,1,1], D[3,3,1,1], √2*D[2,3,1,1], √2*D[1,3,1,1], √2*D[1,2,1,1])
    ∂σᶜ∂ϵᶠ = SMatrix{1,5}(D[1,1,2,2], D[1,1,3,3], √2*D[1,1,2,3], √2*D[1,1,1,3], √2*D[1,1,1,2])
    ∂σᶜ∂ϵᶜ = SMatrix{1,1}(D[1,1,1,1])
    return ∂σᶠ∂ϵᶠ, ∂σᶠ∂ϵᶜ, ∂σᶜ∂ϵᶠ, ∂σᶜ∂ϵᶜ
end
function extract_substiffnesses(stress_state::UniaxialStress, D::Tensor{4,3})
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ∂σᶠ∂ϵᶠ = get_unknowns(stress_state, D)
    ∂σᶠ∂ϵᶜ = SMatrix{8,1}(D[2,2,1,1], D[3,3,1,1], D[2,3,1,1], D[1,3,1,1], D[1,2,1,1], D[3,2,1,1], D[3,1,1,1], D[2,1,1,1])
    ∂σᶜ∂ϵᶠ = SMatrix{1,8}(D[1,1,2,2], D[1,1,3,3], D[1,1,2,3], D[1,1,1,3], D[1,1,1,2], D[1,1,3,2], D[1,1,3,1], D[1,1,2,1])
    ∂σᶜ∂ϵᶜ = SMatrix{1,1}(D[1,1,1,1])
    return ∂σᶠ∂ϵᶠ, ∂σᶠ∂ϵᶜ, ∂σᶜ∂ϵᶠ, ∂σᶜ∂ϵᶜ
end

# PlaneStress 
function extract_substiffnesses(stress_state::PlaneStress, D::SymmetricTensor{4,3})
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ∂σᶠ∂ϵᶠ = get_unknowns(stress_state, D)
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

function extract_substiffnesses(stress_state::PlaneStress, D::Tensor{4,3})
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ∂σᶠ∂ϵᶠ = get_unknowns(stress_state, D)
    ∂σᶠ∂ϵᶜ = @SMatrix [D[3,3,1,1] D[3,3,2,2] D[3,3,1,2] D[3,3,2,1];
                       D[2,3,1,1] D[2,3,2,2] D[2,3,1,2] D[2,3,2,1];
                       D[1,3,1,1] D[1,3,2,2] D[1,3,1,2] D[1,3,2,1];
                       D[3,2,1,1] D[3,2,2,2] D[3,2,1,2] D[3,2,2,1];
                       D[3,1,1,1] D[3,1,2,2] D[3,1,1,2] D[3,1,2,1]]
    
    ∂σᶜ∂ϵᶠ = @SMatrix [D[1,1,3,3] D[1,1,2,3] D[1,1,1,3] D[1,1,3,2] D[1,1,3,1];
                       D[2,2,3,3] D[2,2,2,3] D[2,2,1,3] D[2,2,3,2] D[2,2,3,1];
                       D[1,2,3,3] D[1,2,2,3] D[1,2,1,3] D[1,2,3,2] D[1,2,3,1];
                       D[2,1,3,3] D[2,1,2,3] D[2,1,1,3] D[2,1,3,2] D[2,1,3,1]]

    ∂σᶜ∂ϵᶜ = @SMatrix [D[1,1,1,1] D[1,1,2,2] D[1,1,1,2] D[1,1,2,1];
                       D[2,2,1,1] D[2,2,2,2] D[2,2,1,2] D[2,2,2,1];
                       D[1,2,1,1] D[1,2,2,2] D[1,2,1,2] D[1,2,2,1];
                       D[2,1,1,1] D[2,1,2,2] D[2,1,1,2] D[2,1,2,1]]
    return ∂σᶠ∂ϵᶠ, ∂σᶠ∂ϵᶜ, ∂σᶜ∂ϵᶠ, ∂σᶜ∂ϵᶜ
end
