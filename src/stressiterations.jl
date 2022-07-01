# Based on https://github.com/kimauth/MaterialModels.jl

abstract type AbstractStressState end

# Cases without stress iterations
# Requires 3d strain input, outputs 3d stress
struct FullStressState <: AbstractStressState end
# Requires at least 2d strain input, outputs 2d stress
struct PlaneStrain <: AbstractStressState end
# Requires at least 1d strain input, outputs 1d stress
struct UniaxialStrain <: AbstractStressState end


# Cases with stress iterations
# Only σ11 != 0
# Requires at least 1d strain input, outputs 1d stress
struct UniaxialStress <: AbstractStressState end

# σ33=σ23=σ13=0
# Requires at least 2d strain input, outputs 2d stress
struct PlaneStress <: AbstractStressState end

# σ22=σ33=0
# Requires a full strain input, outputs 3d stress
struct UniaxialNormalStress <: AbstractStressState end

const NoIterationState = Union{FullStressState,PlaneStrain,UniaxialStrain}
const IterationState = Union{UniaxialStress, PlaneStress, UniaxialNormalStress}
const State3D = Union{FullStressState, UniaxialNormalStress}
const State2D = Union{PlaneStress, PlaneStrain}
const State1D = Union{UniaxialStrain, UniaxialStress}

# Translate from reduced input to full tensors
# Note: Also used to translate from unknowns to full tensors
get_full_tensor(::AbstractStressState, a::Tensors.AllTensors{3}) = a
#get_full_tensor(::AbstractStressState, v::Vec{dim,T}) where {dim,T} = Vec{3,T}(i->i>dim ? zero(T) : v[i])
function get_full_tensor(::AbstractStressState, a::Tensor{2,dim,T}) where {dim,T} 
    return Tensor{2,3}((i,j)-> (i<=dim && j<=dim) ? a[i,j] : zero(T))
end
function get_full_tensor(::AbstractStressState, a::SymmetricTensor{2,dim,T}) where {dim,T} 
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


function material_response(
    stress_state::NoIterationState,
    m::AbstractMaterial,
    ϵ::AbstractTensor,
    old::AbstractMaterialState,
    Δt=nothing,
    cache::AbstractMaterialCache=get_cache(m),
    extras::AbstractExtraOutput=NoExtraOutput();
    options::Dict=Dict{Symbol,Any}(),
    )

    ϵ_full = get_full_tensor(stress_state, ϵ)
    σ, dσdϵ, new_state = material_response(m, ϵ_full, old, Δt, cache, extras; options=options)
    return reduce_tensordim(stress_state, σ), reduce_tensordim(stress_state, dσdϵ), new_state, ϵ_full
end

function material_response(
    stress_state::IterationState,
    m::AbstractMaterial,
    ϵ::AbstractTensor,
    old::AbstractMaterialState,
    Δt=nothing,
    cache::AbstractMaterialCache=get_cache(m),
    extras::AbstractExtraOutput=NoExtraOutput();
    options::Dict=Dict{Symbol,Any}(),
    )

    # Newton options, typecast ensures type stability
    tol = Float64(get(options, :stress_state_tol, 1.e-8))
    maxiter = Int(get(options, :stress_state_maxiter, 10))

    ϵ_full = get_full_tensor(stress_state, ϵ)

    for _ in 1:maxiter
        σ_full, dσdϵ_full, new_state = material_response(m, ϵ_full, old, Δt, cache, extras; options=options)
        σ_mandel = get_unknowns(stress_state, σ_full)
        if norm(σ_mandel) < tol
            dσdϵ = reduce_stiffness(stress_state, dσdϵ_full)
            return reduce_tensordim(stress_state, σ_full), dσdϵ, new_state, ϵ_full
        end

        dσdϵ_mandel = get_unknowns(stress_state, dσdϵ_full)
        ϵ_full -= get_full_tensor(stress_state, ϵ, dσdϵ_mandel\σ_mandel)
    end
    throw(NoStressConvergence("Stress iterations with the NewtonSolver did not converge"))
end

reduce_stiffness(::State3D, dσdϵ_full::AbstractTensor{4,3}) = dσdϵ_full

function reduce_stiffness(stress_state, dσdϵ_full::AbstractTensor{4,3})
    ∂σᶠ∂ϵᶠ, ∂σᶠ∂ϵᶜ, ∂σᶜ∂ϵᶠ, ∂σᶜ∂ϵᶜ = extract_substiffnesses(stress_state, dσdϵ_full)
    dσᶜdϵᶜ = ∂σᶜ∂ϵᶜ - ∂σᶜ∂ϵᶠ * (∂σᶠ∂ϵᶠ \ ∂σᶠ∂ϵᶜ)
    return convert_stiffness(dσᶜdϵᶜ, stress_state, dσdϵ_full)
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
function get_full_tensor(::UniaxialStress, ::SymmetricTensor, v::SVector{5})
    s = 1/√2           # 11,   21,     31,     22,   32,     33
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
function get_full_tensor(::UniaxialStress, ϵ::Tensor, v::SVector{8,T}) where T
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
function get_full_tensor(::PlaneStress, ::SymmetricTensor, v::SVector{3})
    s = 1/√2           # 11,21,   31,  22,   32,     33
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
function get_full_tensor(::PlaneStress, ϵ::Tensor, v::SVector{5,T}) where T
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