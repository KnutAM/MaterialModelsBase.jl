module TestMaterials
using MaterialModelsBase
using Tensors, StaticArrays
include(joinpath(@__DIR__, "Newton.jl"))
using .Newton

# Overloaded functions
import MaterialModelsBase: material_response, initial_material_state, get_cache

# Exported materials
export LinearElastic, ViscoElastic, Plastic
# Todo: Also NeoHookean to test nonlinear geometric materials

# LinearElastic material 
struct LinearElastic{T} <: AbstractMaterial 
    G::T
    K::T
end
function get_stiffness(m::LinearElastic)
    I2 = one(SymmetricTensor{2,3})
    Ivol = I2⊗I2
    Isymdev = minorsymmetric(otimesu(I2,I2) - Ivol/3)
    return 2*m.G*Isymdev + m.K*Ivol
end

function material_response(
    m::LinearElastic, 
    ϵ::SymmetricTensor{2},
    old::NoMaterialState, 
    Δt=nothing, 
    ::NoMaterialCache=get_cache(m), 
    ::NoExtraOutput=NoExtraOutput(); 
    options::Dict=Dict{Symbol,Any}())

    dσdϵ = get_stiffness(m)
    σ = dσdϵ⊡ϵ
    return σ, dσdϵ, old
end

# ViscoElastic material (decide model)
struct ViscoElastic{T} <: AbstractMaterial
    E1::LinearElastic{T}
    E2::LinearElastic{T}
    η::T
end
struct ViscoElasticState{T} <: AbstractMaterialState
    ϵv::SymmetricTensor{2,3,T,6}
end
initial_material_state(::ViscoElastic) = ViscoElasticState(zero(SymmetricTensor{2,3}))

# To allow easy automatic differentiation
function get_stress_ϵv(m::ViscoElastic, ϵ::SymmetricTensor{2,3}, old::ViscoElasticState, Δt)
    E1 = get_stiffness(m.E1)
    E2 = get_stiffness(m.E2)
    ϵv = inv(Δt*E2+one(E2)*m.η)⊡(Δt*E2⊡ϵ + m.η*old.ϵv)
    σ = E1⊡ϵ + η*(ϵv-old.ϵv)/Δt
    return σ, ϵv
end

function material_response(
    m::ViscoElastic, 
    ϵ::SymmetricTensor{2},
    old::ViscoElasticState, 
    Δt, # Time step required
    ::NoMaterialCache=get_cache(m), 
    ::NoExtraOutput=NoExtraOutput(); 
    options::Dict=Dict{Symbol,Any}())
    σ, ϵv = get_stress_ϵv(m,ϵ,old,Δt)
    dσdϵ = gradient(ϵ_->get_stress_ev(m,ϵ_,old,Δt)[1], ϵ)
    return σ, dσdϵ, ViscoElasticState(ϵv)
end

# Plastic material (nonlinear isotropic hardening von Mises plasticity)
struct Plastic{T} <: AbstractMaterial
    elastic::LinearElastic{T}
    Y0::T
    H::T
    κ∞::T
end
struct PlasticState{T} <: AbstractMaterial
    ϵp::SymmetricTensor{2,3,T,6}
    κ::T
end

# Helper functions
vonmises(σ) = √(3/2)*norm(dev(σ))
get_stress_from_x(m::Plastic, ϵ, x) = get_stiffness(m.elastic)⊡(ϵ-frommandel(SymmetricTensor{2,3},x))
function residual(x::SVector{8}, m::Plastic, ϵ, old)
    ϵp = frommandel(SymmetricTensor{2,3}, x)
    Δλ = x[7]
    κ = x[8]
    σ = get_stiffness(m.elastic)⊡(ϵ-ϵp)
    ν = gradient(vonmises, σ)
    Rϵp = (ϵp-old.ϵp) - Δλ*ν
    RΦ = vonmises(σ) - (m.Y0 + κ)
    Rκ = κ - (old.κ + Δλ*m.H*(1 - κ/m.κ∞))
    return SVector{8}(append!(tomandel(Rϵp), [RΦ,Rκ]))
end
function calculate_ats(m::Plastic, x::SVector, ϵ::SymmetricTensor{2,3}, old_state, ∂r∂x::SMatrix)
    ∂σ∂ϵ = elastic_stiffness(m)
    
    rf_ϵ(ϵv_) = residual(x, m, frommandel(SymmetricTensor{2,3}, ϵv_), old_state)
    ∂r∂ϵ = ForwardDiff.jacobian(rf_ϵ, SVector{6}(tomandel(ϵ)))

    σ_x(x_) = SVector{6}(tomandel(get_stress_from_x(m, ϵ, x_)))
    ∂σ∂x = ForwardDiff.jacobian(σ_x, x)

    return ∂σ∂ϵ - frommandel(SymmetricTensor{4,3}, ∂σ∂x*(∂r∂x\∂r∂ϵ))
end
# Main routine
function material_response(
    m::Plastic,
    ϵ::SymmetricTensor{2},
    old::PlasticState, 
    Δt, # Time step required
    ::NoMaterialCache=get_cache(m), 
    ::NoExtraOutput=NoExtraOutput(); 
    options::Dict=Dict{Symbol,Any}())
    D_elastic = get_stiffness(m.elastic)
    σ_trial = D_elastic⊡(ϵ-old.ϵp)
    if vonmises(σ_trial) <= (m.Y0+old.κ) # elastic
        return σ_trial, D_elastic, old
    else
        x0 = SVector{8}(append!(tomandel(old.ϵp), [0,old.κ]))
        rf(x_) = residual(x_, m, ϵ, old)
        x,∂r∂x,converged = newtonsolve(x0, rf)
        if converged
            σ = get_stress_from_x(m, ϵ, x)
            dσdϵ = calculate_ats(m, x, ϵ, old, ∂r∂x)
            ϵp = frommandel(SymmetricTensor{2,3}, x)
            κ = x[8]
            return σ, dσdϵ, PlasticState(ϵp, κ)
        else
            throw(NoLocalConvergence("Plastic didn't converge"))
        end
    end
end

end