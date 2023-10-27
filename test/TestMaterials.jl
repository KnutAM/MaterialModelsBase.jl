module TestMaterials
using MaterialModelsBase
using Tensors, StaticArrays

# Overloaded functions
import MaterialModelsBase: material_response, initial_material_state, allocate_material_cache

# Exported materials
export LinearElastic, ViscoElastic
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
    ::NoMaterialCache=allocate_material_cache(m), 
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
    σ = E1⊡ϵ + m.η*(ϵv-old.ϵv)/Δt
    return σ, ϵv
end

function material_response(
    m::ViscoElastic, 
    ϵ::SymmetricTensor{2},
    old::ViscoElasticState, 
    Δt, # Time step required
    ::NoMaterialCache=allocate_material_cache(m), 
    ::NoExtraOutput=NoExtraOutput(); 
    options::Dict=Dict{Symbol,Any}())
    σ, ϵv = get_stress_ϵv(m,ϵ,old,Δt)
    dσdϵ = gradient(ϵ_->get_stress_ϵv(m,ϵ_,old,Δt)[1], ϵ)
    return σ, dσdϵ, ViscoElasticState(ϵv)
end

end