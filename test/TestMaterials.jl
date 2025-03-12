module TestMaterials
using MaterialModelsBase
using Tensors, StaticArrays, ForwardDiff

# Overloaded functions
import MaterialModelsBase as MMB

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

function MMB.material_response(
        m::LinearElastic, ϵ::SymmetricTensor{2},
        old::NoMaterialState, 
        Δt, cache, extras) # Explicit values added to test the defaults
    dσdϵ = get_stiffness(m)
    σ = dσdϵ⊡ϵ
    return σ, dσdϵ, old
end

MMB.get_num_params(::LinearElastic) = 2
function MMB.tovector!(v::Vector, m::LinearElastic)
    v[1] = m.G
    v[2] = m.K
    return v
end
MMB.fromvector(v::Vector, ::LinearElastic) = LinearElastic(v[1], v[2])

function MMB.differentiate_material!(
    diff::MaterialDerivatives,
    m::LinearElastic,
    ϵ::SymmetricTensor,
    args...)
    tomandel!(diff.dσdϵ, get_stiffness(m))

    σ_from_p(p::Vector) = tomandel(get_stiffness(fromvector(p, m))⊡ϵ)
    ForwardDiff.jacobian!(diff.dσdp, σ_from_p, tovector(m))
end

# ViscoElastic material: Note - not physically reasonable model because 
# the volumetric response is also viscous
struct ViscoElastic{T} <: AbstractMaterial
    E1::LinearElastic{T}
    E2::LinearElastic{T}
    η::T
end
struct ViscoElasticState{T} <: AbstractMaterialState
    ϵv::SymmetricTensor{2,3,T,6}
end
MMB.initial_material_state(::ViscoElastic) = ViscoElasticState(zero(SymmetricTensor{2,3}))

# To allow easy automatic differentiation
function get_stress_ϵv(m::ViscoElastic, ϵ::SymmetricTensor{2,3}, old::ViscoElasticState, Δt)
    E1 = get_stiffness(m.E1)
    E2 = get_stiffness(m.E2)
    ϵv = inv(Δt*E2+one(E2)*m.η)⊡(Δt*E2⊡ϵ + m.η*old.ϵv)
    σ = E1⊡ϵ + m.η*(ϵv-old.ϵv)/Δt
    return σ, ϵv
end

function MMB.material_response(
        m::ViscoElastic, ϵ::SymmetricTensor{2},
        old::ViscoElasticState, Δt, args...)
    if Δt == zero(Δt)
        if norm(ϵ) ≈ 0
            return zero(ϵ), NaN*zero(get_stiffness(m.E1)), old
        else
            throw(NoLocalConvergence("Δt=0 is not allowed for non-zero strain"))
        end
    end
    σ, ϵv = get_stress_ϵv(m,ϵ,old,Δt)
    dσdϵ = gradient(ϵ_->get_stress_ϵv(m,ϵ_,old,Δt)[1], ϵ)
    return σ, dσdϵ, ViscoElasticState(ϵv)
end

end