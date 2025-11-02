using MaterialModelsBase
using Test
using Tensors, StaticArrays

const MMB = MaterialModelsBase

# <temporary, move to TestMaterials>
@kwdef struct StVenant{T} <: AbstractMaterial
    G::T
    K::T
end
function MMB.material_response(m::StVenant, F::Tensor{2,3}, old, args...)
    function free_energy(defgrad)
        E = (tdot(defgrad) - one(defgrad)) / 2
        Edev = dev(E)
        return m.G * Edev ⊡ Edev + m.K * tr(E)^2 / 2
    end
    ∂P∂F, P, _ = hessian(free_energy, F, :all)
    return P, ∂P∂F, old
end

@kwdef struct NeoHooke{T} <: AbstractMaterial
    G::T
    K::T
end
function MMB.material_response(m::NeoHooke, F::Tensor{2,3}, old, args...)
    function free_energy(defgrad)
        C = tdot(defgrad)
        detC = det(C)
        ΨG = (m.G / 2) * (tr(C) / cbrt(detC) - 3)
        ΨK = (m.K / 2) * (sqrt(detC) - 1)^2
        return ΨG + ΨK
    end
    ∂P∂F, P, _ = hessian(free_energy, F, :all)
    return P, ∂P∂F, old
end

MMB.get_tensorbase(::Union{StVenant, NeoHooke}) = Tensor{2,3}
# </temporary>

include("TestMaterials.jl")
using .TestMaterials: LinearElastic, ViscoElastic
include("utils4testing.jl")

include("vector_conversion.jl")
include("stressiterations.jl")
include("differentiation.jl")
include("errors.jl")
include("performance.jl")
