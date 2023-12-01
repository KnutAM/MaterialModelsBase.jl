using Tensors
import MaterialModelsBase as MMB

struct Elastic <: MMB.AbstractMaterial
    C::SymmetricTensor{4,3,Float64,36}
end
function Elastic(;G=80e3, K=160e3)
    I2 = one(SymmetricTensor{2,3})
    I4 = one(SymmetricTensor{4,3})
    C = 2G*(I4 - I2 ⊗ I2 / 3) + K*(I2⊗I2)
    return Elastic(C)
end
m = Elastic()

function MMB.material_response(m::Elastic, ϵ, args...)
    return m.C ⊡ ϵ, m.C, MMB.initial_material_state(m)
end

function run_case(m, num_steps)
    stress_state = MMB.UniaxialNormalStress()
    Δϵ = 0.1/num_steps
    Δt = 0.1
    state = MMB.initial_material_state(m)
    extras = MMB.NoExtraOutput()
    cache = MMB.allocate_material_cache(m)
    ϵ = zero(SymmetricTensor{2,3})
    s = 0.0
    for k in 1:num_steps
        #ϵ = SymmetricTensor{2,3}((i,j)->i==j==1 ? Δϵ*k : ϵ[i,j])
        ϵ = SymmetricTensor{2,3}((i,j)->i==j==1 ? Δϵ*k : zero(Δϵ))
        σ, dsde, state, ϵ = MMB.material_response(stress_state, m, ϵ, state, Δt, cache, extras)
        s = (s + σ[1,1])/2
    end
    return s
end


