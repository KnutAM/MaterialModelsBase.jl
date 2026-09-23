module CurrentStressTestMaterials
using MaterialModelsBase
using Tensors
import MaterialModelsBase as MMB

# Well-conditioned isotropic elastic stiffness tensors, so that the `PlaneStress`
# Newton iteration used below converges reliably (unlike an arbitrary random
# 4th-order tensor, which need not be positive definite).
function isotropic_C(G, K)
    I2 = one(SymmetricTensor{2,3})
    I4vol = otimes(I2, I2) / 3
    I4dev = one(SymmetricTensor{4,3}) - I4vol
    return 2G * I4dev + 3K * I4vol
end
function isotropic_C_finite(G, K)
    I2 = one(Tensor{2,3})
    return 2G * otimesu(I2, I2) + K * otimes(I2, I2)
end

# Stateless (NoMaterialState) toy material: exercises the generic
# reduced-dimensional fallback (via `FrozenStressMaterial` and autodiff), given
# only a full-dimensional `stress_from_state` method.
struct ToyElastic{T} <: AbstractMaterial
    C::SymmetricTensor{4,3,T}
end
MMB.initial_material_state(m::ToyElastic{T}) where {T} = MMB.NoMaterialState{T}()
function MMB.material_response(m::ToyElastic, ϵ::SymmetricTensor{2,3}, state, args...)
    return m.C ⊡ ϵ, m.C, state
end
MMB.stress_from_state(m::ToyElastic, ϵ::SymmetricTensor{2,3}, state) = m.C ⊡ ϵ

# Small-strain toy material with a state that unconditionally "evolves" every call
# (unlike real plasticity with a yield surface), so that a frozen-state evaluation
# at a different strain is guaranteed to differ from a fresh `material_response` call.
struct ToyHistory{T} <: AbstractMaterial
    C::SymmetricTensor{4,3,T}
    H::T
end
struct ToyHistoryState{T} <: AbstractMaterialState
    ϵp::SymmetricTensor{2,3,T}
end
MMB.initial_material_state(m::ToyHistory{T}) where {T} = ToyHistoryState(zero(SymmetricTensor{2,3,T}))
function MMB.material_response(m::ToyHistory, ϵ::SymmetricTensor{2,3}, state::ToyHistoryState, args...)
    ϵp_new = state.ϵp + m.H * (ϵ - state.ϵp)
    σ = m.C ⊡ (ϵ - ϵp_new)
    dσdϵ = (1 - m.H) * m.C # Consistent tangent: ϵp_new depends linearly on ϵ
    return σ, dσdϵ, ToyHistoryState(ϵp_new)
end
MMB.stress_from_state(m::ToyHistory, ϵ::SymmetricTensor{2,3}, state::ToyHistoryState) = m.C ⊡ (ϵ - state.ϵp)

# Finite-strain (nonsymmetric `Tensor{2,3}`) analogue, to check that
# `FrozenStressMaterial`'s autodiff also works for the `Tensor` tensor family.
struct ToyHistoryFinite{T} <: AbstractMaterial
    C::Tensor{4,3,T}
    H::T
end
struct ToyHistoryFiniteState{T} <: AbstractMaterialState
    Fp::Tensor{2,3,T}
end
MMB.initial_material_state(m::ToyHistoryFinite{T}) where {T} = ToyHistoryFiniteState(zero(Tensor{2,3,T}))
function MMB.material_response(m::ToyHistoryFinite, F::Tensor{2,3}, state::ToyHistoryFiniteState, args...)
    Fp_new = state.Fp + m.H * (F - state.Fp)
    P = m.C ⊡ (F - Fp_new)
    dPdF = (1 - m.H) * m.C # Consistent tangent: Fp_new depends linearly on F
    return P, dPdF, ToyHistoryFiniteState(Fp_new)
end
MMB.stress_from_state(m::ToyHistoryFinite, F::Tensor{2,3}, state::ToyHistoryFiniteState) = m.C ⊡ (F - state.Fp)

# A plain, stateless material equivalent to `ToyHistory` frozen at a given `ϵp`, used
# as an independent reference to check the generic reduced-dimensional fallback
# (which goes through `FrozenStressMaterial` and autodiff) against MMB's own,
# already-tested, `PlaneStress` Newton iteration on an explicit material.
struct FrozenLinear{T} <: AbstractMaterial
    C::SymmetricTensor{4,3,T}
    ϵp::SymmetricTensor{2,3,T}
end
MMB.initial_material_state(m::FrozenLinear{T}) where {T} = MMB.NoMaterialState{T}()
function MMB.material_response(m::FrozenLinear, ϵ::SymmetricTensor{2,3}, state, args...)
    return m.C ⊡ (ϵ - m.ϵp), m.C, state
end

end # module

import .CurrentStressTestMaterials as CT

@testset "stress_from_state" begin
    @testset "Stateless material, generic reduced-dimensional fallback" begin
        C = CT.isotropic_C(80.e3, 160.e3)
        m = CT.ToyElastic(C)
        state = initial_material_state(m)
        @test state isa MaterialModelsBase.NoMaterialState
        ϵ = rand(SymmetricTensor{2,3})
        @test stress_from_state(m, ϵ, state) ≈ C ⊡ ϵ

        # Reduced-dimensional support follows automatically from the full-dimensional
        # method above, via `FrozenStressMaterial` and autodiff.
        rss = ReducedStressState(PlaneStress(), m)
        ϵ_red = rand(SymmetricTensor{2,2})
        state_red = initial_material_state(rss)
        σ_direct = stress_from_state(rss, ϵ_red, state_red)
        σ_mr, _, _, _ = material_response(rss, ϵ_red, state_red)
        @test σ_direct ≈ σ_mr
    end

    @testset "Stateful material, full dimension" begin
        C = CT.isotropic_C(80.e3, 160.e3)
        m = CT.ToyHistory(C, 0.5)

        state0 = initial_material_state(m)
        ϵ1 = rand(SymmetricTensor{2,3})
        σ1, _, state1 = material_response(m, ϵ1, state0)
        @test stress_from_state(m, ϵ1, state1) ≈ σ1

        # Frozen-state postprocessing: a different strain should give the frozen-ϵp
        # response, NOT a fresh history update.
        ϵ2 = ϵ1 + rand(SymmetricTensor{2,3}) / 10
        σ2_frozen = stress_from_state(m, ϵ2, state1)
        @test σ2_frozen ≈ C ⊡ (ϵ2 - state1.ϵp)
        σ2_true, _, state2_true = material_response(m, ϵ2, state1)
        @test !(σ2_true ≈ σ2_frozen)
        @test state2_true.ϵp != state1.ϵp
    end

    @testset "Stateful material, generic reduced-dimensional fallback" begin
        C = CT.isotropic_C(80.e3, 160.e3)
        m = CT.ToyHistory(C, 0.5)
        rss = ReducedStressState(PlaneStress(), m)

        state0 = initial_material_state(rss)
        ϵ1 = rand(SymmetricTensor{2,2}) / 10
        σ1, _, state1, _ = material_response(rss, ϵ1, state0)
        @test stress_from_state(rss, ϵ1, state1) ≈ σ1

        ϵ2 = ϵ1 + rand(SymmetricTensor{2,2}) / 10
        σ2_frozen = stress_from_state(rss, ϵ2, state1)

        # Independent reference: the frozen material is exactly linear elastic in
        # (ϵ - state1.ϵp), so its plane-stress response can be obtained directly
        # from MMB's own (already-tested) stress-state iteration on an explicit,
        # equivalent material, bypassing `stress_from_state` entirely.
        flin = CT.FrozenLinear(C, state1.ϵp)
        σ2_expected, _, _, _ = material_response(PlaneStress(), flin, ϵ2, initial_material_state(flin))
        @test σ2_frozen ≈ σ2_expected

        σ2_true, _, state2_true, _ = material_response(rss, ϵ2, state1)
        @test !(σ2_true ≈ σ2_frozen)
        @test state2_true.ϵp != state1.ϵp
    end

    @testset "Finite-strain (Tensor) frozen-state fallback" begin
        C = CT.isotropic_C_finite(80.e3, 160.e3)
        m = CT.ToyHistoryFinite(C, 0.5)
        rss = ReducedStressState(PlaneStress(), m)

        state0 = initial_material_state(rss)
        F1 = one(Tensor{2,2}) + rand(Tensor{2,2}) / 20
        P1, _, state1, _ = material_response(rss, F1, state0)
        @test stress_from_state(rss, F1, state1) ≈ P1

        F2 = F1 + rand(Tensor{2,2}) / 20
        P2_frozen = stress_from_state(rss, F2, state1)
        P2_true, _, state2_true, _ = material_response(rss, F2, state1)
        @test !(P2_true ≈ P2_frozen)
        @test state2_true.Fp != state1.Fp
    end
end
