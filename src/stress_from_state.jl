"""
    stress_from_state(m::AbstractMaterial, strain, state::AbstractMaterialState)
    stress_from_state(stress_state::AbstractStressState, m::AbstractMaterial, strain, state::AbstractMaterialState)
    stress_from_state(rss::ReducedStressState, strain, state::AbstractMaterialState)

## Using this interface
Calculate the stress that is energy-conjugated to `strain`, consistent with the *given*
`state`, without invoking any local iteration that would advance history/internal
variables. `state` is normally the already-converged state obtained from a previous
call to `material_response`, e.g. during postprocessing.

!!! warning
    Differentiating this function wrt. `strain` while holding `state` fixed gives a
    frozen-state tangent, which generally differs from `material_response`'s consistent
    tangent whenever internal/history variables would evolve with `strain` (e.g. for a
    plastic or viscous material). It is therefore not a general replacement for
    `material_response` during, e.g., equilibrium iterations.

## Implementing this interface
A material-model developer only needs to implement the full-dimensional method,
`stress_from_state(m::MyMaterial, strain, state::MyMaterialState)`. Support for a reduced-dimensional stress
state (e.g. via [`ReducedStressState`](@ref)) then follows automatically from a generic
fallback, using the tangent obtained by automatic differentiation via `Tensors.gradient`.
A specific reduced-dimensional method,
`stress_from_state(stress_state::AbstractStressState, m::MyMaterial, strain, state::MyMaterialState)`,
can be added when a cheaper, non-autodiff alternative exists.
"""
function stress_from_state end

# Wraps a material `m` and a frozen state `s` as an `AbstractMaterial`, whose
# `material_response` evaluates `stress_from_state(m, strain, s)` (at fixed
# history/internal variables) so that it can ride the existing stress-state Newton
# iteration (e.g. for `PlaneStress`). The tangent needed for that iteration is
# obtained via automatic differentiation. This powers the generic reduced-dimensional
# fallback of `stress_from_state` below.
struct FrozenStressMaterial{MT <: AbstractMaterial, ST <: AbstractMaterialState} <: AbstractMaterial
    m::MT
    s::ST
end
function material_response(fm::FrozenStressMaterial, strain::SecondOrderTensor{3}, old::AbstractMaterialState, args...)
    dσdϵ, σ = Tensors.gradient(e -> stress_from_state(fm.m, e, fm.s), strain, :all)
    return σ, dσdϵ, old
end

# Generic reduced-dimensional fallback: as long as `stress_from_state(m, strain,
# state)` (full-dimensional) is implemented for `m`, this makes `ReducedStressState`
# support "just work", by autodiff-ing through it.
function stress_from_state(stress_state::AbstractStressState, m::AbstractMaterial, strain, state::AbstractMaterialState)
    frozen = FrozenStressMaterial(m, state)
    σ, _, _, _ = material_response(stress_state, frozen, strain, NoMaterialState{eltype(strain)}())
    return σ
end

function stress_from_state(rss::ReducedStressState, strain, state::AbstractMaterialState)
    return stress_from_state(rss.stress_state, rss.material, strain, state)
end
