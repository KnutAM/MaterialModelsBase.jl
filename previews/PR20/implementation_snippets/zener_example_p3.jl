function calculate_viscous_strain(m::Zener, ϵ, old::ZenerState, Δt)
    return (old.ϵv + (Δt * m.G1 / m.η1) * dev(ϵ)) / (1 + Δt * m.G1 / m.η1)
end

function calculate_stress(m::Zener, ϵ, old::ZenerState, Δt)
    ϵv = calculate_viscous_strain(m, ϵ, old, Δt)
    return 3 * m.K * vol(ϵ) + 2 * m.G0 * dev(ϵ) + 2 * m.G1 * (dev(ϵ) - ϵv)
end

function MaterialModelsBase.material_response(m::Zener, ϵ, old::ZenerState, Δt, cache, extras)
    dσdϵ, σ = gradient(e -> calculate_stress(m, e, old, Δt), ϵ, :all)
    ϵv = calculate_viscous_strain(m, ϵ, old, Δt)
    return σ, dσdϵ, ZenerState(ϵv)
end