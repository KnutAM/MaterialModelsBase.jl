function simulate_uniaxial(m::AbstractMaterial, ϵ11_history, time_history)
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    stress_state = UniaxialStress()
    σ11_history = zero(ϵ11_history)
    for i in eachindex(ϵ11_history, time_history)[2:end]
        Δt = time_history[i] - time_history[i-1]
        ϵ = SymmetricTensor{2,1}((ϵ11_history[i],))
        σ, dσdϵ, state = material_response(stress_state, m, ϵ, state, Δt, cache)
        σ11_history[i] = σ[1,1]
    end
    return σ11_history
end