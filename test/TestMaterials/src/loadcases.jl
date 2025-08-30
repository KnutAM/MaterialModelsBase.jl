function symmetric_components(components::NTuple{2, Int})
    return components[1] < components[2] ? reverse(components) : components
end

construct_tensor(ϵ::AbstractTensor, _) = ϵ

function construct_tensor(ϵ::Number, components)
    i, j = symmetric_components(components)
    return SymmetricTensor{2,3}((k, l) -> k == i && l == j ? ϵ : zero(ϵ))
end

function runstrain(m, ϵ_ij_end::Number, components, t_end, num_steps)
    ϵ_end = construct_tensor(ϵ_ij_end, components)
    i, j = symmetric_components(components)
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵ_end * scale
        σ, _, state = material_response(m, ϵ, state, Δt, cache)
        σv[k + 1] = σ[i, j]
    end
    return σv, state
end

function runstrain_diff(m, ϵ_ij_end::Number, components, t_end, num_steps)
    ϵ_end = construct_tensor(ϵ_ij_end, components)
    i, j = symmetric_components(components)
    mind = Tensors.DEFAULT_VOIGT_ORDER[3][j, i]
    mscale = i == j ? 1.0 : 1/√2
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    diff = MaterialDerivatives(m)
    extras = allocate_differentiation_output(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    dσdp = zeros(num_steps + 1, get_num_params(m))
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵ_end * scale
        old_state = state
        σ, dσdϵ, state = material_response(m, ϵ, old_state, Δt, cache, extras)
        differentiate_material!(diff, m, ϵ, old_state, Δt, cache, extras, dσdϵ)
        σv[k + 1] = σ[i, j]
        dσdp[k + 1, :] .= diff.dσdp[mind, :] .* mscale
    end
    return σv, state, dσdp, diff
end

function runstresstate(stress_state, m, ϵend::Union{Number, AbstractTensor}, components, t_end, num_steps)
    ϵt = construct_tensor(ϵend, components)
    i, j = symmetric_components(components)
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵt * scale
        σ, _, state = material_response(stress_state, m, ϵ, state, Δt, cache)
        σv[k + 1] = σ[i, j]
    end
    return σv, state
end

function runstresstate_diff(stress_state, m, ϵend::Union{Number, AbstractTensor}, components, t_end, num_steps)
    ϵt = construct_tensor(ϵend, components)
    i, j = symmetric_components(components)
    mind = Tensors.DEFAULT_VOIGT_ORDER[3][j, i]
    mscale = i == j ? 1.0 : 1/√2
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    diff = StressStateDerivatives(stress_state, m)
    extras = allocate_differentiation_output(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    dσdp = zeros(num_steps + 1, get_num_params(m))
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵt * scale
        old_state = state
        σ, _, state = differentiate_material!(diff, stress_state, m, ϵ, old_state, Δt, cache, extras)
        σv[k + 1] = σ[i, j]
        dσdp[k + 1, :] .= diff.dσdp[mind, :] .* mscale
    end
    return σv, state, dσdp, diff
end
