"""
    obtain_numerical_material_derivative!(deriv, m, ϵ, old, Δt; fdtype = Val{:forward}, kwargs...)

Obtain the numerical derivative of the material `m` at the strain `ϵ` and old state variables, `old`,
for a time step `Δt`. `fdtype` and `kwargs...` are passed to `FiniteDiff.finite_difference_jacobian`. 
"""
function obtain_numerical_material_derivative!(
        deriv::MMB.MaterialDerivatives,
        m::AbstractMaterial, ϵ, old, Δt; 
        fdtype = Val{:forward}, kwargs...
        )
    cache = allocate_material_cache(m)
    p = tovector(m)
    ⁿs = tovector(old)
    e = tovector(ϵ)

    function numjac(f::F, x) where {F}
        sz = (length(f(x)), length(x))
        prod(sz) == 0 && return zeros(sz...)
        return FiniteDiff.finite_difference_jacobian(f, x, fdtype; kwargs...)
    end
    numjac!(J, f::F, x) where {F} = copy!(J, numjac(f, x))

    funs = NamedTuple{(:σ, :s)}(
        (s = svec -> tovector(material_response(m, ϵ, fromvector(svec, old), Δt, cache)[i]), # f(s)
         p = pvec -> tovector(material_response(fromvector(pvec, m), ϵ, old, Δt, cache)[i]), # f(p)
         ϵ = evec -> tovector(material_response(m, fromvector(evec, ϵ), old, Δt, cache)[i])) # f(ϵ)
         for i in (1, 3)
    )

    dⁿsdp = copy(deriv.dsdp) # Copy not needed as dsdp field updated last, but safer in this test routine. 

    # Fill all partial derivatives
    numjac!(deriv.dσdϵ, funs.σ.ϵ, e)
    dσdⁿs = numjac(funs.σ.s, ⁿs)
    #numjac!(deriv. dσdⁿs, funs.σ.s, ⁿs) # not needed in deriv
    numjac!(deriv.dσdp, funs.σ.p, p)

    numjac!(deriv.dsdϵ, funs.s.ϵ, e)
    #numjac!(deriv. dsdⁿs, funs.s.s, ⁿs) # not needed in deriv
    dsdⁿs = numjac(funs.s.s, ⁿs)
    numjac!(deriv.dsdp, funs.s.p, p)

    # Adjust for dependence on previous state variable, i.e. ⁿs = ⁿs(p)
    deriv.dσdp .+= dσdⁿs * dⁿsdp
    deriv.dsdp .+= dsdⁿs * dⁿsdp

end

"""
    obtain_numerical_material_derivative!(ssd, stress_state, m, ϵ, old, Δt; fdtype = Val{:forward}, kwargs...)

Obtain the numerical derivative of the material `m` considering the `stress_state` iterations at the strain 
`ϵ` and old state variables, `old`, for a time step `Δt`. `fdtype` and `kwargs...` are passed to `FiniteDiff.finite_difference_jacobian`. 
"""
function obtain_numerical_material_derivative!(
        ssd::MMB.StressStateDerivatives, 
        stress_state::AbstractStressState, 
        m::AbstractMaterial, ϵ, old, Δt; 
        fdtype = Val{:forward}, kwargs...
        )
    cache = allocate_material_cache(m)
    p = tovector(m)
    ⁿs = tovector(old)

    function numjac(f::F, x) where {F}
        sz = (length(f(x)), length(x))
        prod(sz) == 0 && return zeros(sz...)
        return FiniteDiff.finite_difference_jacobian(f, x, fdtype; kwargs...)
    end
    numjac!(J, f::F, x) where {F} = copy!(J, numjac(f, x))

    funs = NamedTuple{(:σ, :ϵ, :s)}(
        (s = svec -> tovector(material_response(stress_state, m, ϵ, fromvector(svec, old), Δt, cache)[i]), # f(s)
         p = pvec -> tovector(material_response(stress_state, fromvector(pvec, m), ϵ, old, Δt, cache)[i])) # f(p)
         for i in (1, 4, 3)
    )
    
    dⁿsdp = copy(ssd.mderiv.dsdp)

    #numjac!(ssd.mderiv. dσdⁿs, funs.σ.s, ⁿs)
    dσdⁿs = numjac(funs.σ.s, ⁿs)
    numjac!(ssd.dσdp, funs.σ.p, p); ssd.dσdp .+= dσdⁿs * dⁿsdp

    dϵdⁿs = numjac(funs.ϵ.s, ⁿs);
    numjac!(ssd.dϵdp, funs.ϵ.p, p); ssd.dϵdp .+= dϵdⁿs * dⁿsdp
    
    dsdⁿs = numjac(funs.s.s, sv)
    numjac!(ssd.mderiv.dsdp, funs.s.p, p); ssd.mderiv.dsdp .+= dsdⁿs * dⁿsdp
end
