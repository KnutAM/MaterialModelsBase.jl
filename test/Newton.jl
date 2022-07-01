# Temporary solution 
module Newton
using LinearAlgebra
using RecursiveFactorization
using DiffResults
using ForwardDiff
using StaticArrays

struct NewtonCache{T,Tres,Tcfg}
    x::Vector{T}
    result::Tres
    config::Tcfg
    lupivot::Vector{Int}
end

"""
    function NewtonCache(x::AbstractVector, rf!)
    
Create the cache used by the `newtonsolve` and `linsolve!`. 
Only a copy of `x` will be used. 
"""
function NewtonCache(x::AbstractVector, rf!)
    result = DiffResults.JacobianResult(x)
    cfg = ForwardDiff.JacobianConfig(rf!, x, result.value, ForwardDiff.Chunk(length(x)))
    lupivot = Vector{Int}(undef, length(x))
    return NewtonCache(copy(x), result, cfg, lupivot)
end

"""
    getx(cache::NewtonCache)
Extract out the unknown values. This can be used to avoid 
allocations when solving defining the initial guess. 
"""
getx(cache::NewtonCache) = cache.x

"""
    linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)
Solves the linear equation system `Kx=b`, mutating both `K` and `b`.
`b` is mutated to the solution `x`
"""
function linsolve!(K::AbstractMatrix, b::AbstractVector, cache::NewtonCache)
    LU = RecursiveFactorization.lu!(K, cache.lupivot, Val{true}(), Val{false}())
    ldiv!(LU, b)
    return b
end

"""
    newtonsolve(x0::AbstractVector, drdx::AbstractMatrix, rf!, cache::ResidualCache; tol=1.e-6, maxiter=100)
Solve the nonlinear equation system r(x)=0 using the newton-raphson method. 
Returns `x, drdx, true` if converged and `x, drdx, false` otherwise.
# args
- `x0`: Initial guess, not mutated (Unless aliased to `getx(cache)`)
- `rf!`: Residual function. Signature `rf!(r, x)` and mutating the residual `r`
- `cache`: Optional cache that can be preallocated by calling `ResidualCache(x0, rf!)`
# kwargs
- `tol=1.e-6`: Tolerance on `norm(r)`
- `maxiter=100`: Maximum number of iterations before no convergence
"""
function newtonsolve(x0::AbstractVector, rf!, cache::NewtonCache = NewtonCache(x0,rf!); tol=1.e-6, maxiter=100)
    diffresult = cache.result
    x = getx(cache)
    copy!(x, x0)
    cfg = cache.config
    for i = 1:maxiter
        # Disable checktag using Val{false}(). solve_residual should never be differentiated using dual numbers! 
        # This is required when using a different (but equivalent) anynomus function for caching than for running.
        ForwardDiff.jacobian!(diffresult, rf!, diffresult.value, x, cfg, Val{false}())
        err = norm(DiffResults.value(diffresult))
        # Check that we don't try to differentiate:
        i == 1 && check_no_dual(err)
        if err < tol
            drdx = DiffResults.jacobian(diffresult)
            return x, drdx, true
        end
        linsolve!(DiffResults.jacobian(diffresult), DiffResults.value(diffresult), cache)
        x .-= DiffResults.value(diffresult)
    end
    # No convergence
    return x, DiffResults.jacobian(diffresult), false
end

check_no_dual(::Number) = nothing
check_no_dual(::ForwardDiff.Dual) = throw(ArgumentError("newtonsolve cannot be differentiated"))

"""
    newtonsolve(x0::SVector, rf; tol=1.e-6, maxiter=100)
Solve the nonlinear equation system `r(x)=0` using the newton-raphson method.
Returns type: `(converged, x, drdx)`, SVector, SMatrix)` where 
- `converged::Bool` is `true` if converged and `false` otherwise
- `x::SVector` is the solution vector such that `r(x)=0`
- `drdx::SMatrix` is the jacobian at `x`
# args
- `x0`: Vector of with initial guess for unknowns.
- `rf`: Residual function. Signature `r=rf(x::SVector{dim})::SVector{dim}`
# kwargs
- `tol=1.e-6`: Tolerance on `norm(r)`
- `maxiter=100`: Maximum number of iterations before no convergence
"""
function newtonsolve(x::SVector{dim}, rf; tol=1.e-6, maxiter=100) where{dim}
    for _ = 1:maxiter
        r = rf(x)
        err = norm(r)
        drdx = ForwardDiff.jacobian(rf, x)
        if err < tol
            return x, drdx, true
        end
        x -= drdx\r
    end
    return zero(SVector{dim})*NaN, zero(SMatrix{dim,dim})*NaN, false
end
    
export newtonsolve
export NewtonCache
export getx

end