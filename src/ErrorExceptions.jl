"""
    MaterialConvergenceError

Abstract type that can be used to catch errors related to the material not converging. 
"""
abstract type MaterialConvergenceError <: Exception end

"""
    NoLocalConvergence(msg::String)
    NoLocalConvergence(args...)

Throw if the material_response routine doesn't converge internally.
Other arguments than a single `::String`, are converted to `String` with `string`
"""
struct NoLocalConvergence <: MaterialConvergenceError
    msg::String
end
NoLocalConvergence(args...) = NoLocalConvergence(string(args...))

"""
    NoStressConvergence(msg::String)
    NoStressConvergence(args...)

This is thrown if the stress iterations don't converge, see [Stress states](@ref)
Other arguments than a single `::String`, are converted to `String` with `string`
"""
struct NoStressConvergence <: MaterialConvergenceError
    msg::String
end
NoStressConvergence(args...) = NoStressConvergence(string(args...))