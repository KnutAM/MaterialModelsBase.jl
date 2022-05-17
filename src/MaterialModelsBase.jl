module MaterialModelsBase
using Tensors

export material_response, initial_material_state, get_cache                 # Main (mandatory) functions
export AbstractMaterial                                                     # Material parameters
export AbstractMaterialState, NoMaterialState                               # State
export AbstractMaterialCache, NoMaterialCache                               # Cache
export AbstractExtraOutput, NoExtraOutput                                   # Extra output
export MaterialConvergenceError, NoLocalConvergence, NoStressConvergence    # Exceptions

export MaterialDerivatives, differentiate_material!                         # Differentiation routines
export allocate_differentiation_output
export getnumtensorcomponents, getnumstatevars, getnumparam
export material2vector, material2vector!, vector2material                   # Conversion routines

abstract type AbstractMaterial end

# Main material response routine
"""
    material_response(m::AbstractMaterial, ϵ::SymmetricTensor{2}, old::AbstractMaterialState, Δt, cache::AbstractMaterialCache, extras::AbstractExtraOutput; options)

    material_response(m::AbstractMaterial, F::Tensor{2}, old::AbstractMaterialState, Δt, cache::AbstractMaterialCache, extras::AbstractExtraOutput; options)

    material_response(m::AbstractMaterial, u::Vec, old::AbstractMaterialState, Δt, cache::AbstractMaterialCache, extras::AbstractExtraOutput; options)

Calculate the stress/traction, stiffness and updated state variables for the material `m`, given the strain input `ϵ`, the deformation gradient `F`, or the deformation vector `u`

# Mandatory arguments
- `m`: Defines the material and its parameters
- The second input can be one of
    - `ϵ::SymmetricTensor{2}`: The total small strain tensor at the end of the increment
    - `F::Tensor{2}`: The total deformation gradient at the end of the increment
    - `u::Vec`: The deformation at the end of the increment (for cohesive elements)
- `old::AbstractMaterialState`: The material state variables at the end of the last converged increment
    The material initial material state can be obtained by the [`initial_material_state`](@ref) function.

# Optional positional arguments
- `Δt`: The time step in the current increment. The default may depend on the material. 
- `cache::AbstractMaterialCache`: Cache variables that can be used to avoid allocations during each call to the `material_response` function
    This can be created by the [`get_cache`](@ref) function.
- `extras`: Updated with requested extra output. Defaults to the empty struct `NoExtraOutput`

# Optional keyword arguments
- `options`: Additional options that may be specific for each material. Defaults to `nothing`

# Output
- If the second input is the deformation gradient ``\\boldsymbol{F}`` (`F::Tensor{2}`)`, the outputs are
    - `P::Tensor{2}`: First Piola-Kirchhoff stress, ``\\boldsymbol{P}``
    - `dPdF::Tensor{4}`: Algorithmic tangent stiffness tensor, ``\\mathrm{d}\\boldsymbol{P}/\\mathrm{d}\\boldsymbol{F}``
    - `state::AbstractMaterialState`: The updated material state variables at the end of the time increment
- If the second input is the small strain tensor, ``\\boldsymbol{\\epsilon}`` (`ϵ::SymmetricTensor{2}`), the outputs are
    - `σ::SymmetricTensor{2}`: Cauchy stress tensor, ``\\boldsymbol{\\sigma}``
    - `dσdϵ::SymmetricTensor{4}`: Algorithmic tangent stiffness tensor, ``\\mathrm{d}\\boldsymbol{\\sigma}/\\mathrm{d}\\boldsymbol{\\epsilon}``
    - `state::AbstractMaterialState`: The updated material state variables at the end of the time increment
- If the second input is deformation vector, ``\\boldsymbol{u}`` (`u::Vec`), the outputs are
    - `t::Vec`: Traction vector, ``\\boldsymbol{t}``
    - `dtdu::SecondOrderTensor`: Algorithmic tangent stiffness tensor, ``\\mathrm{d}\\boldsymbol{t}/\\mathrm{d}\\boldsymbol{u}``
    - `state::AbstractMaterialState`: The updated material state variables at the end of the time increment

"""
function material_response end

# Initial material state
"""
    initial_material_state(m::AbstractMaterial)

Return the (default) initial `state::AbstractMaterialState` 
of the material `m`. 

Defaults to the empty `NoMaterialState()`
"""
function initial_material_state(::AbstractMaterial) 
    return NoMaterialState()
end

abstract type AbstractMaterialState end
struct NoMaterialState <: AbstractMaterialState end

# Material cache 
"""
    get_cache(m::AbstractMaterial)

Return a `cache::AbstractMaterialCache` that can be used when 
calling the material to reduce allocations

Defaults to the empty `NoMaterialCache()`
"""
function get_cache(::AbstractMaterial) 
    return NoMaterialCache()
end

abstract type AbstractMaterialCache end
struct NoMaterialCache <: AbstractMaterialCache end

# Extra output
"""
    AbstractExtraOutput()

By allocating an `AbstractExtraOutput` type, this type can be mutated
to extract additional information from the internal calculations 
in `material_response` only in cases when this is desired. 
E.g., when calculating derivatives. 
The concrete `NoExtraOutput<:AbstractExtraOutput` exists for the case
when no additional output should be calculated. 
"""
abstract type AbstractExtraOutput end

struct NoExtraOutput <: AbstractExtraOutput end

include("differentiation.jl")

# Convergence errors
abstract type MaterialConvergenceError <: Exception end
Base.showerror(io::IO, e::MaterialConvergenceError) = println(io, e.msg)

"""
    NoLocalConvergence(msg::String)

Throw if the material_response routine doesn't converge internally
"""
struct NoLocalConvergence <: MaterialConvergenceError
    msg::String
end

"""
    NoStressConvergence(msg::String)

Throw if the stress iterations don't converge
"""
struct NoStressConvergence <: MaterialConvergenceError
    msg::String
end

end