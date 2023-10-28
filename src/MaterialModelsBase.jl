module MaterialModelsBase
using Tensors, StaticArrays
import Base: @kwdef

# General interface for <:AbstractMaterial
export AbstractMaterial                                 # Material parameters
export material_response                                # Main (mandatory) functions
export initial_material_state, allocate_material_cache  # Optional, but common functions to define
export AbstractMaterialState, NoMaterialState           # State
export AbstractMaterialCache, NoMaterialCache           # Cache
export AbstractExtraOutput,   NoExtraOutput             # Extra output

# Exceptions
export MaterialConvergenceError, NoLocalConvergence, NoStressConvergence    

# Stress state iterations
export AbstractStressState
export ReducedStressState
export FullStressState, PlaneStrain, UniaxialStrain         # Non-iteration stress states
export PlaneStress, UniaxialStress, UniaxialNormalStress    # Iterative stress state (unless overloaded)
export GeneralStressState                                   # General iterative 3D non-zero stress state
export update_stress_state!                                 # For nonzero stress-conditions

# For parameter identification and differentiation of materials
export material2vector, material2vector!, vector2material                   # Convert to/from parameter vector
export get_num_tensorcomponents, get_num_statevars, get_num_params                # Information about the specific material 
export MaterialDerivatives, differentiate_material!                         # Differentiation routines
export allocate_differentiation_output

abstract type AbstractMaterial end

"""
    material_response(
        [stress_state::AbstractStressState],
        m::AbstractMaterial, 
        strain::Union{SecondOrderTensor,Vec}, 
        old::AbstractMaterialState, 
        Δt::Union{Number,Nothing}=nothing, 
        cache::AbstractMaterialCache=allocate_material_cache(m), 
        extras::AbstractExtraOutput=NoExtraOutput()
        )

Calculate the stress/traction, stiffness and updated state variables 
for the material `m`, given the strain input `strain`.

# Mandatory arguments
- `m`: Defines the material and its parameters
- The second `strain` input can be, e.g.
    - `ϵ::SymmetricTensor{2}`: The total small strain tensor at the end of the increment
    - `F::Tensor{2}`: The total deformation gradient at the end of the increment
    - `u::Vec`: The deformation at the end of the increment (for cohesive elements)
- `old::AbstractMaterialState`: The material state variables at the end of the last 
   converged increment. The material initial material state can be obtained by 
   the [`initial_material_state`](@ref) function.

# Optional positional arguments
- `stress_state`: Use to solve for a reduced stress state, e.g. PlaneStress. 
   See [Stress states](@ref).
- `Δt`: The time step in the current increment. Defaults to `nothing`. 
- `cache::AbstractMaterialCache`: Cache variables that can be used to avoid
  allocations during each call to the `material_response` function. 
  This can be created by the [`allocate_material_cache`](@ref) function.
- `extras`: Updated with requested extra output. 
  Defaults to the empty struct `NoExtraOutput`

# Outputs
1) `stress`, is the stress measure that is energy conjugated to the `strain` (2nd) input.
2) `stiffness`, is the derivative of the `stress` output wrt. the `strain` input. 
3) `new_state`, are the updated state variables
4) `strain`, is only available if `stress_state` is given and returns the full strain tensor

The following are probably the three most common `strain` and `stress` pairs:
- If the second input is the deformation gradient ``\\boldsymbol{F}`` (`F::Tensor{2}`)`, the outputs are
    - `P::Tensor{2}`: First Piola-Kirchhoff stress, ``\\boldsymbol{P}``
    - `dPdF::Tensor{4}`: Algorithmic tangent stiffness tensor, ``\\mathrm{d}\\boldsymbol{P}/\\mathrm{d}\\boldsymbol{F}``
- If the second input is the small strain tensor, ``\\boldsymbol{\\epsilon}`` (`ϵ::SymmetricTensor{2}`), the outputs are
    - `σ::SymmetricTensor{2}`: Cauchy stress tensor, ``\\boldsymbol{\\sigma}``
    - `dσdϵ::SymmetricTensor{4}`: Algorithmic tangent stiffness tensor, ``\\mathrm{d}\\boldsymbol{\\sigma}/\\mathrm{d}\\boldsymbol{\\epsilon}``
- If the second input is deformation vector, ``\\boldsymbol{u}`` (`u::Vec`), the outputs are
    - `t::Vec`: Traction vector, ``\\boldsymbol{t}``
    - `dtdu::SecondOrderTensor`: Algorithmic tangent stiffness tensor, ``\\mathrm{d}\\boldsymbol{t}/\\mathrm{d}\\boldsymbol{u}``
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
    allocate_material_cache(m::AbstractMaterial)

Return a `cache::AbstractMaterialCache` that can be used when 
calling the material to reduce allocations

Defaults to the empty `NoMaterialCache()`
"""
function allocate_material_cache(::AbstractMaterial) 
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
E.g., when calculating derivatives or for multiphysics simulations.
The concrete `NoExtraOutput<:AbstractExtraOutput` exists for the case
when no additional output should be calculated. 
"""
abstract type AbstractExtraOutput end

struct NoExtraOutput <: AbstractExtraOutput end

include("differentiation.jl")
include("stressiterations.jl")
include("ErrorExceptions.jl")

end