# MaterialModelsBase

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KnutAM.github.io/MaterialModelsBase.jl/dev)
[![Build Status](https://github.com/KnutAM/MaterialModelsBase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KnutAM/MaterialModelsBase.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KnutAM/MaterialModelsBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KnutAM/MaterialModelsBase.jl)

Provide interface to "standard" history dependent mechanical (stress-strain) material models that is independent of the implementation.

## Main interface function
```julia
material_response(
    [stress_state::AbstractStressState]             # Optional stress state (e.g. plane stress) if full 3d is not desired. 
    m::AbstractMaterial,                            # Describes the specific material and its parameters
    ϵ::Union{SymmetricTensor{2}, Tensor{2}, Vec},   # ϵ (small strain tensor), F (deformation gradient), or u (displacement jump)
    old::AbstractMaterialState,                     # The old material state
    Δt,                                             # The time step
    cache::AbstractMaterialCache,                   # A cache that can be used to reduce allocations inside material_response
    extras::AbstractExtraOutput)                    # Custom struct whose entries can be mutated to provide extra information from material_response's calculations
```

## Dependencies
The dependencies are limited to [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl) 
and its dependencies to keep the package light-weight. 

## Limitations
* Only mechanical materials: stress-strain (continuum elements) or force-displacement (cohesive elements)
* Gradient-dependent materials are not supported. This restriction would be nice to lift if a suitable interface can be determined. 

## Acknowledgements
The interface was developed based on [MaterialModels.jl](https://github.com/kimauth/MaterialModels.jl)
