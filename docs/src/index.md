```@meta
CurrentModule = MaterialModelsBase
```

# MaterialModelsBase

Documentation for [MaterialModelsBase](https://github.com/KnutAM/MaterialModelsBase.jl).

The main purpose of this package is to provide a unifying interface, 
facilitating the interchanging of material models between researchers. 
The standard use case is described below. In addition, the following 
extra features are included

- Stress iterations
- Differentiation wrt. material parameters


## Standard interface
The main function is the `material_response` function that 
primarly dispatches on the `AbstractMaterial` input type. 
```@docs
material_response
```

Additionally, the following types and functions could be defined for a material
```@docs
initial_material_state
```

```@docs
get_cache
```

```@docs
AbstractExtraOutput
```