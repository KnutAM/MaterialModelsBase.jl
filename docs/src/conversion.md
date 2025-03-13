```@meta
CurrentModule = MaterialModelsBase
```
# Conversion
`MaterialModelsBase` defines an interface for converting parameters and state variables
to and from `AbstractVector`s. This is useful when doing parameter identification, or 
when interfacing with different languages that require information to be passed as arrays.

These function can be divided into those providing information about the material and state variables, and those doing the actual conversion.

## Information functions
```@docs
get_tensorbase
get_num_statevars
get_statevar_eltype
get_num_params
get_params_eltype
get_num_tensorcomponents
```

## Conversion functions
```@docs
tovector!
fromvector
tovector
```
