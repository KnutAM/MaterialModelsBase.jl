```@meta
CurrentModule = MaterialModelsBase
```
# Conversion
`MaterialModelsBase` defines an interface for converting parameters, state variables, and other objects
to and from `AbstractVector`s. This is useful when doing parameter identification, or 
when interfacing with different languages that require information to be passed as arrays.

These function can be divided into those providing information about the material and state variables, and those doing the actual conversion.

## Information functions
The following should be defined for new materials,
```@docs
get_tensorbase
get_vector_length
get_vector_eltype
```

Whereas the following already have default implementations that should work provided that the above are implemented.
```@docs
get_num_statevars
get_num_tensorcomponents
```

## Conversion functions
The following should be defined for new materials, state variables, and alike
```@docs
tovector!
fromvector
```
whereas the following already have default implemenations that should work provided that the above functions are implemented,
```@docs
tovector
```
