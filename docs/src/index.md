```@meta
CurrentModule = MaterialModelsBase
```
# MaterialModelsBase

[MaterialModelsBase](https://github.com/KnutAM/MaterialModelsBase.jl)
provides an implementation-agnostic interface for mechanical (stress-strain)
material models. This facilitates interchanging material models between researchers by having a light-weight common interface.

Consequently, there are two types of user roles for this package (often the same
person takes both roles):

1) Someone writing code that use material models (e.g. finite element code)
2) Someone implementing material models

## Using material models
This section describes how to use a material model defined according to the described interface. The `TestMaterials` module is used to provide some implemented material models (see `test/TestMaterials.jl` in the main repo). Hence, replace `TestMaterials` with the specific material model package(s) you wish to use. 

If we would like to write a function that simulates the uniaxial response
of a material model, we can write a function to do that as
```julia
using MaterialModelsBase, Tensors
function simulate_uniaxial(m::AbstractMaterial, ϵ11_history, time_history)
    state = initial_material_state(material)
    cache = allocate_material_cache(material)
    stress_state = UniaxialStress()
    t_old = 0.0
    σ11_history = similar(ϵ11_history)
    for i in eachindex(ϵ11_history, time_history)
        Δt = time_history[i] - t_old
        ϵ = SymmetricTensor{2,1}((ϵ11_history[i],))
        σ, dσdϵ, state = material_response(stress_state, m, ϵ, state, Δt, cache)
        σ11_history[i] = σ[1,1]
    end
    return σ11_history
end
```

To use this function, we need to define the material parameters for an 
implemented material model in `TestMaterials`, for example
```julia
using TestMaterials
G1=50.e3; G2=80.e3; K1=100.e3; K2=160.e3; η=50e3;
material = ViscoElastic(LinearElastic(G1,K1), LinearElastic(G2,K2), η)
```

And then we can run the simulation
```julia
ϵ11_history  = collect(range(0, 0.01; length=100))  # Ramp to 1 %
time_history = collect(range(0, 0.2; length=100))   # Constant time step
σ11_history  = simulate_uniaxial(material, ϵ11_history, time_history)
```

This example used the stress iterations implemented in `MaterialModelsBase.jl`,
see [Stress States](@ref Stress-states).

If these do not converge, a [`NoStressConvergence`](@ref) exception is thrown. 

The package also contains the exception [`NoLocalConvergence`](@ref), 
which shouldbe thrown from inside implemented material routines to signal 
that something didn't converge and that the caller should consider 
to e.g. reduce the time step or handel the issue in some other way.

## Implementing material models
To implement a material model, at minimum, it is necessary to 
define a material type containing the material parameters, e.g. `MyMaterial`,
and the associated [`material_response(::MyMaterial, args...)`](@ref material_response) function.

For materials with state variables, the function [`initial_material_state`](@ref) should be defined. 

If some special pre-allocated variables are required for the material, the function [`allocate_material_cache`](@ref) should be defined. 


### Advanced use cases
The package also provides an interface for converting `AbstractMaterial`s 
to a vector of material parameters and vice versa. However, to implement 
this part of the interface is not required, if only the standard usage 
is intended for your material model.

Building upon that option, it is also possible to define differentiation
of the material response wrt. to the material parameters, which is useful 
for parameter identification using gradient based methods. For further details, 
see [Differentation of a material](@ref).


## API

### `material_response`
The main function is the `material_response` function that 
primarly dispatches on the `AbstractMaterial` input type. 
Two variants can be called, where the latter allows a reduced 
stress state, see [Stress States](@ref Stress-states) for further details. 
```@docs
material_response(::AbstractMaterial, ::Any, ::Any, ::Any, ::Any)
material_response(::AbstractStressState, ::AbstractMaterial, ::Vararg{Any})
```

### State variables
To support state variables define
```@docs
initial_material_state
```

### Cache
It is possible to pre-allocate a cache to avoid allocations during the material model calculation, to do this define
```@docs
allocate_material_cache
```

### Extra outputs
In some cases, it is necessary to define additional outputs to provide slight 
variations in the material calculation. In this case, a mutable input can be 
given as 
```@docs
AbstractExtraOutput
```

### Exceptions
Finally, the following exceptions are included
```@docs
MaterialConvergenceError
NoLocalConvergence
NoStressConvergence
```
