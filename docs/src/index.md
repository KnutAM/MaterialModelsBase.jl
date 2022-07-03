```@meta
CurrentModule = MaterialModelsBase
```

# [MaterialModelsBase](https://github.com/KnutAM/MaterialModelsBase.jl)
The main purpose of this package is to provide a unifying interface, 
facilitating the interchanging of material models between researchers. 
The basic idea, is to write packages (e.g. finite element simulations), 
that rely only on `MaterialModelsBase.jl`. It will then be easy to run
different material models written by other researchers, to check if your 
implementation is correct or to evaluate different material models. 

## Basic usage
This section describes how to use a material model defined according to the described interface. 
For this example, we will use the case `TestMaterials` module used for testing purposes
(see `test/TestMaterials.jl` in the main repo). Hence, replace `TestMaterials` with the specific
material model package(s) you wish to use. 

Use the required modules and define your material according to the specification
```julia
using MaterialModelsBase, TestMaterials, Tensors
G1=80.e3; G2=8.e3; K1=160.e3; K2=16.e3; η=1000.0
material = ViscoElastic(LinearElastic(G1,K1), LinearElastic(G2,K2), η)
```
As one who will run a code (e.g. a finite element solver or a material point simulator)
that someone else have written, this is all that needs to be done. 
If we would write a material point simulator, we also need to instantiate the material state,
and optionally the cache. 
```julia
state = initial_material_cache(material)
cache = get_cache(material)
```

Now, we can loop through the time history, let's say we want to simulate the uniaxial stress case:
```julia
stress_state = UniaxialStress() 
ϵ11_vector = collect(range(0, 0.01; length=100))   # Linear ramp
Δt = 1.e-3                                  # Fixed time step
for ϵ11 in ϵ11_vector
    ϵ11_tensor = SymmetricTensor{2,1}(ϵ11)
    σ, dσdϵ, state, ϵ_full = material_response(stress_state, material, ϵ11_tensor, state, Δt, cache)
end
```
where `σ` and `dσdϵ` are the reduced outputs for the given `stress_state` (in this case dimension 1).
The `state` is not directly affected by the `stress_state`, and the `ϵ_full` gives the full strain tensor,
such that 
```julia
σ, dσdϵ, state = material_response(material, ϵ_full, old_state, Δt, cache)
```
gives the full stress and stiffness, adhering to the constraint imposed by the `stress_state`. 
Note that in the latter function call, no `stress_state` was given, and then no `ϵ_full` output was given either. 

If the stress iterations would not converge, a 
`NoStressConvergence<:MaterialConvergenceError` 
exception is thrown. The package also contains the exception 
`NoLocalConvergence<:MaterialConvergenceError`, which should
be thrown from inside implemented material routines to signal 
that something didn't converge and that the caller should consider 
to e.g. reduce the time step or handel the issue in some other way. 

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

Finally, the following exceptions are included
```@docs
MaterialConvergenceError
NoLocalConvergence
NoStressConvergence
```
