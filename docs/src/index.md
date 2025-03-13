```@meta
CurrentModule = MaterialModelsBase
```
# MaterialModelsBase

[MaterialModelsBase](https://github.com/KnutAM/MaterialModelsBase.jl)
provides an implementation-agnostic interface for mechanical (stress-strain)
material models. This facilitates interchanging material models between researchers by having a light-weight common interface. 

Key **features** of this package are
* Implement a material model for 3d, use [stress states](@ref Stress-states) to get, e.g.,
  uniaxial stress, plane stress, plane strain, and more without further implementation.
* Conversion routine interface to convert e.g. parameters to a vector and back, allowing    
  interfacing with optimization libraries for parameter identification.
* Differentiation routine interface to allow taking derivatives wrt. material parameters etc.
  to enable gradient-based optimization for parameter identification.

In general, there are two types of user roles for this package (often the same person takes both roles):

1) Someone writing code that use material models (e.g. finite element code)
2) Someone implementing material models

For nr 1, a brief introduction on how to use material models following the `MaterialModelsBase.jl` interface is given below. For nr 2, a first introduction is provided as a [tutorial](@ref basic-implementation).

## Using material models
This section describes how to use a material model defined according to the described interface.
As an example, we use the `Zener` material defined in the [implementation tutorial](@ref basic-implementation). For normal usage different materials are defined in a material models package, such as [MechanicalMaterialModels.jl](https://knutam.github.io/MechanicalMaterialModels.jl/dev/).

If we would like to write a function that simulates the uniaxial response
of a material model, we can write a function to do that as
```@example getstarted
using MaterialModelsBase, Tensors
```
```@example getstarted
include("implementation_snippets/includeshow.jl") #hide
@includeshow "implementation_snippets/simulate_uniaxial.jl" #hide
```

To use this function, we define the material parameters for the 
`Zener` viscoelastic material model whose implementation is demonstrated
[here](@ref basic-implementation).
```@example getstarted
include("implementation_snippets/zener_example.jl") #hide
material = Zener(;K = 100.0, G0 = 20.0, G1 = 80.0, η1 = 10.0)
nothing #hide
```

And then we define the strain and time history, before running the simulation,
```@example getstarted
ϵ11_history  = collect(range(0, 0.01; length=100))  # Ramp to 1 %
time_history = collect(range(0, 1; length=100))   # Constant time step
σ11_history  = simulate_uniaxial(material, ϵ11_history, time_history)
nothing #hide
```

We can also plot the stress-strain result,
```@example getstarted
import CairoMakie as Plt
fig = Plt.Figure()
ax = Plt.Axis(fig[1,1]; xlabel = "strain [%]", ylabel = "stress [MPa]")
Plt.lines!(ax, ϵ11_history * 100, σ11_history)
fig
```

This example used the stress iterations implemented in `MaterialModelsBase.jl`,
see [Stress States](@ref Stress-states).

If these do not converge, a [`NoStressConvergence`](@ref) exception is thrown. 

The package also contains the exception [`NoLocalConvergence`](@ref), 
which should be thrown from inside implemented material routines to signal 
that something didn't converge and that the caller should consider 
to e.g. reduce the time step or handle the issue in some other way.

## API

### `material_response`
The main function is the `material_response` function that 
primarily dispatches on the `AbstractMaterial` input type. 
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
