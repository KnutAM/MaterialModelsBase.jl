# How to implement an `AbstractMaterial`
In this tutorial, we show how a subtype of `AbstractMaterial` can be implemented,
specifically we demonstrate the simple case of a viscoelastic material according 
to the Zener model, illustrated by the following rheological model.

![zener model rheology](zener.svg)

For this model, we then have the stress,
```math
\begin{align*}
\boldsymbol{\sigma} &= K\mathrm{tr}(\epsilon)\boldsymbol{I} + \boldsymbol{\sigma}^\mathrm{dev} \\
\boldsymbol{\sigma}^\mathrm{dev} &= 2 G_0 \boldsymbol{\epsilon}^\mathrm{dev} + 2 G_1 \boldsymbol{\epsilon}_\mathrm{e}^\mathrm{dev} \\
\boldsymbol{\epsilon}_\mathrm{e}^\mathrm{dev} &= \boldsymbol{\epsilon}^\mathrm{dev} - \boldsymbol{\epsilon}_\mathrm{v}^\mathrm{dev}
\end{align*}
```
The stress depends on the viscous strain, $\boldsymbol{\epsilon}_\mathrm{v}^\mathrm{dev}$, which 
is goverened by the evolution law,
```math
\dot{\boldsymbol{\epsilon}}_\mathrm{v}^\mathrm{dev} = \frac{G_1}{\eta_1} \boldsymbol{\epsilon}_\mathrm{e}^\mathrm{dev}
```
We will implement this using the Backward-Euler time integration, such that we have 
```math
\boldsymbol{\epsilon}_\mathrm{v}^\mathrm{dev} = \frac{{}^\mathrm{old}\boldsymbol{\epsilon}_\mathrm{v}^\mathrm{dev} + \Delta t \frac{G_1}{\eta_1} \boldsymbol{\epsilon}^\mathrm{dev}}{1 + \Delta t \frac{G_1}{\eta_1}}
```
Where we will save the viscous strain as a state variable. 

## Basic implementation and usage
In order to define the material, we start by defining a material struct, which subtypes `AbstractMaterial`,
```@example
using MaterialModelsBase, Tensors

@kwdef struct Zener{T} <: AbstractMaterial
    K::T    # Bulk modulus
    G0::T   # Long-term shear modulus
    G1::T   # Shear modulus in viscous chain
    η1::T   # Viscosity in viscous chain
end
nothing #hide
```

```@example Zener
using Markdown
file = "implementation_snippets/zener_example_p1.jl" #hide
include(file) #hide
content = read(file, String)
Markdown.parse(""" 
\`\`\`julia
$content
\`\`\`
""")
```

Next, we define the state struct as well as the `initial_material_state` function,
```@example Zener
struct ZenerState{T} <: AbstractMaterialState
    ϵv::SymmetricTensor{2,3,T,6}
end
MaterialModelsBase.initial_material_state(::Zener) = ZenerState(zero(SymmetricTensor{2,3}))
nothing #hide
```

Before we define some helper functions and the `material_response` function,
```@example Zener
function calculate_viscous_strain(m::Zener, ϵ, old::ZenerState, Δt)
    return (old.ϵv + (Δt * m.G1 / m.η1) * dev(ϵ)) / (1 + Δt * m.G1 / m.η1)
end

function calculate_stress(m::Zener, ϵ, old::ZenerState, Δt)
    ϵv = calculate_viscous_strain(m, ϵ, old, Δt)
    return 3 * m.K * vol(ϵ) + 2 * m.G0 * dev(ϵ) + 2 * m.G1 * (dev(ϵ) - ϵv)
end

function MaterialModelsBase.material_response(m::Zener, ϵ, old::ZenerState, Δt, cache, extras)
    dσdϵ, σ = gradient(e -> calculate_stress(m, e, old, Δt), ϵ, :all)
    ϵv = calculate_viscous_strain(m, ϵ, old, Δt)
    return σ, dσdϵ, ZenerState(ϵv)
end
nothing #hide
```
The main reason for defining the helper functions is to facilitate differentiating the stress 
wrt. the strain input, and to avoid repeating code implementations.

### Using the implementation
After having defined the basic implementation, we can use it to simulate the stress-strain response. We will simulate the case of uniaxial stress, implying that all components of the 
stress, $\boldsymbol{\sigma}$, is zero, except the 11 component. However, in the implementation
above, we only control the strain input, but all strain components, except the 11 component, are 
unknown in the case of uniaxial stress. Therefore, we will use the `UniaxialStress` stress state,
to simulate a ramp of $\epsilon_{11}$, followed by a hold time.

We start by defining a function to calculate the uniaxial material response for any material following the `MaterialModelsBase` interface, given a time and strain history vector:
```@example Zener
# TODO: Save and include this to use the same on the first page #hide
function simulate_uniaxial(m::AbstractMaterial, ϵ11_history, time_history)
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    stress_state = UniaxialStress()
    σ11_history = zero(ϵ11_history)
    for i in eachindex(ϵ11_history, time_history)[2:end]
        Δt = time_history[i] - time_history[i-1]
        ϵ = SymmetricTensor{2,1}((ϵ11_history[i],))
        σ, dσdϵ, state = material_response(stress_state, m, ϵ, state, Δt, cache)
        σ11_history[i] = σ[1,1]
    end
    return σ11_history
end
nothing #hide
```
Next, we define the material properties and load case
```@example Zener
zener_material = Zener(;K = 100.0, G0 = 20.0, G1 = 30.0, η1 = 10.0) # Define material
ϵmax = 0.1  # Maximum strain value
tramp = 0.1 # Ramping time [s]
thold = 0.9 # Hold time [s]
nramp = 20  # Number of steps during ramp
nhold = 40  # Number of steps during hold

# Create the time history
t_history = collect(range(0, tramp, nramp + 1)) # Uniform ramp
append!(t_history, tramp .+ thold * range(0, 1, nhold + 1)[2:end] .^ 2) # Nonuniform hold

# Create the strain history
ϵ_history = collect(range(0, ϵmax, nramp + 1))
append!(ϵ_history, range(ϵmax, ϵmax, nhold + 1)[2:end])
nothing #hide
```
The nonuniform time steps make sense in this case to capture the fast relaxation after the ramp,
but is not required and we could instead use more uniformly spaced time steps to get the same accuracy. First, we simply plot the loading case we supply to see this,
```@example Zener
import CairoMakie as Plt
fig1 = Plt.Figure(;size = (600, 300))
ax1 = Plt.Axis(fig1[1,1]; xlabel = "time [s]", ylabel = "ϵ₁₁ [%]")
Plt.scatter!(ax1, t_history, 100 * ϵ_history)
fig1
```

We are now ready to simulate the response,
```@example Zener
σ_history = simulate_uniaxial(zener_material, ϵ_history, t_history)
nothing #hide
```

and plot it,
```@example Zener
fig2 = Plt.Figure(;size = (600, 300))
ax2 = Plt.Axis(fig2[1,1]; xlabel = "time [s]", ylabel = "σ₁₁ [MPa]")
Plt.lines!(ax2, t_history, σ_history)
fig2
```
