using MaterialModelsBase, Tensors

@kwdef struct Zener{T} <: AbstractMaterial
    K::T    # Bulk modulus
    G0::T   # Long-term shear modulus
    G1::T   # Shear modulus in viscous chain
    η1::T   # Viscosity in viscous chain
end