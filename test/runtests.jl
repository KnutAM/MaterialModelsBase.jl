using MaterialModelsBase
using Test
using Tensors, StaticArrays

const MMB = MaterialModelsBase

include("TestMaterials.jl")
using .TestMaterials

include("stressiterations.jl")
