using MaterialModelsBase
using Test
using Tensors, StaticArrays, ForwardDiff

const MMB = MaterialModelsBase

include("TestMaterials.jl")
using .TestMaterials

include("stressiterations.jl")
