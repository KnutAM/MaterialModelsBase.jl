using MaterialModelsBase
using Test
using Tensors, StaticArrays

const MMB = MaterialModelsBase

include("TestMaterials.jl")
using .TestMaterials
include("utils4testing.jl")

include("stressiterations.jl")
include("differentiation.jl")
include("errors.jl")
include("performance.jl")
