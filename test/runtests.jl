using MaterialModelsBase
using Test
using Tensors, StaticArrays
import MaterialModelsBase as MMB
using TestMaterials: TestMaterials, LinearElastic, ViscoElastic
using FiniteDiff: FiniteDiff

include("utils4testing.jl")

include("vector_conversion.jl")
include("stressiterations.jl")
include("differentiation.jl")
include("errors.jl")
include("performance.jl")
