var documenterSearchIndex = {"docs":
[{"location":"differentiation/#Differentation-of-a-material","page":"Differentiation","title":"Differentation of a material","text":"","category":"section"},{"location":"differentiation/","page":"Differentiation","title":"Differentiation","text":"get_num_tensorcomponents\nget_num_statevars\nget_num_params\nget_parameter_type","category":"page"},{"location":"differentiation/#MaterialModelsBase.get_num_tensorcomponents","page":"Differentiation","title":"MaterialModelsBase.get_num_tensorcomponents","text":"get_num_tensorcomponents(::AbstractMaterial)\n\nReturns the number of independent components for the given material. \n\nIf the material works with the small strain tensor and Cauchy stress, return 6 (default)\nIf the material works with the deformation gradient and the 1st Piola-Kirchhoff stress, return 9\nIf the material is a cohesive material working with vectors, return the number of vector components (e.g. 3)\n\nDefaults to 6 if not overloaded\n\n\n\n\n\n","category":"function"},{"location":"differentiation/#MaterialModelsBase.get_num_statevars","page":"Differentiation","title":"MaterialModelsBase.get_num_statevars","text":"get_num_statevars(m::AbstractMaterial)\n\nReturn the number of state variables. A tensorial state variable should be counted by how many components it has.  E.g. if a state consists of one scalar and one symmetric 2nd order tensor, get_num_statevars should return 7 (if the space dimension is 3).\n\nDefaults to 0 if not overloaded\n\n\n\n\n\n","category":"function"},{"location":"differentiation/#MaterialModelsBase.get_num_params","page":"Differentiation","title":"MaterialModelsBase.get_num_params","text":"get_num_params(m::AbstractMaterial)\n\nReturn the number of material parameters in m. No default value implemented. \n\n\n\n\n\n","category":"function"},{"location":"differentiation/#MaterialModelsBase.get_parameter_type","page":"Differentiation","title":"MaterialModelsBase.get_parameter_type","text":"get_parameter_type(m::AbstractMaterial)\n\nReturn the number type for the scalar material parameters, defaults to Float64\n\n\n\n\n\n","category":"function"},{"location":"differentiation/","page":"Differentiation","title":"Differentiation","text":"material2vector!\nvector2material\nmaterial2vector","category":"page"},{"location":"differentiation/#MaterialModelsBase.material2vector!","page":"Differentiation","title":"MaterialModelsBase.material2vector!","text":"material2vector!(v::AbstractVector, m::AbstractMaterial)\n\nPut the material parameters of m into the vector m.  This is typically used when the parameters should be fitted.\n\n\n\n\n\n","category":"function"},{"location":"differentiation/#MaterialModelsBase.vector2material","page":"Differentiation","title":"MaterialModelsBase.vector2material","text":"vector2material(v::AbstractVector, ::MT) where {MT<:AbstractMaterial}\n\nCreate a material of type MT with the parameters according to v\n\n\n\n\n\n","category":"function"},{"location":"differentiation/#MaterialModelsBase.material2vector","page":"Differentiation","title":"MaterialModelsBase.material2vector","text":"material2vector(m::AbstractMaterial)\n\nOut-of place version of material2vector!. Given get_num_params, this function does not need to be overloaded unless another datatype than Float64 should be used.\n\n\n\n\n\n","category":"function"},{"location":"differentiation/","page":"Differentiation","title":"Differentiation","text":"MaterialDerivatives\nallocate_differentiation_output\ndifferentiate_material!","category":"page"},{"location":"differentiation/#MaterialModelsBase.MaterialDerivatives","page":"Differentiation","title":"MaterialModelsBase.MaterialDerivatives","text":"MaterialDerivatives(m::AbstractMaterial)\n\nA struct that saves all derivative information using a Matrix{T} for each derivative, where T=get_parameter_type(m). The dimensions are obtained from get_num_tensorcomponents,  get_num_statevars, and get_num_params. The values should be updated in differentiate_material! by direct access of the fields, where σ is the stress, ϵ the strain, s and ⁿs are the current  and old state variables, and p the material parameter vector.\n\ndσdϵ\ndσdⁿs\ndσdp\ndsdϵ\ndsdⁿs\ndsdp\n\n\n\n\n\n","category":"type"},{"location":"differentiation/#MaterialModelsBase.allocate_differentiation_output","page":"Differentiation","title":"MaterialModelsBase.allocate_differentiation_output","text":"allocate_differentiation_output(::AbstractMaterial)\n\nWhen calculating the derivatives of a material, it can often be advantageous to have additional  information from the solution procedure inside material_response. This can be obtained via an  AbstractExtraOutput, and allocate_differentiation_output provides a standard function name  for what extra_output::AbstractExtraOutput that should be allocated in such cases.\n\nDefaults to an NoExtraOutput if not overloaded. \n\n\n\n\n\n","category":"function"},{"location":"differentiation/#MaterialModelsBase.differentiate_material!","page":"Differentiation","title":"MaterialModelsBase.differentiate_material!","text":"differentiate_material!(\n    diff::MaterialDerivatives, \n    m::AbstractMaterial, \n    ϵ::Union{SecondOrderTensor, Vec}, \n    old::AbstractMaterialState, \n    Δt,\n    cache::AbstractMaterialCache\n    dσdϵ::AbstractTensor, \n    extra::AbstractExtraOutput\n    )\n\nCalculate the derivatives and save them in diff, see MaterialDerivatives for a description of the fields in diff.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = MaterialModelsBase","category":"page"},{"location":"#MaterialModelsBase","page":"Home","title":"MaterialModelsBase","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"MaterialModelsBase provides an implementation-agnostic interface for mechanical (stress-strain) material models. This facilitates interchanging material models between researchers by having a light-weight common interface.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Consequently, there are two types of user roles for this package (often the same person takes both roles):","category":"page"},{"location":"","page":"Home","title":"Home","text":"Someone writing code that use material models (e.g. finite element code)\nSomeone implementing material models","category":"page"},{"location":"#Using-material-models","page":"Home","title":"Using material models","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This section describes how to use a material model defined according to the described interface. The TestMaterials module is used to provide some implemented material models (see test/TestMaterials.jl in the main repo). Hence, replace TestMaterials with the specific material model package(s) you wish to use. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"If we would like to write a function that simulates the uniaxial response of a material model, we can write a function to do that as","category":"page"},{"location":"","page":"Home","title":"Home","text":"using MaterialModelsBase, Tensors\nfunction simulate_uniaxial(m::AbstractMaterial, ϵ11_history, time_history)\n    state = initial_material_state(material)\n    cache = allocate_material_cache(material)\n    stress_state = UniaxialStress()\n    t_old = 0.0\n    σ11_history = similar(ϵ11_history)\n    for i in eachindex(ϵ11_history, time_history)\n        Δt = time_history[i] - t_old\n        ϵ = SymmetricTensor{2,1}((ϵ11_history[i],))\n        σ, dσdϵ, state = material_response(stress_state, m, ϵ, state, Δt, cache)\n        σ11_history[i] = σ[1,1]\n        t_old += Δt\n    end\n    return σ11_history\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"To use this function, we need to define the material parameters for an  implemented material model in TestMaterials, for example","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TestMaterials\nG1=50.e3; G2=80.e3; K1=100.e3; K2=160.e3; η=50e3;\nmaterial = ViscoElastic(LinearElastic(G1,K1), LinearElastic(G2,K2), η)","category":"page"},{"location":"","page":"Home","title":"Home","text":"And then we can run the simulation","category":"page"},{"location":"","page":"Home","title":"Home","text":"ϵ11_history  = collect(range(0, 0.01; length=100))  # Ramp to 1 %\ntime_history = collect(range(0, 0.2; length=100))   # Constant time step\nσ11_history  = simulate_uniaxial(material, ϵ11_history, time_history)","category":"page"},{"location":"","page":"Home","title":"Home","text":"This example used the stress iterations implemented in MaterialModelsBase.jl, see Stress States.","category":"page"},{"location":"","page":"Home","title":"Home","text":"If these do not converge, a NoStressConvergence exception is thrown. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"The package also contains the exception NoLocalConvergence,  which shouldbe thrown from inside implemented material routines to signal  that something didn't converge and that the caller should consider  to e.g. reduce the time step or handel the issue in some other way.","category":"page"},{"location":"#Implementing-material-models","page":"Home","title":"Implementing material models","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To implement a material model, at minimum, it is necessary to  define a material type containing the material parameters, e.g. MyMaterial, and the associated material_response(::MyMaterial, args...) function.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For materials with state variables, the function initial_material_state should be defined. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"If some special pre-allocated variables are required for the material, the function allocate_material_cache should be defined. ","category":"page"},{"location":"#Advanced-use-cases","page":"Home","title":"Advanced use cases","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package also provides an interface for converting AbstractMaterials  to a vector of material parameters and vice versa. However, to implement  this part of the interface is not required, if only the standard usage  is intended for your material model.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Building upon that option, it is also possible to define differentiation of the material response wrt. to the material parameters, which is useful  for parameter identification using gradient based methods. For further details,  see Differentation of a material.","category":"page"},{"location":"#API","page":"Home","title":"API","text":"","category":"section"},{"location":"#material_response","page":"Home","title":"material_response","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The main function is the material_response function that  primarly dispatches on the AbstractMaterial input type.  Two variants can be called, where the latter allows a reduced  stress state, see Stress States for further details. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"material_response(::AbstractMaterial, ::Any, ::Any, ::Any, ::Any)\nmaterial_response(::AbstractStressState, ::AbstractMaterial, ::Vararg{Any})","category":"page"},{"location":"#MaterialModelsBase.material_response-Tuple{AbstractMaterial, Vararg{Any, 4}}","page":"Home","title":"MaterialModelsBase.material_response","text":"material_response(\n    m::AbstractMaterial, \n    strain::Union{SecondOrderTensor,Vec}, \n    old::AbstractMaterialState, \n    Δt::Union{Number,Nothing}=nothing, \n    cache::AbstractMaterialCache=allocate_material_cache(m), \n    extras::AbstractExtraOutput=NoExtraOutput()\n    )\n\nCalculate the stress/traction, stiffness and updated state variables  for the material m, given the strain input strain.\n\nMandatory arguments\n\nm: Defines the material and its parameters\nThe second strain input can be, e.g.\nϵ::SymmetricTensor{2}: The total small strain tensor at the end of the increment\nF::Tensor{2}: The total deformation gradient at the end of the increment\nu::Vec: The deformation at the end of the increment (for cohesive elements)\nold::AbstractMaterialState: The material state variables at the end of the last   converged increment. The material initial material state can be obtained by   the initial_material_state function.\n\nOptional positional arguments\n\nWhen calling the function, the following arguments are optional.  When implementing a material, it is not necessary to implement these  defaults, but the method signature should allow all arguments to be  compatible with libraries relying on the interface.  Typically, this can done by using args..., e.g.,  material_response(m::MyMat, ϵ, state, args...)\n\nΔt: The time step in the current increment.  Defaults: nothing.\ncache::AbstractMaterialCache: Cache variables that can be used to avoid allocations during each call to the material_response function.  Default: allocate_material_cache(m)\nextras: Updated with requested extra output.  Default: NoExtraOutput (Empty struct)\n\nOutputs\n\nstress, is the stress measure that is energy conjugated to the strain (2nd) input.\nstiffness, is the derivative of the stress output wrt. the strain input. \nnew_state, are the updated state variables\n\nnote: Note\nThe state given as input should not be mutated. That is, someone calling  material_response multiple times with the same input variables should get  the same output each time.\n\nCommon strain and stress pairs are\n\nIf the second input is the small strain tensor, boldsymbolepsilon (ϵ::SymmetricTensor{2}), the outputs are\nσ::SymmetricTensor{2}: Cauchy stress tensor, boldsymbolsigma\ndσdϵ::SymmetricTensor{4}: Algorithmic tangent stiffness tensor, mathrmdboldsymbolsigmamathrmdboldsymbolepsilon\nIf the second input is the deformation gradient boldsymbolF (F::Tensor{2})`, the outputs are\nP::Tensor{2}: First Piola-Kirchhoff stress, boldsymbolP\ndPdF::Tensor{4}: Algorithmic tangent stiffness tensor, mathrmdboldsymbolPmathrmdboldsymbolF\nIf the second input is deformation vector, boldsymbolu (u::Vec), the outputs are\nt::Vec: Traction vector, boldsymbolt\ndtdu::SecondOrderTensor: Algorithmic tangent stiffness tensor, mathrmdboldsymboltmathrmdboldsymbolu\n\n\n\n\n\n","category":"method"},{"location":"#MaterialModelsBase.material_response-Tuple{AbstractStressState, AbstractMaterial, Vararg{Any}}","page":"Home","title":"MaterialModelsBase.material_response","text":"material_response(stress_state::AbstractStressState, m::AbstractMaterial, args...)\n\nTo be able to use material models implemented for 3d stress and strain states in lower-dimensional  simulations, such as 2d plane stress, MaterialModelsBase.jl provides a set of stress states.  For some states, such as plane stress, iterations will be performed to find the correct state. For other states, such as plane strain, the input is only padded with zeros and the out-of-plane  components are removed from the output. \n\nFor someone implementing a material model, it is also possible to use dispatch on both the  stress state and the material to provide an efficient implementation of a reduced stress state. Note that the interface expects the full strain tensor to be given as a fourth output in this case, but it is optional to implement this but such a deviation should be documented as it could cause  problems for users of the material implementation. \n\nThe arguments are the same as for material_response(::AbstractMaterial). However, both a full and reduced strain input is accepted. For a full strain input,  the out-of-plane components are used as an initial guess. For all cases,  the full strain tensor giving the desired reduced response is given as a 4th output.\n\nSee also ReducedStressState.\n\n\n\n\n\n","category":"method"},{"location":"#State-variables","page":"Home","title":"State variables","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To support state variables define","category":"page"},{"location":"","page":"Home","title":"Home","text":"initial_material_state","category":"page"},{"location":"#MaterialModelsBase.initial_material_state","page":"Home","title":"MaterialModelsBase.initial_material_state","text":"initial_material_state(m::AbstractMaterial)\n\nReturn the (default) initial state::AbstractMaterialState  of the material m. \n\nDefaults to the empty NoMaterialState()\n\n\n\n\n\n","category":"function"},{"location":"#Cache","page":"Home","title":"Cache","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"It is possible to pre-allocate a cache to avoid allocations during the material model calculation, to do this define","category":"page"},{"location":"","page":"Home","title":"Home","text":"allocate_material_cache","category":"page"},{"location":"#MaterialModelsBase.allocate_material_cache","page":"Home","title":"MaterialModelsBase.allocate_material_cache","text":"allocate_material_cache(m::AbstractMaterial)\n\nReturn a cache::AbstractMaterialCache that can be used when  calling the material to reduce allocations\n\nDefaults to the empty NoMaterialCache()\n\n\n\n\n\n","category":"function"},{"location":"#Extra-outputs","page":"Home","title":"Extra outputs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In some cases, it is necessary to define additional outputs to provide slight  variations in the material calculation. In this case, a mutable input can be  given as ","category":"page"},{"location":"","page":"Home","title":"Home","text":"AbstractExtraOutput","category":"page"},{"location":"#MaterialModelsBase.AbstractExtraOutput","page":"Home","title":"MaterialModelsBase.AbstractExtraOutput","text":"AbstractExtraOutput()\n\nBy allocating an AbstractExtraOutput type, this type can be mutated to extract additional information from the internal calculations  in material_response only in cases when this is desired.  E.g., when calculating derivatives or for multiphysics simulations. The concrete NoExtraOutput<:AbstractExtraOutput exists for the case when no additional output should be calculated. \n\n\n\n\n\n","category":"type"},{"location":"#Exceptions","page":"Home","title":"Exceptions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Finally, the following exceptions are included","category":"page"},{"location":"","page":"Home","title":"Home","text":"MaterialConvergenceError\nNoLocalConvergence\nNoStressConvergence","category":"page"},{"location":"#MaterialModelsBase.MaterialConvergenceError","page":"Home","title":"MaterialModelsBase.MaterialConvergenceError","text":"MaterialConvergenceError\n\nAbstract type that can be used to catch errors related to the material not converging. \n\n\n\n\n\n","category":"type"},{"location":"#MaterialModelsBase.NoLocalConvergence","page":"Home","title":"MaterialModelsBase.NoLocalConvergence","text":"NoLocalConvergence(msg::String)\nNoLocalConvergence(args...)\n\nThrow if the material_response routine doesn't converge internally. Other arguments than a single ::String, are converted to String with string\n\n\n\n\n\n","category":"type"},{"location":"#MaterialModelsBase.NoStressConvergence","page":"Home","title":"MaterialModelsBase.NoStressConvergence","text":"NoStressConvergence(msg::String)\nNoStressConvergence(args...)\n\nThis is thrown if the stress iterations don't converge, see Stress states Other arguments than a single ::String, are converted to String with string\n\n\n\n\n\n","category":"type"},{"location":"stressiterations/#Stress-states","page":"Stress states","title":"Stress states","text":"","category":"section"},{"location":"stressiterations/","page":"Stress states","title":"Stress states","text":"In many cases, the full 3d stress and strain states are not desired.  In some cases, a specialized reduced material model can be written for these cases, but for advanced models this can be tedious. A stress state iteration procedure is therefore included in this package, since this does not interfere with the internal implementation of the material model. ","category":"page"},{"location":"stressiterations/","page":"Stress states","title":"Stress states","text":"It also allows custom implementations of specific materials for e.g.  plane stress if desired.","category":"page"},{"location":"stressiterations/","page":"Stress states","title":"Stress states","text":"The specific stress state is invoked by defining a stress state of type AbstractStressState, of which the following specific stress states are implemented:","category":"page"},{"location":"stressiterations/","page":"Stress states","title":"Stress states","text":"FullStressState\nPlaneStrain\nPlaneStress\nUniaxialStrain\nUniaxialStress\nUniaxialNormalStress\nGeneralStressState","category":"page"},{"location":"stressiterations/#MaterialModelsBase.FullStressState","page":"Stress states","title":"MaterialModelsBase.FullStressState","text":"FullStressState()\n\nReturn the full stress state, without any constraints.  Equivalent to not giving any stress state to the  material_response function, except that when given,  the full strain (given as input) is also an output which  can be useful if required for consistency with the other  stress states. \n\n\n\n\n\n","category":"type"},{"location":"stressiterations/#MaterialModelsBase.PlaneStrain","page":"Stress states","title":"MaterialModelsBase.PlaneStrain","text":"PlaneStrain()\n\nPlane strain such that if only 2d-components (11, 12, 21, and 22) are given, the remaining strain components are zero. The output is the reduced set,  with the mentioned components. It is possible to give non-zero values for the other strain components, and these will be used for the material evaluation. \n\n\n\n\n\n","category":"type"},{"location":"stressiterations/#MaterialModelsBase.PlaneStress","page":"Stress states","title":"MaterialModelsBase.PlaneStress","text":"PlaneStress(; kwargs...)\n\nPlane stress such that  sigma_33=sigma_23=sigma_13=sigma_32=sigma_31=0 The strain input should be at least 2d (components 11, 12, 21, and 22). A 3d input is also accepted and used as an initial guess for the unknown  out-of-plane strain components. \n\nThe optional keyword arguments are forwarded to IterationSettings.\n\n\n\n\n\n","category":"type"},{"location":"stressiterations/#MaterialModelsBase.UniaxialStrain","page":"Stress states","title":"MaterialModelsBase.UniaxialStrain","text":"UniaxialStrain()\n\nUniaxial strain such that if only the 11-strain component is given, the remaining strain components are zero. The output is the reduced set, i.e.  only the 11-stress-component. It is possible to give non-zero values for the other strain components, and these will be used for the material evaluation. \n\n\n\n\n\n","category":"type"},{"location":"stressiterations/#MaterialModelsBase.UniaxialStress","page":"Stress states","title":"MaterialModelsBase.UniaxialStress","text":"UniaxialStress(; kwargs...)\n\nUniaxial stress such that  sigma_ij=0 forall (ij)neq (11) The strain input can be 1d (SecondOrderTensor{1}). A 3d input is also accepted and used as an initial  guess for the unknown strain components. \n\nThe optional keyword arguments are forwarded to IterationSettings.\n\n\n\n\n\n","category":"type"},{"location":"stressiterations/#MaterialModelsBase.UniaxialNormalStress","page":"Stress states","title":"MaterialModelsBase.UniaxialNormalStress","text":"UniaxialNormalStress(; kwargs...)\n\nThis is a variation of the uniaxial stress state, such that only sigma_22=sigma_33=0 The strain input must be 3d, and the components  epsilon_22 and epsilon_33 are used as initial guesses.  This case is useful when simulating strain-controlled axial-shear experiments. Note that the stress and stiffness outputs are the 3d tensors, and that the  stiffness is not modified to account for the stress constraints.\n\nThe optional keyword arguments are forwarded to IterationSettings.\n\n\n\n\n\n","category":"type"},{"location":"stressiterations/#MaterialModelsBase.GeneralStressState","page":"Stress states","title":"MaterialModelsBase.GeneralStressState","text":"GeneralStressState(σ_ctrl::AbstractTensor{2,3,Bool}, σ::AbstractTensor{2,3,Bool}; kwargs...)\n\nConstruct a general stress state controlled by σ_ctrl whose component is true if that  component is stress-controlled and false if it is strain-controlled. If stress-controlled, σ gives the value to which it is controlled. The current stress, for stress-controlled components can be updated by calling update_stress_state!(s::GeneralStressState, σ). Components in  σ that are not stress-controlled are ignored. \n\nNote that the stress and stiffness outputs are the 3d tensors, and that the  stiffness is not modified to account for the stress constraints.\n\nThe optional keyword arguments are forwarded to IterationSettings.\n\n\n\n\n\n","category":"type"},{"location":"stressiterations/","page":"Stress states","title":"Stress states","text":"To adjust the iteration parameters (number of iterations and tolerance) for stress states requiring iterations, such stress states contain an IterationSettings object,","category":"page"},{"location":"stressiterations/","page":"Stress states","title":"Stress states","text":"MaterialModelsBase.IterationSettings","category":"page"},{"location":"stressiterations/#MaterialModelsBase.IterationSettings","page":"Stress states","title":"MaterialModelsBase.IterationSettings","text":"IterationSettings(;tolerance=1e-8, max_iter=10)\n\nSettings for stress iterations. Constructors for iterative stress states forwards given keyword arguments to this constructor and saves the result.\n\n\n\n\n\n","category":"type"},{"location":"stressiterations/#Reduced-stress-state-wrapper","page":"Stress states","title":"Reduced stress state wrapper","text":"","category":"section"},{"location":"stressiterations/","page":"Stress states","title":"Stress states","text":"When used in finite element codes, it is often convenient to collect both the stress  state and the material type into a single type that is passed to the element routine. The wrapper ReducedStressState is provided for that purpose.","category":"page"},{"location":"stressiterations/","page":"Stress states","title":"Stress states","text":"ReducedStressState","category":"page"},{"location":"stressiterations/#MaterialModelsBase.ReducedStressState","page":"Stress states","title":"MaterialModelsBase.ReducedStressState","text":"ReducedStressState(s::AbstractStressState, m::AbstractMaterial)\n\nCreates a subtype of AbstractMaterial that wraps a stress state and a material, such that  calls to material_response(w::ReducedStressState, args...) gives the same result as  material_response(s, m, args...).  Calls to initial_material_state, allocate_material_cache,  get_num_tensorcomponents, get_num_statevars, get_num_params,  get_parameter_type, material2vector!, material2vector,  and allocate_differentiation_output are forwarded with m as the argument.  vector2material returns ReducedStressState and is supported as well.\n\n\n\n\n\n","category":"type"}]
}
