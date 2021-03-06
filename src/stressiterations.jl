# Based on https://github.com/kimauth/MaterialModels.jl

abstract type AbstractStressState end

# Cases without stress iterations
""" 
    FullStressState()

Return the full stress state, without any constraints. 
Equivalent to not giving any stress state to the 
`material_response` function, except that when given, 
the full strain (given as input) is also an output which 
can be useful if required for consistency with the other 
stress states. 
"""
struct FullStressState <: AbstractStressState end

""" 
    PlaneStrain()

Plane strain such that if only 2d-components (11, 12, 21, and 22) are given,
the remaining strain components are zero. The output is the reduced set, 
with the mentioned components. It is possible to give non-zero values for the
other strain components, and these will be used for the material evaluation. 
"""
struct PlaneStrain <: AbstractStressState end

""" 
    UniaxialStrain()

Uniaxial strain such that if only the 11-strain component is given,
the remaining strain components are zero. The output is the reduced set, i.e. 
only the 11-stress-component. It is possible to give non-zero values for the
other strain components, and these will be used for the material evaluation. 
"""
struct UniaxialStrain <: AbstractStressState end

""" 
    UniaxialStress()

Uniaxial stress such that 
``\\sigma_{ij}=0 \\forall (i,j)\\neq (1,1)``
The strain input can be 1d (`SecondOrderTensor{1}`).
A 3d input is also accepted and used as an initial 
guess for the unknown strain components. 
"""
struct UniaxialStress <: AbstractStressState end

""" 
    PlaneStress()

Plane stress such that 
``\\sigma_{33}=\\sigma_{23}=\\sigma_{13}=\\sigma_{32}=\\sigma_{31}=0``
The strain input should be at least 2d (components 11, 12, 21, and 22).
A 3d input is also accepted and used as an initial guess for the unknown 
out-of-plane strain components. 
"""
struct PlaneStress <: AbstractStressState end

""" 
    UniaxialNormalStress()

This is a variation of the uniaxial stress state, such that only
``\\sigma_{22}=\\sigma_{33}=0``
The strain input must be 3d, and the components 
``\\epsilon_{22}`` and ``\\epsilon_{33}`` are used as initial guesses. 
This case is useful when simulating strain-controlled axial-shear experiments
"""
struct UniaxialNormalStress <: AbstractStressState end

const NoIterationState = Union{FullStressState,PlaneStrain,UniaxialStrain}
const IterationState = Union{UniaxialStress, PlaneStress, UniaxialNormalStress}
const State3D = Union{FullStressState, UniaxialNormalStress}
const State2D = Union{PlaneStress, PlaneStrain}
const State1D = Union{UniaxialStrain, UniaxialStress}

# Translate from reduced input to full tensors
# Note: Also used to translate from unknowns to full tensors
get_full_tensor(::AbstractStressState, a::Tensors.AllTensors{3}) = a
#get_full_tensor(::AbstractStressState, v::Vec{dim,T}) where {dim,T} = Vec{3,T}(i->i>dim ? zero(T) : v[i])
function get_full_tensor(::AbstractStressState, a::Tensor{2,dim,T}) where {dim,T} 
    return Tensor{2,3}((i,j)-> (i<=dim && j<=dim) ? a[i,j] : zero(T))
end
function get_full_tensor(::AbstractStressState, a::SymmetricTensor{2,dim,T}) where {dim,T} 
    return SymmetricTensor{2,3}((i,j)-> (i<=dim && j<=dim) ? a[i,j] : zero(T))
end

# Translate from full tensors to reduced output
reduce_tensordim(::State3D, a::AbstractTensor) = a
reduce_tensordim(::State2D, a::AbstractTensor) = reduce_tensordim(Val{2}(), a)
reduce_tensordim(::State1D, a::AbstractTensor) = reduce_tensordim(Val{1}(), a)
#reduce_tensordim(::Val{dim}, v::Vec{3}) = Vec{dim}(i->v[i])
reduce_tensordim(::Val{dim}, a::Tensor{2}) where dim = Tensor{2,dim}((i,j)->a[i,j])
reduce_tensordim(::Val{dim}, A::Tensor{4}) where dim = Tensor{4,dim}((i,j,k,l)->A[i,j,k,l])
reduce_tensordim(::Val{dim}, a::SymmetricTensor{2}) where dim = SymmetricTensor{2,dim}((i,j)->a[i,j])
reduce_tensordim(::Val{dim}, A::SymmetricTensor{4}) where dim = SymmetricTensor{4,dim}((i,j,k,l)->A[i,j,k,l])


function material_response(
    stress_state::NoIterationState,
    m::AbstractMaterial,
    ??::AbstractTensor,
    args...;
    options::Dict=Dict{Symbol,Any}(),
    )

    ??_full = get_full_tensor(stress_state, ??)
    ??, d??d??, new_state = material_response(m, ??_full, args...; options=options)
    return reduce_tensordim(stress_state, ??), reduce_tensordim(stress_state, d??d??), new_state, ??_full
end

function material_response(
    stress_state::IterationState,
    m::AbstractMaterial,
    ??::AbstractTensor,
    args...;
    options::Dict=Dict{Symbol,Any}(),
    )

    # Newton options, typecast ensures type stability
    tol = Float64(get(options, :stress_state_tol, 1.e-8))
    maxiter = Int(get(options, :stress_state_maxiter, 10))

    ??_full = get_full_tensor(stress_state, ??)

    for _ in 1:maxiter
        ??_full, d??d??_full, new_state = material_response(m, ??_full, args...; options=options)
        ??_mandel = get_unknowns(stress_state, ??_full)
        if norm(??_mandel) < tol
            d??d?? = reduce_stiffness(stress_state, d??d??_full)
            return reduce_tensordim(stress_state, ??_full), d??d??, new_state, ??_full
        end

        d??d??_mandel = get_unknowns(stress_state, d??d??_full)
        ??_full -= get_full_tensor(stress_state, ??, d??d??_mandel\??_mandel)
    end
    throw(NoStressConvergence("Stress iterations with the NewtonSolver did not converge"))
end

reduce_stiffness(::State3D, d??d??_full::AbstractTensor{4,3}) = d??d??_full

function reduce_stiffness(stress_state, d??d??_full::AbstractTensor{4,3})
    ????????????????, ????????????????, ????????????????, ???????????????? = extract_substiffnesses(stress_state, d??d??_full)
    d?????d????? = ???????????????? - ???????????????? * (???????????????? \ ????????????????)
    return convert_stiffness(d?????d?????, stress_state, d??d??_full)
end

convert_stiffness(d?????d?????::SMatrix{1,1}, ::State1D, ::SymmetricTensor) = frommandel(SymmetricTensor{4,1}, d?????d?????)
convert_stiffness(d?????d?????::SMatrix{1,1}, ::State1D, ::Tensor) = frommandel(Tensor{4,1}, d?????d?????)
convert_stiffness(d?????d?????::SMatrix{3,3}, ::State2D, ::SymmetricTensor) = frommandel(SymmetricTensor{4,2}, d?????d?????)
convert_stiffness(d?????d?????::SMatrix{4,4}, ::State2D, ::Tensor) = frommandel(Tensor{4,2}, d?????d?????)


# Conversions to mandel SArray for solving equation system
# Internal numbering in Tensors.jl
# Tensor: 11,21,31,12,22,32,13,23,33
# SymmetricTensor: 11,21,31,22,23,33

# UniaxialStress: only ??11 != 0
# -SymmetricTensor
#         i:  1,  2,  3,  4,  5
# v contains 22, 33, 32, 31, 21
function get_full_tensor(::UniaxialStress, ::SymmetricTensor, v::SVector{5})
    s = 1/???2           # 11,   21,     31,     22,   32,     33
    SymmetricTensor{2,3}((0, v[5]*s, v[4]*s, v[1], v[3]*s, v[2]))
end
function get_unknowns(::UniaxialStress, a::SymmetricTensor{2,3})
    SVector{5}(a[2,2], a[3,3], a[3,2]*???2, a[3,1]*???2, a[2,1]*???2)
end
function get_unknowns(::UniaxialStress, A::SymmetricTensor{4, 3}) 
    @SMatrix [     A[2,2,2,2]    A[2,2,3,3] ???2*A[2,2,3,2] ???2*A[2,2,3,1] ???2*A[2,2,2,1];
                   A[3,3,2,2]    A[3,3,3,3] ???2*A[3,3,3,2] ???2*A[3,3,3,1] ???2*A[3,3,2,1];
                ???2*A[3,2,2,2] ???2*A[3,2,3,3]  2*A[3,2,3,2]  2*A[3,2,3,1]  2*A[3,2,2,1];
                ???2*A[3,1,2,2] ???2*A[3,1,3,3]  2*A[3,1,3,2]  2*A[3,1,3,1]  2*A[3,1,2,1];
                ???2*A[2,1,2,2] ???2*A[2,1,3,3]  2*A[2,1,3,2]  2*A[2,1,3,1]  2*A[2,1,2,1]]
end

# -Tensor
#         i:  1,  2,  3,  4,  5,  6,  7,  8
# v contains 22, 33, 23, 13, 12, 32, 31, 21
function get_full_tensor(::UniaxialStress, ??::Tensor, v::SVector{8,T}) where T
                     # 11,   21,   31,   12,   22,   32,   13,   23,   33
    return Tensor{2,3}((0, v[8], v[7], v[5], v[1], v[6], v[4], v[3], v[2]))
end

function get_unknowns(::UniaxialStress, a::Tensor{2,3})
    SVector{8}(a[2,2], a[3,3], a[2,3], a[1,3], a[1,2], a[3,2], a[3,1], a[2,1])
end

function get_unknowns(::UniaxialStress, A::Tensor{4, 3}) 
    @SMatrix [  A[2,2,2,2] A[2,2,3,3] A[2,2,2,3] A[2,2,1,3] A[2,2,1,2] A[2,2,3,2] A[2,2,3,1] A[2,2,2,1];
                A[3,3,2,2] A[3,3,3,3] A[3,3,2,3] A[3,3,1,3] A[3,3,1,2] A[3,3,3,2] A[3,3,3,1] A[3,3,2,1];
                A[2,3,2,2] A[2,3,3,3] A[2,3,2,3] A[2,3,1,3] A[2,3,1,2] A[2,3,3,2] A[2,3,3,1] A[2,3,2,1];
                A[1,3,2,2] A[1,3,3,3] A[1,3,2,3] A[1,3,1,3] A[1,3,1,2] A[1,3,3,2] A[1,3,3,1] A[1,3,2,1];
                A[1,2,2,2] A[1,2,3,3] A[1,2,2,3] A[1,2,1,3] A[1,2,1,2] A[1,2,3,2] A[1,2,3,1] A[1,2,2,1];
                A[3,2,2,2] A[3,2,3,3] A[3,2,2,3] A[3,2,1,3] A[3,2,1,2] A[3,2,3,2] A[3,2,3,1] A[3,2,2,1];
                A[3,1,2,2] A[3,1,3,3] A[3,1,2,3] A[3,1,1,3] A[3,1,1,2] A[3,1,3,2] A[3,1,3,1] A[3,1,2,1];
                A[2,1,2,2] A[2,1,3,3] A[2,1,2,3] A[2,1,1,3] A[2,1,1,2] A[2,1,3,2] A[2,1,3,1] A[2,1,2,1]]
end

# PlaneStress: ??33=??23=??13=0=??31=??32
# -SymmetricTensor
#         i:  1,  2,  3
# v contains 33, 23, 13
function get_full_tensor(::PlaneStress, ::SymmetricTensor, v::SVector{3})
    s = 1/???2           # 11,21,   31,  22,   32,     33
    SymmetricTensor{2,3}((0, 0, v[3]*s, 0, v[2]*s, v[1]))
end
function get_unknowns(::PlaneStress, a::SymmetricTensor{2,3})
    SVector{3}(a[3,3], a[2,3]*???2, a[3,1]*???2)
end
function get_unknowns(::PlaneStress, A::SymmetricTensor{4, 3}) 
    @SMatrix [     A[3,3,3,3] ???2*A[3,3,3,2] ???2*A[3,3,3,1];
                ???2*A[3,2,3,3]  2*A[3,2,3,2]  2*A[3,2,3,1];
                ???2*A[3,1,3,3]  2*A[3,1,3,2]  2*A[3,1,3,1]]
end

# -Tensor
#         i:  1,  2,  3,  4,  5,  6
# v contains 33, 23, 13, 32, 31
function get_full_tensor(::PlaneStress, ??::Tensor, v::SVector{5,T}) where T
                     # 11,21,   31,12,22,   32,   13,   23,   33
    return Tensor{2,3}((0, 0, v[5], 0, 0, v[4], v[3], v[2], v[1]))
end

function get_unknowns(::PlaneStress, a::Tensor{2,3})
    SVector{5}(a[3,3], a[2,3], a[1,3], a[3,2], a[3,1])
end

function get_unknowns(::PlaneStress, A::Tensor{4, 3}) 
    @SMatrix [  A[3,3,3,3] A[3,3,2,3] A[3,3,1,3] A[3,3,3,2] A[3,3,3,1];
                A[2,3,3,3] A[2,3,2,3] A[2,3,1,3] A[2,3,3,2] A[2,3,3,1];
                A[1,3,3,3] A[1,3,2,3] A[1,3,1,3] A[1,3,3,2] A[1,3,3,1];
                A[3,2,3,3] A[3,2,2,3] A[3,2,1,3] A[3,2,3,2] A[3,2,3,1];
                A[3,1,3,3] A[3,1,2,3] A[3,1,1,3] A[3,1,3,2] A[3,1,3,1]]
end

# UniaxialNormalStress: ??22=??33=0
# -SymmetricTensor and Tensor
#         i:  1,  2
# v contains 22, 33
function get_full_tensor(::UniaxialNormalStress, ::SymmetricTensor, v::SVector{2})
                       # 11,21,31,   22,32,   33
    SymmetricTensor{2,3}((0, 0, 0, v[1], 0, v[2]))
end
function get_full_tensor(::UniaxialNormalStress, ??::Tensor, v::SVector{2})
                 # 11,21,31,12,   22,32,13,23,  33
return Tensor{2,3}((0, 0, 0, 0, v[1], 0, 0, 0, v[2]))
end
function get_unknowns(::UniaxialNormalStress, a::AbstractTensor{2,3})
    SVector{2}(a[2,2], a[3,3])
end
function get_unknowns(::UniaxialNormalStress, A::AbstractTensor{4, 3}) 
    @SMatrix [A[2,2,2,2] A[2,2,3,3];
              A[3,3,2,2] A[3,3,3,3]]
end

# Extract each part of the stiffness tensor as SMatrix
# UniaxialStress
function extract_substiffnesses(stress_state::UniaxialStress, D::SymmetricTensor{4,3})
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ???????????????? = get_unknowns(stress_state, D)
    ???????????????? = SMatrix{5,1}(D[2,2,1,1], D[3,3,1,1], ???2*D[2,3,1,1], ???2*D[1,3,1,1], ???2*D[1,2,1,1])
    ???????????????? = SMatrix{1,5}(D[1,1,2,2], D[1,1,3,3], ???2*D[1,1,2,3], ???2*D[1,1,1,3], ???2*D[1,1,1,2])
    ???????????????? = SMatrix{1,1}(D[1,1,1,1])
    return ????????????????, ????????????????, ????????????????, ????????????????
end
function extract_substiffnesses(stress_state::UniaxialStress, D::Tensor{4,3})
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ???????????????? = get_unknowns(stress_state, D)
    ???????????????? = SMatrix{8,1}(D[2,2,1,1], D[3,3,1,1], D[2,3,1,1], D[1,3,1,1], D[1,2,1,1], D[3,2,1,1], D[3,1,1,1], D[2,1,1,1])
    ???????????????? = SMatrix{1,8}(D[1,1,2,2], D[1,1,3,3], D[1,1,2,3], D[1,1,1,3], D[1,1,1,2], D[1,1,3,2], D[1,1,3,1], D[1,1,2,1])
    ???????????????? = SMatrix{1,1}(D[1,1,1,1])
    return ????????????????, ????????????????, ????????????????, ????????????????
end

# PlaneStress 
function extract_substiffnesses(stress_state::PlaneStress, D::SymmetricTensor{4,3})
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ???????????????? = get_unknowns(stress_state, D)
    ???????????????? = @SMatrix [   D[3,3,1,1]    D[3,3,2,2] ???2*D[3,3,2,1];
                       ???2*D[3,2,1,1] ???2*D[3,2,2,2]  2*D[3,2,2,1];
                       ???2*D[3,1,1,1] ???2*D[3,1,2,2]  2*D[3,1,2,1]]
    
    ???????????????? = @SMatrix [   D[1,1,3,3] ???2*D[1,1,3,2] ???2*D[1,1,3,1];
                          D[2,2,3,3] ???2*D[2,2,3,2] ???2*D[2,2,3,1];
                       ???2*D[2,1,3,3]  2*D[2,1,3,2]  2*D[2,1,3,1]]

    ???????????????? = @SMatrix [   D[1,1,1,1]    D[1,1,2,2] ???2*D[1,1,2,1];
                          D[2,2,1,1]    D[2,2,2,2] ???2*D[2,2,2,1];
                       ???2*D[2,1,1,1] ???2*D[2,1,2,2]  2*D[2,1,2,1]]

    return ????????????????, ????????????????, ????????????????, ????????????????
end

function extract_substiffnesses(stress_state::PlaneStress, D::Tensor{4,3})
    # f=strain free, stress constrained
    # c=strain constrained, stress free
    ???????????????? = get_unknowns(stress_state, D)
    ???????????????? = @SMatrix [D[3,3,1,1] D[3,3,2,2] D[3,3,1,2] D[3,3,2,1];
                       D[2,3,1,1] D[2,3,2,2] D[2,3,1,2] D[2,3,2,1];
                       D[1,3,1,1] D[1,3,2,2] D[1,3,1,2] D[1,3,2,1];
                       D[3,2,1,1] D[3,2,2,2] D[3,2,1,2] D[3,2,2,1];
                       D[3,1,1,1] D[3,1,2,2] D[3,1,1,2] D[3,1,2,1]]
    
    ???????????????? = @SMatrix [D[1,1,3,3] D[1,1,2,3] D[1,1,1,3] D[1,1,3,2] D[1,1,3,1];
                       D[2,2,3,3] D[2,2,2,3] D[2,2,1,3] D[2,2,3,2] D[2,2,3,1];
                       D[1,2,3,3] D[1,2,2,3] D[1,2,1,3] D[1,2,3,2] D[1,2,3,1];
                       D[2,1,3,3] D[2,1,2,3] D[2,1,1,3] D[2,1,3,2] D[2,1,3,1]]

    ???????????????? = @SMatrix [D[1,1,1,1] D[1,1,2,2] D[1,1,1,2] D[1,1,2,1];
                       D[2,2,1,1] D[2,2,2,2] D[2,2,1,2] D[2,2,2,1];
                       D[1,2,1,1] D[1,2,2,2] D[1,2,1,2] D[1,2,2,1];
                       D[2,1,1,1] D[2,1,2,2] D[2,1,1,2] D[2,1,2,1]]
    return ????????????????, ????????????????, ????????????????, ????????????????
end