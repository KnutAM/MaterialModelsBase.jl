_getdim(::MaterialModelsBase.State1D) = 1
_getdim(::MaterialModelsBase.State2D) = 2
_getdim(::MaterialModelsBase.State3D) = 3
_getdim(::AbstractTensor{order,dim}) where {order,dim} = dim

tensortype(::SymmetricTensor) = SymmetricTensor
tensortype(::Tensor) = Tensor

# Create GeneralStressState from regular states to check compatibility
function MMB.GeneralStressState(::UniaxialStress, TT::Type{<:SecondOrderTensor{3}})
    σ = zero(TT)
    σ_ctrl = tensortype(σ){2,3,Bool}((i,j)->!(i==j==1))
    return MMB.GeneralStressState(σ_ctrl, σ)
end
function MMB.GeneralStressState(::PlaneStress, TT::Type{<:SecondOrderTensor{3}})
    σ = zero(TT)
    σ_ctrl = tensortype(σ){2,3,Bool}((i,j)->(i==3 || j==3))
    return MMB.GeneralStressState(σ_ctrl, σ)
end
function MMB.GeneralStressState(::UniaxialNormalStress, TT::Type{<:SecondOrderTensor{3}})
    σ = zero(TT)
    σ_ctrl = tensortype(σ){2,3,Bool}((i,j)->(i==j∈(2,3)))
    return MMB.GeneralStressState(σ_ctrl, σ)
end

function run_timehistory!(σv::Vector, ϵv_full::Vector, state, cache, s, m, ϵv, t)
    local dσdϵ
    for i in 1:(length(ϵv)-1)
        σ, dσdϵ, state, ϵ_full = material_response(s, m, ϵv[i+1], state, t[i+1]-t[i], cache)
        σv[i] = σ
        ϵv_full[i] = ϵ_full
    end
    return dσdϵ
end

function setup_run_timehistory(m::AbstractMaterial, ϵv::Vector{<:AbstractTensor})
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    σv = zeros(eltype(ϵv), length(ϵv)-1)
    ϵv_full = zeros(tensortype(first(ϵv)){2,3}, length(ϵv)-1)
    return state, cache, σv, ϵv_full
end

function run_timehistory(s, m::AbstractMaterial, ϵv, t = collect(range(0, 1, length(ϵv))))
    state, cache, σv, ϵv_full = setup_run_timehistory(m, ϵv)
    dσdϵ = run_timehistory!(σv, ϵv_full, state, cache, s, m, ϵv, t)
    return σv, ϵv_full, dσdϵ
end
