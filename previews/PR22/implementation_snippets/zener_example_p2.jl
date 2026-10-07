struct ZenerState{T} <: AbstractMaterialState
    ϵv::SymmetricTensor{2,3,T,6}
end
MaterialModelsBase.initial_material_state(::Zener) = ZenerState(zero(SymmetricTensor{2,3}))