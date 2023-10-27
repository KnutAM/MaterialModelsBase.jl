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

all_states = (
    FullStressState(), UniaxialStrain(), PlaneStrain(),
    UniaxialStress(), PlaneStress(), UniaxialNormalStress())
iter_states = ( UniaxialStress(), PlaneStress(), UniaxialNormalStress())
iter_mandel = ( ([2,3,4,5,6,7,8,9],[2,3,4,5,6]),
                ([3,4,5,7,8], [3,4,5]),
                ([2,3], [2,3]))


function run_timehistory(s::MMB.AbstractStressState, m::AbstractMaterial, ϵv::Vector{<:AbstractTensor}, t = collect(range(0,1;length=length(ϵ))))
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    σv = eltype(ϵv)[]
    ϵv_full = tensortype(first(ϵv))[]
    local dσdϵ
    for i in 2:length(ϵv)
        σ, dσdϵ, state, ϵ_full = material_response(s, m, ϵv[i], state, t[i]-t[i-1], cache)
        push!(σv, σ)
        push!(ϵv_full, ϵ_full)
    end
    return σv, ϵv_full, dσdϵ
end

@testset "conversions" begin
    # Test that
    # 1) All conversions are invertible (convert back and fourth)
    # 2) Are compatible with the expected mandel dynamic tensors 
    for stress_state in all_states
        for TT in (Tensor, SymmetricTensor)
            ared = rand(TT{2,_getdim(stress_state)})
            afull = MMB.expand_tensordim(stress_state, ared)
            @test _getdim(afull) == 3    # Full dimensional tensor
            @test norm(ared) ≈ norm(afull)    # Only add zeros
            # Test invertability of conversions
            @test ared ≈ MMB.reduce_tensordim(stress_state, afull)
        end
    end

    for (state, inds) in zip(iter_states, iter_mandel)
        for (TT, ii) in zip((Tensor, SymmetricTensor), inds)
            a = rand(TT{2,3})
            A = rand(TT{4,3})
            ax = MMB.get_unknowns(state, a)
            Ax = MMB.get_unknowns(state, A)
            # Corresponding to regular mandel
            @test ax ≈ tomandel(a)[ii] 
            @test Ax ≈ tomandel(A)[ii,ii]
            
            am = tomandel(a)
            am[setdiff(1:length(am), ii)] .= 0
            ac = frommandel(typeof(a), am)
            # Test that other comps become zero, and the rest are converted correctly
            @test ac ≈ MMB.get_full_tensor(state, a, MMB.get_unknowns(state, a)) 
        end
    end
end

@testset "stiffness_calculations" begin
    for (state, inds) in zip(iter_states, iter_mandel)
        for (TT, ii) in zip((Tensor, SymmetricTensor), inds)
            dσdϵ = rand(TT{4,3}) + one(TT{4,3})
            dσᶜdϵᶜ = MMB.reduce_stiffness(state, dσdϵ)
            if _getdim(state) < 3   # Otherwise, conversion is direct    
                D_mandel = tomandel(dσdϵ)
                jj = setdiff(1:size(D_mandel,1), ii)
                @test dσᶜdϵᶜ ≈ frommandel(TT{4,_getdim(state)}, inv(inv(D_mandel)[jj,jj]))
            else
                @test dσdϵ == dσᶜdϵᶜ
            end
        end
    end
end

@testset "elastic_stress_states" begin
    G = 80.e3           # Shear modulus (μ)
    K = 160.e3          # Bulk modulus
    E = 9*K*G/(3*K+G)   # Young's modulus
    ν = E/2G - 1        # Poisson's ratio
    λ = K - 2G/3        # Lame parameter
    m = LinearElastic(G, K)
    old = initial_material_state(m)
    # UniaxialStress 
    Δϵ = 1.e-3
    σ, dσdϵ, state, ϵfull = material_response(UniaxialStress(), m, SymmetricTensor{2,1}((Δϵ,)), old, 0.0)
    @test σ[1,1] ≈ E*Δϵ
    @test dσdϵ[1,1,1,1] ≈ E 
    @test ϵfull[2,2] ≈ ϵfull[3,3]
    @test ϵfull[2,2] ≈ -ν*Δϵ
    gen = MMB.GeneralStressState(UniaxialStress(), SymmetricTensor{2,3})
    ϵfull_ = SymmetricTensor{2,3}((i,j)->i==j==1 ? Δϵ : rand())
    σ_, dσdϵ_ = material_response(gen, m, ϵfull_, old)
    @test σ_[1,1] ≈ σ[1,1]

    # UniaxialStrain
    σ, dσdϵ, state, ϵfull = material_response(UniaxialStrain(), m, SymmetricTensor{2,1}((Δϵ,)), old, 0.0)
    @test σ[1,1] ≈ (2G+λ)*Δϵ
    @test dσdϵ[1,1,1,1] ≈ (2G+λ)
    σ_full, dσdϵ_full, _ = material_response(m, ϵfull, old, 0.0)
    @test σ_full[2,2] ≈ λ*Δϵ

    # PlaneStress
    ϵ = rand(SymmetricTensor{2,2})
    σ, dσdϵ, state, ϵfull = material_response(PlaneStress(), m, ϵ, old, 0.0)
    Dvoigt = (E/(1-ν^2))*[1 ν 0; ν 1 0; 0 0 (1-ν)/2]
    @test tovoigt(dσdϵ) ≈ Dvoigt 
    @test σ ≈ dσdϵ⊡ϵ
    σ_full, dσdϵ_full, _ = material_response(m, ϵfull, old, 0.0)
    @test σ_full[1:2,1:2] ≈ σ
    gen = MMB.GeneralStressState(PlaneStress(), SymmetricTensor{2,3})
    ϵfull_ = SymmetricTensor{2,3}((i,j)->(i<3 && j<3) ? ϵ[i,j] : rand())
    σ_, dσdϵ_ = material_response(gen, m, ϵfull_, old)
    @test σ_[1:2,1:2] ≈ σ

    # PlaneStrain
    ϵ = rand(SymmetricTensor{2,2})
    σ, dσdϵ, state, ϵfull = material_response(PlaneStrain(), m, ϵ, old, 0.0)
    Dvoigt = [2G+λ λ 0; λ 2G+λ 0; 0 0 G]
    @test tovoigt(dσdϵ) ≈ Dvoigt
    @test σ ≈ dσdϵ⊡ϵ
    σ_full, dσdϵ_full, _ = material_response(m, ϵfull, old, 0.0)
    @test σ_full[1:2,1:2] ≈ σ 

    # Stress state wrapper
    σ_w, dσdϵ_w, state_w, ϵfull_w = material_response(ReducedStressState(PlaneStrain(), m), ϵ, old, 0.0)
    @test σ_w == σ
    @test dσdϵ_w == dσdϵ
    @test ϵfull_w == ϵfull
end

@testset "visco_elastic" begin
    # Test a nonlinear, state and rate dependent, material by running stress iterations.
    # Check that state variables and time dependence are handled correctly
    G = 80.e3           # Shear modulus (μ)
    K = 160.e3          # Bulk modulus
    E = 9*K*G/(3*K+G)   # Young's modulus
    material = ViscoElastic(LinearElastic(G, K), LinearElastic(G*0.5, K*0.5), E*0.1)
    
    stress_state = UniaxialStress()
    N = 100
    ϵ_end = 0.1
    ϵv = collect((SymmetricTensor{2,1}((ϵ_end*(i-1)/N,)) for i in 1:N+1))
    tfast, tslow = (1.e-10, 1.e+10)
    σvf, ϵv_fullf, dσdϵf = run_timehistory(stress_state, material, ϵv, collect(range(0.,tfast; length=N+1)))
    σvs, ϵv_fulls, dσdϵs = run_timehistory(stress_state, material, ϵv, collect(range(0.,tslow; length=N+1)))
    @test dσdϵf[1,1,1,1] ≈ 1.5E # Fast limit response
    @test dσdϵs[1,1,1,1] ≈ 1.0E # Slow limit response

    # Test ReducedStressState wrapper 
    w = ReducedStressState(stress_state, material)
    @test initial_material_state(material) == initial_material_state(w)
    @test allocate_material_cache(material) == allocate_material_cache(w)
end

