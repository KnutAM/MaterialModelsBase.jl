all_states = (
    FullStressState(), UniaxialStrain(), PlaneStrain(),
    UniaxialStress(), PlaneStress(), UniaxialNormalStress())
gss_ctrl = SymmetricTensor{2,3,Bool}((true, true, falses(4)...))
iter_states = ( UniaxialStress(), PlaneStress(), UniaxialNormalStress(), 
                (SymmetricTensor = GeneralStressState(gss_ctrl, 0.0 * gss_ctrl),
                 Tensor = GeneralStressState(Tensor{2,3}(gss_ctrl), 0.0 * Tensor{2,3}(gss_ctrl)))
                 )
iter_mandel = ( ([2,3,4,5,6,7,8,9],[2,3,4,5,6]),
                ([3,4,5,7,8], [3,4,5]),
                ([2,3], [2,3]),
                ([1,6,9], [1,6]))

@testset "conversions" begin
    # Test that
    # 1) All conversions are invertible (convert back and fourth)
    # 2) Are compatible with the expected mandel dynamic tensors 
    for _stress_state in all_states
        for TT in (Tensor, SymmetricTensor)
            stress_state = if isa(_stress_state, NamedTuple)
                _stress_state[nameof(TT)]
            else
                _stress_state
            end
            ared = rand(TT{2,_getdim(stress_state)})
            afull = MMB.expand_tensordim(stress_state, ared)
            @test _getdim(afull) == 3    # Full dimensional tensor
            if TT == SymmetricTensor
                @test norm(ared) ≈ norm(afull)    # Only add zeros
            else # Add ones to diagonal
                @test sqrt(norm(ared)^2 + (3 - _getdim(ared))) ≈ norm(afull)
            end
            # Test invertability of conversions
            @test ared ≈ MMB.reduce_tensordim(stress_state, afull)
        end
    end

    for (_stress_state, inds) in zip(iter_states, iter_mandel)
        for (TT, ii) in zip((Tensor, SymmetricTensor), inds)
            state = if isa(_stress_state, NamedTuple)
                _stress_state[nameof(TT)]
            else
                _stress_state
            end
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
    for (_stress_state, inds) in zip(iter_states, iter_mandel)
        for (TT, ii) in zip((Tensor, SymmetricTensor), inds)
            state = if isa(_stress_state, NamedTuple)
                _stress_state[nameof(TT)]
            else
                _stress_state
            end
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
    gen = GeneralStressState(UniaxialStress(), SymmetricTensor{2,3})
    ϵfull_ = SymmetricTensor{2,3}((i,j)->i==j==1 ? Δϵ : rand())
    σ_, dσdϵ_ = material_response(gen, m, ϵfull_, old)
    @test σ_[1,1] ≈ σ[1,1]

    # UniaxialStrain
    σ, dσdϵ, state, ϵfull = material_response(UniaxialStrain(), m, SymmetricTensor{2,1}((Δϵ,)), old, 0.0)
    @test σ[1,1] ≈ (2G+λ)*Δϵ
    @test dσdϵ[1,1,1,1] ≈ (2G+λ)
    σ_full, dσdϵ_full, _ = material_response(m, ϵfull, old, 0.0)
    @test σ_full[2,2] ≈ λ*Δϵ

    # UniaxialNormalStress
    ϵ11_value = rand()
    ϵ11 = SymmetricTensor{2,1}((ϵ11_value,))
    ϵ = SymmetricTensor{2,3}((i,j)-> i==j==1 ? ϵ11_value : 0.0)
    σ1, dσdϵ1, _, ϵfull1 = material_response(UniaxialStress(), m, ϵ11, old, 0.0)
    σ2, dσdϵ2, _, ϵfull2 = material_response(UniaxialStrain(), m, ϵ11, old, 0.0)
    σ3, dσdϵ3, _, ϵfull3 = material_response(UniaxialNormalStress(), m, ϵ, old, 0.0)
    @test σ1[1,1] ≈ σ3[1,1]
    @test abs(σ3[2,2]) < 1e-8 
    @test abs(σ3[3,3]) < 1e-8
    @test dσdϵ3[1,1,1,1] ≈ dσdϵ2[1,1,1,1]

    # PlaneStress
    ϵ = rand(SymmetricTensor{2,2})
    σ, dσdϵ, state, ϵfull = material_response(PlaneStress(), m, ϵ, old, 0.0)
    Dvoigt = (E/(1-ν^2))*[1 ν 0; ν 1 0; 0 0 (1-ν)/2]
    @test tovoigt(dσdϵ) ≈ Dvoigt 
    @test σ ≈ dσdϵ⊡ϵ
    σ_full, dσdϵ_full, _ = material_response(m, ϵfull, old, 0.0)
    @test σ_full[1:2,1:2] ≈ σ
    gen = GeneralStressState(PlaneStress(), SymmetricTensor{2,3})
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

    # - Non-zero stress conditions taken from PlaneStrain above
    gen = GeneralStressState(PlaneStress(), SymmetricTensor{2,3})
    update_stress_state!(gen, σ_full)
    σ_gen, dσdϵ_gen, _, ϵfull_gen = material_response(gen, m, ϵ, old, 0.0)
    @test σ_gen[1:2, 1:2] ≈ σ
    @test σ_gen ≈ σ_full
    @test ϵfull_gen ≈ ϵfull
    # The 3d material stiffness is given, so this should match the plane strain output
    @test dσdϵ_gen[1:2, 1:2, 1:2, 1:2] ≈ dσdϵ

    # Stress state wrapper
    ϵ = rand(SymmetricTensor{2,2})
    σ, dσdϵ, state, ϵfull = material_response(PlaneStrain(), m, ϵ, old, 0.0)
    σ_w, dσdϵ_w, state_w, ϵfull_w = material_response(ReducedStressState(PlaneStrain(), m), ϵ, old, 0.0)
    @test σ_w == σ
    @test dσdϵ_w == dσdϵ
    @test ϵfull_w == ϵfull
end

@testset "hyperelastic" begin
    G = 80.e3           # Shear modulus (μ)
    K = 160.e3          # Bulk modulus
    E = 9*K*G/(3*K+G)   # Young's modulus
    ν = E/2G - 1        # Poisson's ratio
    λ = K - 2G/3        # Lame parameter

    Δϵ = 1e-6   # Small strain to compare with linear case
    rtol = 1e-5 # Relative tolerance to compare with linear case
    m = NeoHooke(;G, K)
    old = initial_material_state(m)
    # UniaxialStress
    P, dPdF, state, Ffull = material_response(UniaxialStress(), m, Tensor{2,1}((1 + Δϵ,)), old, 0.0)
    @test isapprox(P[1,1], E*Δϵ; rtol)
    @test isapprox(dPdF[1,1,1,1], E; rtol)
    @test Ffull[2,2] ≈ Ffull[3,3]
    @test Ffull[2,2] ≈ 1-ν*Δϵ
    
    # UniaxialStrain
    P, dPdF, state, Ffull = material_response(UniaxialStrain(), m, Tensor{2,1}((1 + Δϵ,)), old, 0.0)
    @test isapprox(P[1,1], (2G+λ)*Δϵ; rtol)
    @test isapprox(dPdF[1,1,1,1], (2G+λ); rtol)
    P_full, dPdF_full, _ = material_response(m, Ffull, old, 0.0)
    @test isapprox(P_full[2,2], λ*Δϵ; rtol)

    # PlaneStress
    ϵ = Δϵ * rand(SymmetricTensor{2,2})
    F = one(Tensor{2,2}) + ϵ
    P, dPdF, state, Ffull = material_response(PlaneStress(), m, F, old, 0.0)
    Dvoigt = (E/(1-ν^2))*[1 ν 0; ν 1 0; 0 0 (1-ν)/2]
    @test isapprox(tovoigt(symmetric(dPdF)), Dvoigt; rtol)
    @test isapprox(P, fromvoigt(SymmetricTensor{4,2}, Dvoigt)⊡ϵ; rtol)
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

