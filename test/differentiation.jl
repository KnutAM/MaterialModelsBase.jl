@testset "differentiation_utils" begin
    for TT in (Tensor, SymmetricTensor)
        ϵmock = TT == Tensor ? ones(TT{2,3}) : TT{2,3}((i,j) -> i==j ? 1.0 : 1/√2)
        for stress_state in (
                UniaxialStress(), UniaxialNormalStress(), PlaneStress(),
                GeneralStressState(rand(TT{2, 3, Bool}), zero(TT{2, 3})),
                )
            sc = MMB.stress_controlled_indices(stress_state, ϵmock)
            ec = MMB.strain_controlled_indices(stress_state, ϵmock)
            @test length(union(Set(sc), Set(ec))) == Tensors.n_components(TT{2, 3})
            @test tomandel(ϵmock)[sc] ≈ MMB.get_unknowns(stress_state, ϵmock)
        end
        for stress_state in (UniaxialStrain(), PlaneStrain(), FullStressState())
            sc = MMB.stress_controlled_indices(stress_state, ϵmock)
            ec = MMB.strain_controlled_indices(stress_state, ϵmock)
            @test length(sc) == 0
            @test length(ec) == Tensors.n_components(TT{2,3})
        end
    end
end

@testset "differentiation" begin
    @testset "Analytical for LinearElastic" begin
        G = rand()
        K = G + rand()
        m = LinearElastic(G, K)
        ϵ = rand(SymmetricTensor{2,3})
        diff = MaterialDerivatives(m)
        IxI = one(SymmetricTensor{2,3}) ⊗ one(SymmetricTensor{2,3})
        dσdϵ = 2G*(one(SymmetricTensor{4,3}) - IxI/3) + K*IxI
        dσdG = 2*(one(SymmetricTensor{4,3}) - IxI/3) ⊡ ϵ
        dσdK = IxI ⊡ ϵ

        old = NoMaterialState{Float64}(); Δt = nothing
        cache = NoMaterialCache(); extras = allocate_differentiation_output(m)
        differentiate_material!(diff, m, ϵ, old, Δt, cache, extras, dσdϵ)
        @test diff.dσdϵ ≈ tomandel(dσdϵ)
        @test diff.dσdp[:,1] ≈ tomandel(dσdG)
        @test diff.dσdp[:,2] ≈ tomandel(dσdK)
    end

    @testset "Numerical (3D material)" begin
        elastic = LinearElastic(0.52, 0.77)
        viscoelastic = ViscoElastic(elastic, LinearElastic(0.33, 0.54), 0.36)
        for m in (elastic, viscoelastic)    
            @testset "Initial response" begin
                ϵ = rand(SymmetricTensor{2,3}) * 1e-3
                Δt = 1e-2
                test_derivative(m, ϵ, initial_material_state(m), Δt; 
                    numdiffsettings = (fdtype = Val{:central},),
                    comparesettings = ()
                    )
                diff = MaterialDerivatives(m)
                copy!(diff.dsdp, rand(size(diff.dsdp)...))
                test_derivative(m, ϵ, initial_material_state(m), Δt; 
                    numdiffsettings = (fdtype = Val{:central},),
                    comparesettings = (),
                    diff)
            end
            @testset "After shear loading" begin
                ϵ21 = 0.01; num_steps = 10; t_end = 0.01
                stressfun(p) = runstrain(fromvector(p, m), ϵ21, (2, 1), t_end, num_steps)[1]
                dσ21_dp_num = FiniteDiff.finite_difference_jacobian(stressfun, tovector(m), Val{:central}; relstep = 1e-6)
                σv, state, dσ21_dp, diff = runstrain_diff(m, ϵ21, (2, 1), t_end, num_steps)
                @test σv ≈ stressfun(tovector(m))
                @test dσ21_dp ≈ dσ21_dp_num
            end
            @testset "FullStressState" begin
                ϵ21 = 0.01; num_steps = 10; t_end = 0.01
                ss = FullStressState()
                # Check that we get the same result for runstresstate and runstrain
                σ_ss, s_ss = runstresstate(ss, m, ϵ21, (2, 1), t_end, num_steps)
                σ, s = runstrain(m, ϵ21, (2, 1), t_end, num_steps)
                @test σ_ss ≈ σ
                @test tovector(s_ss) ≈ tovector(s)
            end
            for (stress_state, ij) in (
                    (UniaxialStress(), (1,1)), (UniaxialStrain(), (1,1)), 
                    (UniaxialNormalStress(), (1,1)), (UniaxialNormalStress(), (2,1)),
                    (PlaneStress(), (2, 2)), (PlaneStrain(), (2, 1)),
                    (GeneralStressState(SymmetricTensor{2,3,Bool}((true, false, false, false, true, true)), rand(SymmetricTensor{2,3})), (2,2))
                    )
                @testset "$(nameof(typeof(stress_state))), (i,j) = ($(ij[1]), $(ij[2]))" begin
                    ϵij = 0.01; num_steps = 10; t_end = 0.01
                    stressfun(p) = runstresstate(stress_state, fromvector(p, m), ϵij, ij, t_end, num_steps)[1]
                    dσij_dp_num = FiniteDiff.finite_difference_jacobian(stressfun, tovector(m), Val{:central}; relstep = 1e-6)
                    σv, state, dσij_dp, diff = runstresstate_diff(stress_state, m, ϵij, ij, t_end, num_steps)
                    @test dσij_dp ≈ dσij_dp_num
                end
            end
        end
    end
end
