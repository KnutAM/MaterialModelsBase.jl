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
    G = rand()
    K = G + rand()
    m = LinearElastic(G, K)
    ϵ = rand(SymmetricTensor{2,3})
    diff = MaterialDerivatives(m)
    IxI = one(SymmetricTensor{2,3}) ⊗ one(SymmetricTensor{2,3})
    dσdϵ = 2G*(one(SymmetricTensor{4,3}) - IxI/3) + K*IxI
    dσdG = 2*(one(SymmetricTensor{4,3}) - IxI/3) ⊡ ϵ
    dσdK = IxI ⊡ ϵ

    old = NoMaterialState(); Δt = nothing
    cache = NoMaterialCache(); extras = allocate_differentiation_output(m)
    differentiate_material!(diff, m, ϵ, old, Δt, cache, extras, dσdϵ)
    @test diff.dσdϵ ≈ tomandel(dσdϵ)
    @test diff.dσdp[:,1] ≈ tomandel(dσdG)
    @test diff.dσdp[:,2] ≈ tomandel(dσdK)
end