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
    differentiate_material!(diff, m, ϵ, old, Δt, cache, dσdϵ, extras)
    @test diff.dσdϵ ≈ tomandel(dσdϵ)
    @test diff.dσdp[:,1] ≈ tomandel(dσdG)
    @test diff.dσdp[:,2] ≈ tomandel(dσdK)
end