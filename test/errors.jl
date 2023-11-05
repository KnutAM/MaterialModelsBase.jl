@testset "Errors" begin
    # Test multiple arguments 
    args = ("This", " ", "is", " a ", "test")
    msg = string(args...)
    @test NoLocalConvergence(args...).msg == msg
    @test NoStressConvergence(args...).msg == msg

    # Test that "impossible" stress iteration throws 
    # the correct error
    m = LinearElastic(80e3, 150e3)
    ss = PlaneStress(;max_iter = 1, tolerance=1e-100)
    ϵ = rand(SymmetricTensor{2,2})
    @test_throws NoStressConvergence material_response(ss, m, ϵ, NoMaterialState())
end