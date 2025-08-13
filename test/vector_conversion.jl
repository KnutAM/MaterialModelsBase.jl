function test_invertible_conversion(x)
    v1 = tovector(x)
    # Basic test
    @test isa(v1, AbstractVector)
    @test v1 ≈ tovector(fromvector(v1, x))

    # Ensure that we are not using the values in 'x'
    v2 = rand(length(v1))
    @test v2 ≈ tovector(fromvector(v2, x))

    # Ensure that offset is working as intended
    v3 = zeros(length(v1) + 2)
    r1, r2 = rand(2)
    v3[1] = r1
    v3[end] = r2
    v4 = copy(v3) # Do copy here to keep 2:end-1 filled with zeros
    tovector!(v3, x; offset = 1)
    @test v3[1] == r1
    @test v3[end] == r2
    @test v3[2:end-1] ≈ v1
    @test v4 === tovector!(v4, fromvector(v3, x; offset = 1); offset = 1)
    @test v3 ≈ v4
end

@testset "vector_conversion" begin
    @testset "tensors" begin
        @testset for TT in (Tensor{2,2}, Tensor{4,3}, SymmetricTensor{2,3}, SymmetricTensor{4,2})
            test_invertible_conversion(rand(TT))
        end    
    end
    @testset "materials" begin
        @testset for m in (LinearElastic(rand(), rand()),)
            test_invertible_conversion(m)
            # TODO: For state variables as well
        end
    end
end
