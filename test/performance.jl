# Checks to ensure that e.g. calls are allocation-free
# If functions that shouldn't allocate do, this could be a 
# sign of type-instability, would have caught e.g. 
# https://github.com/KnutAM/MaterialModelsBase.jl/pull/10

function get_run_allocations(ss, m, ϵv, t = collect(range(0, 1, length(ϵv))))
    state, cache, σv, ϵv_full = setup_run_timehistory(m, ϵv)
    run_timehistory!(σv, ϵv_full, state, cache, ss, m, ϵv, t) # Compilation
    return @allocated run_timehistory!(σv, ϵv_full, state, cache, ss, m, ϵv, t)
end

@testset "performance" begin
    # Load cases
    N = 100
    load_cases = (
        (UniaxialStress(), 
        collect((SymmetricTensor{2,1}((0.1*(i-1)/N,)) for i in 1:N+1))),
    )
    # Materials    
    G = 80.e3; K = 160.e3; η = 20.e3
    linear_elastic = LinearElastic(G, K)
    visco_elastic = ViscoElastic(linear_elastic, LinearElastic(G*0.5, K*0.5), η)
    materials = (linear_elastic, visco_elastic)

    for (stress_state, ϵv) in load_cases
        for m in materials
            allocs = get_run_allocations(stress_state, m, ϵv)
            allocs > 0 && println(allocs, " allocations for m::", typeof(m), ", ss::", typeof(stress_state))
            @test allocs == 0
        end
    end
end
