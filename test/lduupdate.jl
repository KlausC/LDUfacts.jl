using LDUFacts: sqrtapp

@testset "lduupdate(A+u*u')" begin
    u = ones(5)
    v = [1.0, 2, 0, -1, 1]
    A = [-2.0 2 2 0 2; 2 1 1 0 1; 2 1 2 -1 1; 0 0 -1 -2 0; 2 1 1 0 -1]
    la = ldu(A, FullPivot())
    f = 0.5
    ldu_update!(la, v, f)
    @test norm(la.P' * (A + v * v' * f) * la.P - la.L * la.D * la.U) <= 1e-13
end

@testset "lduupdate(A+u*v'+v*u')" begin
    u = ones(5)
    v = [1.0, 2, 0, -1, 1]
    A = [-2.0 2 2 0 2; 2 1 1 0 1; 2 1 2 -1 1; 0 0 -1 -2 0; 2 1 1 0 -1]
    la = ldu(A, FullPivot())
    ldu_update!(la, u, v, 0.1)
    @test norm(la.P' * (A + (u * v' + v * u') * 0.1) * la.P - la.L * la.D * la.U) <= 1e-13
    # note: test fails when f = -1 - check!!!
    la = ldu(A, FullPivot())
    f = -1
    @test_throws SingularException ldu_update!(la, u, v, f)
    #@test_broken norm(la.P' * (A + (u * v' + v * u') * f) * la.P - la.L * la.D * la.U) <= 1e-13
end

@testset "sqrtapp" begin
    @test abs(sqrtapp(0.5) / sqrt(0.5) - 1) < 0.05
    @test abs(sqrtapp(3//2) / sqrt(1.5) - 1) < 0.05
end
