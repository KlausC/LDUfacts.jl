
Random.seed!(2)

@testset "lduupdate(A+u*u')" begin
    u = ones(5)
    v = [1.0, 2, 0, -1, 1]
    A = Matrix(float.(Symmetric(rand(-2:2, 5,5))))
    la = ldu(A, FullPivot())
    f = 0.5
    ldu_update!(la, v, f)
    @test norm(la.P' * (A + v * v' * f) * la.P - la.L * la.D * la.U) <= 1e-13
end

@testset "lduupdate(A+u*v'+v*u')" begin
    u = ones(5)
    v = [1.0, 2, 0, -1, 1]
    A = Matrix(float.(Symmetric(rand(-2:2, 5,5))))
    la = ldu(A, FullPivot())
    ldu_update!(la, u, v, 0.1)
    @test norm(la.P' * (A + (u * v' + v * u') * 0.1) * la.P - la.L * la.D * la.U) <= 1e-13
    # note: test fails when f = 1 or f = -1 - check!!!
end