
using Random

Random.seed!(1)
essential(x) = abs(x) < 1e-13 ? zero(x) : x

res(A, la) = la.P' * A * la.P - la.L * la.D * la.U

@testset "throwing" begin
   A = [0.0 1; 1 0]
   @test_throws SingularException ldu(A, NoPivot())
   @test_throws SingularException ldu(A, DiagonalPivot())
   @test rank(ldu(A, FullPivot())) == 2
end

@testset "ldu(3,2) NoPivot" begin
    A = Matrix(Symmetric([1. 0 2; 0 1 1; 0 0 1]))
    la = ldu(A, NoPivot(); check = false)
    @test issuccess(la)
    @test rank(la) == 3
    @test la.p == [1, 2, 3]
    @test norm(res(A, la)) <= 1e-13
    @test Matrix(la.P) == I(3)
end
@testset "ldu(3,2) DiagonalPivot" begin
    A = Matrix(Symmetric([1. 0 2; 0 1 1; 0 0 1]))
    la = ldu(A, DiagonalPivot(); check = false)
    @test issuccess(la)
    @test rank(la) == 3
    @test la.p == [1, 3, 2]
    @test norm(res(A, la)) <= 1e-13
    @test Matrix(la.P) == [1. 0 0; 0 0 1; 0 1 0]
end
@testset "ldu(3,2) FullPivot" begin
    A = Matrix(Symmetric([1. 0 2; 0 1 1; 0 0 1]))
    la = ldu(A, FullPivot(); check = false)
    @test issuccess(la)
    @test rank(la) == 3
    @test la.p == [1, 2, 3]
    @test norm(res(A, la)) <= 1e-13
    @test Matrix(la.P) == [1. 0 0; 0 1 0; 1 0 1]
end

@testset "ldu!(7,10), $ps" for ps in (NoPivot(), DiagonalPivot(), FullPivot())
    R = float.(rand(-1:1, 10 ,7))
    A = R * Diagonal([1,1,1,1,-1,-1,-1]) * R'
    la = ldu(copy(A), ps, tol=-1.0, check=false)
    @test (norm(res(A, la)) < 1e-14) == issuccess(la)
    @test issuccess(la) == !(ps isa NoPivot)
    issuccess(la) && @test rank(la) == 7
end

@testset "ldu!(rational)" begin
    A = [20 4 11 -23 20; 4 -38 -25 -46 -1; 11 -25 12 -7 27; -23 -46 -7 10 14; 20 -1 27 14 -3] .// 4
    la = ldu!(copy(A), FullPivot(), tol = 0)
    @test rank(la) == 5
    @test issuccess(la)
    @test iszero(res(A, la))
end
