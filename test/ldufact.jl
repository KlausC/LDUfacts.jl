using LDUFacts: piv_to_perm, perm_to_piv, LDUPerm

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
    R = [
         0  -1   0   0   0  -1  -1
        -1   1   0   0  -1  -1   0
         1   1   1   1  -1   1   0
         1   0   1   0   0   0   1
         0   0   0  -1   0   0   0
         0  -1  -1   0  -1  -1   0
         0   0   0   1   1   0  -1
        -1   1  -1  -1   1   0  -1
         0   1  -1  -1  -1   0   0
        -1  -1   0   1   0   1  -1
    ]
    R = float.(R)
    A = R * Diagonal([1,1,1,1,-1,-1,-1]) * R'
    la = ldu(copy(A), ps, tol=-1, check=false)
    @test norm(res(A, la)) < 1e-14
    @test issuccess(la)
    issuccess(la) && @test rank(la) == 7
end

@testset "ldu!(rational)" begin
    A = [20 4 11 -23 20; 4 -38 -25 -46 -1; 11 -25 12 -7 27; -23 -46 -7 10 14; 20 -1 27 14 -3] .// 4
    la = ldu!(copy(A), FullPivot(), tol = 0)
    @test rank(la) == 5
    @test issuccess(la)
    @test iszero(res(A, la))
end

@testset "simple complex" begin
    A = [
        -3//1+0//1*im   3//2-2//1*im  -1//1+2//1*im
         3//2+2//1*im   2//1+0//1*im  -3//2-1//2*im
        -1//1-2//1*im  -3//2+1//2*im  -3//2+0//1*im
    ]
    la = ldu(A, DiagonalPivot(), tol = 0)
    @test issuccess(la)
    @test rank(la) == 3
    @test iszero(res(A, la))
    @test iszero(inv(A) - inv(la))
    @test det(A) == det(la)
end

@testset "advanced complex and rational" begin
    A = [
        2//1+0//1*im   2//1-3//2*im   0//1-2//1*im   3//1+3//1*im  -1//1+1//1*im
        2//1+3//2*im   1//1+0//1*im  -1//2+1//2*im  -3//2+2//1*im   1//1+0//1*im
        0//1+2//1*im  -1//2-1//2*im  -1//1+0//1*im   1//1-3//2*im  -3//2-3//2*im
        3//1-3//1*im  -3//2-2//1*im   1//1+3//2*im   3//1+0//1*im   1//1+1//2*im
       -1//1-1//1*im   1//1+0//1*im  -3//2+3//2*im   1//1-1//2*im  -1//1+0//1*im
    ]
    A = big.(A)
    la = ldu(A, FullPivot(), tol = 0)
    @test issuccess(la)
    @test rank(la) == 5
    @test iszero(res(A, la))
    @test inv(la) == inv(A)
    @test logdet(la) ≈ logdet(A)
    b = big.([1+0im, 2, 3, 4, 5])
    @test la \ b == A \ b
end

@testset "PivotLike" begin
    A = [20 4 11 -23 20; 4 -38 -25 -46 -1; 11 -25 12 -7 27; -23 -46 -7 10 14; 20 -1 27 14 -3] .// 4 
    la = ldu(A, FullPivot())
    ps = PivotLike(la)
    lb = ldu(A, ps)
    @test la.piv == lb.piv
    @test la.pam == lb.pam
    @test la.ame == lb.ame
end

@testset "perm_to_piv" begin
    perm = [2, 4, 3, 9, 5, 1, 10, 7, 6, 8]
    piv = perm_to_piv(perm)
    @test piv_to_perm(piv) == perm
end

@testset "mut_by_revperm" begin
    P = LDUPerm([1, 2], [2, 0], [1+2im, 0])
    b = [10+0im, 20]
    MP = Matrix(P)
    @test MP == [1 0; 1+2im 1]
    @test P * b == MP * b
    @test P * [b b] == MP * [b b]
    @test [b'; b'] * P' == [b'; b'] * MP'
end
