module LDUFacts

using LinearAlgebra

using LinearAlgebra: checksquare
using LinearAlgebra: PivotingStrategy, NoPivot

struct LDU{T<:AbstractMatrix}
    data::T
end

struct LDUPivoted{S,T<:AbstractMatrix{S},F<:Real}
    factors::T
    piv::Vector{Int}
    rank::Int
    tol::F
    pam::Vector{Int}
    ame::Vector{S}
    Q::Matrix{S}
end

struct LDUPerm{S}
    piv::Vector{Int}
    pam::Vector{Int}
    ame::S
end

struct DiagonalPivot <: PivotingStrategy end
struct FullPivot <: PivotingStrategy end

function ldu_update!(ldu::LDU, u::AbstractVector, f::Real)
    A = ldu.data
    n, m = size(A)
    @assert n == m
    @assert length(u) == m
    for k = 1:n
        d1 = real(A[k,k])
        d2 = f
        u1 = u[k]
        dis = d1 + d2 * abs2(u1)
        abs2(dis) < abs2(d1) / 100 && continue
        c = d1 / dis
        s = d2 * adjoint(u1) / dis
        for i = k+1:n
            ui = u[i] - A[k,i] * u1
            A[k,i] = A[k,i] * c + u[i] * s
            u[i] = ui
        end
        A[k,k] = dis
        u[k] = 0
        f = d1 * d2 / dis
    end
    ldu
end

function ldu_update!(ldu::LDU, u::AbstractVector, v::AbstractVector, f::Real)
    g = sqrt(norm(u) / norm(v) * 2)
    u = u / g
    v = v * (g / 2)
    upv = u + v
    umv = u - v
    ldu_update!(ldu, upv, f)
    ldu_update!(ldu, ump, -f)
end

function ldu_update(A, u, f)
    u = float.(collect(u))
    ldu = ldu_update!(LDU(copy(A.data)), u, f)
    (ldu, u)
end

function matrix(ldu::LDU)
    Adjoint(UnitUpperTriangular(ldu.data)) * Diagonal(diag(ldu.data)) * UnitUpperTriangular(ldu.data)
end

"""
    ldu!(A::Matrix, Val(false/true)[, tol=tolerance])

Perform a decomposition `A = P * L * D * L' * P'`, where `D` is real diagonal, `L` is lower unit triangular,
and `P` is a permutation matrix.

If the pivoting indicator is `Val(false)`, no search for optimal pivot element is performed and `P == I`.
The operation fails, if a diagonal element is zero.

If the pivoting indicator is `Val(true)`, the diagonal absolutely biggest diagonal element of the remaining matrix
is searched in each step and the rows/columsn are swapped in order to move this to the pivot position.
"""
function ldu!(A::Matrix{T}, ::Val{P}; tol::Real=0.0) where {T,P}
    n = checksquare(A)
    piv = collect(1:n)
    rank = n
    stop = tol < 0 ? eps(float(maximum(abs, diag(A))) * n) : tol

    @inbounds for k = 1:n
        if P
            _pivotstep!(A, k, n, piv)
        end
        for i = 1:min(k-1,rank)
            A[i,k] /= A[i,i]
        end

        akk = A[k,k]
        sing = abs(akk) <= stop
        akk = sing ? zero(akk) : inv(akk)
        for j = k+1:n
            s = A[k,j]
            for i = 1:k-1
                s -= adjoint(A[i,k]) * A[i,j]
            end
            A[k,j] = s
            A[j,j] -= abs2(s) * akk
        end
        if sing
            rank = min(rank, k - 1)
            # break
        end
    end
    LDUPivoted(A, piv, rank, stop, Int[], T[], zeros(T, 0, 0))
end

function _pivotstep!(A, k, n, piv)
    j = k
    a = abs2(A[k,k])
    for i = k+1:n
        b = abs2(A[i,i])
        if b > a
            a = b
            j = i
        end
    end
    _swap_upper!(A, k, j, piv, zeros(Float64, n, n))
end

function _swap_upper!(A::AbstractMatrix, i::Int, j::Int, piv, Q)
    n = size(A, 1)
    i >= j && return

    println("swap_upper($i, $j)")
    @inbounds begin
        for k = 1:i-1
            A[k,i], A[k,j] = A[k,j], A[k,i]
        end
        A[i,i], A[j,j] = A[j,j], A[i,i]
        for k = i+1:j-1
            A[i,k], A[k,j] = adjoint(A[k,j]), adjoint(A[i,k])
        end
        A[i,j] = adjoint(A[i,j])
        for k = j+1:n
            A[j,k], A[i,k] = A[i,k], A[j,k]
        end
    end
    piv[i], piv[j] = piv[j], piv[i]
    for k = 1:n
        Q[i,k], Q[j,k] = Q[j,k], Q[i,k]
    end
    nothing
end

function Base.getproperty(lf::LDUPivoted, s::Symbol)
    s == :U && return UnitUpperTriangular(lf.factors)
    s == :L && return UnitUpperTriangular(lf.factors)'
    s == :D && return Diagonal(diag(lf.factors))
    s == :d && return diag(lf.factors)
    s == :p && return lf.piv
    s == :P && return LDUPerm(lf.piv, lf.pam, lf.ame)
    getfield(lf, s)
end

function LinearAlgebra.nullspace(la::LDUPivoted)
    pp = invperm(la.p)
    n = length(pp)
    inv(la.U)[pp,lp.rank+1:n]
end

function imagespace(la::LDUPivoted)
    pp = invperm(la.p)
    la.L[pp,1:lp.rank]
end

_absapp(a::Real) = abs(a)
absapp(a::Real) = float(abs(a))
function absapp(a::Complex)
    c, r = minmax(_absapp(real(a)), _absapp(imag(a)))
    if iszero(r)
        zero(c) / one(c)
    else
        x = (c / r) ^ 2
        q = x + 1
        x /= 2
        c = x - x * x / (oftype(x, 14) / oftype(x, 5)) + 1
        #c = (q / c + c) / 2
        #c = (q / c + c) / 2
        c * r
    end
end


function ldu!(A::Matrix{T}, ps::PivotingStrategy; iter::Integer=0, tol::Real=0.0) where T
    n = checksquare(A)
    nn = iter <= 0 ? n : min(iter, n)
    piv = collect(1:n)
    pam = zeros(Int, n)
    ame = zeros(T, n)
    Q = Matrix{T}(I(n))

    rank = n
    stop = tol < 0 ? eps(float(maximum(abs, diag(A))) * n) : tol

    for k = 1:nn
        _pivotstep!(A, k, n, ps, piv, pam, ame, Q)
        for i = 1:min(k-1,rank)
            A[i,k] /= A[i,i]
        end
        
        akk = A[k,k]
        sing = rank < n || abs(akk) <= stop
        akk = sing ? zero(akk) : inv(akk)
        
        for j = k+1:n
            akj = adjoint(A[k,j]) * akk
            for i = k+1:j
                A[i,j] -= akj * A[k,i]
            end
        end
        if sing
            rank = min(rank, k - 1)
            #break
        end
    end
    for k = rank+nn+1:nn
        for i = 1:rank
            A[i,k] /= A[i,i]
        end
    end
    LDUPivoted(A, piv, rank, stop, pam, ame, Q)
end

function _pivotstep!(A, k, n, ::FullPivot, piv, pam, ame, Q)
    a = zero(A[k,k])
    jj = ii = k
    for j = k:n
        for i = j:-1:k
            b = abs2(A[i,j])
            if b > a
                a = b
                ii = i
                jj = j
            end
        end
    end
    println("pivot($k) = ($ii, $jj)")
    _swap_upper!(A, k, ii, piv, Q)
    if ii != jj
        _fix_upper!(A, k, jj, piv, pam, ame, Q)
    end
end

function _pivotstep!(A, k, n, ::DiagonalPivot, piv, pam, ame, Q)
    a = zero(A[k,k])
    jj = k
    for j = k:n
        b = abs2(A[j,j])
        if b > a
            a = b
            jj = j
        end
    end
    println("pivot($k) = ($jj, $jj)")
    _swap_upper!(A, k, jj, piv, Q)
end
function _pivotstep!(A, k, n, ::NoPivot, piv, pam, ame, Q)
    nothing
end

function opteps(d, a)
    adjoint(d) / absapp(d) * ifelse(a < 0, -1, 1)
end

function optepsd(d, a)
    copysign(abs2(d) / absapp(d), a)
end

function _fix_upper!(A::AbstractMatrix, i::Int, j::Int, piv, pam, ame, Q)
    n = size(A, 1)
    i >= j && return
    println("fix_upper($i, $j)")
    disp(A, i)
    d = A[i,j]
    aii = A[i,i]
    ajj = A[j,j]
    eps = opteps(d, aii + ajj)
    eps2 = adjoint(eps) * eps
    epsd = optepsd(d, aii + ajj)
    @assert eps * d ≈ epsd "eps * d... $eps * $d != $epsd"

    begin
        for k = 1:i-1
            A[k,i] += eps * A[k,j]
        end
        A[i,i] = eps2 * ajj + aii + epsd * 2
        println("eps = $eps, A[$i,$i] = $eps2 * $ajj + $aii + $epsd * 2 = $(A[i,i])")
        for k = i+1:j-1
            A[i,k] += adjoint(eps * A[k,j])
        end
        A[i,j] = adjoint(eps) * ajj + d
        for k = j+1:n
            A[i,k] += adjoint(eps) * A[j,k]
        end
        pi = piv[i]
        pam[pi] = piv[j]
        ame[pi] = eps
        Q[j,i] = eps
    end

    disp(A, i)
    nothing
end

LinearAlgebra.Matrix(p::LDUPerm) = _matrix(p.piv, p.pam, p.ame)

function _matrix(piv, pam, ame)
    n = length(piv)
    A = Matrix{eltype(ame)}(I(n))
    for j = 1:length(pam)
        if pam[j] != 0
            A[pam[j],j] = ame[j]
        end
    end
    A[:,piv]
end

function disp(A, i)
    display(UpperTriangular(A))
    s = eigvals(Symmetric(A[i:end,i:end]))
    display([diag(A)[1:i-1]; s]')
end

end # module