
Base.propertynames(::P, private::Bool=false) where P<:LDUPivoted = (:U,:L,:D,:d,:p,:P, (private ? fieldnames(P) : ())...)
function Base.getproperty(lf::LDUPivoted, s::Symbol)
    s == :U && return UnitUpperTriangular(lf.factors)
    s == :L && return UnitUpperTriangular(lf.factors)'
    s == :D && return Diagonal(diag(lf.factors))
    s == :d && return diag(lf.factors)
    s == :p && return permutation(lf.piv)
    s == :P && return LDUPerm(lf.piv, lf.pam, lf.ame)
    getfield(lf, s)
end

LinearAlgebra.rank(lf::LDUPivoted) = lf.rank
LinearAlgebra.issuccess(lf::LDUPivoted) = lf.info == 0

function permutation(piv::AbstractVector{T}) where T<:Integer
    permbypiv!(collect(Base.OneTo(T(length(piv)))), piv)
end

const PAM = Int[]
const AME = Float64[]

function permbypiv!(A::StridedVector, piv::AbstractVector{<:Integer}, pam::AbstractVector{<:Integer}=PAM, B::AbstractVector=AME; dims=1)
    n = length(piv)
    nn = length(A)
    ma = length(pam)
    n > nn && throw(ArgumentError("cannot permute vector of length $nn by permutation of length $n"))
    for i = 1:n
        pi = piv[i]
        if i != pi
            A[i], A[pi] = A[pi], A[i]
        end
        i > ma && continue
        pa = pam[i]
        if pa != 0
            A[i] += A[pa] * B[i]
        end
    end
    A
end

function permbypiv!(A::StridedMatrix, piv::AbstractVector{<:Integer}, pam::AbstractVector{<:Integer}=PAM, B::AbstractVector=AME; dims = 1)
    n = length(piv)
    ma = length(pam)
    if dims == 1 || dims isa Colon
        nn, m = size(A)
        n > nn && throw(ArgumentError("cannot permute $nn rows by permutation of length $n"))
        for i = 1:n
            j = piv[i]
            if i != j
                for k = 1:m
                    A[i,k], A[j,k] = A[j,k], A[i,k]
                end
            end
            i > ma && continue
            pa = pam[i]
            if pa != 0
                for k = 1:m
                    A[i, k] += A[pa,k] * adjoint(B[i])
                end
            end
        end
    end
    if dims == 2 || dims isa Colon
        m, nn = size(A)
        n > nn && throw(ArgumentError("cannot permute $nn columns by permutation of length $n"))
        for i = 1:n
            j = piv[i]
            if i != j
                for k = 1:m
                    A[k,i], A[k,j] = A[k,j], A[k,i]
                end
            end
            i > ma && continue
            pa = pam[i]
            if pa != 0
                for k = 1:m
                    A[k,i] += A[k,pa] * B[i]
                end
            end
        end
    end
    A
end

function LinearAlgebra.nullspace(la::LDUPivoted)
    pp = invA(la.p)
    n = length(pp)
    inv(la.U)[pp,lp.rank+1:n]
end

function imagespace(la::LDUPivoted)
    pp = invA(la.p)
    la.L[pp,1:lp.rank]
end

absapp(a::Real) = abs(a)

"""
    absapp(x)

Squareroot-less approximation to `abs(x)`. That may differ from `abs` for complex arguments.
"""
function absapp(a::Complex{T}) where T
    s, r = minmax(absapp(real(a)), absapp(imag(a)))
    if iszero(r)
        zero(s) / one(s)
    else
        x = (s / r) ^ 2
        #q = x + 1
        x /= 2
        c = x - x * x / (oftype(x, 14) / oftype(x, 5)) + 1
        #c = (q / c + c) / 2
        #c = (q / c + c) / 2
        c = c * r
    end
end

splitexp(a::AbstractFloat) = frexp(a)
function splitexp(a::Union{Integer,Rational})
    q, x = frexp(float(a))
    x >= 0 ? a / 2^x : a * 2^(-x), x
end

function sqrtapp(a::Real)
    q, x = splitexp(a)
    if isodd(x)
        q += q
        x -= 1
    end
    x ÷= 2
    c = (12 - (3 - q)^2) / 8
    c = (q / c + c) / 2
    x >= 0 ? c * 2^x : c / 2^(-x)
end

default_tol(A::AbstractMatrix, ::PivotingStrategy) = eps(float(maximum(abs, diag(A))) * size(A, 1))
default_tol(A::AbstractMatrix, ::FullPivot) =  eps(float(maximum(abs, A)) * size(A, 1))

"""
    ldu(A::Matrix, ps::Union{FullPivot,DiagonalPivot,NoPivot}[; tol=tolerance, check=true]) -> LDUPivoted
    ldu!(A::Matrix, ps::Union{FullPivot,DiagonalPivot,NoPivot}[; tol=tolerance, check=true]) -> LDUPivoted

Perform a decomposition `P' * A * P = L * D * L'`, where `D` is real diagonal, `L` is lower unit triangular,
and `P` is a quasi permutation matrix.

If the pivoting indicator is `NoPivot()`, no search for optimal pivot element is performed and `P == I`.
The operation fails, if a diagonal element except the last one is below or equal `tol`.
If `tol < 0`, a default of `maximum(abs, A) * n * eps` for full pivoting or `maximum(diag(A)) * n * eps` is used.

If the pivoting indicator is `DiagonalPivot()`, the diagonal absolutely biggest diagonal element of the remaining matrix
is searched in each step and the rows/columsn are swapped in order to move this to the pivot position.
The operation fails, if any element beyond the rank is exceeds `tol`.

If the pivoting indicator is `FullPivot()`, the absolutely biggest element in the remaining matrix is searched in each step.
The row/columns are swapped to move the row index to pivoting position. Then a multiple of the column index is added
to the pivot row/column. In this case, `P` includes these 2x2 triangular linear transformations.
The operation succeeds always.

In case no composition could be found, an exception is thrown or the user's is responsible to check success depending on `check`.

The resulting factorization object has type `LDUPivoted`, with public properties:

    D   diagonal factor matrix
    d   D as vector
    L   lower unit triangular factor
    U   upper unit triangular factor ( U == L')
    P   permutation transformation (of type LDUPerm)
    tol used tolerance used as pivot threshold

The following access methods for the result type

    rank()      best found rank
    issuccess() true iff decomposition is valid.
"""
function ldu(A::StridedMatrix, ps::PivotingStrategy; tol::Real=0.0, check::Bool=true)
    ldu!(copy(A), ps; tol, check)
end

function ldu!(A::StridedMatrix{T}, ps::P; iter::Integer=0, tol::Real=0.0, check::Bool=true) where {T,P<:PivotingStrategy}
    n = checksquare(A)
    nn = iter <= 0 ? n : min(iter, n)
    piv = collect(1:n)
    na = P <: FullPivot ? n : 0
    pam = zeros(Int, na)
    ame = zeros(T, na)
    rank = n

    stop = tol >= 0 ? tol : default_tol(A, ps)
    stop2 = stop^2

    for k = 1:nn
        _pivotstep!(A, k, ps, piv, pam, ame)
        
        akk = A[k,k]
        if abs2(akk) <= stop2
            rank = k - 1
            break
        end
        for j = k+1:n
            akj = adjoint(A[k,j]) / akk
            for i = k+1:j
                @inbounds A[i,j] -= akj * A[k,i]
            end
        end
        for i = 1:min(k-1,rank)
            @inbounds A[i,k] /= A[i,i]
        end
    end
    for k = rank+1:nn
        for i = 1:rank
            @inbounds A[i,k] /= A[i,i]
        end
    end
    info = verifycheck(A, stop2, ps, check, rank)
    LDUPivoted(A, piv, rank, stop, info, pam, ame)
end

function _pivotstep!(A, k, ::FullPivot, piv, pam, ame)
    n = size(A, 1)
    a = zero(A[k,k])
    jj = ii = k
    @inbounds for j = k:n
        for i = j:-1:k
            b = abs2(A[i,j])
            if b > a
                a = b
                ii = i
                jj = j
            end
        end
    end
    # println("pivot($k) = ($ii, $jj)")
    _swap_upper!(A, k, ii, piv)
    if ii != jj
        _addto_upper!(A, k, jj, pam, ame)
    end
end

function _pivotstep!(A, k, ::DiagonalPivot, piv, pam, ame)
    n = size(A, 1)
    a = zero(A[k,k])
    jj = k
    for j = k:n
        b = abs2(A[j,j])
        if b > a
            a = b
            jj = j
        end
    end
    #println("pivot($k) = ($jj, $jj)")
    _swap_upper!(A, k, jj, piv)
end
_pivotstep!(A, k, ::NoPivot, piv, pam, ame) = nothing

function _swap_upper!(A::AbstractMatrix, i::Int, j::Int, piv)
    n = size(A, 1)
    i >= j && return

    #println("swap_upper($i, $j)")
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
    @inbounds piv[i] = j
    nothing
end

function opteps(d, a)
    adjoint(d) / absapp(d) * ifelse(a < 0, -1, 1)
end

function optepsd(d, a)
    copysign(abs2(d) / absapp(d), a)
end

function _addto_upper!(A::AbstractMatrix, i::Int, j::Int, pam, ame)
    n = size(A, 1)
    i >= j && return
    #println("fix_upper($i, $j)")
    #disp(A, i)
    d = A[i,j]
    aii = A[i,i]
    ajj = A[j,j]
    eps = opteps(d, aii + ajj)
    eps2 = adjoint(eps) * eps
    epsd = optepsd(d, aii + ajj)
    @assert eps * d ≈ epsd "eps * d... $eps * $d != $epsd"

    @inbounds begin
        for k = 1:i-1
            A[k,i] += eps * A[k,j]
        end
        A[i,i] = eps2 * ajj + aii + epsd * 2
        #println("eps = $eps, A[$i,$i] = $eps2 * $ajj + $aii + $epsd * 2 = $(A[i,i])")
        for k = i+1:j-1
            A[i,k] += adjoint(eps * A[k,j])
        end
        A[i,j] = adjoint(eps) * ajj + d
        for k = j+1:n
            A[i,k] += adjoint(eps) * A[j,k]
        end
        pam[i] = j
        ame[i] = eps
    end
    nothing
end

function verifycheck(A::AbstractMatrix{T}, stop2, ::P, check::Bool, rank::Integer) where {T,P<:PivotingStrategy}
    info = 0
    m, n = size(A)
    if !(P <: FullPivot)
        a = zero(T)
        for j = rank+1:n
            for i = rank+1:j
                j > m && break
                b = abs2(A[i,j])
                if b > a
                    a = b
                end
            end
        end
        if a > stop2
            info = 1
        end
    end
    if check && info != 0
        throw(SingularException(rank+1))
    end
    info
end

function LinearAlgebra.Matrix(p::LDUPerm{S}) where S
    n = length(p.piv)
    A = Matrix{S}(I(n))
    rmul!(A, p)
end

adjoint(p::S) where {T,S<:LDUPerm{T}} = Adjoint{T,S}(p)

rmul!(A::StridedArray, p::LDUPerm) = permbypiv!(A, p.piv, p.pam, p.ame, dims = 2)
function lmul!(P::Adjoint{<:Any,<:LDUPerm}, A::StridedArray)
    p = P.parent
    permbypiv!(A, p.piv, p.pam, p.ame, dims = 1)
end
(*)(A::AbstractArray, p::LDUPerm) = rmul!(copy(A), p)
(*)(p::Adjoint{<:Any,<:LDUPerm}, A::AbstractMatrix) = lmul!(p, copy(A))
(*)(p::Adjoint{<:Any,<:LDUPerm}, A::AbstractVector) = lmul!(p, copy(A))
