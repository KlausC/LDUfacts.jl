
Base.propertynames(::P, private::Bool=false) where P<:LDUPivoted = (:U,:L,:D,:d,:p,:P, (private ? fieldnames(P) : ())...)
function Base.getproperty(lf::LDUPivoted, s::Symbol)
    s == :U && return UnitUpperTriangular(lf.factors)
    s == :L && return UnitUpperTriangular(lf.factors)'
    s == :D && return Diagonal(diag(lf.factors))
    s == :d && return diag(lf.factors)
    s == :p && return piv_to_perm(lf.piv)
    s == :P && return LDUPerm(lf.piv, lf.pam, lf.ame)
    getfield(lf, s)
end

LinearAlgebra.rank(lf::LDUPivoted) = lf.rank
LinearAlgebra.issuccess(lf::LDUPivoted) = lf.info == 0

function piv_to_perm(piv::AbstractVector{T}) where T<:Integer
    mult_by_perm!(collect(Base.OneTo(T(length(piv)))), piv)
end

const PAM = Int[]
const AME = Float64[]

Base.size(p::LDUPivoted, x...) = size(p.factors, x...)

function mult_by_perm!(A::StridedArray, piv::AbstractVector{<:Integer}, pam::AbstractVector{<:Integer}=PAM, B::AbstractVector=AME; dims = 1)
    n = length(piv)
    ma = length(pam)
    if dims == 1 || dims isa Colon
        nn = size(A, 1)
        m = size(A, 2)
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
                bi = adjoint(B[i])
                for k = 1:m
                    A[i, k] += A[pa,k] * bi
                end
            end
        end
    end
    if dims == 2 || dims isa Colon
        m = size(A, 1)
        nn = size(A, 2)
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
                bi = B[i]
                for k = 1:m
                    A[k,i] += A[k,pa] * bi
                end
            end
        end
    end
    A
end

function mult_by_revperm!(A::StridedArray, piv::AbstractVector{<:Integer}, pam::AbstractVector{<:Integer}=PAM, B::AbstractVector=AME; dims = 1)
    n = length(piv)
    ma = length(pam)
    if dims == 1 || dims isa Colon
        nn = size(A, 1)
        m = size(A, 2)
        n > nn && throw(ArgumentError("cannot permute $nn rows by permutation of length $n"))
        for i = n:-1:1
            if i <= ma
                pa = pam[i]
                if pa != 0
                    bi = B[i]
                    for k = 1:m
                        A[pa, k] += A[i,k] * bi
                    end
                end
            end
            j = piv[i]
            if i != j
                for k = 1:m
                    A[i,k], A[j,k] = A[j,k], A[i,k]
                end
            end
        end
    end
    if dims == 2 || dims isa Colon
        m = size(A, 1)
        nn = size(A, 2)
        n > nn && throw(ArgumentError("cannot permute $nn columns by permutation of length $n"))
        for i = n:-1:1
            if i <= ma
                pa = pam[i]
                if pa != 0
                    bi = adjoint(B[i])
                    for k = 1:m
                        A[k,pa] += A[k,i] * bi
                    end
                end
            end
            j = piv[i]
            if i != j
                for k = 1:m
                    A[k,i], A[k,j] = A[k,j], A[k,i]
                end
            end
        end
    end
    A
end

"""
    absapp(x)

Squareroot-less approximation to `abs(x)`. That may differ from `abs` for complex arguments.
"""
absapp(a::Real) = abs(a)
function absapp(a::Complex{T}) where T
    s, r = minmax(absapp(real(a)), absapp(imag(a)))
    if iszero(r)
        zero(s) / one(s)
    else
        x = (s / r) ^ 2
        #q = x + 1
        x /= 2
        #c = x - x * x / (oftype(x, 14) / oftype(x, 5)) + 1
        #c = (q / c + c) / 2
        #c = (q / c + c) / 2
        c = x + 1
        c = c * r
        s / r * s / 2 + r
    end
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
function ldu(A::AbstractMatrix, ps::PivotingStrategy; tol::Real=0.0, check::Bool=true)
    ldu!(copy(A), ps; tol, check)
end

function ldu!(A::Symmetric{T}, ps::P; iter::Integer=0, tol::Real=0.0, check::Bool=true) where {T,P<:PivotingStrategy}
    _ldu!(A.uplo == 'U' ? A.data : transpose(A.data), ps, iter, float(tol), check)
end

function ldu!(A::Hermitian{T}, ps::P; iter::Integer=0, tol::Real=0.0, check::Bool=true) where {T,P<:PivotingStrategy}
    _ldu!(A.uplo == 'U' ? A.data : adjoint(A.data), ps, iter, float(tol), check)
end

function ldu!(A::StridedMatrix{T}, ps::P; iter::Integer=0, tol::Real=0.0, check::Bool=true) where {T,P<:PivotingStrategy}
    _ldu!(A, ps, iter, float(tol), check)
end

function _ldu!(A::SATMatrix{T}, ps::P, iter::Integer, tol::AbstractFloat, check::Bool) where {T,P<:PivotingStrategy}
    n = checksquare(A)
    nn = iter <= 0 ? n : min(iter, n)
    piv = collect(1:n)
    P <: PivotLike && length(ps.piv) != n && throw(ArgumentError("PivotLike with wrong dimension"))
    na = P <: FullPivot ? n : P <: PivotLike ? length(ps.pam) : 0
    pam = zeros(Int, na)
    ame = zeros(T, na)
    rank = n
    stop = tol >= 0 ? tol : default_tol(A, ps)

    for k = 1:nn
        _pivotstep!(A, k, ps, piv, pam, ame)
        
        akk = A[k,k]
        if pivabs(akk) <= stop
            rank = k - 1
            break
        end
        for j = k+1:n
            akj = A[k,j] / akk
            for i = k+1:j-1
                @inbounds A[i,j] -= akj * adjoint(A[k,i])
            end
            @inbounds A[j,j] -= real(akj * adjoint(A[k,j]))
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
    info = verifycheck(A, stop, ps, check, rank)
    LDUPivoted(A, piv, rank, stop, info, pam, ame)
end

pivabs(x::T) where T<:Union{Real,Complex{<:Real}} = abs2(x)
pivabs(x::Union{Rational,Complex{<:Rational}}) = iszero(x) ? zero(x) : 1 / pivsum(x)
pivsum(x::Rational) = iszero(x.num) ? zero(x.num) : abs(x.den) + abs(x.num)
pivsum(x::Complex{<:Rational}) = pivsum(real(x)) + pivsum(imag(x))

function _pivotstep!(A, k, ::FullPivot, piv, pam, ame)
    n = size(A, 1)
    a = zero(real(eltype(A)))
    jj = ii = k
    @inbounds for j = k:n
        for i = j:-1:k
            b = pivabs(A[i,j])
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

function _pivotstep!(A, k, p::PivotLike, piv, pam, ame)
    ii = p.piv[k]
    jj = k > length(p.pam) || p.pam[k] == 0 ? ii : p.pam[k]
    # println("pivot($k) = ($ii, $jj)")
    _swap_upper!(A, k, ii, piv)
    if ii != jj
        _addto_upper!(A, k, jj, pam, ame)
    end
end

function _pivotstep!(A, k, ::DiagonalPivot, piv, pam, ame)
    n = size(A, 1)
    a = zero(real(eltype(A)))
    jj = k
    for j = k:n
        b = pivabs(A[j,j])
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
    rsum = real(aii) + real(ajj)
    eps = opteps(d, rsum)
    eps2 = adjoint(eps) * eps
    epsd = optepsd(d, rsum)
    @assert eps * d ≈ epsd "eps * d... $eps * $d != $epsd"

    @inbounds begin
        for k = 1:i-1
            A[k,i] += eps * A[k,j]
        end
        A[i,i] = real(eps2 * ajj + aii + epsd * 2)
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

function verifycheck(A::AbstractMatrix{T}, stop, ::P, check::Bool, rank::Integer) where {T,P<:PivotingStrategy}
    info = 0
    m, n = size(A)
    if !(P <: FullPivot)
        a = zero(real(eltype(A)))
        for j = rank+1:n
            for i = rank+1:j
                j > m && break
                b = pivabs(A[i,j])
                if b > a
                    a = b
                end
            end
        end
        if a > stop
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

rmul!(A::StridedArray, p::LDUPerm) = mult_by_perm!(A, p.piv, p.pam, p.ame, dims = 2)
lmul!(p::LDUPerm, A::StridedArray) = mult_by_revperm!(A, p.piv, p.pam, p.ame, dims = 1)
function lmul!(P::Adjoint{<:Any,<:LDUPerm}, A::StridedArray)
    p = P.parent
    mult_by_perm!(A, p.piv, p.pam, p.ame, dims = 1)
end
function rmul!(A::StridedArray, P::Adjoint{<:Any,<:LDUPerm})
    p = P.parent
    mult_by_revperm!(A, p.piv, p.pam, p.ame, dims = 2)
end
(*)(A::AbstractArray, p::LDUPerm) = rmul!(copy(A), p)
(*)(p::Adjoint{<:Any,<:LDUPerm}, A::AbstractMatrix) = lmul!(p, copy(A))
(*)(p::Adjoint{<:Any,<:LDUPerm}, A::AbstractVector) = lmul!(p, copy(A))
(*)(p::LDUPerm, A::AbstractArray) = lmul!(p, copy(A))
(*)(p::LDUPerm, A::AbstractVector) = lmul!(p, copy(A))
(*)(A::AbstractMatrix, p::Adjoint{<:Any,<:LDUPerm}) = rmul!(copy(A), p)

function ldiv!(p::LDUPivoted, A::StridedArray)
    lmul!(p.P', A)
    ldiv!(p.L, A)
    for i = 1:rank(p)
        A[i,:] = p.factors[i,i] \ A[i,:]
    end
    ldiv!(p.U, A)
    lmul!(p.P, A)
end

function rdiv!(A::StridedArray, p::LDUPivoted)
    rmul!(A, p.P)
    rdiv!(A, p.U)
    for i = 1:rank(p)
        A[:,i] = A[:,i] / p.factors[i,i]
    end
    rdiv!(A, p.L)
    rmul!(A, p.P')
end

Base.inv(p::LDUPivoted{T}) where T = p \ Matrix{T}(I(size(p, 1)))

function det(p::LDUPivoted)
    prod(real.(p.d))
end
function logabsdet(p::LDUPivoted{T}) where T
    sig = one(real(T))
    das = zero(float(real(T)))
    n = size(p.factors, 1)
    for k = 1:n
        x = p.factors[k,k]
        if real(x) < 0
            sig *= -1
        end
        das += log(abs(x))
    end
    das, sig
end
function logdet(p::LDUPivoted{T}) where T
    d, s = logabsdet(p)
    s <= 0 && throw(DomainError(s))
    d
end

"""
    perm_to_piv(perm)

Given a permutation vector `perm`, create an integer vector `p` of same length, such that the
product of involutions `(k, p[k])` is equivalent with `perm`.

Invariant:
    piv_to_perm(perm_to_piv(perm)) == perm

If `k == p[k]` that is identity. The order of operations in the product is right to left.

If `perm` is not a permutation, the result is undefined.
"""
function perm_to_piv(perm::AbstractVector{<:Integer})
    ax = axes(perm, 1)
    w = similar(perm, ax)
    p = similar(perm, ax)
    @inbounds for k in ax
        w[k] = p[k] = k
    end
    @inbounds for wb in ax
        la = perm[wb]
        lb = p[wb]
        if la == lb
            p[wb] = wb
        elseif la in ax
            wa = w[la]
            p[wa] = lb
            p[wb] = wa
            w[la] = wb
            w[lb] = wa
        end
    end
    p
end