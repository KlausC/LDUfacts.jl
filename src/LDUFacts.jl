module LDUFacts

using LinearAlgebra

using LinearAlgebra: checksquare

struct LDU{T<:AbstractMatrix}
    data::T
end

struct LDUPivoted{T<:AbstractMatrix,F<:Real}
    factors::T
    piv::Vector{Int}
    rank::Int
    tol::F
end

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
        s = d2 * conj(u1) / dis
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

function ldu!(A::Matrix, ::Val{Pivoted}=Val(true); tol::Real=0.0) where Pivoted
    n = checksquare(A)
    piv = collect(1:n)
    rank = n
    stop = tol < 0 ? eps(float(maximum(abs, diag(A))) * n) : tol

    @inbounds for k = 1:n
        if Pivoted
            j = k
            a = abs2(A[k,k])
            for i = k+1:n
                b = abs2(A[i,i])
                if b > a
                    a = b
                    j = i
                end
            end
            swap_upper!(A, k, j, piv)
        end
        for i = 1:k-1
            A[i,k] /= A[i,i]
        end

        akk = A[k,k]
        sing = abs(akk) <= stop
        akk = sing ? zero(akk) : inv(akk)
        for j = k+1:n
            s = A[k,j]
            for i = 1:k-1
                s -= conj(A[i,k]) * A[i,j]
            end
            A[k,j] = s
            A[j,j] -= abs2(s) * akk
        end
        if sing
            rank = k - 1
            break
        end
    end
    LDUPivoted(A, piv, rank, stop)
end

function swap_upper!(A::AbstractMatrix, i::Int, j::Int, piv)
    m, n = size(A)
    @assert m == n
    @assert i <= j
    i == j && return
    @inbounds begin
        for k = 1:i-1
            A[k,i], A[k,j] = A[k,j], A[k,i]
        end
        A[i,i], A[j,j] = A[j,j], A[i,i]
        for k = i+1:j-1
            A[i,k], A[k,j] = conj(A[k,j]), conj(A[i,k])
        end
        A[i,j] = conj(A[i,j])
        for k = j+1:n
            A[j,k], A[i,k] = A[i,k], A[j,k]
        end
    end
    piv[i], piv[j] = piv[j], piv[i]
    nothing
end

function Base.getproperty(lf::LDUPivoted, s::Symbol)
    s == :U && return UnitUpperTriangular(lf.factors)
    s == :L && return UnitUpperTriangular(lf.factors)'
    s == :D && return Diagonal(diag(lf.factors))
    s == :d && return diag(lf.factors)
    s == :p && return lf.piv
    getfield(lf, s)
end

end # module
