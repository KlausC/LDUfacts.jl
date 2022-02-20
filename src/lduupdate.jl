
"""
    ldu_update!(ldu::LDUPivoted, u::Vector, f::Real)

Given the LDU factorization `ldu: P' * A * P = L * D * L'` of a symmetric real or hermitian matrix `A`, update
`ldu` to obtain a factorization of `A + u * u' * f` without changing `L`.

This attempt may fail with a `SingularException` if the new pivot elements becomes zero.
"""
function ldu_update!(ldu::LDUPivoted, u::AbstractVector, f::Real)
    A = ldu.factors
    n = checksquare(A)
    length(u) == n || throw(ArgumentError("vector size $(length(u)) does not match matrix size $n"))
    u = ldu.P' * u

    for k = 1:n
        d1 = real(A[k,k])
        d2 = f
        u1 = u[k]
        dis = d1 + d2 * abs2(u1)
        A[k,k] = dis
        iszero(dis) && throw(SingularException(k))
        c = d1 / dis
        s = d2 * adjoint(u1) / dis
        for i = k+1:n
            ui = u[i] - A[k,i] * u1
            A[k,i] = A[k,i] * c + u[i] * s
            u[i] = ui
        end
        u[k] = 0
        f = d1 * d2 / dis
    end
    ldu
end

"""
    ldu_update!(ldu::LDUPivoted, u::Vector, v::Vector, f::Real)

Given the LDU factorization `ldu: P' * A * P = L * D * L'` of a symmetric real or hermitian matrix `A`, update
`ldu` to obtain a factorization of `A + (u * v' + v * u') * f` without changing `L`.

This attempt may fail with a `SingularException` if the new pivot elements becomes zero.
"""
function ldu_update!(ldu::LDUPivoted, u::AbstractVector, v::AbstractVector, f::Real)
    un = maximum(abs, u)
    vn = maximum(abs, v)
    (iszero(un) || iszero(vn) ) && return ldu
    g = sqrtapp(un / vn)
    u = u / g
    v = v * g
    f = f / 2
    upv = u + v
    umv = u - v
    ldu_update!(ldu, upv, f)
    ldu_update!(ldu, umv, -f)
end

splitexp(a::AbstractFloat) = frexp(a)
function splitexp(a::Union{Integer,Rational})
    q, x = frexp(float(a))
    x >= 0 ? a / 2^x : a * 2^(-x), x
end

"""
    sqrtapp(x)

Rational approximation of `sqrt(x)`.
"""
function sqrtapp(a::Real)
    q, x = splitexp(a)
    if isodd(x)
        q += q
        x -= 1
    end
    x ÷= 2
    c = (12 - (3 - q)^2) / 8
    # c = (q / c + c) / 2
    x >= 0 ? c * 2^x : c / 2^(-x)
end
