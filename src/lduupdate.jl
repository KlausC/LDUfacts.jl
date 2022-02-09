
function ldu_update!(ldu::LDU, u::AbstractVector, f::Real)
    A = ldu.data
    n = checksquare(A)
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
