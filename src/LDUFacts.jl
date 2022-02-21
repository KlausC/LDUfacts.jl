module LDUFacts

using LinearAlgebra

using LinearAlgebra: checksquare
using LinearAlgebra: PivotingStrategy, NoPivot
import LinearAlgebra: rmul!, lmul!, ldiv!, rdiv!, adjoint, det, logdet, logabsdet
import Base: *, \, /, size, inv

export ldu, ldu!, ldu_update!
export NoPivot, DiagonalPivot, FullPivot, PivotLike

struct LDUPivoted{T,V<:AbstractMatrix{T},F<:Real} <: Factorization{T}
    factors::V
    piv::Vector{Int}
    rank::Int
    tol::F
    info::Int
    pam::Vector{Int}
    ame::Vector{T}
end

struct LDUPerm{T}
    piv::Vector{Int}
    pam::Vector{Int}
    ame::Vector{T}
    LDUPerm(piv, pam, ame::AbstractVector{T}) where T = new{T}(piv, pam, Vector{T}(ame))
end

struct DiagonalPivot <: PivotingStrategy end
struct FullPivot <: PivotingStrategy end
struct PivotLike{T} <: PivotingStrategy
    piv::Vector{Int}
    pam::Vector{Int}
    ame::Vector{T}
    PivotLike(la::LDUPivoted{T}) where T = new{T}(la.piv, la.pam, la.ame)
end

const SATMatrix{T} = Union{StridedMatrix{T}, Adjoint{T,<:StridedMatrix}, Transpose{T,<:StridedMatrix}}

include("ldufact.jl")
include("lduupdate.jl")

end # module