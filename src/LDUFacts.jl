module LDUFacts

using LinearAlgebra

using LinearAlgebra: checksquare
using LinearAlgebra: PivotingStrategy, NoPivot
import LinearAlgebra: rmul!, lmul!, adjoint
import Base: (*)

export ldu, ldu!, ldu_update!
export NoPivot, DiagonalPivot, FullPivot

struct LDUPivoted{S,T<:AbstractMatrix{S},F<:Real}
    factors::T
    piv::Vector{Int}
    rank::Int
    tol::F
    info::Int
    pam::Vector{Int}
    ame::Vector{S}
end

struct LDUPerm{T,S}
    piv::Vector{Int}
    pam::Vector{Int}
    ame::S
    LDUPerm(piv, pam, ame::S) where {T,S<:AbstractVector{T}} = new{T,S}(piv, pam, ame)
end

struct DiagonalPivot <: PivotingStrategy end
struct FullPivot <: PivotingStrategy end

include("ldufact.jl")
include("lduupdate.jl")

end # module