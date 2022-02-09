using LDUFacts
using LinearAlgebra

using Test

@testset "LDUFacts.jl" begin
    include("ldufact.jl")
    include("lduupdate.jl")
end
