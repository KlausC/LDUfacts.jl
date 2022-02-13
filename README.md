# LDUFacts

[![Build Status][gha-img]][gha-url]     [![Coverage Status][codecov-img]][codecov-url]

## Introduction

This package is dedicated to the L-D-Lt factorization of symmetric real and Hermitian matrices.

In contrast to Cholesky factorization, which is appropriate for most use cases, it has the following features:

- handle also the indefinite case (Cholesky requires positive (semi)-definite)
- square-root free algorithm
- exact operation with real and complex rational types
- exact operation with user-defined field types
- option of full pivot search

## Algorithm

Factors `P, L, D` of matrix `A` are found in a way, that `P' * A * P = L * D * L'`.

Here `D` is diagonal and real, `L` is lower unit triangular, and `P` is a 'simple' square matrix.
With diagonal pivot search, `P` is a permutation matrix, with full pivot search up to `n-1` nonzero additional matrix elements may occur.
Without pivot search, `P` is the unit matrix.

When not a full pivot search is done, it is possible, that the algorithm fails before the factors are found.

For positive definite matrices, the no-pivot version is guaranteed to succeed. For positive semi-definite matrices,
a diagonal pivot search always succeeds, for infefinite matrices, a full pivot search is required for that.

There are examples of regular matrices, (e.g. `[0. 1; 1 0]`), which do not have any factorization where `P` is a permutation. For those cases
the full pivot search is required.

## Usage

    ]add LDUFacts

    using LDUFacts
    
    A = Matrix(Symmetric(rand(5, 5)))
    la = ldu(A, FullPivot())
    
    issuccess(la)
    rank(la)
    la.L
    la.d
    la.P' * A * la.P - la.L * la.D * la.L' # should be close to zero

[gha-img]: https://github.com/KlausC/LDUFacts.jl/actions/workflows/CI.yml/badge.svg
[gha-url]: https://github.com/KlausC/LDUFacts.jl/actions/workflows/CI.yml

[codecov-img]: https://codecov.io/gh/KlausC/LDUFacts.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/KlausC/LDUFacts.jl
