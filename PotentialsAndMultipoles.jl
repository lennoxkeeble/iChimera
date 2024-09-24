# we write covariant vectors with underscores (e.g., for BL coordinates x^μ = xBL x_μ = x_BL)
module PotentialsAndMultipoles
using LinearAlgebra
using Combinatorics
using BSplineKit
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using LsqFit
using ..Kerr
using ..BLTimeEvolution
using ..FourierFitGSL
using ..CircularNonEquatorial
using ..Deriv2, ..Deriv3, ..Deriv4, ..Deriv5, ..Deriv6
using ..ParameterizedDerivs, ..MinoTimeDerivs
import ..HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ..HarmonicCoords
using ..SymmetricTensors
using JLD2
using FileIO
using ..AnalyticCoordinateDerivs
using ..AnalyticMultipoleDerivs

# define some useful functions
otimes4d(a::Float64, b::Vector) = [a^2 a*b[1] a*b[2] a*b[3]; a*b[1] b[1]*b[1] b[1]*b[2] b[1]*b[3]; a*b[2] b[2]*b[1] b[2]*b[2] b[2]*b[3]; a*b[3] b[3]*b[1] b[3]*b[2] b[3]*b[3]]    # tensor product of two vectors
otimes(a::Vector, b::Vector) = [a[i] * b[j] for i=1:size(a, 1), j=1:size(b, 1)]    # tensor product of two vectors
otimes(a::Vector) = [a[i] * a[j] for i=1:size(a, 1), j=1:size(a, 1)]    # tensor product of a vector with itself
dot3d(u::AbstractVector{Float64}, v::AbstractVector{Float64}) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3]
norm2_3d(u::AbstractVector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
norm_3d(u::AbstractVector{Float64}) = sqrt(norm2_3d(u))
dot4d(u::AbstractVector{Float64}, v::AbstractVector{Float64}) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4]
norm2_4d(u::AbstractVector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3] + u[4] * u[4]
norm_4d(u::AbstractVector{Float64}) = sqrt(norm2_4d(u))

const ημν::Matrix{Float64} = [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]    # minkowski metric
const ηij::Matrix{Float64} = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]    # spatial part of minkowski metric
const t0::Float64 = 0.0
δ(x,y) = ==(x,y)   # delta function

# define vector and scalar potentials for self-force calculation - underscore denotes covariant indices
K(xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_tt_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) + 1.0                   # outputs K00 (Eq. 54)
K_i(xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_tr_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)                       # outputs Ki vector, i.e., Ki for i ∈ {1, 2, 3} (Eq. 55)
K_ij(xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_rr_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) - ηij                # outputs Kij matrix (Eq. 56)
K_μν(xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_μν_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) - ημν                # outputs Kμν matrix
Q(xH::AbstractArray, a::Float64, M::Float64, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = gTT_H(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) + 1.0                          # outputs Q^00 (Eq. 54)
Qi(xH::AbstractArray, a::Float64, M::Float64, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = gTR_H(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)                               # outputs Q^i vector, i.e., Q^i for i ∈ {1, 2, 3} (Eq. 55)
Qij(xH::AbstractArray, a::Float64, M::Float64, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = gRR_H(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) - ηij                        # outputs diagonal of Q^ij matrix (Eq. 56)
Qμν(xH::AbstractArray, a::Float64, M::Float64, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = gμν_H(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) - ημν                        # outputs Qμν matrix

# ### NewKludge derivatives of the potential as written in the paper ###

# # define partial derivatives of K (in harmonic coordinates)
# # ∂ₖK: outputs float
# function ∂K_∂xk(xH::AbstractArray, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int)   # Eq. A12
#     ∂K=0.0
#     @inbounds for μ=1:4
#         for i=1:3
#             ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, μ) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, 1, i+1) * jBLH[i, k]   # i → i + 1 to go from spatial indices to spacetime indices
#         end
#     end
#     return ∂K
# end

# # ∂ₖKᵢ: outputs float. Note: rH = norm(xH).
# function ∂Ki_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int)   # Eq. A13
#     ∂K=0.0
#     @inbounds for m=1:3   # start with iteration over m to not over-count last terms
#         ∂K += 2.0 * g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, m+1) * HarmonicCoords.HessBLH(xH, rH, a, M, m)[i, k]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices
#         @inbounds for μ=1:4, n=1:3
#             ∂K += ((g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, m+1, n+1) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, m+1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, 1, n+1))/2) * jBLH[n, k] * jBLH[m, i]   # first term of Eq. A13
#         end
#     end
#     return ∂K
# end

# # ∂ₖKᵢⱼ: outputs float. Note: rH = norm(xH).
# function ∂Kij_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int, j::Int)   # Eq. A14
#     ∂K=0.0
#     @inbounds for m=1:3
#         for l=1:3   # iterate over m and l first to avoid over-counting
#             ∂K += 2.0 * g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, l+1, m+1) * HarmonicCoords.HessBLH(xH, rH, a, M, m)[j, k] * jBLH[l, i]  # last term Eq. A14
#             @inbounds for μ=1:4, n=1:3
#                 ∂K += ((g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, l+1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, m+1, n+1) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, m+1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, l+1, n+1))/2) * jBLH[n, k] * jBLH[m, j] * jBLH[l, i]   # first term of Eq. A14
#             end
#         end
#     end
#     return ∂K
# end

## Corrected NewKludge derivatives of the potential ###

# define partial derivatives of K (in harmonic coordinates)
# ∂ₖK: outputs float
@views function ∂K_∂xk(xH::AbstractArray, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int)   # Eq. A12
    return 2 * (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 2) * jBLH[1, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 3) * jBLH[2, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 4) * jBLH[3, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 2) * jBLH[1, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 3) * jBLH[2, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 4) * jBLH[3, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 2) * jBLH[1, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 3) * jBLH[2, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 4) * jBLH[3, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 2) * jBLH[1, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 3) * jBLH[2, k] +
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 4) * jBLH[3, k])
end


# ∂ₖKᵢ: outputs float. Note: rH = norm(xH).
@views function ∂Ki_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int)   # Eq. A13
    # m = 1
    ∂K = g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * HarmonicCoords.HessBLH(xH, rH, a, M, 1)[k, i]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices

    # μ = 1, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 2)) * jBLH[1, k] * jBLH[1, i]
    # μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 3)) * jBLH[2, k] * jBLH[1, i]
    # μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 4)) * jBLH[3, k] * jBLH[1, i]
    
    # μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 2)) * jBLH[1, k] * jBLH[1, i]
    # μ = 2, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 3)) * jBLH[2, k] * jBLH[1, i]
    # μ = 2, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 4)) * jBLH[3, k] * jBLH[1, i]

    # μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 2)) * jBLH[1, k] * jBLH[1, i]
    # μ = 3, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 3)) * jBLH[2, k] * jBLH[1, i]
    # μ = 3, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 4)) * jBLH[3, k] * jBLH[1, i]
    
    # μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 2)) * jBLH[1, k] * jBLH[1, i]
    # μ = 4, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 3)) * jBLH[2, k] * jBLH[1, i]
    # μ = 4, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 4)) * jBLH[3, k] * jBLH[1, i]

    # m = 2
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * HarmonicCoords.HessBLH(xH, rH, a, M, 2)[k, i]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices

    # μ = 1, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 2)) * jBLH[1, k] * jBLH[2, i]
    # μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 3)) * jBLH[2, k] * jBLH[2, i]
    # μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 4)) * jBLH[3, k] * jBLH[2, i]
    
    # μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 2)) * jBLH[1, k] * jBLH[2, i]
    # μ = 2, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 3)) * jBLH[2, k] * jBLH[2, i]
    # μ = 2, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 4)) * jBLH[3, k] * jBLH[2, i]

    # μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 2)) * jBLH[1, k] * jBLH[2, i]
    # μ = 3, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 3)) * jBLH[2, k] * jBLH[2, i]
    # μ = 3, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 4)) * jBLH[3, k] * jBLH[2, i]
    
    # μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 2)) * jBLH[1, k] * jBLH[2, i]
    # μ = 4, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 3)) * jBLH[2, k] * jBLH[2, i]
    # μ = 4, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 4)) * jBLH[3, k] * jBLH[2, i]
    
    # m = 3
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * HarmonicCoords.HessBLH(xH, rH, a, M, 3)[k, i]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices

    # μ = 1, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 2)) * jBLH[1, k] * jBLH[3, i]
    # μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 3)) * jBLH[2, k] * jBLH[3, i]
    # μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 1, 4)) * jBLH[3, k] * jBLH[3, i]
    
    # μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 2)) * jBLH[1, k] * jBLH[3, i]
    # μ = 2, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 3)) * jBLH[2, k] * jBLH[3, i]
    # μ = 2, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 1, 4)) * jBLH[3, k] * jBLH[3, i]

    # μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 2)) * jBLH[1, k] * jBLH[3, i]
    # μ = 3, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 3)) * jBLH[2, k] * jBLH[3, i]
    # μ = 3, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 1, 4)) * jBLH[3, k] * jBLH[3, i]
    
    # μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 2) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 2)) * jBLH[1, k] * jBLH[3, i]
    # μ = 4, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 3) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 3)) * jBLH[2, k] * jBLH[3, i]
    # μ = 4, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 4) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 1, 4)) * jBLH[3, k] * jBLH[3, i]

    return ∂K
end


# ∂ₖKᵢⱼ: outputs float. Note: rH = norm(xH).
@views function ∂Kij_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int, j::Int)   # Eq. A14
    # m = 1

    # l = 1, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 1)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * (Hess_BLH_l[k, i] * jBLH[1, j] + Hess_BLH_l[k, j] * jBLH[1, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14

    # l = 1, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    

    # l = 1, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    

    # l = 1, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[1, j]   # first term of Eq. A14
    
    # l = 2, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 2)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * (Hess_BLH_l[k, i] * jBLH[1, j] + Hess_BLH_l[k, j] * jBLH[1, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14

    # l = 2, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    

    # l = 2, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    

    # l = 2, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[1, j]   # first term of Eq. A14

    # l = 3, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 3)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * (Hess_BLH_l[k, i] * jBLH[1, j] + Hess_BLH_l[k, j] * jBLH[1, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14

    # l = 3, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    

    # l = 3, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    

    # l = 3, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[1, j]   # first term of Eq. A14

    # m = 2

    # l = 1, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 1)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * (Hess_BLH_l[k, i] * jBLH[2, j] + Hess_BLH_l[k, j] * jBLH[2, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14

    # l = 1, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    

    # l = 1, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    

    # l = 1, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[2, j]   # first term of Eq. A14
    
    # l = 2, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 2)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * (Hess_BLH_l[k, i] * jBLH[2, j] + Hess_BLH_l[k, j] * jBLH[2, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14

    # l = 2, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    

    # l = 2, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    

    # l = 2, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[2, j]   # first term of Eq. A14

    # l = 3, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 3)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * (Hess_BLH_l[k, i] * jBLH[2, j] + Hess_BLH_l[k, j] * jBLH[2, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14

    # l = 3, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    

    # l = 3, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    

    # l = 3, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[2, j]   # first term of Eq. A14

    # m = 3

    # l = 1, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 1)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * (Hess_BLH_l[k, i] * jBLH[3, j] + Hess_BLH_l[k, j] * jBLH[3, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14

    # l = 1, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    

    # l = 1, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    

    # l = 1, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 2)) * jBLH[1, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 3)) * jBLH[2, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 1, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 2, 4)) * jBLH[3, k] * jBLH[1, i] * jBLH[3, j]   # first term of Eq. A14
    
    # l = 2, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 2)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * (Hess_BLH_l[k, i] * jBLH[3, j] + Hess_BLH_l[k, j] * jBLH[3, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14

    # l = 2, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    

    # l = 2, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    

    # l = 2, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 2)) * jBLH[1, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 3)) * jBLH[2, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 2, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 3, 4)) * jBLH[3, k] * jBLH[2, i] * jBLH[3, j]   # first term of Eq. A14

    # l = 3, μ = 1, n = 1
    Hess_BLH_l = HarmonicCoords.HessBLH(xH, rH, a, M, 3)
    ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * (Hess_BLH_l[k, i] * jBLH[3, j] + Hess_BLH_l[k, j] * jBLH[3, i])
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 1, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14

    # l = 3, μ = 2, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 2, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    

    # l = 3, μ = 3, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 3, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    

    # l = 3, μ = 4, n = 1
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 2) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 2)) * jBLH[1, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 2
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 3) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 3)) * jBLH[2, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    # l = 3, μ = 1, n = 3
    ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 4) + 
    g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, 4, 4, 4)) * jBLH[3, k] * jBLH[3, i] * jBLH[3, j]   # first term of Eq. A14
    return ∂K
end

# ∂ₖKᵢⱼ: outputs float. Note: rH = norm(xH).
function ∂Kij_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int, j::Int)   # Eq. A14
    ∂K=0.0
    @inbounds for m=1:3
        for l=1:3   # iterate over m and l first to avoid over-counting
            ∂K += g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, l+1, m+1) * (HarmonicCoords.HessBLH(xH, rH, a, M, l)[k, i] * jBLH[m, j] + HarmonicCoords.HessBLH(xH, rH, a, M, l)[k, j] * jBLH[m, i])  # last term Eq. A14
            @inbounds for μ=1:4, n=1:3
                ∂K += (g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, l+1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, m+1, n+1) + g_μν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, m+1) * Γαμν(t0, xBL[1], xBL[2], xBL[3], a, M, μ, l+1, n+1)) * jBLH[n, k] * jBLH[l, i] * jBLH[m, j]   # first term of Eq. A14
            end
        end
    end
    return ∂K
end

# define GR Γ factor, v_H = contravariant velocity in harmonic coordinates
Γ(vH::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = 1.0 / sqrt(1.0 - norm2_3d(vH) - K(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) - 2.0 * dot(K_i(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ), vH) - transpose(vH) * K_ij(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) * vH)   # Eq. A3

# define projection operator
Pαβ(vH::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = ημν + Qμν(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) + Γ(vH, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * otimes4d(1.0, vH)   # contravariant, Eq. A1
P_αβ(vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) =  ημν + K_μν(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) + Γ(vH, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * otimes4d(1.0, vH)   # cοvariant, Eq. A2 (note that we take both contravariant and covariant velocities as arguments)

# define STF projections 
STF(u::Vector, i::Int, j::Int) = u[i] * u[j] - dot(u, u) * δ(i, j) /3.0                                                                     # STF projection x^{<ij>}
STF(u::Vector, v::Vector, i::Int, j::Int) = (u[i] * v[j] + u[j] * v[i])/2.0 - dot(u, v)* δ(i, j) /3.0                                       # STF projection of two distinct vectors
STF(u::Vector, i::Int, j::Int, k::Int) = u[i] * u[j] * u[k] - (1.0/5.0) * dot(u, u) * (δ(i, j) * u[k] + δ(j, k) * u[i] + δ(k, i) * u[j])    # STF projection x^{<ijk>} (Eq. 46)

# define mass-ratio parameter
η(q::Float64) = q/((1+q)^2)   # q = mass ratio
mTot(m::Float64, M::Float64) = m + M;
δm(m::Float64, M::Float64) = M - m;

# TO-DO: SINCE WE SET M=1 and m=q (currently, at least) WE SHOULD, FOR CLARITY, REMOVE M FROM THESE EQUATIONS AND WRITE m->q

# define multipole moments
M_ij(x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int) = η(m/M) * (1.0+m) * STF(x_H, i, j)  # quadrupole mass moment Eq. 48
ddotMij(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int) = η(m/M) * (1.0+m) * ((-2.0δ(i, j)/3.0) * (dot(x_H, a_H) +
dot(v_H, v_H)) + x_H[j] * a_H[i] + 2.0 * v_H[i] * v_H[j] + x_H[i] * a_H[j])   # Eq. 7.17

M_ijk(x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) = η(m/M) * (1.0 - m) * STF(x_H, i, j, k)  # octupole mass moment Eq. 48
ddotMijk(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) = -η(m/M) * (1.0 - m) * ((-4.0/5.0) * (dot(x_H, v_H)) *
(δ(i, j) * v_H[k] + δ(j, k) * v_H[i] + δ(k, i) * v_H[j]) - (2.0/5.0) * (dot(x_H, a_H) + dot(v_H, v_H)) * (δ(i, j) * x_H[k] + δ(j, k) * x_H[i] + δ(k, i) * x_H[j]) -
(1.0/5.0) * dot(x_H, x_H) * (δ(i, j) * a_H[k] + δ(j, k) * a_H[i] + δ(k, i) * a_H[j]) + 2.0 * v_H[k] * (x_H[j] * v_H[i] + x_H[i] * v_H[j]) + x_H[k] * (x_H[j] * a_H[i] +
2.0 * v_H[i] * v_H[j] + x_H[i] * a_H[j]) + x_H[i] * x_H[j] * a_H[k])   # Eq. 7.19

# second derivative of Mijkl, as defined in Eq. 85 (LONG EXPRESSION COPIED FROM MMA)
ddotMijkl(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int, l::Int) = (1.0+m)*η(m/M)*(2.0*(x_H[j]*v_H[i] +
x_H[i]*v_H[j])*(x_H[l]*v_H[k] + x_H[k]*v_H[l]) - (4.0*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3])*(x_H[k]*δ(j,l)*v_H[i] + x_H[j]*δ(k,l)*v_H[i] + x_H[k]*δ(i,l)*v_H[j] +
x_H[i]*δ(k,l)*v_H[j] + x_H[j]*δ(i,l)*v_H[k] + x_H[i]*δ(j,l)*v_H[k] + x_H[l]*(δ(j,k)*v_H[i] + δ(i,k)*v_H[j] + δ(i,j)*v_H[k]) + (x_H[k]*δ(i,j) + x_H[j]*δ(i,k) +
x_H[i]*δ(j,k))*v_H[l]))/7. - (2.0*(x_H[i]*x_H[l]*δ(j,k) + x_H[k]*(x_H[l]*δ(i,j) + x_H[j]*δ(i,l) + x_H[i]*δ(j,l)) + x_H[j]*(x_H[l]*δ(i,k) + x_H[i]*δ(k,l)))*(v_H[1]^2 +
v_H[2]^2 + v_H[3]^2 + x_H[1]*a_H[1] + x_H[2]*a_H[2] + x_H[3]*a_H[3]))/7. + ((δ(i,l)*δ(j,k) + δ(i,k)*δ(j,l) + δ(i,j)*δ(k,l))*(8.0*(x_H[1]*v_H[1] + x_H[2]*v_H[2] +
x_H[3]*v_H[3])^2 + 4.0*(x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*(v_H[1]^2 + v_H[2]^2 + v_H[3]^2 + x_H[1]*a_H[1] + x_H[2]*a_H[2] + x_H[3]*a_H[3])))/35. +
x_H[k]*x_H[l]*(2.0*v_H[i]*v_H[j] + x_H[j]*a_H[i] + x_H[i]*a_H[j]) + x_H[i]*x_H[j]*(2.0*v_H[k]*v_H[l] + x_H[l]*a_H[k] + x_H[k]*a_H[l]) - ((x_H[1]^2 + x_H[2]^2 +
x_H[3]^2)*(δ(k,l)*(2.0*v_H[i]*v_H[j] + x_H[j]*a_H[i] + x_H[i]*a_H[j]) + δ(j,l)*(2.0*v_H[i]*v_H[k] + x_H[k]*a_H[i] + x_H[i]*a_H[k]) + δ(i,l)*(2.0*v_H[j]*v_H[k] +
x_H[k]*a_H[j] + x_H[j]*a_H[k]) + δ(j,k)*(2.0*v_H[i]*v_H[l] + x_H[l]*a_H[i] + x_H[i]*a_H[l]) + δ(i,k)*(2.0*v_H[j]*v_H[l] + x_H[l]*a_H[j] + x_H[j]*a_H[l]) +
δ(i,j)*(2.0*v_H[k]*v_H[l] + x_H[l]*a_H[k] + x_H[k]*a_H[l])))/7.)

# define some objects useful for efficient calculation of current quadrupole and its derivatives
const ρ::Vector{Int} = [1, 2, 3]   # spacial indices
const spatial_indices_3::Array = [[x, y, z] for x=1:3, y=1:3, z=1:3]   # array where each element kl = [[k, l, i] for i=1:3]
const εkl::Array{Vector} = [[levicivita(spatial_indices_3[k, l, i]) for i = 1:3] for k=1:3, l=1:3]   # array where each element kl = [e_{kli} for i=1:3]

function S_ij(x_H::AbstractArray, xH::AbstractArray, vH::AbstractArray, m::Float64, M::Float64, i::Int, j::Int)   # Eq. 49
    s_ij=0.0
    @inbounds for k=1:3
        for l=1:3
            s_ij +=  STF(εkl[k, l], x_H, i, j) * xH[k] * vH[l]
        end
    end
    return η(m/M) * (1.0 - m) * s_ij
end

function dotSij(aH::AbstractArray, v_H::AbstractArray, vH::AbstractArray, x_H::AbstractArray, xH::AbstractArray, m::Float64, M::Float64, i::Int, j::Int)
    S=0.0
    @inbounds for k=1:3
        for l=1:3
            S += -2.0δ(i, j) * (vH[l] * (xH[k] * dot(εkl[k, l], v_H) + vH[k] * dot(εkl[k, l], x_H)) + xH[k] * aH[l] * dot(εkl[k, l], x_H)) + 3.0 * vH[l] * (εkl[k, l][i] * (xH[k] * v_H[j] + x_H[j] * vH[k]) + εkl[k, l][j] * (xH[k] * v_H[i] + x_H[i] * vH[k])) + 3.0 * xH[k] * aH[l] * (εkl[k, l][i] * x_H[j] + εkl[k, l][j] * x_H[i])
        end
    end
    return -η(m/M) * (1.0 - m) * S / 6.0
end


# first derivative of Sijk, as defined in Eq. 86 (LONG EXPRESSION COPIED FROM MMA)
dotSijk(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) =((1.0+m)*η(m/M)*((δ(j,k)*(-2.0*x_H[i]*(x_H[1]*
εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] +
x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 +
x_H[2]^2 + x_H[3]^2)*εkl[1,1][k]) + 5.0*(x_H[i]*x_H[k]*εkl[1,1][j] + x_H[j]*(x_H[k]*εkl[1,1][i] + x_H[i]*εkl[1,1][k])))*v_H[1]^2 + (δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] +
x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) -
(x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][k]) +
5.0*(x_H[i]*x_H[k]*εkl[1,2][j] + x_H[j]*(x_H[k]*εkl[1,2][i] + x_H[i]*εkl[1,2][k])))*v_H[1]*v_H[2] + (δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] +
x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 +
x_H[2]^2 + x_H[3]^2)*εkl[2,1][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][k]) +
5.0*(x_H[i]*x_H[k]*εkl[2,1][j] + x_H[j]*(x_H[k]*εkl[2,1][i] + x_H[i]*εkl[2,1][k])))*v_H[1]*v_H[2] + (δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] +
x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 +
x_H[2]^2 + x_H[3]^2)*εkl[2,2][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][k]) +
5.0*(x_H[i]*x_H[k]*εkl[2,2][j] + x_H[j]*(x_H[k]*εkl[2,2][i] + x_H[i]*εkl[2,2][k])))*v_H[2]^2 + x_H[1]*v_H[1]*(-2.0*δ(j,k)*(εkl[1,1][i]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] +
x_H[3]*v_H[3]) + x_H[i]*(εkl[1,1][1]*v_H[1] + εkl[1,1][2]*v_H[2] + εkl[1,1][3]*v_H[3]) + (x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3])*v_H[i]) -
2.0*δ(k,i)*(εkl[1,1][j]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,1][1]*v_H[1] + εkl[1,1][2]*v_H[2] + εkl[1,1][3]*v_H[3]) + (x_H[1]*εkl[1,1][1] +
x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3])*v_H[i]) - 2.0*δ(i,j)*(εkl[1,1][k]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,1][1]*v_H[1] + εkl[1,1][2]*v_H[2] +
εkl[1,1][3]*v_H[3]) + (x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3])*v_H[i]) + 5.0*(εkl[1,1][k]*(x_H[j]*v_H[i] + x_H[i]*v_H[j]) + x_H[k]*(εkl[1,1][j]*v_H[i] +
εkl[1,1][i]*v_H[j]) + (x_H[j]*εkl[1,1][i] + x_H[i]*εkl[1,1][j])*v_H[k])) + x_H[1]*v_H[2]*(-2.0*δ(j,k)*(εkl[1,2][i]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) +
x_H[i]*(εkl[1,2][1]*v_H[1] + εkl[1,2][2]*v_H[2] + εkl[1,2][3]*v_H[3]) + (x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3])*v_H[i]) -
2.0*δ(k,i)*(εkl[1,2][j]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,2][1]*v_H[1] + εkl[1,2][2]*v_H[2] + εkl[1,2][3]*v_H[3]) +
(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3])*v_H[i]) - 2.0*δ(i,j)*(εkl[1,2][k]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) +
x_H[i]*(εkl[1,2][1]*v_H[1] + εkl[1,2][2]*v_H[2] + εkl[1,2][3]*v_H[3]) + (x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3])*v_H[i]) +
5.0*(εkl[1,2][k]*(x_H[j]*v_H[i] + x_H[i]*v_H[j]) + x_H[k]*(εkl[1,2][j]*v_H[i] + εkl[1,2][i]*v_H[j]) + (x_H[j]*εkl[1,2][i] + x_H[i]*εkl[1,2][j])*v_H[k])) +
x_H[2]*v_H[1]*(-2.0*δ(j,k)*(εkl[2,1][i]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,1][1]*v_H[1] + εkl[2,1][2]*v_H[2] + εkl[2,1][3]*v_H[3]) +
(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3])*v_H[i]) - 2.0*δ(k,i)*(εkl[2,1][j]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) +
x_H[i]*(εkl[2,1][1]*v_H[1] + εkl[2,1][2]*v_H[2] + εkl[2,1][3]*v_H[3]) + (x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3])*v_H[i]) -
2.0*δ(i,j)*(εkl[2,1][k]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,1][1]*v_H[1] + εkl[2,1][2]*v_H[2] + εkl[2,1][3]*v_H[3]) + (x_H[1]*εkl[2,1][1] +
x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3])*v_H[i]) + 5.0*(εkl[2,1][k]*(x_H[j]*v_H[i] + x_H[i]*v_H[j]) + x_H[k]*(εkl[2,1][j]*v_H[i] + εkl[2,1][i]*v_H[j]) +
(x_H[j]*εkl[2,1][i] + x_H[i]*εkl[2,1][j])*v_H[k])) + x_H[2]*v_H[2]*(-2.0*δ(j,k)*(εkl[2,2][i]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,2][1]*v_H[1] +
εkl[2,2][2]*v_H[2] + εkl[2,2][3]*v_H[3]) + (x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3])*v_H[i]) - 2.0*δ(k,i)*(εkl[2,2][j]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] +
x_H[3]*v_H[3]) + x_H[i]*(εkl[2,2][1]*v_H[1] + εkl[2,2][2]*v_H[2] + εkl[2,2][3]*v_H[3]) + (x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3])*v_H[i]) -
2.0*δ(i,j)*(εkl[2,2][k]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,2][1]*v_H[1] + εkl[2,2][2]*v_H[2] + εkl[2,2][3]*v_H[3]) + (x_H[1]*εkl[2,2][1] +
x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3])*v_H[i]) + 5.0*(εkl[2,2][k]*(x_H[j]*v_H[i] + x_H[i]*v_H[j]) + x_H[k]*(εkl[2,2][j]*v_H[i] + εkl[2,2][i]*v_H[j]) + (x_H[j]*εkl[2,2][i] +
x_H[i]*εkl[2,2][j])*v_H[k])) + x_H[1]*(δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][i]) +
δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] +
x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][k]) + 5.0*(x_H[i]*x_H[k]*εkl[1,1][j] + x_H[j]*(x_H[k]*εkl[1,1][i] +
x_H[i]*εkl[1,1][k])))*a_H[1] + x_H[2]*(δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][i]) +
δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] +
x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][k]) + 5.0*(x_H[i]*x_H[k]*εkl[2,1][j] + x_H[j]*(x_H[k]*εkl[2,1][i] +
x_H[i]*εkl[2,1][k])))*a_H[1] + x_H[1]*(δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][i]) +
δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] +
x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][k]) + 5.0*(x_H[i]*x_H[k]*εkl[1,2][j] + x_H[j]*(x_H[k]*εkl[1,2][i] +
x_H[i]*εkl[1,2][k])))*a_H[2] + x_H[2]*(δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][i]) +
δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] +
x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][k]) + 5.0*(x_H[i]*x_H[k]*εkl[2,2][j] + x_H[j]*(x_H[k]*εkl[2,2][i] +
x_H[i]*εkl[2,2][k])))*a_H[2]))/15.


end