# we write covariant vectors with underscores (e.g., for BL coordinates x^μ = xBL x_μ = x_BL)
include("/home/lkeeble/GRSuite/Testing/Test_modules/ParameterizedDerivs.jl");

module SelfForce_numerical
using LinearAlgebra
using Combinatorics
using BSplineKit
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using LsqFit
using Dierckx
using ..Kerr
using ..ParameterizedDerivs
using ..MinoTimeDerivs
using ..Deriv2
using ..Deriv3
using ..Deriv4
using ..Deriv5
using ..Deriv6
using ..FiniteDiff_5
using ..HJEvolution
using ..MinoEvolution
using ..ConstructSymmetricArrays
using ..SelfForce
import ..HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ..HarmonicCoords
using JLD2
using FileIO

# define some useful functions
otimes(a::Vector, b::Vector) = [a[i] * b[j] for i=1:size(a, 1), j=1:size(b, 1)]    # tensor product of two vectors
otimes(a::Vector) = [a[i] * a[j] for i=1:size(a, 1), j=1:size(a, 1)]    # tensor product of a vector with itself
dot3d(u::Vector{Float64}, v::Vector{Float64}) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3]
norm2_3d(u::Vector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
norm_3d(u::Vector{Float64}) = sqrt(norm2_3d(u))
dot4d(u::Vector{Float64}, v::Vector{Float64}) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4]
norm2_4d(u::Vector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3] + u[4] * u[4]
norm_4d(u::Vector{Float64}) = sqrt(norm2_4d(u))

ημν = [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]    # minkowski metric
ηij = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]    # spatial part of minkowski metric
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

# ## NewKludge derivatives of the potential as written in the paper ###

# # define partial derivatives of K (in harmonic coordinates)
# # ∂ₖK: outputs float
# function ∂K_∂xk(xH::AbstractArray, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int)   # Eq. A12
#     ∂K=0.0
#     @inbounds for μ=1:4
#         for i=1:3
#             ∂K += g_μν(0., xBL..., a, M, 1, μ) * Γαμν(0., xBL..., a, M, μ, 1, i+1) * jBLH[i, k]   # i → i + 1 to go from spatial indices to spacetime indices
#         end
#     end
#     return ∂K
# end

# # ∂ₖKᵢ: outputs float. Note: rH = norm(xH).
# function ∂Ki_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int)   # Eq. A13
#     ∂K=0.0
#     @inbounds for m=1:3   # start with iteration over m to not over-count last terms
#         ∂K += 2.0 * g_μν(0., xBL..., a, M, 1, m+1) * HarmonicCoords.HessBLH(xH, rH, a, M, m)[i, k]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices
#         @inbounds for μ=1:4, n=1:3
#             ∂K += ((g_μν(0., xBL..., a, M, μ, 1) * Γαμν(0., xBL..., a, M, μ, m+1, n+1) + g_μν(0., xBL..., a, M, μ, m+1) * Γαμν(0., xBL..., a, M, μ, 1, n+1))/2) * jBLH[n, k] * jBLH[m, i]   # first term of Eq. A13
#         end
#     end
#     return ∂K
# end

# # ∂ₖKᵢⱼ: outputs float. Note: rH = norm(xH).
# function ∂Kij_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int, j::Int)   # Eq. A14
#     ∂K=0.0
#     @inbounds for m=1:3
#         for l=1:3   # iterate over m and l first to avoid over-counting
#             ∂K += 2.0 * g_μν(0., xBL..., a, M, l+1, m+1) * HarmonicCoords.HessBLH(xH, rH, a, M, m)[j, k] * jBLH[l, i]  # last term Eq. A14
#             @inbounds for μ=1:4, n=1:3
#                 ∂K += ((g_μν(0., xBL..., a, M, μ, l+1) * Γαμν(0., xBL..., a, M, μ, m+1, n+1) + g_μν(0., xBL..., a, M, μ, m+1) * Γαμν(0., xBL..., a, M, μ, l+1, n+1))/2) * jBLH[n, k] * jBLH[m, j] * jBLH[l, i]   # first term of Eq. A14
#             end
#         end
#     end
#     return ∂K
# end

### Corrected NewKludge derivatives of the potential ###

# define partial derivatives of K (in harmonic coordinates)
# ∂ₖK: outputs float
function ∂K_∂xk(xH::AbstractArray, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int)   # Eq. A12
    ∂K=0.0
    @inbounds for μ=1:4
        for i=1:3
            ∂K += 2 * g_μν(0., xBL..., a, M, 1, μ) * Γαμν(0., xBL..., a, M, μ, 1, i+1) * jBLH[i, k]   # i → i + 1 to go from spatial indices to spacetime indices
        end
    end
    return ∂K
end

# ∂ₖKᵢ: outputs float. Note: rH = norm(xH).
function ∂Ki_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int)   # Eq. A13
    ∂K=0.0
    @inbounds for m=1:3   # start with iteration over m to not over-count last terms
        ∂K += g_μν(0., xBL..., a, M, 1, m+1) * HarmonicCoords.HessBLH(xH, rH, a, M, m)[k, i]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices
        @inbounds for μ=1:4, n=1:3
            ∂K += (g_μν(0., xBL..., a, M, μ, 1) * Γαμν(0., xBL..., a, M, μ, m+1, n+1) + g_μν(0., xBL..., a, M, μ, m+1) * Γαμν(0., xBL..., a, M, μ, 1, n+1)) * jBLH[n, k] * jBLH[m, i]   # first term of Eq. A13
        end
    end
    return ∂K
end

# ∂ₖKᵢⱼ: outputs float. Note: rH = norm(xH).
function ∂Kij_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int, j::Int)   # Eq. A14
    ∂K=0.0
    @inbounds for m=1:3
        for l=1:3   # iterate over m and l first to avoid over-counting
            ∂K += g_μν(0., xBL..., a, M, l+1, m+1) * (HarmonicCoords.HessBLH(xH, rH, a, M, l)[k, i] * jBLH[m, j] + HarmonicCoords.HessBLH(xH, rH, a, M, l)[k, j] * jBLH[m, i])  # last term Eq. A14
            @inbounds for μ=1:4, n=1:3
                ∂K += (g_μν(0., xBL..., a, M, μ, l+1) * Γαμν(0., xBL..., a, M, μ, m+1, n+1) + g_μν(0., xBL..., a, M, μ, m+1) * Γαμν(0., xBL..., a, M, μ, l+1, n+1)) * jBLH[n, k] * jBLH[l, i] * jBLH[m, j]   # first term of Eq. A14
            end
        end
    end
    return ∂K
end

# define GR Γ factor, v_H = contravariant velocity in harmonic coordinates
Γ(vH::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = 1.0 / sqrt(1.0 - norm2_3d(vH) - K(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) - 2.0 * dot(K_i(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ), vH) - transpose(vH) * K_ij(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) * vH)   # Eq. A3

# define projection operator
Pαβ(vH::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = ημν + Qμν(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) + Γ(vH, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * otimes(vcat([1], vH))   # contravariant, Eq. A1
P_αβ(vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) =  ημν + K_μν(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) + Γ(vH, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * otimes(vcat([1], vH))   # cοvariant, Eq. A2 (note that we take both contravariant and covariant velocities as arguments)

# define STF projections 
STF(u::Vector, i::Int, j::Int) = u[i] * u[j] - dot(u, u) * δ(i, j) /3.0                                                                     # STF projection x^{<ij>}
STF(u::Vector, v::Vector, i::Int, j::Int) = (u[i] * v[j] + u[j] * v[i])/2.0 - dot(u, v)* δ(i, j) /3.0                                       # STF projection of two distinct vectors
STF(u::Vector, i::Int, j::Int, k::Int) = u[i] * u[j] * u[k] - (1.0/5.0) * dot(u, u) * (δ(i, j) * u[k] + δ(j, k) * u[i] + δ(k, i) * u[j])    # STF projection x^{<ijk>} (Eq. 46)

# define mass-ratio parameter
η(q::Float64) = q/((1+q)^2)   # q = mass ratio

# TO-DO: SINCE WE SET M=1 and m=q (currently, at least) WE SHOULD, FOR CLARITY, REMOVE M FROM THESE EQUATIONS AND WRITE m->q

# define multipole moments
M_ij(x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int) = η(m/M) * (1.0+m) * STF(x_H, i, j)  # quadrupole mass moment Eq. 48
ddotMij(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int) = η(m/M) * (1.0+m) * ((-2.0δ(i, j)/3.0) * (dot(x_H, a_H) + dot(v_H, v_H)) + x_H[j] * a_H[i] + 2.0 * v_H[i] * v_H[j] + x_H[i] * a_H[j])   # Eq. 7.17

M_ijk(x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) = η(m/M) * (1.0-m) * STF(x_H, i, j, k)  # octupole mass moment Eq. 48
ddotMijk(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) = η(m/M) * (1.0-m) * ((-4.0/5.0) * (dot(x_H, v_H)) * (δ(i, j) * v_H[k] + δ(j, k) * v_H[i] + δ(k, i) * v_H[j]) - (2.0/5.0) * (dot(x_H, a_H) + dot(v_H, v_H)) * (δ(i, j) * x_H[k] + δ(j, k) * x_H[i] + δ(k, i) * x_H[j]) - (1.0/5.0) * dot(x_H, x_H) * (δ(i, j) * a_H[k] + δ(j, k) * a_H[i] + δ(k, i) * a_H[j]) + 2.0 * v_H[k] * (x_H[j] * v_H[i] + x_H[i] * v_H[j]) + x_H[k] * (x_H[j] * a_H[i] + 2.0 * v_H[i] * v_H[j] + x_H[i] * a_H[j]) + x_H[i] * x_H[j] * a_H[k])   # Eq. 7.19

# second derivative of Mijkl, as defined in Eq. 85 (LONG EXPRESSION COPIED FROM MMA)
ddotMijkl(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int, l::Int) = (1.0+m)*η(m/M)*(2.0*(x_H[j]*v_H[i] + x_H[i]*v_H[j])*(x_H[l]*v_H[k] + x_H[k]*v_H[l]) - (4.0*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3])*(x_H[k]*δ(j,l)*v_H[i] + x_H[j]*δ(k,l)*v_H[i] + x_H[k]*δ(i,l)*v_H[j] + x_H[i]*δ(k,l)*v_H[j] + x_H[j]*δ(i,l)*v_H[k] + x_H[i]*δ(j,l)*v_H[k] + x_H[l]*(δ(j,k)*v_H[i] + δ(i,k)*v_H[j] + δ(i,j)*v_H[k]) + (x_H[k]*δ(i,j) + x_H[j]*δ(i,k) + x_H[i]*δ(j,k))*v_H[l]))/7. - (2.0*(x_H[i]*x_H[l]*δ(j,k) + x_H[k]*(x_H[l]*δ(i,j) + x_H[j]*δ(i,l) + x_H[i]*δ(j,l)) + x_H[j]*(x_H[l]*δ(i,k) + x_H[i]*δ(k,l)))*(v_H[1]^2 + v_H[2]^2 + v_H[3]^2 + x_H[1]*a_H[1] + x_H[2]*a_H[2] + x_H[3]*a_H[3]))/7. + ((δ(i,l)*δ(j,k) + δ(i,k)*δ(j,l) + δ(i,j)*δ(k,l))*(8.0*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3])^2 + 4.0*(x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*(v_H[1]^2 + v_H[2]^2 + v_H[3]^2 + x_H[1]*a_H[1] + x_H[2]*a_H[2] + x_H[3]*a_H[3])))/35. + x_H[k]*x_H[l]*(2.0*v_H[i]*v_H[j] + x_H[j]*a_H[i] + x_H[i]*a_H[j]) + x_H[i]*x_H[j]*(2.0*v_H[k]*v_H[l] + x_H[l]*a_H[k] + x_H[k]*a_H[l]) - ((x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*(δ(k,l)*(2.0*v_H[i]*v_H[j] + x_H[j]*a_H[i] + x_H[i]*a_H[j]) + δ(j,l)*(2.0*v_H[i]*v_H[k] + x_H[k]*a_H[i] + x_H[i]*a_H[k]) + δ(i,l)*(2.0*v_H[j]*v_H[k] + x_H[k]*a_H[j] + x_H[j]*a_H[k]) + δ(j,k)*(2.0*v_H[i]*v_H[l] + x_H[l]*a_H[i] + x_H[i]*a_H[l]) + δ(i,k)*(2.0*v_H[j]*v_H[l] + x_H[l]*a_H[j] + x_H[j]*a_H[l]) + δ(i,j)*(2.0*v_H[k]*v_H[l] + x_H[l]*a_H[k] + x_H[k]*a_H[l])))/7.)

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
    return η(m/M) * (1.0-m) * s_ij
end

function dotSij(aH::AbstractArray, v_H::AbstractArray, vH::AbstractArray, x_H::AbstractArray, xH::AbstractArray, m::Float64, M::Float64, i::Int, j::Int)
    S=0.0
    @inbounds for k=1:3
        for l=1:3
            S += -2.0δ(i, j) * (vH[l] * (xH[k] * dot(εkl[k, l], v_H) + vH[k] * dot(εkl[k, l], x_H)) + xH[k] * aH[l] * dot(εkl[k, l], x_H)) + 3.0 * vH[l] * (εkl[k, l][i] * (xH[k] * v_H[j] + x_H[j] * vH[k]) + εkl[k, l][j] * (xH[k] * v_H[i] + x_H[i] * vH[k])) + 3.0 * xH[k] * aH[l] * (εkl[k, l][i] * x_H[j] + εkl[k, l][j] * x_H[i])
        end
    end
    return η(m/M) * (1.0-m) * S / 6.0
end

# first derivative of Sijk, as defined in Eq. 86 (LONG EXPRESSION COPIED FROM MMA)
dotSijk(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) =((1.0+m)*η(m/M)*((δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][k]) + 5.0*(x_H[i]*x_H[k]*εkl[1,1][j] + x_H[j]*(x_H[k]*εkl[1,1][i] + x_H[i]*εkl[1,1][k])))*v_H[1]^2 + (δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][k]) + 5.0*(x_H[i]*x_H[k]*εkl[1,2][j] + x_H[j]*(x_H[k]*εkl[1,2][i] + x_H[i]*εkl[1,2][k])))*v_H[1]*v_H[2] + (δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][k]) + 5.0*(x_H[i]*x_H[k]*εkl[2,1][j] + x_H[j]*(x_H[k]*εkl[2,1][i] + x_H[i]*εkl[2,1][k])))*v_H[1]*v_H[2] + (δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][k]) + 5.0*(x_H[i]*x_H[k]*εkl[2,2][j] + x_H[j]*(x_H[k]*εkl[2,2][i] + x_H[i]*εkl[2,2][k])))*v_H[2]^2 + x_H[1]*v_H[1]*(-2.0*δ(j,k)*(εkl[1,1][i]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,1][1]*v_H[1] + εkl[1,1][2]*v_H[2] + εkl[1,1][3]*v_H[3]) + (x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3])*v_H[i]) - 2.0*δ(k,i)*(εkl[1,1][j]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,1][1]*v_H[1] + εkl[1,1][2]*v_H[2] + εkl[1,1][3]*v_H[3]) + (x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3])*v_H[i]) - 2.0*δ(i,j)*(εkl[1,1][k]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,1][1]*v_H[1] + εkl[1,1][2]*v_H[2] + εkl[1,1][3]*v_H[3]) + (x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3])*v_H[i]) + 5.0*(εkl[1,1][k]*(x_H[j]*v_H[i] + x_H[i]*v_H[j]) + x_H[k]*(εkl[1,1][j]*v_H[i] + εkl[1,1][i]*v_H[j]) + (x_H[j]*εkl[1,1][i] + x_H[i]*εkl[1,1][j])*v_H[k])) + x_H[1]*v_H[2]*(-2.0*δ(j,k)*(εkl[1,2][i]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,2][1]*v_H[1] + εkl[1,2][2]*v_H[2] + εkl[1,2][3]*v_H[3]) + (x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3])*v_H[i]) - 2.0*δ(k,i)*(εkl[1,2][j]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,2][1]*v_H[1] + εkl[1,2][2]*v_H[2] + εkl[1,2][3]*v_H[3]) + (x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3])*v_H[i]) - 2.0*δ(i,j)*(εkl[1,2][k]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[1,2][1]*v_H[1] + εkl[1,2][2]*v_H[2] + εkl[1,2][3]*v_H[3]) + (x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3])*v_H[i]) + 5.0*(εkl[1,2][k]*(x_H[j]*v_H[i] + x_H[i]*v_H[j]) + x_H[k]*(εkl[1,2][j]*v_H[i] + εkl[1,2][i]*v_H[j]) + (x_H[j]*εkl[1,2][i] + x_H[i]*εkl[1,2][j])*v_H[k])) + x_H[2]*v_H[1]*(-2.0*δ(j,k)*(εkl[2,1][i]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,1][1]*v_H[1] + εkl[2,1][2]*v_H[2] + εkl[2,1][3]*v_H[3]) + (x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3])*v_H[i]) - 2.0*δ(k,i)*(εkl[2,1][j]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,1][1]*v_H[1] + εkl[2,1][2]*v_H[2] + εkl[2,1][3]*v_H[3]) + (x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3])*v_H[i]) - 2.0*δ(i,j)*(εkl[2,1][k]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,1][1]*v_H[1] + εkl[2,1][2]*v_H[2] + εkl[2,1][3]*v_H[3]) + (x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3])*v_H[i]) + 5.0*(εkl[2,1][k]*(x_H[j]*v_H[i] + x_H[i]*v_H[j]) + x_H[k]*(εkl[2,1][j]*v_H[i] + εkl[2,1][i]*v_H[j]) + (x_H[j]*εkl[2,1][i] + x_H[i]*εkl[2,1][j])*v_H[k])) + x_H[2]*v_H[2]*(-2.0*δ(j,k)*(εkl[2,2][i]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,2][1]*v_H[1] + εkl[2,2][2]*v_H[2] + εkl[2,2][3]*v_H[3]) + (x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3])*v_H[i]) - 2.0*δ(k,i)*(εkl[2,2][j]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,2][1]*v_H[1] + εkl[2,2][2]*v_H[2] + εkl[2,2][3]*v_H[3]) + (x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3])*v_H[i]) - 2.0*δ(i,j)*(εkl[2,2][k]*(x_H[1]*v_H[1] + x_H[2]*v_H[2] + x_H[3]*v_H[3]) + x_H[i]*(εkl[2,2][1]*v_H[1] + εkl[2,2][2]*v_H[2] + εkl[2,2][3]*v_H[3]) + (x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3])*v_H[i]) + 5.0*(εkl[2,2][k]*(x_H[j]*v_H[i] + x_H[i]*v_H[j]) + x_H[k]*(εkl[2,2][j]*v_H[i] + εkl[2,2][i]*v_H[j]) + (x_H[j]*εkl[2,2][i] + x_H[i]*εkl[2,2][j])*v_H[k])) + x_H[1]*(δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[1,1][1] + x_H[2]*εkl[1,1][2] + x_H[3]*εkl[1,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,1][k]) + 5.0*(x_H[i]*x_H[k]*εkl[1,1][j] + x_H[j]*(x_H[k]*εkl[1,1][i] + x_H[i]*εkl[1,1][k])))*a_H[1] + x_H[2]*(δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[2,1][1] + x_H[2]*εkl[2,1][2] + x_H[3]*εkl[2,1][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,1][k]) + 5.0*(x_H[i]*x_H[k]*εkl[2,1][j] + x_H[j]*(x_H[k]*εkl[2,1][i] + x_H[i]*εkl[2,1][k])))*a_H[1] + x_H[1]*(δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[1,2][1] + x_H[2]*εkl[1,2][2] + x_H[3]*εkl[1,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[1,2][k]) + 5.0*(x_H[i]*x_H[k]*εkl[1,2][j] + x_H[j]*(x_H[k]*εkl[1,2][i] + x_H[i]*εkl[1,2][k])))*a_H[2] + x_H[2]*(δ(j,k)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][i]) + δ(k,i)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][j]) + δ(i,j)*(-2.0*x_H[i]*(x_H[1]*εkl[2,2][1] + x_H[2]*εkl[2,2][2] + x_H[3]*εkl[2,2][3]) - (x_H[1]^2 + x_H[2]^2 + x_H[3]^2)*εkl[2,2][k]) + 5.0*(x_H[i]*x_H[k]*εkl[2,2][j] + x_H[j]*(x_H[k]*εkl[2,2][i] + x_H[i]*εkl[2,2][k])))*a_H[2]))/15.

# fill pre-allocated arrays with the appropriate derivatives of the mass and current moments for trajectory evolution, i.e., to compute self-force
function multipole_moments_tr!(vH::AbstractArray, xH::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, Mij::AbstractArray, Mijk::AbstractArray, Sij::AbstractArray)
    @inbounds Threads.@threads for i=1:3
        for j=1:3
            Mij[i, j] = M_ij.(x_H, m, M, i, j)
            Sij[i, j] = S_ij.(x_H, xH, vH, m, M, i, j)
            @inbounds for k=1:3
                Mijk[i, j, k] = M_ijk.(x_H, m, M, i, j, k)
            end
        end
    end
end


# fill pre-allocated arrays with the appropriate derivatives of the mass and current moments for trajectory evolution, i.e., to compute self-force
function moments_tr!(aH::AbstractArray, a_H::AbstractArray, vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, Mij2::AbstractArray, Mijk2::AbstractArray, Sij1::AbstractArray)
    @inbounds Threads.@threads for i=1:3
        for j=1:3
            Mij2[i, j] = ddotMij.(a_H, v_H, x_H, m, M, i, j)
            Sij1[i, j] = dotSij.(aH, v_H, vH, x_H, xH, m, M, i, j)
            @inbounds for k=1:3
                Mijk2[i, j, k] = ddotMijk.(a_H, v_H, x_H, m, M, i, j, k)
            end
        end
    end
end

# fill pre-allocated arrays with the appropriate derivatives of the mass and current moments for waveform computation
function moments_wf!(aH::AbstractArray, a_H::AbstractArray, vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, Mij2::AbstractArray, Mijk2::AbstractArray, Mijkl2::AbstractArray, Sij1::AbstractArray, Sijk1::AbstractArray)
    @inbounds Threads.@threads for i=1:3
        for j=1:3
            Mij2[i, j] = ddotMij.(a_H, v_H, x_H, m, M, i, j)
            Sij1[i, j] = dotSij.(aH, v_H, vH, x_H, xH, m, M, i, j)
            @inbounds for k=1:3
                Mijk2[i, j, k] = ddotMijk.(a_H, v_H, x_H, m, M, i, j, k)
                Sijk1[i, j, k] = dotSijk.(a_H, v_H, x_H, m, M, i, j, k)
                @inbounds for l=1:3
                    Mijkl2[i, j, k, l] = ddotMijkl.(a_H, v_H, x_H, m, M, i, j, k, l)
                end
            end
        end
    end
end

const index_pairs::Matrix{Tuple{Int64, Int64}} = [(i, j) for i=1:3, j=1:3];

# calculate time derivatives of the mass and current moments for trajectory evolution in BL time, i.e., to compute self-force
function moment_derivs_tr_BL!(h::Float64, compute_at::Int64, nPoints::Int64, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    @inbounds Threads.@threads for i=1:3
        @inbounds for j=1:3
            Mij5[i, j] = FiniteDiff_5.compute_third_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            Mij6[i, j] = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            Mij7[i, j] = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            Mij8[i, j] = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mij2data[i, j], h, nPoints)

            Sij5[i, j] = FiniteDiff_5.compute_fourth_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            Sij6[i, j] = FiniteDiff_5.compute_fifth_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            @inbounds for k=1:3
                Mijk7[i, j, k] = FiniteDiff_5.compute_fifth_derivative(compute_at, Mijk2data[i, j, k], h, nPoints)
                Mijk8[i, j, k] = FiniteDiff_5.compute_sixth_derivative(compute_at, Mijk2data[i, j, k], h, nPoints)
            end 
        end
    end
end


# calculate time derivatives of the mass and current moments for trajectory evolution in mino time, i.e., to compute self-force
function moment_derivs_tr_Mino!(h::Float64, compute_at::Int64, nPoints::Int64, a::Float64, M::Float64, E::Float64, L::Float64, x::Vector{Float64}, dx::Vector{Float64}, d2x::Vector{Float64}, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    # compute time derivatives of x^{μ}(t)
    ### to make faster will want these arrays pre-allocated
    d3x = [Deriv3.d3r_dt(d2x, dx, x, a), Deriv3.d3θ_dt(d2x, dx, x, a), Deriv3.d3ϕ_dt(d2x, dx, x, a)]
    d4x = [Deriv4.d4r_dt(d3x, d2x, dx, x, a), Deriv4.d4θ_dt(d3x, d2x, dx, x, a), Deriv4.d4ϕ_dt(d3x, d2x, dx, x, a)]
    d5x = [Deriv5.d5r_dt(d4x, d3x, d2x, dx, x, a), Deriv5.d5θ_dt(d4x, d3x, d2x, dx, x, a), Deriv5.d5ϕ_dt(d4x, d3x, d2x, dx, x, a)]

    # compute BL time derivatives of mino time λ(t)
    dλ_dt = MinoTimeDerivs.dλ_dt(x, a, M, E, L)
    d2λ_dt = MinoTimeDerivs.d2λ_dt(dx, x, a, M, E, L)
    d3λ_dt = MinoTimeDerivs.d3λ_dt(d2x, dx, x, a, M, E, L)
    d4λ_dt = MinoTimeDerivs.d4λ_dt(d3x, d2x, dx, x, a, M, E, L)
    d5λ_dt = MinoTimeDerivs.d5λ_dt(d4x, d3x, d2x, dx, x, a, M, E, L)
    d6λ_dt = MinoTimeDerivs.d6λ_dt(d5x, d4x, d3x, d2x, dx, x, a, M, E, L)

    @inbounds Threads.@threads for indices in ConstructSymmetricArrays.traj_indices
        if length(indices)==2
            i, j = indices
            # the naming is a bit confusing in the code block below. For example take f(t) = Mij2 = d2Mij_dt2. We now wish to compute higher order derivatives of f(t) from the re-parameterized function 
            # f(λ(t)). We do this by first computing d^{n}f/dλ^{n} numerically from finite difference formulas, and d^{n}λ/dt^{n} from analytix expressions. We can then use these to compute d^{n}f_dt^{n} 
            # via the chain rule (e.g., see the file "ParameterizedDerivs.jl")

            # mass quadrupole
            dMij2_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d3λ = FiniteDiff_5.compute_third_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d4λ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d5λ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d6λ = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mij2data[i, j], h, nPoints)

            Mij5[i, j] = ParameterizedDerivs.d3f_dt(dMij2_dλ, dλ_dt, dMij2_d2λ, d2λ_dt, dMij2_d3λ, d3λ_dt)
            Mij6[i, j] = ParameterizedDerivs.d4f_dt(dMij2_dλ, dλ_dt, dMij2_d2λ, d2λ_dt, dMij2_d3λ, d3λ_dt, dMij2_d4λ, d4λ_dt)
            Mij7[i, j] = ParameterizedDerivs.d5f_dt(dMij2_dλ, dλ_dt, dMij2_d2λ, d2λ_dt, dMij2_d3λ, d3λ_dt, dMij2_d4λ, d4λ_dt, dMij2_d5λ, d5λ_dt)
            Mij8[i, j] = ParameterizedDerivs.d6f_dt(dMij2_dλ, dλ_dt, dMij2_d2λ, d2λ_dt, dMij2_d3λ, d3λ_dt, dMij2_d4λ, d4λ_dt, dMij2_d5λ, d5λ_dt, dMij2_d6λ, d6λ_dt)

            # current quadrupole
            dSij1_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            dSij1_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            dSij1_d3λ = FiniteDiff_5.compute_third_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            dSij1_d4λ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            dSij1_d5λ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Sij1data[i, j], h, nPoints)

            Sij5[i, j] = ParameterizedDerivs.d4f_dt(dSij1_dλ, dλ_dt, dSij1_d2λ, d2λ_dt, dSij1_d3λ, d3λ_dt, dSij1_d4λ, d4λ_dt)
            Sij6[i, j] = ParameterizedDerivs.d5f_dt(dSij1_dλ, dλ_dt, dSij1_d2λ, d2λ_dt, dSij1_d3λ, d3λ_dt, dSij1_d4λ, d4λ_dt, dSij1_d5λ, d5λ_dt)
        else
            # mass octupole
            i, j, k = indices
            dMijk2_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d3λ = FiniteDiff_5.compute_third_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d4λ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d5λ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d6λ = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)

            Mijk7[i, j, k] = ParameterizedDerivs.d5f_dt(dMijk2_dλ, dλ_dt, dMijk2_d2λ, d2λ_dt, dMijk2_d3λ, d3λ_dt, dMijk2_d4λ, d4λ_dt, dMijk2_d5λ, d5λ_dt)
            Mijk8[i, j, k] = ParameterizedDerivs.d6f_dt(dMijk2_dλ, dλ_dt, dMijk2_d2λ, d2λ_dt, dMijk2_d3λ, d3λ_dt, dMijk2_d4λ, d4λ_dt, dMijk2_d5λ, d5λ_dt, dMijk2_d6λ, d6λ_dt)
        end
    end

    # symmetrize moments
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij5); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij6);
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij7); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij8);
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij5); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij6);
    ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk7); ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk8);
end

@views function moment_derivs_wf_Mino!(a::Float64, E::Float64, L::Float64, M::Float64, x::AbstractArray, dx::AbstractArray,
    Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray,
    Sij2::AbstractArray, Sijk3::AbstractArray, nPoints::Int64, h::Float64)
    dλ_dt = MinoTimeDerivs.dλ_dt.(x, a, M, E, L)
    d2λ_dt = MinoTimeDerivs.d2λ_dt.(dx, x, a, M, E, L)
    @inbounds Threads.@threads for indices in ConstructSymmetricArrays.waveform_indices
        for compute_at in 1:nPoints
            if length(indices)==2
                i, j = indices
                # current quadrupole
                dSij1_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sij1data[i, j], h, nPoints)
                Sij2[i, j][compute_at] = ParameterizedDerivs.df_dt(dSij1_dλ, dλ_dt[compute_at])
            elseif length(indices)==3
                # mass octupole
                i, j, k = indices
                dMijk2_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
                Mijk3[i, j, k][compute_at] = ParameterizedDerivs.df_dt(dMijk2_dλ, dλ_dt[compute_at])

                dSijk1_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sijk1data[i, j, k], h, nPoints)
                dSijk1_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Sijk1data[i, j, k], h, nPoints)
                Sijk3[i, j, k][compute_at] = ParameterizedDerivs.d2f_dt(dSijk1_dλ, dλ_dt[compute_at], dSijk1_d2λ, d2λ_dt[compute_at])
            else
                i, j, k, l = indices
                dMijkl2_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijkl2data[i, j, k, l], h, nPoints)
                dMijkl2_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Mijkl2data[i, j, k, l], h, nPoints)
                Mijkl4[i, j, k, l][compute_at] = ParameterizedDerivs.d2f_dt(dMijkl2_dλ, dλ_dt[compute_at], dMijkl2_d2λ, d2λ_dt[compute_at])
            end
        end
    end

    # # symmetrize moments
    # ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij2);
    # ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk3); ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Sijk3);
    # ConstructSymmetricArrays.SymmetrizeFourIndexTensor!(Mijkl4);
end


function compute_waveform_moments_and_derivs_Mino!(a::Float64, E::Float64, L::Float64, m::Float64, M::Float64, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, 
    xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray, v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, 
    λ::Vector{Float64}, r::Vector{Float64}, rdot::Vector{Float64}, rddot::Vector{Float64}, θ::Vector{Float64}, θdot::Vector{Float64}, θddot::Vector{Float64}, ϕ::Vector{Float64},
    ϕdot::Vector{Float64}, ϕddot::Vector{Float64}, Mij2data::AbstractArray, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, 
    Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nPoints::Int64, h::Float64)

    # convert trajectories to BL coords
    @inbounds Threads.@threads for i=1:nPoints
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]);
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]);
    end
    @inbounds Threads.@threads for i=1:nPoints
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
        x_H[i] = xH[i]
        rH[i] = norm_3d(xH[i]);
    end
    @inbounds Threads.@threads for i=1:nPoints
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        v_H[i] = vH[i]; 
        v[i] = norm_3d(vH[i]);
    end
    @inbounds Threads.@threads for i=1:nPoints
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
        a_H[i] = aH[i]
    end

    SelfForce_numerical.moments_wf!(aH[1:nPoints], a_H[1:nPoints], vH[1:nPoints], v_H[1:nPoints], xH[1:nPoints], x_H[1:nPoints], m, M, Mij2data, Mijk2data, Mijkl2data, Sij1data, Sijk1data)
    SelfForce_numerical.moment_derivs_wf_Mino!(a, E, L, M, xBL, vBL, Mijk2data, Mijkl2data, Sij1data, Sijk1data, Mijk3, Mijkl4, Sij2, Sijk3, nPoints, h)
end                     


# returns hij array at some time t specified as an index (rather than a time in seconds)
function hij!(hij::AbstractArray, nPoints::Int, r::Float64, Θ::Float64, Φ::Float64, Mij2::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray)
    # n ≡ unit vector pointing in direction of far away observer
    nx = sin(Θ) * cos(Φ)
    ny = sin(Θ) * sin(Φ)
    nz = cos(Θ)
    n = [nx, ny, nz]

    # calculate perturbations in TT gauge (Eq. 84)
    @inbounds Threads.@threads for t=1:nPoints
        for i=1:3
            @inbounds for j=1:3
                @views hij[i, j, t] = 0    # set all entries to zero

                @views hij[i, j, t] += 2.0 * Mij2[i, j, t] / r    # first term in Eq. 84 

                @inbounds for k=1:3
                    @views hij[i, j, t] += 2.0 * Mijk3[i, j, k, t] * n[k] / (3.0r)    # second term in Eq. 84

                    @inbounds for l=1:3
                        @views hij[i, j, t] += 4.0 * (εkl[k, l][i] * Sij2[j, k, t] * n[l] + εkl[k, l][j] * Sij2[i, k, t] * n[l]) / (3.0r) + Mijkl4[i, j, k, l, t] * n[k] * n[l] / (6.0r)    # third and fourth terms in Eq. 84
                        
                        @inbounds for m=1:3
                            @views hij[i, j, t] += (εkl[k, l][i] * Sijk3[j, k, m, t] * n[l] * n[m] + εkl[k, l][j] * Sijk3[i, k, m, t] * n[l] * n[m]) / (2.0r)
                        end
                    end
                end
            end
        end
    end
end

# calculate radiation reaction potentials
function Vrr(t::Float64, xH::AbstractArray, Mij5::AbstractArray, Mij7::AbstractArray, Mijk7::AbstractArray)    # Eq. 44
    V = 0.0
    @inbounds for i=1:3
        for j=1:3
            V += -xH[i] * xH[j] * Mij5[i, j] / 5.0 - dot(xH, xH) * xH[i] * xH[j] * Mij7[i, j] / 70.0   # first and last term in Eq. 44
            @inbounds for k=1:3
                V+= xH[i] * xH[j] * xH[k] * Mijk7[i, j, k] / 189.0   # 2nd term in Eq. 44
            end
        end
    end
    return V
end

function ∂Vrr_∂t(t::Float64, xH::AbstractArray, Mij6::AbstractArray, Mij8::AbstractArray, Mijk8::AbstractArray)    # Eq. 7.25
    V = 0.0
    @inbounds for i=1:3
        for j=1:3
            V += -xH[i] * xH[j] * Mij6[i, j] / 5.0 - dot(xH, xH) * xH[i] * xH[j] * Mij8[i, j] / 70.0   # first and last term in Eq. 7.25
            @inbounds for k=1:3
                V+= xH[i] * xH[j] * xH[k] * Mijk8[i, j, k] / 189.0   # 2nd term in Eq. 7.25
            end
        end
    end
    return V
end

function ∂Vrr_∂a(t::Float64, xH::AbstractArray, Mij5::AbstractArray, Mij7::AbstractArray, Mijk7::AbstractArray, a::Int)    # Eq. 7.30
    V = 0.0
    @inbounds for i=1:3
        for j=1:3
            V += (-2.0/5.0) * xH[j] * Mij5[i, j] * δ(i, a) + (3.0/189.0) * xH[i] * xH[j] * Mijk7[a, i, j] - (1.0/35.0) * (xH[a] * xH[i] * xH[j] * Mij7[i, j] + dot(xH, xH) * xH[j] * Mij7[i, j] * δ(i, a))   # Eq. 7.31
        end
    end
    return V
end

function Virr(t::Float64, xH::AbstractArray, Mij6::AbstractArray, Sij5::AbstractArray)   # Eq. 45
    V = [0., 0., 0.]  
    @inbounds Threads.@threads for i=1:3
        for j=1:3, k=1:3   # dummy indices
            V[i] += STF(xH, i, j, k) * Mij6[j, k] / 21.0    # first term Eq. 45
            @inbounds for l=1:3   # dummy indices in second term in Eq. 45
                V[i] += -4.0 * εkl[i, j][k] * xH[j] * xH[l] * Sij5[k, l]  / 45.0
            end 
        end
    end
    return V
end

function ∂Virr_∂t(t::Float64, xH::AbstractArray, Mij7::AbstractArray, Sij6::AbstractArray, i::Int)   # Eq. 7.26
    V = 0.0
    @inbounds for j=1:3
        for k=1:3   # dummy indices
            V += STF(xH, i, j, k) * Mij7[j, k] / 21.0    # first term Eq. 7.26
            @inbounds for l=1:3   # dummy indices in second term in Eq. 7.26
                V += -4.0 * εkl[i, j][k] * xH[j] * xH[l] * Sij6[k, l]  / 45.0
            end 
        end
    end
    return V
end

function ∂Virr_∂a(t::Float64, xH::AbstractArray, Mij6::AbstractArray, Sij5::AbstractArray, i::Int, a::Int)   # Eq. 45
    # use numerical derivatives to calculate RR potentials
    V = 0.0   
    @inbounds for j=1:3
        for k=1:3   # dummy indices
            V += (Mij6[j, k] / 21.0) * ((δ(i, a) * xH[j] * xH[k] +  xH[i] * δ(j, a) * xH[k] + xH[i] * xH[j] * δ(k, a)) - (1.0/5.0) * (2.0 * xH[a] * (δ(i, j) * xH[k] + δ(j, k) * xH[i] + δ(k, i) * xH[j]) + dot(xH, xH) * (δ(i, j) * δ(k, a) + δ(j, k) * δ(i, a) + δ(k, i) * δ(j, a))))   # first term Eq. 7.34 (first line)
            @inbounds for l=1:3   # dummy indices in second term in Eq. 45
                V += -4.0 * εkl[i, j][k] * (δ(j, a) * xH[l] + xH[j] * δ(l, a)) * Sij5[k, l] / 45.0
            end 
        end
    end
    return V
end

# # chimera code
# function ∂Virr_∂a(t::Float64, xH::AbstractArray, Mij6::AbstractArray, Sij5::AbstractArray, i::Int, a::Int)   # Eq. 45
#     # use numerical derivatives to calculate RR potentials
#     V = 0.0   
#     @inbounds for j=1:3
#         for k=1:3   # dummy indices
#             V += (Mij6[j, k] / 21.0) * (δ(i,a) * xH[j] * xH[k] + 0.6 * (xH[i] * xH[j] * δ(a,k) + xH[a] * xH[j] * δ(i,k))) - 0.4 * dot3d(xH, xH) * Mij6[i, a]
#             @inbounds for l=1:3   # dummy indices in second term in Eq. 45
#                 V += -4.0 * εkl[i, j][k] * (δ(j, a) * xH[l] + xH[j] * δ(l, a)) * Sij5[k, l] / 45.0
#             end 
#         end
#     end
#     return V
# end

# compute self-acceleration pieces
function A_RR(t::Float64, xH::AbstractArray, v::Float64, vH::AbstractArray, ∂Vrr_∂t::Float64, ∂Vrr_∂a::SVector{3, Float64}, ∂Virr_∂a::SMatrix{3, 3, Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    aRR = (1.0 - v^2) * ∂Vrr_∂t   # first term in Eq. A4
    @inbounds for i=1:3
        aRR += 2.0 * vH[i] * ∂Vrr_∂a[i]   # second term Eq. A4
        @inbounds for j=1:3
            aRR += -4.0 * vH[i] * vH[j] * ∂Virr_∂a[j, i]   # third term Eq. A4
        end
    end
    return aRR
end

function Ai_RR(t::Float64, xH::AbstractArray, v::Float64, v_H::AbstractArray, vH::AbstractArray, ∂Vrr_∂t::Float64, ∂Virr_∂t::SVector{3, Float64}, ∂Vrr_∂a::SVector{3, Float64}, ∂Virr_∂a::SMatrix{3, 3, Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, i::Int)
    aiRR = -(1 + v^2) * ∂Vrr_∂a[i] + 2.0 * v_H[i] * ∂Vrr_∂t - 4.0 * ∂Virr_∂t[i]    # first, second, and last term in Eq. A5
    @inbounds for j=1:3
        aiRR += 2.0 * v_H[i] * vH[j] * ∂Vrr_∂a[j] - 4.0 * vH[j] * (∂Virr_∂a[i, j] - ∂Virr_∂a[j, i])    # third and fourth terms in Eq. A5
    end
    return aiRR
end

function A1_β(t::Float64, xH::AbstractArray, v::Float64, v_H::AbstractArray, vH::AbstractArray, xBL::AbstractArray, rH::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    ∂Vrr_∂t = SelfForce_numerical.∂Vrr_∂t(t, xH, Mij6, Mij8, Mijk8)
    ∂Vrr_∂a = @SVector [SelfForce_numerical.∂Vrr_∂a(t, xH, Mij5, Mij7, Mijk7, i) for i =1:3]
    ∂Virr_∂t = @SVector [SelfForce_numerical.∂Virr_∂t(t, xH, Mij7, Sij6, i) for i =1:3]
    ∂Virr_∂a = @SMatrix [SelfForce_numerical.∂Virr_∂a(t, xH, Mij6, Sij5, j, i) for j=1:3, i=1:3]
    return [i==1 ? A_RR(t, xH, v, vH, ∂Vrr_∂t, ∂Vrr_∂a, ∂Virr_∂a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6) : Ai_RR(t, xH, v, v_H, vH, ∂Vrr_∂t, ∂Virr_∂t, ∂Vrr_∂a, ∂Virr_∂a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, i-1) for i = 1:4]
end

function B_RR(xH::Vector{Float64}, Qi::Vector{Float64}, ∂K_∂xk::SVector{3, Float64}, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    return dot(Qi, ∂K_∂xk)   # Eq. A6
end

function Bi_RR(xH::Vector{Float64}, Qij::AbstractArray, ∂K_∂xk::SVector{3, Float64}, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    return -2.0 * (ηij + Qij) * ∂K_∂xk   # Eq. A9
end
function C_RR(xH::Vector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂K_∂xk::SVector{3, Float64}, ∂Ki_∂xk::SMatrix{3, 3, Float64}, Q::Float64, Qi::Vector{Float64}, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    C = 0.0
    @inbounds for i=1:3
        C += 2.0 * (1.0 - Q) * vH[i] * ∂K_∂xk[i]
        @inbounds for j=1:3
            C += 2.0 * Qi[i] * vH[j] * (∂Ki_∂xk[i, j] - ∂Ki_∂xk[j, i])
        end
    end
    return C
end

function Ci_RR(xH::Vector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂K_∂xk::SVector{3, Float64}, ∂Ki_∂xk::SMatrix{3, 3, Float64}, Qi::Vector{Float64}, Qij::AbstractArray, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)   # Eq. A10
    C = @MVector [0., 0., 0.]
    @inbounds for j=1:3
        @inbounds for i=1:3
            C[i] += 4.0 * Qi[i] * vH[j] * ∂K_∂xk[j]
        end
        C .+= 4.0 * (ηij + Qij) * vH[j] * ([(∂Ki_∂xk[j, k] - ∂Ki_∂xk[k, j]) for k=1:3]) 
    end
    return C
end

function D_RR(xH::Vector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂Ki_∂xk::SMatrix{3, 3, Float64}, ∂Kij_∂xk::SArray{Tuple{3, 3, 3}, Float64, 3, 27}, Q::Float64, Qi::Vector{Float64}, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    D = 0.0
    @inbounds for i=1:3
        @inbounds for j=1:3
            D += 2.0 * (1.0 - Q) * vH[i] * vH[j] * ∂Ki_∂xk[i, j]
            @inbounds for k=1:3
                D += -Qi[i] * vH[j] * vH[k] * (∂Kij_∂xk[j, k, i] + ∂Kij_∂xk[k, j, i] - ∂Kij_∂xk[i, j, k]) 
            end
        end
    end
    return D
end

function Di_RR(xH::Vector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂Ki_∂xk::SMatrix{3, 3, Float64}, ∂Kij_∂xk::SArray{Tuple{3, 3, 3}, Float64, 3, 27}, Qi::Vector{Float64}, Qij::AbstractArray, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)   # Eq. A11
    D = @MVector [0., 0., 0.]
    @inbounds for j=1:3
        @inbounds for k=1:3
            @inbounds for i=1:3
                D[i] += 4.0 * Qi[i] * vH[j] * vH[k] * ∂Ki_∂xk[j, k]
            end
            D .+= 2.0 * (ηij + Qij) * vH[j] * vH[k] * [(∂Kij_∂xk[j, k, l] + ∂Kij_∂xk[k, j, l] - ∂Kij_∂xk[l, j, k]) for l=1:3]
        end
    end
    return D
end

# computes the four self-acceleration components A^{2}_{β} (Eqs. 62 - 63)
function A2_β(t::Float64, xH::AbstractArray, vH::AbstractArray, xBL::AbstractArray, rH::Float64, a::Float64, M::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mijk7::AbstractArray, Sij5::AbstractArray, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    jBLH = HarmonicCoords.jBLH(xH, a, M)
    ∂K_∂xk = @SVector [SelfForce_numerical.∂K_∂xk(xH, xBL, jBLH, a, M, g_μν, Γαμν, j) for j=1:3];
    ∂Ki_∂xk = @SMatrix [SelfForce_numerical.∂Ki_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, j, k) for j=1:3, k=1:3];
    ∂Kij_∂xk = @SArray [SelfForce_numerical.∂Kij_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, j, k, l) for j=1:3, k=1:3, l=1:3]
    Q = SelfForce_numerical.Q(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)
    Qi = SelfForce_numerical.Qi(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)
    Qij = SelfForce_numerical.Qij(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)

    BRR = B_RR(xH, Qi, ∂K_∂xk, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    BiRR = Bi_RR(xH, Qij, ∂K_∂xk, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    CRR = C_RR(xH, vH, xBL, ∂K_∂xk, ∂Ki_∂xk, Q, Qi, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    CiRR = Ci_RR(xH, vH, xBL, ∂K_∂xk, ∂Ki_∂xk, Qi, Qij, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    DRR = D_RR(xH, vH, xBL, ∂Ki_∂xk, ∂Kij_∂xk, Q, Qi, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    DiRR = Di_RR(xH, vH, xBL, ∂Ki_∂xk, ∂Kij_∂xk, Qi, Qij, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    VRR = Vrr(t, xH,  Mij5, Mij7, Mijk7)
    ViRR = Virr(t, xH, Mij6, Sij5)

    A2_t = (BRR + CRR + DRR) * VRR + dot((BiRR + CiRR + DiRR), ViRR)   # Eq. 62
    A2_i = -2.0 * (BRR + CRR + DRR) * ViRR - (BiRR + CiRR + DiRR) * VRR / 2.0  # Eq. 63

    return vcat(A2_t, A2_i)
end

# compute self-acceleration in harmonic coordinates and transform components back to BL
function aRRα(aSF_H::Vector{Float64}, aSF_BL::Vector{Float64}, t::Float64, xH::Vector{Float64}, v::Float64, v_H::Vector{Float64}, vH::Vector{Float64}, xBL::Vector{Float64}, rH::Float64, a::Float64, M::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    aSF_H[:] = -Γ(v_H, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * Pαβ(v_H, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ) * (A1_β(t, xH, v, v_H, vH, xBL, rH, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6) + A2_β(t, xH, vH, xBL, rH, a, M, Mij5, Mij6, Mij7, Mijk7, Sij5, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ))
    aSF_BL[1] = aSF_H[1]
    aSF_BL[2:4] = HarmonicCoords.aHtoBL(xH, zeros(3), aSF_H[2:4], a, M)
end


# returns the self-acceleration 4-vector
function selfAcc!(aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray, v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, t::Vector{Float64}, r::Vector{Float64}, dr_dt::Vector{Float64}, d2r_dt2::Vector{Float64}, θ::Vector{Float64}, dθ_dt::Vector{Float64}, d2θ_dt2::Vector{Float64}, ϕ::Vector{Float64}, dϕ_dt::Vector{Float64}, d2ϕ_dt2::Vector{Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, a::Float64, M::Float64, m::Float64, compute_at::Int64, h::Float64)
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);      # divide by (dt/dτ)² to get accelerations wrt BL time
    end
    @inbounds Threads.@threads for i in eachindex(t)
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
        x_H[i] = xH[i]
        rH[i] = norm_3d(xH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        v_H[i] = vH[i]; 
        v[i] = norm_3d(vH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
        a_H[i] = aH[i]
    end
    
    # calculate ddotMijk, ddotMijk, dotSij "analytically"
    SelfForce_numerical.moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)

    # calculate moment derivatives numerically at t = tF
    # SelfForce_numerical.moment_derivs_tr!(t, Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)

    # SelfForce_numerical.moment_derivs_tr!(t, Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, compute_at, nHarm, Ωr, Ωθ, Ωϕ, fit_fname_param)

    SelfForce_numerical.moment_derivs_tr!(h, compute_at, size(t, 1), Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)


    # calculate self force in BL and harmonic coordinates
    SelfForce_numerical.aRRα(aSF_H, aSF_BL, 0.0, xH[compute_at], v[compute_at], v_H[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, M, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ)
end


# returns the self-acceleration 4-vector
function selfAcc_mino!(a::Float64, M::Float64, E::Float64, L::Float64, aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray,
    x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray, v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, t::Vector{Float64}, r::Vector{Float64}, dr_dt::Vector{Float64}, 
    d2r_dt2::Vector{Float64}, θ::Vector{Float64}, dθ_dt::Vector{Float64}, d2θ_dt2::Vector{Float64}, ϕ::Vector{Float64}, dϕ_dt::Vector{Float64}, d2ϕ_dt2::Vector{Float64}, Mij5::AbstractArray, 
    Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray,
    Sij1_data::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function,
    gΦΦ::Function, m::Float64, compute_at::Int64, h::Float64)
       
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);      # divide by (dt/dτ)² to get accelerations wrt BL time
    end
    @inbounds Threads.@threads for i in eachindex(t)
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
        x_H[i] = xH[i]
        rH[i] = norm_3d(xH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        v_H[i] = vH[i]; 
        v[i] = norm_3d(vH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
        a_H[i] = aH[i]
    end
    
    # calculate ddotMijk, ddotMijk, dotSij "analytically"
    SelfForce_numerical.moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)

    SelfForce_numerical.moment_derivs_tr_Mino!(h, compute_at, size(t, 1), a, M, E, L, xBL[compute_at], vBL[compute_at], aBL[compute_at], Mij2_data, Mijk2_data, Sij1_data, Mij5, 
    Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)


    # calculate self force in BL and harmonic coordinates
    SelfForce_numerical.aRRα(aSF_H, aSF_BL, 0.0, xH[compute_at], v[compute_at], v_H[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, M, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, 
    Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ)
end

function EvolveConstants(Δt::Float64, a::Float64, t::Float64, r::Float64, θ::Float64, ϕ::Float64, Γ::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, aSF_BL::Vector{Float64}, EE::AbstractArray, Edot::AbstractArray, LL::AbstractArray, Ldot::AbstractArray, QQ::AbstractArray, Qdot::AbstractArray, CC::AbstractArray, Cdot::AbstractArray, pArray::AbstractArray, ecc::AbstractArray, θmin::AbstractArray, M::Float64, nPoints::Int64)
    #### ELQ ####
    push!(Edot, (- Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 1, 1) * aSF_BL[1] - Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 4, 1) * aSF_BL[4])/Γ)    # Eq. 30
    push!(Ldot, (Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 1, 4) * aSF_BL[1] + Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 4, 4) * aSF_BL[4])/Γ)    # Eq. 31
    
    dQ_dt = 0
    @inbounds for α=1:4, β=1:4
        dQ_dt += 2 * Kerr.KerrMetric.ξ_μν(t, r, θ, ϕ, a, M, α, β) * (α==1 ? 1. : α==2 ? rdot : α==3 ? θdot : ϕdot) * aSF_BL[β]    # Eq. 32
    end
    push!(Qdot, dQ_dt)

    push!(Cdot, dQ_dt + 2 * (a * last(EE) - last(LL)) * (last(Ldot) - a * last(Edot)))

    # constants of motion
    append!(EE, ones(nPoints) * (last(EE) + last(Edot) * Δt))
    append!(LL, ones(nPoints) * (last(LL) + last(Ldot) * Δt))
    append!(QQ, ones(nPoints) * (last(QQ) + last(Qdot) * Δt))
    append!(CC, ones(nPoints) * (last(CC) + last(Cdot) * Δt))

    # since we compute the self force at the end of each piecewise geodesic, the flux in between will be zero
    append!(Edot, zeros(nPoints-1))
    append!(Ldot, zeros(nPoints-1))
    append!(Qdot, zeros(nPoints-1))
    append!(Cdot, zeros(nPoints-1))

    #### p, e, θmin ####

    # computing p, e, θmin_BL from updated constants
    pp, ee, θθ = Kerr.ConstantsOfMotion.peθ(a, last(EE), last(LL), last(QQ), last(CC), 1.0)
    append!(pArray, ones(nPoints) * pp)
    append!(ecc, ones(nPoints) * ee)
    append!(θmin, ones(nPoints) * θθ)
end

Z_1(a::Float64, M::Float64) = 1 + (1 - a^2 / M^2)^(1/3) * ((1 + a / M)^(1/3) + (1 - a / M)^(1/3))
Z_2(a::Float64, M::Float64) = sqrt(3 * (a / M)^2 + Z_1(a, M)^2)
LSO_r(a::Float64, M::Float64) = M * (3 + Z_2(a, M) - sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # retrograde LSO
LSO_p(a::Float64, M::Float64) = M * (3 + Z_2(a, M) + sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # prograde LSO

dr2_dt2(t::Float64, r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64, M::Float64) = -(drdt)^2 * Kerr.KerrMetric.Γrrr(t, r, θ, ϕ, a, M) - 2.0*drdt*dθdt*Kerr.KerrMetric.Γrrθ(t, r, θ, ϕ, a, M)-Kerr.KerrMetric.Γrtt(t, r, θ, ϕ, a, M)-
dϕdt*(2.0Kerr.KerrMetric.Γrtϕ(t, r, θ, ϕ, a, M) + dϕdt*Kerr.KerrMetric.Γrϕϕ(t, r, θ, ϕ, a, M))+2.0drdt^2*(dϕdt * Kerr.KerrMetric.Γtrϕ(t, r, θ, ϕ, a, M)+Kerr.KerrMetric.Γttr(t, r, θ, ϕ, a, M)) + dθdt*(-dθdt*Kerr.KerrMetric.Γrθθ(t, r, θ, ϕ, a, M)+
2.0drdt*(Kerr.KerrMetric.Γttθ(t, r, θ, ϕ, a, M)+dϕdt*Kerr.KerrMetric.Γtθϕ(t, r, θ, ϕ, a, M)))


dθ2_dt2(t::Float64, r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64, M::Float64) = 2.0drdt * dθdt * dϕdt * Kerr.KerrMetric.Γtrϕ(t, r, θ, ϕ, a, M)+2.0drdt * dθdt * Kerr.KerrMetric.Γttr(t, r, θ, ϕ, a, M)+
2.0dθdt^2*Kerr.KerrMetric.Γttθ(t, r, θ, ϕ, a, M)+2.0dθdt^2*dϕdt * Kerr.KerrMetric.Γtθϕ(t, r, θ, ϕ, a, M)-drdt^2*Kerr.KerrMetric.Γθrr(t, r, θ, ϕ, a, M)-2.0drdt*dθdt*Kerr.KerrMetric.Γθrθ(t, r, θ, ϕ, a, M)-Kerr.KerrMetric.Γθtt(t, r, θ, ϕ, a, M)-
2.0dϕdt*Kerr.KerrMetric.Γθtϕ(t, r, θ, ϕ, a, M)-dθdt^2*Kerr.KerrMetric.Γθθθ(t, r, θ, ϕ, a, M)-dϕdt^2*Kerr.KerrMetric.Γθϕϕ(t, r, θ, ϕ, a, M)

dϕ2_dt2(t::Float64, r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64, M::Float64) = 2.0*(drdt*dϕdt^2*Kerr.KerrMetric.Γtrϕ(t, r, θ, ϕ, a, M)+
drdt*dϕdt*Kerr.KerrMetric.Γttr(t, r, θ, ϕ, a, M)+dθdt*dϕdt*Kerr.KerrMetric.Γttθ(t, r, θ, ϕ, a, M)+dθdt*dϕdt^2*Kerr.KerrMetric.Γtθϕ(t, r, θ, ϕ, a, M)-drdt*dϕdt*Kerr.KerrMetric.Γϕrϕ(t, r, θ, ϕ, a, M)-drdt*Kerr.KerrMetric.Γϕtr(t, r, θ, ϕ, a, M)-
dθdt*Kerr.KerrMetric.Γϕtθ(t, r, θ, ϕ, a, M)-dθdt*dϕdt*Kerr.KerrMetric.Γϕθϕ(t, r, θ, ϕ, a, M))

function compute_inspiral!(τOrbit::Float64, nPoints::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, h::Float64=0.15, saveat::Float64=0.5, Δti::Float64=1.0, reltol::Float64=1e-16, abstol::Float64=1e-16; data_path::String="Data/")
    # create arrays for trajectory
    t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    tdot = Float64[]; rdot = Float64[]; θdot = Float64[]; ϕdot = Float64[];
    tddot = Float64[]; rddot = Float64[]; θddot = Float64[]; ϕddot = Float64[];
    
    # initialize data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    fit_array_length = 11
    xBL = [Float64[] for i in 1:fit_array_length]
    vBL = [Float64[] for i in 1:fit_array_length]
    aBL = [Float64[] for i in 1:fit_array_length]
    xH = [Float64[] for i in 1:fit_array_length]
    x_H = [Float64[] for i in 1:fit_array_length]
    vH = [Float64[] for i in 1:fit_array_length]
    v_H = [Float64[] for i in 1:fit_array_length]
    v = zeros(fit_array_length)
    rH = zeros(fit_array_length)
    aH = [Float64[] for i in 1:fit_array_length]
    a_H = [Float64[] for i in 1:fit_array_length]
    Mij5 = zeros(3, 3)
    Mij6 = zeros(3, 3)
    Mij7 = zeros(3, 3)
    Mij8 = zeros(3, 3)
    Mijk7 = zeros(3, 3, 3)
    Mijk8 = zeros(3, 3, 3)
    Sij5 = zeros(3, 3)
    Sij6 = zeros(3, 3)
    aSF_BL_temp = zeros(4)
    aSF_H_temp = zeros(4)

    function geodesicEq(du, u, params, t)
        ddt = Kerr.KerrGeodesics.tddot(u..., du..., params...) + aSF_BL_temp[1]
        ddr = Kerr.KerrGeodesics.rddot(u..., du..., params...) + aSF_BL_temp[2]
        ddθ = Kerr.KerrGeodesics.θddot(u..., du..., params...) + aSF_BL_temp[3]
        ddϕ = Kerr.KerrGeodesics.ϕddot(u..., du..., params...) + aSF_BL_temp[4]
        @SArray [ddt, ddr, ddθ, ddϕ]
    end

    # orbital parameters
    params = @SArray [a, M];

    # define periastron and apastron
    rp = p * M / (1 + e);
    ra = p * M / (1 - e);

    # calculate integrals of motion from orbital parameters - TO-DO: CHOOSE A CONVENTION ON HOW TO HANDLE UNITS, E.G., M=1.0? USE SCHMIDT OR NEW KLUDGE FUNCTION?
    EEi, LLi, QQi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi)   # dimensionless constants

    # store orbital params in arrays
    EE = ones(nPoints) * EEi; 
    Edot = zeros(nPoints-1);
    LL = ones(nPoints) * LLi; 
    Ldot = zeros(nPoints-1);
    CC = ones(nPoints) * QQi;    # note that C in the new kludge is equal to Schmidt's Q
    Cdot = zeros(nPoints-1);
    QQ = ones(nPoints) * (CC[1] + (LL[1] - a * EE[1])^2);    # Eq. 17
    Qdot = zeros(nPoints-1);

    # square roots of negative numbers
    pArray = ones(nPoints) * p;                      # semilatus rectum p(t)
    ecc = ones(nPoints) * e;                            # eccentricity e(t)
    θmin = ones(nPoints) * θi;                         # θmin_BL(t)

    # initial conditions for Kerr geodesic trajectory
    ri = ra;
    # ics = Kerr.KerrGeodesics.boundKerr_ics(a, M, m, EEi, LLi, ri, θi, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ);
    ics = Kerr.KerrGeodesics.boundKerr_ics(a, M, EEi, LLi, ri, θi, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ);
    τ0 = 0.0; Δτ = nPoints * saveat     ### note that his gives (n+1) points in the geodesic since it starts at t=0
    τF = τ0 + Δτ; params = [a, M];
    n=1:nPoints |> collect
    rLSO = LSO_p(a, M)

    while τOrbit > τF
        println("Time = $(τ0); τOrbit = $(τOrbit)")
        τspan = (τ0, τF)   ## overshoot into the future for fit

        # stop when it reaches LSO
        condition(u, t , integrator) = u[6] - rLSO # Is zero when r = rLSO (to 5 d.p)
        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(condition, affect!)

        # numerically solve for geodesic motion
        prob = SecondOrderODEProblem(geodesicEq, ics..., τspan, params);
        
        # println(saveat)
        if e==0.0
            sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat, callback = cb);
        else
            sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat);
        end

        # AutoTsit5(DP8())
        # deconstruct solution - ignore part of solution which overshoots for fit
        ttdot = sol[1,:];
        rrdot = sol[2,:];
        θθdot = sol[3,:];
        ϕϕdot = sol[4,:];
        tt = sol[5,:];
        rr = sol[6,:];
        θθ = sol[7,:];
        ϕϕ= sol[8,:];

        # println(length(tt))
        # println(τspan)
        # break out of loop when LSO reached- either the integration terminated, or there is a repeated value of t (due to ODE solver feature)
        if (length(sol[1, :]) < nPoints+1) | !all(≠(0), diff(tt))
            println("Integration terminated at t = $(last(t))")
            break
        elseif length(tt)>nPoints+1
            # deconstruct solution
            ttdot = sol[1, 1:nPoints+1];
            rrdot = sol[2, 1:nPoints+1];
            θθdot = sol[3, 1:nPoints+1];
            ϕϕdot = sol[4, 1:nPoints+1];
            tt = sol[5, 1:nPoints+1];
            rr = sol[6, 1:nPoints+1];
            θθ = sol[7, 1:nPoints+1];
            ϕϕ= sol[8, 1:nPoints+1];
            τF = sol.t[nPoints+1]
        end

        # save initial conditions for mulitpole fit before updating initial conditions for next geodesic piece
        icsMultipoleFit = ics

        # save endpoints for initial conditions of next geodesic
        ics = [@SArray[last(ttdot), last(rrdot), last(θθdot), last(ϕϕdot)], @SArray[last(tt), last(rr), last(θθ), last(ϕϕ)]];

        # update evolution times for next geodesic piece
        τ0 = τF
        τF += Δτ

        # remove last elements to not apply SF twice at end/start points
        pop!(ttdot); pop!(rrdot); pop!(θθdot); pop!(ϕϕdot); pop!(tt); pop!(rr); pop!(θθ); pop!(ϕϕ);

        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt τ)
        ttddot = Kerr.KerrGeodesics.tddot.(tt, rr, θθ, ϕϕ, ttdot, rrdot, θθdot, ϕϕdot, params...);
        rrddot = Kerr.KerrGeodesics.rddot.(tt, rr, θθ, ϕϕ, ttdot, rrdot, θθdot, ϕϕdot, params...);
        θθddot = Kerr.KerrGeodesics.θddot.(tt, rr, θθ, ϕϕ, ttdot, rrdot, θθdot, ϕϕdot, params...);
        ϕϕddot = Kerr.KerrGeodesics.ϕddot.(tt, rr, θθ, ϕϕ, ttdot, rrdot, θθdot, ϕϕdot, params...);

        # store parts of trajectory
        append!(t, tt); append!(tdot, ttdot); append!(tddot, ttddot); append!(r, rr); append!(rdot, rrdot); append!(rddot, rrddot); append!(θ, θθ); append!(θdot, θθdot); append!(θddot, θθddot); append!(ϕ, ϕϕ); append!(ϕdot, ϕϕdot); append!(ϕddot, ϕϕddot);
        
        ###### COMPUTE SELF-FORCE ######

        # we first compute the fundamental frequencies in order to determine over what interval of time a fit needs to be carried out
        ω = Kerr.ConstantsOfMotion.KerrFreqs(a, last(pArray), last(ecc), last(θmin)); Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];
        τFit = minimum(@. 2π/ω[1:3]);
        nPointsMultipoleFit = floor(Int, 1 + τFit/h)

        # Our fitting method is as follows: we want to perform our fit over the 'future' and 'past' of the point at which we wish to compute the self-force. In other words, we would like 
        # to perform a fit to data, and take the values of the fit at the center of the arrays (this has obvious benefits since interpolation/numerical differentiation schemes often
        # have unwieldly "edge" effects. To achieve this centering, we first evolve a geodesic from τ0 to τF-τFit/2, in order to obtain initial condiditions of the trajectory at the 
        # time τ=τF-τFit/2. We then use these initial conditions to evolve a geodesic from τF-τFit/2 to τF+τFit/2, which places the point at which we want to compute the self force 
        # at the center of the data array. Then we use this data to carry out a "fourier fit". Note that all the geodesic data we compute is solely for the computation of the self-force,
        # and will be discarded thereafter.
        
        # begin by carrying out fit from τ0 to τF-τFit/2
        saveat_multipole_fit = h
        τSpanMultipoleFit = (τspan[2] - saveat_multipole_fit * (nPointsMultipoleFit ÷ 2), τspan[2] + saveat_multipole_fit * (nPointsMultipoleFit ÷ 2))    # this range ensures that τF is the center point
        compute_at = Int(1 + (nPointsMultipoleFit÷2))    # this will be the index of τF in the trajectory data arrays

        τspanFit0 = (τ0, τSpanMultipoleFit[1])
        prob = SecondOrderODEProblem(geodesicEq, icsMultipoleFit..., τspanFit0, params);
        
        if e==0.0
            sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat, callback = cb);
        else
            sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat);
        end

        ttdot = sol[1, :];
        rrdot = sol[2, :];
        θθdot = sol[3, :];
        ϕϕdot = sol[4, :];
        tt = sol[5, :];
        rr = sol[6, :];
        θθ = sol[7, :];
        ϕϕ= sol[8, :];

        icsMultipoleFit = [@SArray[last(ttdot), last(rrdot), last(θθdot), last(ϕϕdot)], @SArray[last(tt), last(rr), last(θθ), last(ϕϕ)]];    # initial conditions at τ=τF

        # fit from τF-τFit/2 to τF+τFit/2
        prob = SecondOrderODEProblem(geodesicEq, icsMultipoleFit..., τSpanMultipoleFit, params);
        
        if e==0.0
            sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_multipole_fit, callback = cb);
        else
            sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_multipole_fit);
        end
        ttdot = sol[1, (compute_at-5):(compute_at+5)];
        dr_dt = @. sol[2, (compute_at-5):(compute_at+5)] / ttdot;
        dθ_dt = sol[3, (compute_at-5):(compute_at+5)] / ttdot;
        dϕ_dt = sol[4, (compute_at-5):(compute_at+5)] / ttdot;
        tt = sol[5, (compute_at-5):(compute_at+5)];
        rr = sol[6, (compute_at-5):(compute_at+5)];
        θθ = sol[7, (compute_at-5):(compute_at+5)];
        ϕϕ= sol[8, (compute_at-5):(compute_at+5)];

        ττ=sol.t[(compute_at-5):(compute_at+5)];
        compute_at_2=6

        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        r_ddot = HJEvolution.dr2_dt2.(tt, rr, θθ, ϕϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
        θ_ddot = HJEvolution.dθ2_dt2.(tt, rr, θθ, ϕϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
        ϕ_ddot = HJEvolution.dϕ2_dt2.(tt, rr, θθ, ϕϕ, dr_dt, dθ_dt, dϕ_dt, a, M)


        println("Check that we are computing the self-force at the right value:\n τF=$(τspan[2]), compute_at=$(ττ[compute_at_2])")
        println("Length of solution = $(size(ttdot, 1))")

        # calculate SF at each point of trajectory and take the sum
        SelfForce_numerical.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL, vBL, aBL, xH, x_H, rH, vH, v_H, aH, a_H, v, ττ, ttdot, rr, rrdot, rrddot, θθ, θθdot, θθddot, ϕϕ, ϕϕdot, ϕϕddot, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m,compute_at_2, h);
        selfAcc!(aSF_H, aSF_BL, xBL, vBL, aBL, xH, x_H, rH, vH, v_H, aH, a_H, v, t, r, dr_dt, d2r_dt2, θ, dθ_dt, d2θ_dt2, ϕ, dϕ_dt, d2ϕ_dt2, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m, compute_at, h)
        EvolveConstants(saveat, a, tt[compute_at_2], rr[compute_at_2], θθ[compute_at_2], ϕϕ[compute_at_2], ttdot[compute_at_2], rrdot[compute_at_2], θθdot[compute_at_2], ϕϕdot[compute_at_2], aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPoints)
        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)
    end

    # delete final "extra" energies and fluxes
    delete_first = size(EE, 1) - (nPoints-1)
    deleteat!(EE, delete_first:(delete_first+nPoints-1))
    deleteat!(LL, delete_first:(delete_first+nPoints-1))
    deleteat!(QQ, delete_first:(delete_first+nPoints-1))
    deleteat!(CC, delete_first:(delete_first+nPoints-1))
    deleteat!(pArray, delete_first:(delete_first+nPoints-1))
    deleteat!(ecc, delete_first:(delete_first+nPoints-1))
    deleteat!(θmin, delete_first:(delete_first+nPoints-1))

    delete_first = size(Edot, 1) - (nPoints-2)
    deleteat!(Edot, delete_first:(delete_first+nPoints-2))
    deleteat!(Ldot, delete_first:(delete_first+nPoints-2))
    deleteat!(Qdot, delete_first:(delete_first+nPoints-2))
    deleteat!(Cdot, delete_first:(delete_first+nPoints-2))

    # save data 
    mkpath(data_path)

    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end

    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # number of data points
    n_OrbPoints = size(r, 1)

    # save trajectory
    τRange = 0.0:saveat:(n_OrbPoints-1) * saveat |> collect
    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

function compute_inspiral_HJE!(tOrbit::Float64, nPoints::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, h::Float64=0.15, reltol::Float64=1e-12, abstol::Float64=1e-10; data_path::String="Data/")
    # create arrays for trajectory
    t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[];
    
    # initialize data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    fit_array_length = 11
    xBL = [Float64[] for i in 1:fit_array_length]
    vBL = [Float64[] for i in 1:fit_array_length]
    aBL = [Float64[] for i in 1:fit_array_length]
    xH = [Float64[] for i in 1:fit_array_length]
    x_H = [Float64[] for i in 1:fit_array_length]
    vH = [Float64[] for i in 1:fit_array_length]
    v_H = [Float64[] for i in 1:fit_array_length]
    v = zeros(fit_array_length)
    rH = zeros(fit_array_length)
    aH = [Float64[] for i in 1:fit_array_length]
    a_H = [Float64[] for i in 1:fit_array_length]
    Mij5 = zeros(3, 3)
    Mij6 = zeros(3, 3)
    Mij7 = zeros(3, 3)
    Mij8 = zeros(3, 3)
    Mijk7 = zeros(3, 3, 3)
    Mijk8 = zeros(3, 3, 3)
    Sij5 = zeros(3, 3)
    Sij6 = zeros(3, 3)
    aSF_BL_temp = zeros(4)
    aSF_H_temp = zeros(4)

    # compute apastron
    ra = p * M / (1 - e);

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi, CCi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)   

    # store orbital params in arrays
    EE = ones(nPoints) * EEi; 
    Edot = zeros(nPoints-1);
    LL = ones(nPoints) * LLi; 
    Ldot = zeros(nPoints-1);
    CC = ones(nPoints) * CCi;
    Cdot = zeros(nPoints-1);
    QQ = ones(nPoints) * QQi
    Qdot = zeros(nPoints-1);
    pArray = ones(nPoints) * p;
    ecc = ones(nPoints) * e;
    θmin = ones(nPoints) * θi;
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    ics = HJEvolution.HJ_ics(ra, p, e, M);

    rLSO = LSO_p(a, M)
    while tOrbit > t0
        # orbital parameters during current piecewise geodesic
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θmin); e_t = last(ecc);
        print("Completion: $(100 * t0/tOrbit)%   \r")
        flush(stdout)   

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t * M / (1.0 - e_t); rp=p_t * M / (1.0 + e_t);
        A = M / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t) / M; p4 = r4 * (1.0 + e_t) / M    # Above Eq. 96

        # array of params for ODE solver
        params = @SArray [a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm]

        # to compute the self force at a point, we must overshoot the solution into the future
        tF = t0 + (nPoints-1) * h + (fit_array_length÷2) * h   # evolve geodesic up to tF
        total_num_points = nPoints+(fit_array_length÷2)   # total number of points in geodesic since we overshoot
        Δti=h;    # initial time step for geodesic integration

        saveat_t = t0:h:tF |> collect    # saveat time array for solver
        tspan=(t0, tF)

        # stop when it reaches LSO
        condition(u, t , integrator) = u[1] - rLSO # Is zero when r = rLSO (to 5 d.p)
        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(condition, affect!)

        # numerically solve for geodesic motion
        prob = ODEProblem(HJEvolution.geodesicEq, ics, tspan, params);
        
        if e==0.0
            sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t, callback = cb);
        else
            sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);
        end

        
        # deconstruct solution
        tt = sol.t;
        psi = sol[1, :];
        chi = mod.(sol[2, :], 2π);
        ϕϕ = sol[3, :];

        if (length(sol[1, :]) < total_num_points)
            println("Integration terminated at t = $(last(t))")
            println("(nPoints+1) - len(sol) = $(nPoints+1-length(sol[1,:]))")
            break
        elseif length(tt)>total_num_points
            tt = sol.t[:total_num_points];
            psi = sol[1, 1:total_num_points];
            chi = mod.(sol[2, 1:total_num_points], 2π);
            ϕϕ = sol[3, 1:total_num_points];
        end

        # compute time derivatives
        psi_dot = HJEvolution.psi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        chi_dot = HJEvolution.chi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        ϕ_dot = HJEvolution.phi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)

        # compute BL coordinates t, r, θ and their time derivatives
        rr = HJEvolution.r.(psi, p_t, e_t, M)
        θθ = [acos((π/2<chi[i]<1.5π) ? -sqrt(HJEvolution.z(chi[i], θmin_t)) : sqrt(HJEvolution.z(chi[i], θmin_t))) for i in eachindex(chi)]

        r_dot = HJEvolution.dr_dt.(psi_dot, psi, p_t, e_t, M);
        θ_dot = HJEvolution.dθ_dt.(chi_dot, chi, θθ, θmin_t);
        v_spatial = [[r_dot[i], θ_dot[i], ϕ_dot[i]] for i in eachindex(tt)];
        Γ = @. HJEvolution.Γ(tt, rr, θθ, ϕϕ, v_spatial, a, M)

        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        r_ddot = HJEvolution.dr2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)
        θ_ddot = HJEvolution.dθ2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)
        ϕ_ddot = HJEvolution.dϕ2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)

        ###### MIGHT WANT TO USE VIEWS TO OPTIMIZE A BIT AND AVOID MAKING COPIES IN EACH CALL BELOW ######

        # store trajectory, ignoring the overshot piece
        append!(t, tt[1:nPoints]); append!(dt_dτ, Γ[1:nPoints]); append!(r, rr[1:nPoints]); append!(dr_dt, r_dot[1:nPoints]); append!(d2r_dt2, r_ddot[1:nPoints]); 
        append!(θ, θθ[1:nPoints]); append!(dθ_dt, θ_dot[1:nPoints]); append!(d2θ_dt2, θ_ddot[1:nPoints]); append!(ϕ, ϕϕ[1:nPoints]); 
        append!(dϕ_dt, ϕ_dot[1:nPoints]); append!(d2ϕ_dt2, ϕ_ddot[1:nPoints]);
        
        ###### COMPUTE SELF-FORCE ######
        fit_index_0 = nPoints - (fit_array_length÷2); fit_index_1 = nPoints + (fit_array_length÷2); compute_at=fit_array_length÷2+1;
        
        SelfForce_numerical.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL, vBL, aBL, xH, x_H, rH, vH, v_H, aH, a_H, v, tt[fit_index_0:fit_index_1], 
        rr[fit_index_0:fit_index_1], r_dot[fit_index_0:fit_index_1], r_ddot[fit_index_0:fit_index_1], θθ[fit_index_0:fit_index_1], 
        θ_dot[fit_index_0:fit_index_1], θ_ddot[fit_index_0:fit_index_1], ϕϕ[fit_index_0:fit_index_1], ϕ_dot[fit_index_0:fit_index_1], 
        ϕ_ddot[fit_index_0:fit_index_1], Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, 
        Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m, compute_at, h);

        EvolveConstants(tt[nPoints]-tt[1], a, tt[nPoints], rr[nPoints], θθ[nPoints], ϕϕ[nPoints], Γ[nPoints], r_dot[nPoints], θ_dot[nPoints], ϕ_dot[nPoints], aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPoints)
        
        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)

        # update next ics for next piece
        t0 = tt[nPoints+1];
        ics = @SArray [psi[nPoints+1], chi[nPoints+1], ϕϕ[nPoints+1]]
    end

    # delete final "extra" energies and fluxes
    delete_first = size(EE, 1) - (nPoints-1)
    deleteat!(EE, delete_first:(delete_first+nPoints-1))
    deleteat!(LL, delete_first:(delete_first+nPoints-1))
    deleteat!(QQ, delete_first:(delete_first+nPoints-1))
    deleteat!(CC, delete_first:(delete_first+nPoints-1))
    deleteat!(pArray, delete_first:(delete_first+nPoints-1))
    deleteat!(ecc, delete_first:(delete_first+nPoints-1))
    deleteat!(θmin, delete_first:(delete_first+nPoints-1))

    delete_first = size(Edot, 1) - (nPoints-2)
    deleteat!(Edot, delete_first:(delete_first+nPoints-2))
    deleteat!(Ldot, delete_first:(delete_first+nPoints-2))
    deleteat!(Qdot, delete_first:(delete_first+nPoints-2))
    deleteat!(Cdot, delete_first:(delete_first+nPoints-2))

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end


    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # number of data points
    n_OrbPoints = size(r, 1)

    # save trajectory
    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end


function compute_inspiral_Mino!(tOrbit::Float64, nPoints::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, h::Float64=0.15, reltol::Float64=1e-12, abstol::Float64=1e-10; data_path::String="Data/")
    # create arrays for trajectory
    t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[];
    Mij2_wf = [Float64[] for i=1:3, j=1:3];
    Mijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf = [Float64[] for i=1:3, j=1:3];
    Sijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    
    # initialize data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    fit_array_length = 11;
    total_num_points = nPoints + fit_array_length÷2;
    # arrays for self force computation
    xBL = [Float64[] for i in 1:fit_array_length]
    vBL = [Float64[] for i in 1:fit_array_length]
    aBL = [Float64[] for i in 1:fit_array_length]
    xH = [Float64[] for i in 1:fit_array_length]
    x_H = [Float64[] for i in 1:fit_array_length]
    vH = [Float64[] for i in 1:fit_array_length]
    v_H = [Float64[] for i in 1:fit_array_length]
    v = zeros(fit_array_length)
    rH = zeros(fit_array_length)
    aH = [Float64[] for i in 1:fit_array_length]
    a_H = [Float64[] for i in 1:fit_array_length]

    # arrays for waveform computation
    xBL_wf = [Float64[] for i in 1:nPoints]
    vBL_wf = [Float64[] for i in 1:nPoints]
    aBL_wf = [Float64[] for i in 1:nPoints]
    xH_wf = [Float64[] for i in 1:nPoints]
    x_H_wf = [Float64[] for i in 1:nPoints]
    vH_wf = [Float64[] for i in 1:nPoints]
    v_H_wf = [Float64[] for i in 1:nPoints]
    v_wf = zeros(nPoints)
    rH_wf = zeros(nPoints)
    aH_wf = [Float64[] for i in 1:nPoints]
    a_H_wf = [Float64[] for i in 1:nPoints]


    # arrays for multipole moments
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Mijkl2_data = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    Sijk1_data= [Float64[] for i=1:3, j=1:3, k=1:3]

    # "temporary" mulitpole arrays which contain the multipole data for a given piecewise geodesic
    Mij2_wf_temp = [zeros(nPoints) for i=1:3, j=1:3];
    Mijk3_wf_temp = [zeros(nPoints) for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf_temp = [zeros(nPoints) for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf_temp = [zeros(nPoints) for i=1:3, j=1:3];
    Sijk3_wf_temp = [zeros(nPoints) for i=1:3, j=1:3, k=1:3];

    # arrays for self-force computation
    Mij5 = zeros(3, 3)
    Mij6 = zeros(3, 3)
    Mij7 = zeros(3, 3)
    Mij8 = zeros(3, 3)
    Mijk7 = zeros(3, 3, 3)
    Mijk8 = zeros(3, 3, 3)
    Sij5 = zeros(3, 3)
    Sij6 = zeros(3, 3)
    aSF_BL_temp = zeros(4)
    aSF_H_temp = zeros(4)

    # compute apastron
    ra = p * M / (1 - e);

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi, CCi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)   

    # store orbital params in arrays
    EE = ones(nPoints) * EEi; 
    Edot = zeros(nPoints-1);
    LL = ones(nPoints) * LLi; 
    Ldot = zeros(nPoints-1);
    CC = ones(nPoints) * CCi;
    Cdot = zeros(nPoints-1);
    QQ = ones(nPoints) * QQi
    Qdot = zeros(nPoints-1);
    pArray = ones(nPoints) * p;
    ecc = ones(nPoints) * e;
    θmin = ones(nPoints) * θi;
    # initial condition for Kerr geodesic trajectory
    λ0 = 0.0;
    t0 = 0.0;
    ics = MinoEvolution.Mino_ics(t0, ra, p, e, M);
    Δλi=h/10;    # initial time step for geodesic integration

    rLSO = LSO_p(a, M)
    while tOrbit > t0
        # orbital parameters during current piecewise geodesic
        E_λ = last(EE); L_λ = last(LL); C_λ = last(CC); Q_λ = last(QQ); p_λ = last(pArray); θmin_λ = last(θmin); e_λ = last(ecc);
        print("Completion: $(100 * t0/tOrbit)%   \r")
        flush(stdout)
    
        # to compute the self force at a point, we must overshoot the solution into the future
        saveat_λ = range(start=λ0, step=h, length=total_num_points)|>collect
        λspan=(saveat_λ[1], last(saveat_λ))
        
        sol = MinoEvolution.compute_kerr_geodesic_inspiral(a, p_λ, e_λ, θmin_λ, E_λ, L_λ, Q_λ, C_λ, M, ics, saveat_λ, λspan, Δλi, reltol, abstol)

        tt=sol[2,:]

        if (length(tt) ≠ total_num_points)
            println("Integration terminated at t = $(last(t))")
            println("total_num_points - len(sol) = $(total_num_points-length(sol[1,:]))")
            println(sol[1,:])
            println(saveat_λ)
            break
        else
            λλ=sol[1,:]; rr=sol[3,:]; θθ=sol[4,:]; ϕϕ=sol[5,:]; r_dot=sol[6,:]; θ_dot=sol[7,:]; ϕ_dot=sol[8,:]; r_ddot=sol[9,:]; θ_ddot=sol[10,:]; ϕ_ddot=sol[11,:]; ΓΓ = sol[13,:];
            psi = sol[14,:]; chi = sol[15,:];
        end

        ###### MIGHT WANT TO USE VIEWS TO OPTIMIZE A BIT AND AVOID MAKING COPIES IN EACH CALL BELOW ######
        SelfForce_numerical.compute_waveform_moments_and_derivs_Mino!(a, E_λ, L_λ, m, M, xBL_wf, vBL_wf, aBL_wf, xH_wf, x_H_wf, rH_wf, vH_wf, v_H_wf, aH_wf, a_H_wf, v_wf, λλ[1:nPoints], rr[1:nPoints], r_dot[1:nPoints], r_ddot[1:nPoints], θθ[1:nPoints], θ_dot[1:nPoints], 
            θ_ddot[1:nPoints], ϕϕ[1:nPoints], ϕ_dot[1:nPoints], ϕ_ddot[1:nPoints], Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp, nPoints, h)

        # store trajectory, ignoring the overshot piece
        append!(t, tt[1:nPoints]); append!(dt_dτ, ΓΓ[1:nPoints]); append!(r, rr[1:nPoints]); append!(dr_dt, r_dot[1:nPoints]); append!(d2r_dt2, r_ddot[1:nPoints]); 
        append!(θ, θθ[1:nPoints]); append!(dθ_dt, θ_dot[1:nPoints]); append!(d2θ_dt2, θ_ddot[1:nPoints]); append!(ϕ, ϕϕ[1:nPoints]); 
        append!(dϕ_dt, ϕ_dot[1:nPoints]); append!(d2ϕ_dt2, ϕ_ddot[1:nPoints]);
        
        # store multipole data for waveforms — note that we only save the independent components
        @inbounds Threads.@threads for indices in ConstructSymmetricArrays.waveform_indices
            if length(indices)==2
                i1, i2 = indices
                append!(Mij2_wf[i1, i2], Mij2_data[i1, i2])
                append!(Sij2_wf[i1, i2], Sij2_wf_temp[i1, i2])
            elseif length(indices)==3
                i1, i2, i3 = indices
                append!(Mijk3_wf[i1, i2, i3], Mijk3_wf_temp[i1, i2, i3])
                append!(Sijk3_wf[i1, i2, i3], Sijk3_wf_temp[i1, i2, i3])
            else
                i1, i2, i3, i4 = indices
                append!(Mijkl4_wf[i1, i2, i3, i4], Mijkl4_wf_temp[i1, i2, i3, i4])
            end
        end

        ###### COMPUTE SELF-FORCE ######
        fit_index_0 = nPoints - (fit_array_length÷2); fit_index_1 = nPoints + (fit_array_length÷2); compute_at=fit_array_length÷2+1;
        SelfForce_numerical.selfAcc_mino!(a, M, E_λ, L_λ, aSF_H_temp, aSF_BL_temp, xBL, vBL, aBL, xH, x_H, rH, vH, v_H, aH, a_H, v, tt[fit_index_0:fit_index_1], 
        rr[fit_index_0:fit_index_1], r_dot[fit_index_0:fit_index_1], r_ddot[fit_index_0:fit_index_1], θθ[fit_index_0:fit_index_1], 
        θ_dot[fit_index_0:fit_index_1], θ_ddot[fit_index_0:fit_index_1], ϕϕ[fit_index_0:fit_index_1], ϕ_dot[fit_index_0:fit_index_1], 
        ϕ_ddot[fit_index_0:fit_index_1], Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, 
        Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, m, compute_at, h);

        EvolveConstants(tt[nPoints]-tt[1], a, tt[nPoints], rr[nPoints], θθ[nPoints], ϕϕ[nPoints], ΓΓ[nPoints], r_dot[nPoints], θ_dot[nPoints], ϕ_dot[nPoints], aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPoints)
        
        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)

        # update next ics for next piece
        λ0 = λλ[nPoints+1];
        t0 = tt[nPoints+1];
        ics = @SArray [t0, psi[nPoints+1], chi[nPoints+1], ϕϕ[nPoints+1]]
    end

    # delete final "extra" energies and fluxes
    delete_first = size(EE, 1) - (nPoints-1)
    deleteat!(EE, delete_first:(delete_first+nPoints-1))
    deleteat!(LL, delete_first:(delete_first+nPoints-1))
    deleteat!(QQ, delete_first:(delete_first+nPoints-1))
    deleteat!(CC, delete_first:(delete_first+nPoints-1))
    deleteat!(pArray, delete_first:(delete_first+nPoints-1))
    deleteat!(ecc, delete_first:(delete_first+nPoints-1))
    deleteat!(θmin, delete_first:(delete_first+nPoints-1))

    delete_first = size(Edot, 1) - (nPoints-2)
    deleteat!(Edot, delete_first:(delete_first+nPoints-2))
    deleteat!(Ldot, delete_first:(delete_first+nPoints-2))
    deleteat!(Qdot, delete_first:(delete_first+nPoints-2))
    deleteat!(Cdot, delete_first:(delete_first+nPoints-2))

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_mino_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end


    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_mino_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_mino_fdm.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end


    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_mino_fdm.jld2"
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_mino_fdm.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_mino_fdm.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end


function load_Mino_trajectory(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, h::Float64, reltol::Float64, data_path::String)
    # load ODE solution
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_mino_fdm.txt"
    sol = readdlm(ODE_filename)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; dr_dt=sol[5,:]; dθ_dt=sol[6,:]; dϕ_dt=sol[7,:]; d2r_dt2=sol[8,:]; d2θ_dt2=sol[9,:]; d2ϕ_dt2=sol[10,:]; dt_dτ=sol[11,:]
    return t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, h::Float64, reltol::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_mino_fdm.txt"
    constants=readdlm(constants_filename)
    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_mino_fdm.txt"
    constants_derivs = readdlm(constants_derivs_filename)
    EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :]
    Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]
    return EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin
end


function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, t::Vector{Float64}, a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, 
    h::Float64, reltol::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=data_path *  "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_mino_fdm.jld2"
    waveform_data = load(waveform_filename)["data"]
    Mij2 = waveform_data["Mij2"]; ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij2);
    Mijk3 = waveform_data["Mijk3"]; ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk3);
    Mijkl4 = waveform_data["Mijkl4"]; ConstructSymmetricArrays.SymmetrizeFourIndexTensor!(Mijkl4);
    Sij2 = waveform_data["Sij2"]; ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij2);
    Sijk3 = waveform_data["Sijk3"]; ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Sijk3);

    # compute h_{ij} tensor
    hij = [zeros(length(t)) for i=1:3, j=1:3];
    SelfForce.hij!(hij, t, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    # project h_{ij} tensor
    h_plus = SelfForce.hplus(hij, Θ, Φ);
    h_cross = SelfForce.hcross(hij, Θ, Φ);
    return h_plus, h_cross
end

end