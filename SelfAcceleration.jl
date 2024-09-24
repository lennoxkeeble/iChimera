#= 

    In this module we write the analytic expressions for the self-force. See Eqs. 54-56, 57, 61-63, and A1-A14 in arXiv:1109.0572v2

=#
module SelfAcceleration
using LinearAlgebra
using Combinatorics
import ..HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ..HarmonicCoords
using StaticArrays
using ..PotentialsRR

# define some useful functions
otimes(a::AbstractVector{Float64}, b::AbstractVector{Float64}) = [a[i] * b[j] for i=1:size(a, 1), j=1:size(b, 1)]    # tensor product of two vectors
otimes(a::AbstractVector{Float64}) = [a[i] * a[j] for i=1:size(a, 1), j=1:size(a, 1)]    # tensor product of a vector with itself
dot3d(u::AbstractVector{Float64}, v::AbstractVector{Float64}) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3]
norm2_3d(u::AbstractVector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
norm_3d(u::AbstractVector{Float64}) = sqrt(norm2_3d(u))
dot4d(u::AbstractVector{Float64}, v::AbstractVector{Float64}) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4]
norm2_4d(u::AbstractVector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3] + u[4] * u[4]
norm_4d(u::AbstractVector{Float64}) = sqrt(norm2_4d(u))

ημν = [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]    # minkowski metric
ηij = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]    # spatial part of minkowski metric
δ(x::Int, y::Int)::Int = x == y ? 1 : 0

### KERR POTENTIALS, Γ-FACTOR AND PROJECTION OPERATOR ###

# define vector and scalar potentials for self-force calculation - underscore denotes covariant indices
K(xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_tt_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) + 1.0                   # outputs K00 (Eq. 54)
K_i(xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_tr_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)                       # outputs Ki vector, i.e., Ki for i ∈ {1, 2, 3} (Eq. 55)
K_ij(xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_rr_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) - ηij                # outputs Kij matrix (Eq. 56)
K_μν(xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_μν_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) - ημν                # outputs Kμν matrix
Q(xH::AbstractArray, a::Float64, M::Float64, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = gTT_H(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) + 1.0                          # outputs Q^00 (Eq. 54)
Qi(xH::AbstractArray, a::Float64, M::Float64, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = gTR_H(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)                               # outputs Q^i vector, i.e., Q^i for i ∈ {1, 2, 3} (Eq. 55)
Qij(xH::AbstractArray, a::Float64, M::Float64, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = gRR_H(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) - ηij                        # outputs diagonal of Q^ij matrix (Eq. 56)
Qμν(xH::AbstractArray, a::Float64, M::Float64, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = gμν_H(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) - ημν                        # outputs Qμν matrix

# define partial derivatives of K (in harmonic coordinates)
# ∂ₖK: outputs float
function ∂K_∂xk(xH::AbstractArray, xBL::AbstractArray, jBLH::AbstractArray, HessBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int)   # Eq. A12
    ∂K=0.0
    @inbounds for μ=1:4
        for i=1:3
            ∂K += 2 * g_μν(0., xBL..., a, M, 1, μ) * Γαμν(0., xBL..., a, M, μ, 1, i+1) * jBLH[i, k]   # i → i + 1 to go from spatial indices to spacetime indices
        end
    end
    return ∂K
end

# ∂ₖKᵢ: outputs float. Note: rH = norm(xH).
function ∂Ki_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, HessBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int)   # Eq. A13
    ∂K=0.0
    @inbounds for m=1:3   # start with iteration over m to not over-count last terms
        ∂K += g_μν(0., xBL..., a, M, 1, m+1) * HessBLH[m][k, i]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices
        @inbounds for μ=1:4, n=1:3
            ∂K += (g_μν(0., xBL..., a, M, μ, 1) * Γαμν(0., xBL..., a, M, μ, m+1, n+1) + g_μν(0., xBL..., a, M, μ, m+1) * Γαμν(0., xBL..., a, M, μ, 1, n+1)) * jBLH[n, k] * jBLH[m, i]   # first term of Eq. A13
        end
    end
    return ∂K
end

# ∂ₖKᵢⱼ: outputs float. Note: rH = norm(xH).
function ∂Kij_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, HessBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int, j::Int)   # Eq. A14
    ∂K=0.0
    @inbounds for m=1:3
        @inbounds for l=1:3   # iterate over m and l first to avoid over-counting
            ∂K += g_μν(0., xBL..., a, M, l+1, m+1) * (HessBLH[l][k, i] * jBLH[m, j] + HessBLH[l][k, j] * jBLH[m, i])  # last term Eq. A14
            @inbounds for μ=1:4, n=1:3
                ∂K += (g_μν(0., xBL..., a, M, μ, l+1) * Γαμν(0., xBL..., a, M, μ, m+1, n+1) + g_μν(0., xBL..., a, M, μ, m+1) * Γαμν(0., xBL..., a, M, μ, l+1, n+1)) * jBLH[n, k] * jBLH[l, i] * jBLH[m, j]   # first term of Eq. A14
            end
        end
    end
    return ∂K
end

# define GR Γ factor, v_H = contravariant velocity in harmonic coordinates
Γ(vH::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = 1.0 / sqrt(1.0 - SelfAcceleration.norm2_3d(vH) - K(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) - 2.0 * dot(K_i(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ), vH) - transpose(vH) * K_ij(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) * vH)   # Eq. A3

# define projection operator
Pαβ(vH::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = ημν + Qμν(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) + Γ(vH, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * otimes(vcat([1], vH))   # contravariant, Eq. A1
P_αβ(vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) =  ημν + K_μν(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) + Γ(vH, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * otimes(vcat([1], vH))   # cοvariant, Eq. A2 (note that we take both contravariant and covariant velocities as arguments)

### SELF-ACCELERATION PIECES ###

# compute self-acceleration pieces
function A_RR(t::Float64, xH::AbstractArray, v::Float64, vH::AbstractArray, ∂Vrr_∂t::Float64, ∂Vrr_∂a::SVector{3, Float64}, ∂Virr_∂a::SMatrix{3, 3, Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    aRR = (1.0 - v^2) * ∂Vrr_∂t   # first term in Eq. A4
    @inbounds for i=1:3
        aRR += 2.0 * vH[i] * ∂Vrr_∂a[i]   # second term Eq. A4
        @inbounds for j=1:3
            aRR += -4.0 * vH[i] * vH[j] * ∂Virr_∂a[i, j]   # third term Eq. A4
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
    ∂Vrr_∂t = PotentialsRR.∂Vrr_∂t(t, xH, Mij6, Mij8, Mijk8)
    ∂Vrr_∂a = @SVector [PotentialsRR.∂Vrr_∂a(t, xH, Mij5, Mij7, Mijk7, i) for i =1:3]
    ∂Virr_∂t = @SVector [PotentialsRR.∂Virr_∂t(t, xH, Mij7, Sij6, i) for i =1:3]
    ∂Virr_∂a = @SMatrix [PotentialsRR.∂Virr_∂a(t, xH, Mij6, Sij5, i, j) for i=1:3, j=1:3]
    return [i==1 ? A_RR(t, xH, v, vH, ∂Vrr_∂t, ∂Vrr_∂a, ∂Virr_∂a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6) : Ai_RR(t, xH, v, v_H, vH, ∂Vrr_∂t, ∂Virr_∂t, ∂Vrr_∂a, ∂Virr_∂a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, i-1) for i = 1:4]
end

function B_RR(xH::AbstractVector{Float64}, Qi::AbstractVector{Float64}, ∂K_∂xk::SVector{3, Float64}, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    return dot(Qi, ∂K_∂xk)   # Eq. A6
end

function Bi_RR(xH::AbstractVector{Float64}, Qij::AbstractArray, ∂K_∂xk::SVector{3, Float64}, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    return -2.0 * (ηij + Qij) * ∂K_∂xk   # Eq. A9
end
function C_RR(xH::AbstractVector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂K_∂xk::SVector{3, Float64}, ∂Ki_∂xk::SMatrix{3, 3, Float64}, Q::Float64, Qi::AbstractVector{Float64}, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    C = 0.0
    @inbounds for i=1:3
        C += 2.0 * (1.0 - Q) * vH[i] * ∂K_∂xk[i]
        @inbounds for j=1:3
            C += 2.0 * Qi[i] * vH[j] * (∂Ki_∂xk[i, j] - ∂Ki_∂xk[j, i])
        end
    end
    return C
end

function Ci_RR(xH::AbstractVector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂K_∂xk::SVector{3, Float64}, ∂Ki_∂xk::SMatrix{3, 3, Float64}, Qi::AbstractVector{Float64}, Qij::AbstractArray, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)   # Eq. A10
    C = @MVector [0., 0., 0.]
    @inbounds for j=1:3
        @inbounds for i=1:3
            C[i] += 4.0 * Qi[i] * vH[j] * ∂K_∂xk[j]
        end
        C .+= 4.0 * (ηij + Qij) * vH[j] * ([(∂Ki_∂xk[j, k] - ∂Ki_∂xk[k, j]) for k=1:3]) 
    end
    return C
end

function D_RR(xH::AbstractVector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂Ki_∂xk::SMatrix{3, 3, Float64}, ∂Kij_∂xk::SArray{Tuple{3, 3, 3}, Float64, 3, 27}, Q::Float64, Qi::AbstractVector{Float64}, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
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

function Di_RR(xH::AbstractVector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂Ki_∂xk::SMatrix{3, 3, Float64}, ∂Kij_∂xk::SArray{Tuple{3, 3, 3}, Float64, 3, 27}, Qi::AbstractVector{Float64}, Qij::AbstractArray, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)   # Eq. A11
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
    HessBLH = [HarmonicCoords.HessBLH(xH, rH, a, M, m) for m=1:3]
    ∂K_∂xk = @SVector [SelfAcceleration.∂K_∂xk(xH, xBL, jBLH, HessBLH, a, M, g_μν, Γαμν, j) for j=1:3];
    ∂Ki_∂xk = @SMatrix [SelfAcceleration.∂Ki_∂xk(xH, rH, xBL, jBLH, HessBLH, a, M, g_μν, Γαμν, j, k) for j=1:3, k=1:3];
    ∂Kij_∂xk = @SArray [SelfAcceleration.∂Kij_∂xk(xH, rH, xBL, jBLH, HessBLH, a, M, g_μν, Γαμν, j, k, l) for j=1:3, k=1:3, l=1:3]
    Q = SelfAcceleration.Q(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)
    Qi = SelfAcceleration.Qi(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)
    Qij = SelfAcceleration.Qij(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)

    BRR = B_RR(xH, Qi, ∂K_∂xk, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    BiRR = Bi_RR(xH, Qij, ∂K_∂xk, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    CRR = C_RR(xH, vH, xBL, ∂K_∂xk, ∂Ki_∂xk, Q, Qi, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    CiRR = Ci_RR(xH, vH, xBL, ∂K_∂xk, ∂Ki_∂xk, Qi, Qij, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    DRR = D_RR(xH, vH, xBL, ∂Ki_∂xk, ∂Kij_∂xk, Q, Qi, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    DiRR = Di_RR(xH, vH, xBL, ∂Ki_∂xk, ∂Kij_∂xk, Qi, Qij, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    VRR = PotentialsRR.Vrr(t, xH,  Mij5, Mij7, Mijk7)
    ViRR = PotentialsRR.Virr(t, xH, Mij6, Sij5)

    A2_t = (BRR + CRR + DRR) * VRR + dot((BiRR + CiRR + DiRR), ViRR)   # Eq. 62
    A2_i = -2.0 * (BRR + CRR + DRR) * ViRR - (BiRR + CiRR + DiRR) * VRR / 2.0  # Eq. 63

    return vcat(A2_t, A2_i)
end

# compute self-acceleration in harmonic coordinates and transform components back to BL
function aRRα(aSF_H::AbstractVector{Float64}, aSF_BL::AbstractVector{Float64}, t::Float64, xH::AbstractVector{Float64}, v::Float64, v_H::AbstractVector{Float64}, vH::AbstractVector{Float64},
    xBL::AbstractVector{Float64}, rH::Float64, a::Float64, M::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray,
    Sij5::AbstractArray, Sij6::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function,
    gThTh::Function, gΦΦ::Function)

    aSF_H[:] = -Γ(v_H, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * Pαβ(v_H, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ) * (A1_β(t, xH, v, v_H, vH, xBL, rH, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6) + A2_β(t, xH, vH, xBL, rH, a, M, Mij5, Mij6, Mij7, Mijk7, Sij5, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ))
    aSF_BL[1] = aSF_H[1]
    aSF_BL[2:4] = HarmonicCoords.aHtoBL(xH, zeros(3), aSF_H[2:4], a, M)
end

module FiniteDifferences
using ...SelfAcceleration
using ...SelfAcceleration
using ...HarmonicCoords
using ...EstimateMultipoleDerivs
using ...MultipoleFDM

# # returns the self-acceleration 4-vector
# @views function selfAcc!(aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray,
#     v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, t::AbstractVector{Float64}, r::AbstractVector{Float64}, dr_dt::AbstractVector{Float64}, d2r_dt2::AbstractVector{Float64}, θ::AbstractVector{Float64}, dθ_dt::AbstractVector{Float64},
#     d2θ_dt2::AbstractVector{Float64}, ϕ::AbstractVector{Float64}, dϕ_dt::AbstractVector{Float64}, d2ϕ_dt2::AbstractVector{Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray,
#     Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function,
#     g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, a::Float64, M::Float64, m::Float64, compute_at::Int64, h::Float64)
#     # convert trajectories to BL coords
#     @inbounds for i in eachindex(t)
#         xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
#         vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
#         aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);      # divide by (dt/dτ)² to get accelerations wrt BL time

#         xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M);
#         vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
#         aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M);

#         rH[i] = SelfAcceleration.norm_3d(xH[i]);
#         v[i] = SelfAcceleration.norm_3d(vH[i]);

#         x_H[i] = xH[i];
#         v_H[i] = vH[i];
#         a_H[i] = aH[i];

#     end
    
#     # calculate Multipoles.ddotMijk, Multipoles.ddotMijk, Multipoles.dotSij "analytically"
#     EstimateMultipoleDerivs.moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)
#     EstimateMultipoleDerivs.FiniteDifferences.moment_derivs_tr!(h, compute_at, size(t, 1), Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)

#     # calculate self force in BL and harmonic coordinates
#     SelfAcceleration.aRRα(aSF_H, aSF_BL, 0.0, xH[compute_at], v[compute_at], v_H[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, M, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6,
#     Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ)
# end


# returns the self-acceleration 4-vector
@views function selfAcc_mino!(a::Float64, M::Float64, E::Float64, L::Float64, C::Float64, aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray,
    x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray, v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, t::AbstractArray, r::AbstractArray, dr_dt::AbstractArray, 
    d2r_dt2::AbstractArray, θ::AbstractArray, dθ_dt::AbstractArray, d2θ_dt2::AbstractArray, ϕ::AbstractArray, dϕ_dt::AbstractArray, d2ϕ_dt2::AbstractArray, Mij5::AbstractArray, 
    Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray,
    Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, 
    gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, m::Float64, compute_at::Int64, h::Float64)

    # convert trajectories to BL coords
    @inbounds for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);      # divide by (dt/dτ)² to get accelerations wrt BL time
  
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M);
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M);

        rH[i] = SelfAcceleration.norm_3d(xH[i]);
        v[i] = SelfAcceleration.norm_3d(vH[i]);

        x_H[i] = xH[i];
        v_H[i] = vH[i];
        a_H[i] = aH[i];

    end

    # calculate first and second derivative of moments analytically
    EstimateMultipoleDerivs.analytic_moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)

    MultipoleFDM.diff_moments_tr_Mino!(a, E, L, C, M, xBL[compute_at], sign(dr_dt[compute_at]), sign(dθ_dt[compute_at]), Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, compute_at, size(r, 1), h)

    # calculate self force in BL and harmonic coordinates
    SelfAcceleration.aRRα(aSF_H, aSF_BL, 0.0, xH[compute_at], v[compute_at], v_H[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, M, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, 
    Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ)
end
end


module FourierFit
using ...SelfAcceleration
using ...HarmonicCoords
using ...EstimateMultipoleDerivs
using ...MultipoleFitting

# returns the self-acceleration 4-vector
@views function selfAcc_Mino!(aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray,
    v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, λ::AbstractVector{Float64}, r::AbstractVector{Float64}, rdot::AbstractVector{Float64}, rddot::AbstractVector{Float64}, θ::AbstractVector{Float64}, θdot::AbstractVector{Float64},
    θddot::AbstractVector{Float64}, ϕ::AbstractVector{Float64}, ϕdot::AbstractVector{Float64}, ϕddot::AbstractVector{Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray,
    Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray,
    Sij1_data::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function,
    gThTh::Function, gΦΦ::Function, a::Float64, E::Float64, L::Float64, C::Float64, M::Float64, m::Float64, compute_at::Int64, nHarm::Int64, γr::Float64, γθ::Float64, γϕ::Float64, nPoints::Int64, n_freqs::Int64,
    chisq::AbstractVector{Float64}, fit::String)
    
    # convert trajectories to BL coords
    @inbounds for i in eachindex(λ)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]);             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]);      # divide by (dt/dτ)² to get accelerations wrt BL time
  
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M);
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M);

        rH[i] = SelfAcceleration.norm_3d(xH[i]);
        v[i] = SelfAcceleration.norm_3d(vH[i]);

        x_H[i] = xH[i];
        v_H[i] = vH[i];
        a_H[i] = aH[i];

    end
    
    # calculate first and second derivative of moments analytically
    EstimateMultipoleDerivs.analytic_moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)

    MultipoleFitting.fit_moments_tr_Mino!(a, E, L, C, M, λ, xBL[compute_at], sign(rdot[compute_at]), sign(θdot[compute_at]), Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6,
    compute_at, nHarm, γr, γθ, γϕ, nPoints, n_freqs, chisq, fit)
    
    # calculate self force in BL and harmonic coordinates
    SelfAcceleration.aRRα(aSF_H, aSF_BL, 0.0, xH[compute_at], v[compute_at], v_H[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, M, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ)
end

# returns the self-acceleration 4-vector
@views function selfAcc!(aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray,
    v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, t::AbstractVector{Float64}, r::AbstractVector{Float64}, rdot::AbstractVector{Float64}, rddot::AbstractVector{Float64}, θ::AbstractVector{Float64}, θdot::AbstractVector{Float64},
    θddot::AbstractVector{Float64}, ϕ::AbstractVector{Float64}, ϕdot::AbstractVector{Float64}, ϕddot::AbstractVector{Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray,
    Mijk8::AbstractArray,Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray,
    Sij1_data::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function,
    gThTh::Function, gΦΦ::Function, a::Float64, M::Float64, m::Float64, compute_at::Int64, nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::AbstractVector{Float64}, fit::String)
    
    # convert trajectories to BL coords
    @inbounds for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]);             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]);      # divide by (dt/dτ)² to get accelerations wrt BL time
    
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M);
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M);

        rH[i] = SelfAcceleration.norm_3d(xH[i]);
        v[i] = SelfAcceleration.norm_3d(vH[i]);

        x_H[i] = xH[i];
        v_H[i] = vH[i];
        a_H[i] = aH[i];

    end
    
    # calculate first and second derivative of moments analytically
    EstimateMultipoleDerivs.analytic_moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)
    # estimate higher order derivative moments
    MultipoleFitting.fit_moments_tr_BL!(t, Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, compute_at, nHarm, Ωr, Ωθ, Ωϕ, nPoints, n_freqs, chisq, fit)

    # calculate self force in BL and harmonic coordinates
    SelfAcceleration.aRRα(aSF_H, aSF_BL, 0.0, xH[compute_at], v[compute_at], v_H[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, M, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ)
end

end
end