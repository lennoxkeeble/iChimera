#= 

    In this module we write the analytic expressions for computing the self-force from the high order derivatives of the multipole moments and the position, velocity and acceleration in harmonic coordinates. 
    See Eqs. 54-56, 57, 61-63, and A1-A14 in arXiv:1109.0572v2. Note that our implemenation of Eqs. A12 - A14 differ slightly by factors of 2 in places and symmetrization operators.

=#

module SelfAcceleration
using LinearAlgebra
using Combinatorics
import ..HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ..HarmonicCoords
using StaticArrays
using ..RRPotentials
import ..Kerr.KerrMetric: g_μν, Γαμν

"""
# Common Arguments in this module
- `xBL::AbstractVector{Float64}`: Boyer-Lindquist coordinates, xBL = [r, θ, ϕ].
- `xH::AbstractVector{Float64}`: Harmonic coordinates, xH = [x, y, z].
- `vH::AbstractArray`: velocity in Harmonic coordinates.
- `v::Float64`: velocity v = sqrt(vx^2 + vy^2 + vz^2).
- `aH::AbstractArray`: acceleration in Harmonic coordinates.
- `rH::Float64`: rH = sqrt(xH^2 + yH^2 + zH^2).
- `jBLH::AbstractArray`: Jacobian of the transformation from BL to Harmonic coordinates.
- `HessBLH::AbstractArray`: Hessian of the transformation from BL to Harmonic coordinates.
- `Mij5::AbstractArray`: fifth derivative of the mass quadrupole (Eq. 48).
- `Mij6::AbstractArray`: sixth derivative of the mass quadrupole (Eq. 48).
- `Mij7::AbstractArray`: seventh derivative of the mass quadrupole (Eq. 48).
- `Mij8::AbstractArray`: eighth derivative of the mass quadrupole (Eq. 48).
- `Mijk7::AbstractArray`: seventh derivative of the mass quadrupole (Eq. 48).
- `Mijk8::AbstractArray`: eighth derivative of the mass quadrupole (Eq. 48).
- `Sij5::AbstractArray`: fifth derivative of the current quadrupole (Eq. 49).
- `Sij6::AbstractArray`: sixth derivative of the current quadrupole (Eq. 49).
- `∂Vrr_∂t::Float64`: time derivative of the radiation reaction potential (Eq. 44).
- `∂Vrr_∂a::AbstractVector{Float64}`: radiation reaction potential derivative with respect to the harmonic spatial coordinates.
- `∂Virr_∂t::AbstractVector{Float64}`: time derivative of the spatial components of the radiation reaction potential (Eq. 45).
- `∂Virr_∂a::AbstractArray`: spatial radiation reaction potential derivatives with respect to the harmonic spatial coordinates.
- `∂K_∂xk::AbstractVector{Float64}`: partial derivative of "Kerr potential" K with respect to the harmonic spatial coordinates (Eqs. 54-56, A12-A14).
- `∂Ki_∂xk::AbstractArray`: partial derivative of "Kerr potential" K_i with respect to the harmonic spatial coordinates (Eqs. 54-56, A12-A14).
- `∂Kij_∂xk::AbstractArray`: partial derivative of "Kerr potential" K_ij with respect to the harmonic spatial coordinates (Eqs. 54-56, A12-A14).
- `Q::Float64`: Kerr potential tt component (Eq. 54).
- `Qi::AbstractVector{Float64}`: Kerr potential ti components (Eq. 55).
- `Qij::AbstractArray`: Kerr potential ij (spatial) components (Eq. 56).
- `aSF_H::AbstractArray`: self-acceleration (Eq. 57) in Harmonic coordinates.
- `aSF_BL::AbstractArray`: self-acceleration in Boyer-Lindquist coordinates.
- `a::Float64`: Kerr black hole spin parameter.
"""

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

# define vector and scalar potentials for self-force calculation - underscore denotes covariant indices
K(xH::AbstractArray, a::Float64) = g_tt_H(xH, a) + 1.0                         # outputs K00 (Eq. 54)
K_i(xH::AbstractArray, a::Float64) = g_tr_H(xH, a)                             # outputs Ki vector, i.e., Ki for i ∈ {1, 2, 3} (Eq. 55)
K_ij(xH::AbstractArray, a::Float64) = g_rr_H(xH, a) - ηij                      # outputs Kij matrix (Eq. 56)
K_μν(xH::AbstractArray, a::Float64) = g_μν_H(xH, a) - ημν                      # outputs Kμν matrix
Q(xH::AbstractArray, a::Float64) = gTT_H(xH, a) + 1.0                          # outputs Q^00 (Eq. 54)
Qi(xH::AbstractArray, a::Float64) = gTR_H(xH, a)                               # outputs Q^i vector, i.e., Q^i for i ∈ {1, 2, 3} (Eq. 55)
Qij(xH::AbstractArray, a::Float64) = gRR_H(xH, a) - ηij                        # outputs diagonal of Q^ij matrix (Eq. 56)
Qμν(xH::AbstractArray, a::Float64) = gμν_H(xH, a) - ημν                        # outputs Qμν matrix

# define partial derivatives of K (in harmonic coordinates)
# ∂ₖK: outputs float
function ∂K_∂xk(xH::AbstractArray, xBL::AbstractArray, jBLH::AbstractArray, HessBLH::AbstractArray, a::Float64, k::Int)   # Eq. A12
    ∂K=0.0
    @inbounds for μ=1:4
        for i=1:3
            ∂K += 2 * g_μν(xBL[1], xBL[2], xBL[3], a, 1, μ) * Γαμν(xBL[1], xBL[2], xBL[3], a, μ, 1, i+1) * jBLH[i, k]   # i → i + 1 to go from spatial indices to spacetime indices
        end
    end
    return ∂K
end

# ∂ₖKᵢ: outputs float.
function ∂Ki_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, HessBLH::AbstractArray, a::Float64, k::Int, i::Int)   # Eq. A13
    ∂K=0.0
    @inbounds for m=1:3   # start with iteration over m to not over-count last terms
        ∂K += g_μν(xBL[1], xBL[2], xBL[3], a, 1, m+1) * HessBLH[m][k, i]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices
        @inbounds for μ=1:4, n=1:3
            ∂K += (g_μν(xBL[1], xBL[2], xBL[3], a, μ, 1) * Γαμν(xBL[1], xBL[2], xBL[3], a, μ, m+1, n+1) + g_μν(xBL[1], xBL[2], xBL[3], a, μ, m+1) * Γαμν(xBL[1], xBL[2], xBL[3], a, μ, 1, n+1)) * jBLH[n, k] * jBLH[m, i]   # first term of Eq. A13
        end
    end
    return ∂K
end

# ∂ₖKᵢⱼ: outputs float.
function ∂Kij_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, HessBLH::AbstractArray, a::Float64, k::Int, i::Int, j::Int)   # Eq. A14
    ∂K=0.0
    @inbounds for m=1:3
        @inbounds for l=1:3   # iterate over m and l first to avoid over-counting
            ∂K += g_μν(xBL[1], xBL[2], xBL[3], a, l+1, m+1) * (HessBLH[l][k, i] * jBLH[m, j] + HessBLH[l][k, j] * jBLH[m, i])  # last term Eq. A14
            @inbounds for μ=1:4, n=1:3
                ∂K += (g_μν(xBL[1], xBL[2], xBL[3], a, μ, l+1) * Γαμν(xBL[1], xBL[2], xBL[3], a, μ, m+1, n+1) + g_μν(xBL[1], xBL[2], xBL[3], a, μ, m+1) * Γαμν(xBL[1], xBL[2], xBL[3], a, μ, l+1, n+1)) * jBLH[n, k] * jBLH[l, i] * jBLH[m, j]   # first term of Eq. A14
            end
        end
    end
    return ∂K
end

# define relativistic Γ factor
Γ(vH::AbstractArray, xH::AbstractArray, a::Float64) = 1.0 / sqrt(1.0 - SelfAcceleration.norm2_3d(vH) - K(xH, a) - 2.0 * dot(K_i(xH, a), vH) - transpose(vH) * K_ij(xH, a) * vH)   # Eq. A3

# define projection operator
Pαβ(vH::AbstractArray, xH::AbstractArray, a::Float64) = ημν + Qμν(xH, a) + Γ(vH, xH, a)^2 * otimes(vcat([1], vH))   # contravariant, Eq. A1
P_αβ(vH::AbstractArray, xH::AbstractArray, a::Float64) =  ημν + K_μν(xH, a) + Γ(vH, xH, a)^2 * otimes(vcat([1], vH))   # cοvariant, Eq. A2 (note that we take both contravariant and covariant velocities as arguments)

### SELF-ACCELERATION PIECES ###
# compute self-acceleration pieces
function A_RR(xH::AbstractArray, v::Float64, vH::AbstractArray, ∂Vrr_∂t::Float64, ∂Vrr_∂a::SVector{3, Float64}, ∂Virr_∂a::SMatrix{3, 3, Float64})
    aRR = (1.0 - v^2) * ∂Vrr_∂t   # first term in Eq. A4
    @inbounds for i=1:3
        aRR += 2.0 * vH[i] * ∂Vrr_∂a[i]   # second term Eq. A4
        @inbounds for j=1:3
            aRR += -4.0 * vH[i] * vH[j] * ∂Virr_∂a[i, j]   # third term Eq. A4
        end
    end
    return aRR
end

function Ai_RR(xH::AbstractArray, v::Float64,  vH::AbstractArray, ∂Vrr_∂t::Float64, ∂Virr_∂t::SVector{3, Float64}, ∂Vrr_∂a::SVector{3, Float64}, ∂Virr_∂a::SMatrix{3, 3, Float64}, i::Int)
    aiRR = -(1 + v^2) * ∂Vrr_∂a[i] + 2.0 * vH[i] * ∂Vrr_∂t - 4.0 * ∂Virr_∂t[i]    # first, second, and last term in Eq. A5
    @inbounds for j=1:3
        aiRR += 2.0 * vH[i] * vH[j] * ∂Vrr_∂a[j] - 4.0 * vH[j] * (∂Virr_∂a[i, j] - ∂Virr_∂a[j, i])    # third and fourth terms in Eq. A5
    end
    return aiRR
end

function A1_β(xH::AbstractArray, v::Float64,  vH::AbstractArray, rH::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    ∂Vrr_∂t = RRPotentials.∂Vrr_∂t(xH, Mij6, Mij8, Mijk8)
    ∂Vrr_∂a = @SVector [RRPotentials.∂Vrr_∂a(xH, Mij5, Mij7, Mijk7, i) for i =1:3]
    ∂Virr_∂t = @SVector [RRPotentials.∂Virr_∂t(xH, Mij7, Sij6, i) for i =1:3]
    ∂Virr_∂a = @SMatrix [RRPotentials.∂Virr_∂a(xH, Mij6, Sij5, i, j) for i=1:3, j=1:3]
    return [i==1 ? A_RR(xH, v, vH, ∂Vrr_∂t, ∂Vrr_∂a, ∂Virr_∂a) : Ai_RR(xH, v, vH, ∂Vrr_∂t, ∂Virr_∂t, ∂Vrr_∂a, ∂Virr_∂a, i-1) for i = 1:4]
end

function B_RR(xH::AbstractVector{Float64}, Qi::AbstractVector{Float64}, ∂K_∂xk::SVector{3, Float64}, a::Float64)
    return dot(Qi, ∂K_∂xk)   # Eq. A6
end

function Bi_RR(xH::AbstractVector{Float64}, Qij::AbstractArray, ∂K_∂xk::SVector{3, Float64}, a::Float64)
    return -2.0 * (ηij + Qij) * ∂K_∂xk   # Eq. A9
end

# Eq. A7
function C_RR(xH::AbstractVector{Float64}, vH::AbstractArray, ∂K_∂xk::SVector{3, Float64}, ∂Ki_∂xk::SMatrix{3, 3, Float64}, Q::Float64, Qi::AbstractVector{Float64}, rH::Float64, a::Float64)
    C = 0.0
    @inbounds for i=1:3
        C += 2.0 * (1.0 - Q) * vH[i] * ∂K_∂xk[i]
        @inbounds for j=1:3
            C += 2.0 * Qi[i] * vH[j] * (∂Ki_∂xk[i, j] - ∂Ki_∂xk[j, i])
        end
    end
    return C
end

# Eq. A10
function Ci_RR(xH::AbstractVector{Float64}, vH::AbstractArray, ∂K_∂xk::SVector{3, Float64}, ∂Ki_∂xk::SMatrix{3, 3, Float64}, Qi::AbstractVector{Float64}, Qij::AbstractArray, rH::Float64, a::Float64)   # Eq. A10
    C = @MVector [0., 0., 0.]
    @inbounds for j=1:3
        @inbounds for i=1:3
            C[i] += 4.0 * Qi[i] * vH[j] * ∂K_∂xk[j]
        end
        C .+= 4.0 * (ηij + Qij) * vH[j] * ([(∂Ki_∂xk[j, k] - ∂Ki_∂xk[k, j]) for k=1:3]) 
    end
    return C
end

# Eq. A8
function D_RR(xH::AbstractVector{Float64}, vH::AbstractArray, ∂Ki_∂xk::SMatrix{3, 3, Float64}, ∂Kij_∂xk::SArray{Tuple{3, 3, 3}, Float64, 3, 27}, Q::Float64, Qi::AbstractVector{Float64}, rH::Float64, a::Float64)
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

# Eq. A11
function Di_RR(xH::AbstractVector{Float64}, vH::AbstractArray, ∂Ki_∂xk::SMatrix{3, 3, Float64}, ∂Kij_∂xk::SArray{Tuple{3, 3, 3}, Float64, 3, 27}, Qi::AbstractVector{Float64}, Qij::AbstractArray, rH::Float64, a::Float64)   # Eq. A11
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
function A2_β(xH::AbstractArray, vH::AbstractArray, xBL::AbstractArray, rH::Float64, a::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mijk7::AbstractArray, Sij5::AbstractArray)
    jBLH = HarmonicCoords.jBLH(xH, a)
    HessBLH = [HarmonicCoords.HessBLH(xH, rH, a, m) for m=1:3]
    ∂K_∂xk = @SVector [SelfAcceleration.∂K_∂xk(xH, xBL, jBLH, HessBLH, a, j) for j=1:3];
    ∂Ki_∂xk = @SMatrix [SelfAcceleration.∂Ki_∂xk(xH, rH, xBL, jBLH, HessBLH, a, j, k) for j=1:3, k=1:3];
    ∂Kij_∂xk = @SArray [SelfAcceleration.∂Kij_∂xk(xH, rH, xBL, jBLH, HessBLH, a, j, k, l) for j=1:3, k=1:3, l=1:3]
    Q = SelfAcceleration.Q(xH, a)
    Qi = SelfAcceleration.Qi(xH, a)
    Qij = SelfAcceleration.Qij(xH, a)

    BRR = B_RR(xH, Qi, ∂K_∂xk, a)
    BiRR = Bi_RR(xH, Qij, ∂K_∂xk, a)

    CRR = C_RR(xH, vH, ∂K_∂xk, ∂Ki_∂xk, Q, Qi, rH, a)
    CiRR = Ci_RR(xH, vH, ∂K_∂xk, ∂Ki_∂xk, Qi, Qij, rH, a)

    DRR = D_RR(xH, vH, ∂Ki_∂xk, ∂Kij_∂xk, Q, Qi, rH, a)
    DiRR = Di_RR(xH, vH, ∂Ki_∂xk, ∂Kij_∂xk, Qi, Qij, rH, a)

    VRR = RRPotentials.Vrr(xH,  Mij5, Mij7, Mijk7)
    ViRR = RRPotentials.Virr(xH, Mij6, Sij5)

    A2_t = (BRR + CRR + DRR) * VRR + dot((BiRR + CiRR + DiRR), ViRR)   # Eq. 62
    A2_i = -2.0 * (BRR + CRR + DRR) * ViRR - (BiRR + CiRR + DiRR) * VRR / 2.0  # Eq. 63

    return vcat(A2_t, A2_i)
end

# compute self-acceleration in harmonic coordinates and transform components back to BL
function aRRα(aSF_H::AbstractVector{Float64}, aSF_BL::AbstractVector{Float64}, xH::AbstractVector{Float64}, v::Float64, vH::AbstractVector{Float64},
    xBL::AbstractVector{Float64}, rH::Float64, a::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray,
    Sij5::AbstractArray, Sij6::AbstractArray)

    aSF_H[:] = -Γ(vH, xH, a)^2 * Pαβ(vH, xH, a) * (A1_β(xH, v, vH, rH, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6) + A2_β(xH, vH, xBL, rH, a, Mij5, Mij6, Mij7, Mijk7, Sij5))
    aSF_BL[1] = aSF_H[1]
    aSF_BL[2:4] = HarmonicCoords.aHtoBL(xH, zeros(3), aSF_H[2:4], a)
end

module FiniteDifferences
using ...SelfAcceleration
using ...HarmonicCoords
using ...EstimateMultipoleDerivs
using ...MultipoleFDM
"""
# Common Arguments in this module
- `compute_at::Int64`: index at which to compute the self-acceleration. This is always at the center of the finite differences stencil.
- `h::Float64`: step size for finite differences.
"""

# mutates the self-acceleration 4-vector
@views function selfAcc_mino!(a::Float64, E::Float64, L::Float64, C::Float64, aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray,
    rH::AbstractArray, vH::AbstractArray,  aH::AbstractArray, v::AbstractArray, t::AbstractArray, r::AbstractArray, dr_dt::AbstractArray, 
    d2r_dt2::AbstractArray, θ::AbstractArray, dθ_dt::AbstractArray, d2θ_dt2::AbstractArray, ϕ::AbstractArray, dϕ_dt::AbstractArray, d2ϕ_dt2::AbstractArray, Mij5::AbstractArray, 
    Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray,
    Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray, q::Float64, compute_at::Int64, h::Float64)

    # convert trajectories to BL coords
    @inbounds for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
        aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);
  
        HarmonicCoords.xBLtoH!(xH[i], xBL[i], a);
        HarmonicCoords.vBLtoH!(vH[i], xH[i], vBL[i], a); 
        HarmonicCoords.aBLtoH!(aH[i], xH[i], vBL[i], aBL[i], a);

        rH[i] = SelfAcceleration.norm_3d(xH[i]);
        v[i] = SelfAcceleration.norm_3d(vH[i]);

        xH[i] = xH[i];
        vH[i] = vH[i];
        aH[i] = aH[i];

    end

    # calculate first and second derivative of multipole moments from analytic expressions
    EstimateMultipoleDerivs.analytic_moment_derivs_tr!(aH, vH, xH, q, Mij2_data, Mijk2_data, Sij1_data)

    # estimate higher order derivatives of the multipole moments via finite differences
    MultipoleFDM.diff_moments_tr_Mino!(a, E, L, C, xBL[compute_at], sign(dr_dt[compute_at]), sign(dθ_dt[compute_at]), Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, compute_at, size(r, 1), h)

    # calculate self force in BL and harmonic coordinates
    SelfAcceleration.aRRα(aSF_H, aSF_BL, xH[compute_at], v[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)
end
end


module FourierFit
using ...SelfAcceleration
using ...HarmonicCoords
using ...EstimateMultipoleDerivs
using ...MultipoleFitting
"""
# Common Arguments in this module
- `compute_at::Int64`: index at which to compute the self-acceleration. This is always at the center of the time series array which we fit to its fourier series expansion.
- `nHarm::Int64`: number of (radial) harmonic frequencies.
- `ωr::Float64`: radial frequency wrt Mino time.
- `ωθ::Float64`: polar frequency wrt Mino time.
- `ωϕ::Float64`: azimuthal frequency wrt Mino time.
- `Ωr::Float64`: radial frequency wrt BL time.
- `Ωθ::Float64`: polar frequency wrt BL time.
- `Ωϕ::Float64`: azimuthal frequency wrt BL time.
- `nPoints::Int64`: number of points in the time series array.
- `n_freqs::Int64`: number of non-zero fitting frequencies in the fourier series expansion.
- `chisq::AbstractVector{Float64}`: chi-squared value of the fit. In practice this is just the output of the GSL solver, so one only passes as argument [0.0].
- `fit::String`: type of fitting to perform. Either "GSL" for fits using Julia's GSL wrapper, or "Julia" to use Julia's base least squares solver.
"""

# mutates the self-acceleration 4-vector
@views function selfAcc_Mino!(aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, rH::AbstractArray, vH::AbstractArray,
    aH::AbstractArray, v::AbstractArray, λ::AbstractVector{Float64}, r::AbstractVector{Float64}, rdot::AbstractVector{Float64}, rddot::AbstractVector{Float64}, θ::AbstractVector{Float64}, θdot::AbstractVector{Float64},
    θddot::AbstractVector{Float64}, ϕ::AbstractVector{Float64}, ϕdot::AbstractVector{Float64}, ϕddot::AbstractVector{Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray,
    Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray, a::Float64, q::Float64, E::Float64, L::Float64, C::Float64, compute_at::Int64, nHarm::Int64,
    ωr::Float64, ωθ::Float64, ωϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::AbstractVector{Float64}, fit::String)
    
    # convert trajectories to BL coords
    @inbounds for i in eachindex(λ)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]);
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]);
  
        HarmonicCoords.xBLtoH!(xH[i], xBL[i], a);
        HarmonicCoords.vBLtoH!(vH[i], xH[i], vBL[i], a); 
        HarmonicCoords.aBLtoH!(aH[i], xH[i], vBL[i], aBL[i], a);

        rH[i] = SelfAcceleration.norm_3d(xH[i]);
        v[i] = SelfAcceleration.norm_3d(vH[i]);
    end
    
    # calculate first and second derivative of multipole moments from analytic expressions
    EstimateMultipoleDerivs.analytic_moment_derivs_tr!(aH, vH, xH, q, Mij2_data, Mijk2_data, Sij1_data)

    # estimate higher order derivatives of the multipole moments via fourier fits
    MultipoleFitting.fit_moments_tr_Mino!(a, E, L, C, λ, xBL[compute_at], sign(rdot[compute_at]), sign(θdot[compute_at]), Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6,
    compute_at, nHarm, ωr, ωθ, ωϕ, nPoints, n_freqs, chisq, fit)
    
    # calculate self force in BL and harmonic coordinates
    SelfAcceleration.aRRα(aSF_H, aSF_BL, xH[compute_at], v[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)
end

# mutates the self-acceleration 4-vector
@views function selfAcc!(aSF_H::AbstractArray, aSF_BL::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, rH::AbstractArray, vH::AbstractArray,
    aH::AbstractArray, v::AbstractArray, t::AbstractVector{Float64}, r::AbstractVector{Float64}, rdot::AbstractVector{Float64}, rddot::AbstractVector{Float64}, θ::AbstractVector{Float64}, θdot::AbstractVector{Float64},
    θddot::AbstractVector{Float64}, ϕ::AbstractVector{Float64}, ϕdot::AbstractVector{Float64}, ϕddot::AbstractVector{Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray,
    Mijk8::AbstractArray,Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray, a::Float64, q::Float64, compute_at::Int64, nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64,
    nPoints::Int64, n_freqs::Int64, chisq::AbstractVector{Float64}, fit::String)
    
    # convert trajectories to BL coords
    @inbounds for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]);
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]);
    
        HarmonicCoords.xBLtoH!(xH[i], xBL[i], a);
        HarmonicCoords.vBLtoH!(vH[i], xH[i], vBL[i], a); 
        HarmonicCoords.aBLtoH!(aH[i], xH[i], vBL[i], aBL[i], a);

        rH[i] = SelfAcceleration.norm_3d(xH[i]);
        v[i] = SelfAcceleration.norm_3d(vH[i]);
    end
    
    # calculate first and second derivative of multipole moments from analytic expressions
    EstimateMultipoleDerivs.analytic_moment_derivs_tr!(aH, vH, xH, q, Mij2_data, Mijk2_data, Sij1_data)

    # estimate higher order derivatives of the multipole moments via fourier fits
    MultipoleFitting.fit_moments_tr_BL!(t, Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, compute_at, nHarm, Ωr, Ωθ, Ωϕ, nPoints, n_freqs, chisq, fit)

    # calculate self force in BL and harmonic coordinates
    SelfAcceleration.aRRα(aSF_H, aSF_BL, xH[compute_at], v[compute_at], vH[compute_at], xBL[compute_at], rH[compute_at], a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)
end

end
end