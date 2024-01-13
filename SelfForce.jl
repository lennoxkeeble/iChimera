# we write covariant vectors with underscores (e.g., for BL coordinates x^μ = xBL x_μ = x_BL)
module SelfForce
using LinearAlgebra
using Combinatorics
using BSplineKit
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ..Kerr

import ..HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ..HarmonicCoords

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

# define partial derivatives of K (in harmonic coordinates)
# ∂ₖK: outputs float
function ∂K_∂xk(xH::AbstractArray, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int)   # Eq. A12
    ∂K=0.0
    @inbounds for μ=1:4
        for i=1:3
            ∂K += g_μν(0., xBL..., a, M, 1, μ) * Γαμν(0., xBL..., a, M, μ, 1, i+1) * jBLH[i, k]   # i → i + 1 to go from spatial indices to spacetime indices
        end
    end
    return ∂K
end

# ∂ₖKᵢ: outputs float. Note: rH = norm(xH).
function ∂Ki_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int)   # Eq. A13
    ∂K=0.0
    @inbounds for m=1:3   # start with iteration over m to not over-count last terms
        ∂K += 2.0 * g_μν(0., xBL..., a, M, 1, m+1) * HarmonicCoords.HessBLH(xH, rH, a, M, m)[i, k]   # last term Eq. A13, m → m + 1 to go from spatial indices to spacetime indices
        @inbounds for μ=1:4, n=1:3
            ∂K += ((g_μν(0., xBL..., a, M, μ, 1) * Γαμν(0., xBL..., a, M, μ, m+1, n+1) + g_μν(0., xBL..., a, M, μ, m+1) * Γαμν(0., xBL..., a, M, μ, 1, n+1))/2) * jBLH[n, k] * jBLH[m, i]   # first term of Eq. A13
        end
    end
    return ∂K
end

# ∂ₖKᵢⱼ: outputs float. Note: rH = norm(xH).
function ∂Kij_∂xk(xH::AbstractArray, rH::Float64, xBL::AbstractArray, jBLH::AbstractArray, a::Float64, M::Float64, g_μν::Function, Γαμν::Function, k::Int, i::Int, j::Int)   # Eq. A14
    ∂K=0.0
    @inbounds for m=1:3
        for l=1:3   # iterate over m and l first to avoid over-counting
            ∂K += 2.0 * g_μν(0., xBL..., a, M, l+1, m+1) * HarmonicCoords.HessBLH(xH, rH, a, M, m)[i, k] * jBLH[l, i]  # last term Eq. A14
            @inbounds for μ=1:4, n=1:3
                ∂K += ((g_μν(0., xBL..., a, M, μ, l+1) * Γαμν(0., xBL..., a, M, μ, m+1, n+1) + g_μν(0., xBL..., a, M, μ, m+1) * Γαμν(0., xBL..., a, M, μ, l+1, n+1))/2) * jBLH[n, k] * jBLH[m, i] * jBLH[l, i]   # first term of Eq. A14
            end
        end
    end
    return ∂K
end

# define GR Γ factor, v_H = contravariant velocity in harmonic coordinates
Γ(vH::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = 1.0 / sqrt(1.0 - norm2_3d(vH) - K(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) - 2.0 * dot(K_i(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ), vH) - sum(transpose(vH) * K_ij(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) * vH))   # Eq. A3

# define projection operator  # OUTSTANDING QUESTION: SCO 4-VELOCITY = (1,vⁱ)? (SEE EQS. 57, A1)
Pαβ(vH::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function) = ημν + Qμν(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ) + Γ(vH, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2.0 * otimes(vcat([1], vH))   # contravariant, Eq. A1
P_αβ(vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) =  ημν + K_μν(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ) + Γ(vH, xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2.0 * otimes(vcat([1], vH))   # cοvariant, Eq. A2 (note that we take both contravariant and covariant velocities as arguments)

# define STF projections 
STF(u::Vector, i::Int, j::Int) = u[i] * u[j] - dot(u, u) * δ(i, j) /3.0                                                                   # STF projection x^{<ij>}
STF(u::Vector, v::Vector, i::Int, j::Int) = (u[i] * v[j] + u[j] * v[i])/2.0 - dot(u, v)* δ(i, j) /3.0                                                         # STF projection of two distinct vectors
STF(u::Vector, i::Int, j::Int, k::Int) = u[i] * u[j] * u[k] - (1.0/5.0) * dot(u, u) * (δ(i, j) * u[k] + δ(j, k) * u[i] + δ(k, i) * u[j])    # STF projection x^{<ijk>} (Eq. 46)

# define mass-ratio parameters
μ(m::Float64, M::Float64) = m * M / (m + M)
η(q::Float64) = q/((1+q)^2)   # q = mass ratio
η2(q::Float64) = q/(1+q)

# define multipole moments
# M_ij(x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int) = η(m/M) * m * STF(x_H, i, j)  # quadrupole mass moment Eq. 48
M_ij(x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int) = η2(m/M) * STF(x_H, i, j)  # quadrupole mass moment Eq. 48
# ddotMij(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int) = η(m/M) * m * ((-2.0δ(i, j)/3.0) * (dot(x_H, a_H) + dot(v_H, v_H)) + x_H[j] * a_H[i] + 2.0 * v_H[i] * v_H[j] + x_H[i] * a_H[j])   # Eq. 7.17
ddotMij(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int) = η2(m/M) * ((-2.0δ(i, j)/3.0) * (dot(x_H, a_H) + dot(v_H, v_H)) + x_H[j] * a_H[i] + 2.0 * v_H[i] * v_H[j] + x_H[i] * a_H[j])   # Eq. 7.17


M_ijk(x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) = η(m/M) * (M - m) * STF(x_H, i, j, k)  # octupole mass moment Eq. 48
# ddotMijk(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) = η(m/M) * (M-m) * ((-12.0/5.0) * (dot(x_H, v_H)) * (δ(i, j) * v_H[k] + δ(j, k) * v_H[i] + δ(k, i) * v_H[j]) - (6/5) * (dot(x_H, a_H) + dot(v_H, v_H)) * (δ(i, j) * x_H[k] + δ(j, k) * x_H[i] + δ(k, i) * x_H[j]) - (3/5) * dot(x_H, x_H) * (δ(i, j) * a_H[k] + δ(j, k) * a_H[i] + δ(k, i) * a_H[j]) + 2.0 * v_H[k] * (x_H[j] * v_H[i] + x_H[i] * v_H[j]) + x_H[k] * (x_H[j] * a_H[i] + 2.0 * v_H[i] * v_H[j] + x_H[i] * a_H[j]) + x_H[i] * x_H[j] * a_H[k])   # Eq. 7.19
ddotMijk(a_H::AbstractArray, v_H::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, i::Int, j::Int, k::Int) = η(m/M) * (M-m) * ((-4.0/5.0) * (dot(x_H, v_H)) * (δ(i, j) * v_H[k] + δ(j, k) * v_H[i] + δ(k, i) * v_H[j]) - (2.0/5.0) * (dot(x_H, a_H) + dot(v_H, v_H)) * (δ(i, j) * x_H[k] + δ(j, k) * x_H[i] + δ(k, i) * x_H[j]) - (1.0/5.0) * dot(x_H, x_H) * (δ(i, j) * a_H[k] + δ(j, k) * a_H[i] + δ(k, i) * a_H[j]) + 2.0 * v_H[k] * (x_H[j] * v_H[i] + x_H[i] * v_H[j]) + x_H[k] * (x_H[j] * a_H[i] + 2.0 * v_H[i] * v_H[j] + x_H[i] * a_H[j]) + x_H[i] * x_H[j] * a_H[k])   # Eq. 7.19

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
    return η(m/M) * (M - m) * s_ij
end

function dotSij(aH::AbstractArray, v_H::AbstractArray, vH::AbstractArray, x_H::AbstractArray, xH::AbstractArray, m::Float64, M::Float64, i::Int, j::Int)
    S=0.0
    @inbounds for k=1:3
        for l=1:3
            S += -2.0δ(i, j) * (vH[l] * (xH[k] * dot(εkl[k, l], v_H) + vH[k] * dot(εkl[k, l], x_H)) + xH[k] * aH[l] * dot(εkl[k, l], x_H)) + 3.0 * vH[l] * (εkl[k, l][i] * (xH[k] * v_H[j] + x_H[j] * vH[k]) + εkl[k, l][j] * (xH[k] * v_H[i] + x_H[i] * vH[k])) + 3.0 * xH[k] * aH[l] * (εkl[k, l][i] * x_H[j] + εkl[k, l][j] * x_H[i])
        end
    end
    return η(m/M) * (M-m) * S / 6.0
end

# numerically compute the nth derivative of a given BSplineKit interpolator at some x, where n ≤ BSplineOrder
function ND(x::Float64, itp, n::Int)
    return diff(itp, Derivative(n))(x)
end

# construct fill pre-allocated arrays with the appropriate derivatives of the mass and current moments
function moments!(aH::AbstractArray, a_H::AbstractArray, vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, Mij2::AbstractArray, Mijk2::AbstractArray, Sij1::AbstractArray)
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

# calculate time derivatives of the moments. allocate memory to data arrays. TO-DO: FIND MORE DETAIL ON IMPORTANCE OF SPLINE ORDER
function moment_derivs!(tdata::AbstractArray, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    @inbounds Threads.@threads for i=1:3
        @inbounds for j=1:3
            MijSpline = interpolate(tdata, Mij2data[i, j], BSplineOrder(4))
            @views Mij5[i, j, :] = ND.(tdata, Ref(MijSpline), 3)  # differentiate 2nd derivative 5-2=3 times
            MijSpline = interpolate(tdata, Mij2data[i, j], BSplineOrder(5))
            @views Mij6[i, j, :] = ND.(tdata, Ref(MijSpline), 4)   # differentiate 2nd derivative 6-2=4 times
            @views MijSpline = interpolate(tdata, Mij2data[i, j], BSplineOrder(6))
            Mij7[i, j, :] = ND.(tdata, Ref(MijSpline), 5)   # differentiate 2nd derivative 7-2=5 times
            @views MijSpline = interpolate(tdata, Mij2data[i, j], BSplineOrder(7))
            Mij8[i, j, :] = ND.(tdata, Ref(MijSpline), 6)   # differentiate 2nd derivative 8-2=6 times

            SijSpline = interpolate(tdata, Sij1data[i, j], BSplineOrder(5))
            @views Sij5[i, j, :] = ND.(tdata, Ref(SijSpline), 4)   # differentiate 1st derivative 5-1=4 times
            SijSpline = interpolate(tdata, Sij1data[i, j], BSplineOrder(6))
            @views Sij6[i, j, :] = ND.(tdata, Ref(SijSpline), 5)   # differentiate 1st derivative 5-1=4 times

            @inbounds for k=1:3
                MijkSpline = interpolate(tdata, Mijk2data[i, j, k], BSplineOrder(6))
                @views Mijk7[i, j, k, :] = ND.(tdata, Ref(MijkSpline), 5)   # differentiate 2nd derivative 7-2=5 times
                MijkSpline = interpolate(tdata, Mijk2data[i, j, k], BSplineOrder(7))
                @views Mijk8[i, j, k, :] = ND.(tdata, Ref(MijkSpline), 6)   # differentiate 2nd derivative 8-2=6 times
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
            εijk = εkl[i, j][k]
            @inbounds for l=1:3   # dummy indices in second term in Eq. 45
                V[i] += -4.0 * εijk * xH[j] * xH[l] * Sij5[k, l]  / 45.0
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
            εijk = εkl[i, j][k]
            @inbounds for l=1:3   # dummy indices in second term in Eq. 7.26
                V += -4.0 * εijk * xH[j] * xH[l] * Sij6[k, l]  / 45.0
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
            εijk = εkl[i, j][k]
            @inbounds for l=1:3   # dummy indices in second term in Eq. 45
                V += -4.0 * εijk * (δ(j, a) * xH[l] + xH[j] * δ(l, a)) * Sij5[k, l]  / 45.0
            end 
        end
    end
    return V
end

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
        aiRR += 2.0 * v_H[i] * vH[j] * ∂Vrr_∂a[j] - 8.0 * vH[j] * (∂Virr_∂a[i, j] - ∂Virr_∂a[j, i]) / 2.0   # third and fourth terms in Eq. A5
    end
    return aiRR
end

function A1_β(t::Float64, xH::AbstractArray, v::Float64, v_H::AbstractArray, vH::AbstractArray, xBL::AbstractArray, rH::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    ∂Vrr_∂t = SelfForce.∂Vrr_∂t(t, xH, Mij6, Mij8, Mijk8)
    ∂Vrr_∂a = @SVector [SelfForce.∂Vrr_∂a(t, xH, Mij5, Mij7, Mijk7, i) for i =1:3]
    ∂Virr_∂t = @SVector [SelfForce.∂Virr_∂t(t, xH, Mij7, Sij6, i) for i =1:3]
    ∂Virr_∂a = @SMatrix [SelfForce.∂Virr_∂a(t, xH, Mij6, Sij5, j, i) for j=1:3, i=1:3]
    return [i==1 ? A_RR(t, xH, v, vH, ∂Vrr_∂t, ∂Vrr_∂a, ∂Virr_∂a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6) : Ai_RR(t, xH, v, v_H, vH, ∂Vrr_∂t, ∂Virr_∂t, ∂Vrr_∂a, ∂Virr_∂a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, i-1) for i = 1:4]
end

function B_RR(xH::Vector{Float64}, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    jBLH = HarmonicCoords.jBLH(xH, a, M)
    xBL = HarmonicCoords.xHtoBL(xH, a, M)
    return dot(Qi(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ), [∂K_∂xk(xH, xBL, jBLH, a, M, g_μν, Γαμν, k) for k=1:3])   # Eq. A6
end

function Bi_RR(xH::Vector{Float64}, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    jBLH = HarmonicCoords.jBLH(xH, a, M)
    xBL = HarmonicCoords.xHtoBL(xH, a, M)
    return -2.0 * (ηij + Qij(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)) * [∂K_∂xk(xH, xBL, jBLH, a, M, g_μν, Γαμν, k) for k=1:3]   # Eq. A9
end
function C_RR(xH::Vector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂K_∂xk::SVector{3, Float64}, ∂Ki_∂xk::SMatrix{3, 3, Float64}, Q::Float64, Qi::Vector{Float64}, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    C = 0.0
    @inbounds for i=1:3
        C += 2.0 * (1.0 - Q) * vH[i] * ∂K_∂xk[i]
        @inbounds for j=1:3
            C += 4.0 * Qi[i] * vH[j] * (∂Ki_∂xk[i, j] - ∂Ki_∂xk[j, i]) / 2.0
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
        C .+= 8.0 * (ηij + Qij) * vH[j] * ([(∂Ki_∂xk[j, k] - ∂Ki_∂xk[k, j])/2.0 for k=1:3]) 
    end
    return C
end

function D_RR(xH::Vector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂Ki_∂xk::SMatrix{3, 3, Float64}, ∂Kij_∂xk::SArray{Tuple{3, 3, 3}, Float64, 3, 27}, Q::Float64, Qi::Vector{Float64}, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    D = 0.0
    @inbounds for i=1:3
        for j=1:3
            D += 2.0 * (1.0 - Q) * vH[i] * vH[j] * ∂Ki_∂xk[i, j]
            @inbounds for k=1:3
                D += -Qi[i] * vH[j] * vH[k] * (2.0 * (∂Kij_∂xk[j, k, i] + ∂Kij_∂xk[k, j, i]) / 2.0 - ∂Kij_∂xk[i, j, k]) 
            end
        end
    end
    return D
end

function Di_RR(xH::Vector{Float64}, vH::AbstractArray, xBL::AbstractArray, ∂Ki_∂xk::SMatrix{3, 3, Float64}, ∂Kij_∂xk::SArray{Tuple{3, 3, 3}, Float64, 3, 27}, Qi::Vector{Float64}, Qij::AbstractArray, rH::Float64, a::Float64, M::Float64, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)   # Eq. A11
    D = @MVector [0., 0., 0.]
    @inbounds for j=1:3
        for k=1:3
            @inbounds for i=1:3
                D[i] += 4.0 * Qi[i] * vH[j] * vH[k] * ∂Ki_∂xk[j, k]
            end
            D .+= 2.0 * (ηij + Qij) * vH[j] * vH[k] * [(2.0 * (∂Kij_∂xk[j, k, l] + ∂Kij_∂xk[k, j, l]) / 2.0 - ∂Kij_∂xk[l, j, k]) for l=1:3]
        end
    end
    return D
end

# computes the four self-acceleration components A^{2}_{β} (Eqs. 62 - 63)
function A2_β(t::Float64, xH::AbstractArray, vH::AbstractArray, xBL::AbstractArray, rH::Float64, a::Float64, M::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mijk7::AbstractArray, Sij5::AbstractArray, Γαμν::Function, g_μν::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    jBLH = HarmonicCoords.jBLH(xH, a, M)
    ∂K_∂xk = @SVector [SelfForce.∂K_∂xk(xH, xBL, jBLH, a, M, g_μν, Γαμν, j) for j=1:3];
    ∂Ki_∂xk = @SMatrix [SelfForce.∂Ki_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, j, k) for j=1:3, k=1:3];
    ∂Kij_∂xk = @SArray [SelfForce.∂Kij_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, j, k, l) for j=1:3, k=1:3, l=1:3]
    Q = SelfForce.Q(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)
    Qi = SelfForce.Qi(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)
    Qij = SelfForce.Qij(xH, a, M, gTT, gTΦ, gRR, gThTh, gΦΦ)

    BRR = B_RR(xH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    BiRR = Bi_RR(xH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    CRR = C_RR(xH, vH, xBL, ∂K_∂xk, ∂Ki_∂xk, Q, Qi, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    CiRR = Ci_RR(xH, vH, xBL, ∂K_∂xk, ∂Ki_∂xk, Qi, Qij, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    DRR = D_RR(xH, vH, xBL, ∂Ki_∂xk, ∂Kij_∂xk, Q, Qi, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)
    DiRR = Di_RR(xH, vH, xBL, ∂Ki_∂xk, ∂Kij_∂xk, Qi, Qij, rH, a, M, Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ)

    VRR = Vrr(t, xH,  Mij5, Mij7, Mijk7)
    ViRR = Virr(t, xH, Mij6, Sij5)

    A2_t = (BRR + CRR + DRR) * VRR + dot((BiRR + CiRR + DiRR), ViRR)   # Eq. 62
    A2_i = -2.0 * (BRR + CRR + DRR) * ViRR - (1.0/2.0) * (BiRR + CiRR + DiRR) * VRR   # Eq. 63

    return [A2_t, A2_i...]
end

# compute self-acceleration in harmonic coordinates and transform components back to BL
function aRRα(n::Vector{Int}, aSF::AbstractArray, t::Float64, xH::AbstractArray, v::AbstractArray, v_H::AbstractArray, vH::AbstractArray, xBL::AbstractArray, rH::Vector{Float64}, a::Float64, M::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function)
    @inbounds Threads.@threads for i in n
        aSF[:, i] = -Γ(v_H[i], xH[i], a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)^2 * Pαβ(v_H[i], xH[i], a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ) * (A1_β(t, xH[i], v[i], v_H[i], vH[i], xBL[i], rH[i], Mij5[:, :, i], Mij6[:, :, i], Mij7[:, :, i], Mij8[:, :, i], Mijk7[:, :, :, i], Mijk8[:, :, :, i], Sij5[:, :, i], Sij6[:, :, i]) + A2_β(t, xH[i], vH[i], xBL[i], rH[i], a, M, Mij5[:, :, i], Mij6[:, :, i], Mij7[:, :, i], Mijk7[:, :, :, i], Sij5[:, :, i], Γαμν, g_μν, gTT, gTΦ, gRR, gThTh, gΦΦ))
        aSF[2:4, i] = HarmonicCoords.aHtoBL(xH[i], zeros(3), aSF[2:4, i], a, M)
    end
end


# returns the self-acceleration 4-vector
function selfAcc!(n::Vector{Int}, aSF::AbstractArray, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray, v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, t::Vector{Float64}, tdot::Vector{Float64}, r::Vector{Float64}, rdot::Vector{Float64}, rddot::Vector{Float64}, θ::Vector{Float64}, θdot::Vector{Float64}, θddot::Vector{Float64}, ϕ::Vector{Float64}, ϕdot::Vector{Float64}, ϕddot::Vector{Float64}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray, Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, a::Float64, M::Float64, m::Float64)
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in n
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]) / tdot[i];             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]) / (tdot[i]^2);      # divide by (dt/dτ)² to get accelerations wrt BL time
    end
    @inbounds Threads.@threads for i in n
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
        x_H[i] = xH[i]
        rH[i] = norm_3d(xH[i]);
    end
    @inbounds Threads.@threads for i in n
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        v_H[i] = vH[i]; 
        v[i] = norm_3d(vH[i]);
    end
    @inbounds Threads.@threads for i in n
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
        a_H[i] = aH[i]
    end
    
    # calculate ddotMijk, ddotMijk, dotSij "analytically"
    SelfForce.moments!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)

    # calculate moment derivatives numerically at t = tF
    SelfForce.moment_derivs!(t, Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)

    return SelfForce.aRRα(n, aSF, 0.0, xH, v, v_H, vH, xBL, rH, a, M, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ)
end

Z_1(a::Float64, M::Float64) = 1 + (1 - a^2 / M^2)^(1/3) * ((1 + a / M)^(1/3) + (1 - a / M)^(1/3))
Z_2(a::Float64, M::Float64) = sqrt(3 * (a / M)^2 + Z_1(a, M)^2)
LSO_r(a::Float64, M::Float64) = M * (3 + Z_2(a, M) - sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # retrograde LSO
LSO_p(a::Float64, M::Float64) = M * (3 + Z_2(a, M) + sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # prograde LSO

function compute_inspiral!(τOrbit::Float64, nPoints::Int, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, saveat::Float64=0.5, Δti::Float64=1.0, reltol::Float64=1e-16, abstol::Float64=1e-16; data_path::String="Data/")
    # create arrays for trajectory
    t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    tdot = Float64[]; rdot = Float64[]; θdot = Float64[]; ϕdot = Float64[];
    tddot = Float64[]; rddot = Float64[]; θddot = Float64[]; ϕddot = Float64[];
    
    # initialize data arrays
    aSF = Vector{Matrix{Float64}}()
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    xBL = [Float64[] for i in 1:nPoints]
    vBL = [Float64[] for i in 1:nPoints]
    aBL = [Float64[] for i in 1:nPoints]
    xH = [Float64[] for i in 1:nPoints]
    x_H = [Float64[] for i in 1:nPoints]
    vH = [Float64[] for i in 1:nPoints]
    v_H = [Float64[] for i in 1:nPoints]
    v = zeros(nPoints)
    rH = zeros(nPoints)
    aH = [Float64[] for i in 1:nPoints]
    a_H = [Float64[] for i in 1:nPoints]
    Mij5 = zeros(3, 3, nPoints)
    Mij6 = zeros(3, 3, nPoints)
    Mij7 = zeros(3, 3, nPoints)
    Mij8 = zeros(3, 3, nPoints)
    Mijk7 = zeros(3, 3, 3, nPoints)
    Mijk8 = zeros(3, 3, 3, nPoints)
    Sij5 = zeros(3, 3, nPoints)
    Sij6 = zeros(3, 3, nPoints)
    aSF_temp = zeros(4, nPoints)
    aSF_avg = zeros(4)

    function geodesicEq!(ddu, du, u, params, t)
        ddu[1] = Kerr.KerrGeodesics.tddot(u..., du..., params...) + aSF_avg[1]
        ddu[2] = Kerr.KerrGeodesics.rddot(u..., du..., params...) + aSF_avg[2]
        ddu[3] = Kerr.KerrGeodesics.θddot(u..., du..., params...) + aSF_avg[3]
        ddu[4] = Kerr.KerrGeodesics.ϕddot(u..., du..., params...) + aSF_avg[4]
    end

    # orbital parameters
    params = [a, M];

    # define periastron and apastron
    rp = p * M / (1 + e);
    ra = p * M / (1 - e);

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi)   # dimensionless constants

    # initial conditions for Kerr geodesic trajectory
    ri = ra;
    ics = Kerr.KerrGeodesics.boundKerr_ics(a, M, m, EEi, LLi, ri, θi, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ);
    τ0 = 0.0; Δτ = nPoints * saveat * M; τF = τ0 + Δτ; params = [a, M];
    n=1:nPoints |> collect
    rLSO = LSO_p(a, M)
    while τOrbit > τF
        τspan = (τ0, τF)

        # stop when it reaches LSO
        condition(u, t , integrator) = u[6] - rLSO # Is zero when r = rLSO
        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(condition, affect!)

        # numerically solve for geodesic motion
        prob = SecondOrderODEProblem(geodesicEq!, ics..., τspan, params);
        
        sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat, callback = cb);

        # deconstruct solution
        ttdot = sol[1, :];
        rrdot = sol[2, :];
        θθdot = sol[3, :];
        ϕϕdot = sol[4, :];
        tt = sol[5, :];
        rr = sol[6, :];
        θθ = sol[7, :];
        ϕϕ= sol[8, :];

        # break out of loop when LSO reached
        if length(tt) < nPoints
            τF = τspan[1] + saveat * (length(tt)-1)
            ttddot = Kerr.KerrGeodesics.tddot.(tt, rr, θθ, ϕϕ, ttdot, rrdot, θθdot, ϕϕdot, params...);
            rrddot = Kerr.KerrGeodesics.rddot.(tt, rr, θθ, ϕϕ, ttdot, rrdot, θθdot, ϕϕdot, params...);
            θθddot = Kerr.KerrGeodesics.θddot.(tt, rr, θθ, ϕϕ, ttdot, rrdot, θθdot, ϕϕdot, params...);
            ϕϕddot = Kerr.KerrGeodesics.ϕddot.(tt, rr, θθ, ϕϕ, ttdot, rrdot, θθdot, ϕϕdot, params...);
            append!(t, tt); append!(tdot, ttdot); append!(tddot, ttddot); append!(r, rr); append!(rdot, rrdot); append!(rddot, rrddot); append!(θ, θθ); append!(θdot, θθdot); append!(θddot, θθddot); append!(ϕ, ϕϕ); append!(ϕdot, ϕϕdot); append!(ϕddot, ϕϕddot);
            println("LSO reached at t = $(last(tt))")
            break
        end

        # save endpoints for initial conditions of next geodesic
        ics = [[last(ttdot), last(rrdot), last(θθdot), last(ϕϕdot)], [last(tt), last(rr), last(θθ), last(ϕϕ)]];
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

        
        # calculate SF at each point of trajectory and take the sum
        SelfForce.selfAcc!(n, aSF_temp, xBL, vBL, aBL, xH, x_H, rH, vH, v_H, aH, a_H, v, tt, ttdot, rr, rrdot, rrddot, θθ, θθdot, θθddot, ϕϕ, ϕϕdot, ϕϕddot, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m);

        # updated averaged SF
        aSF_avg .= sum(aSF_temp, dims = 2) / nPoints

        # store self force values
        push!(aSF, aSF_temp)
    end

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF = hcat(aSF...)
    SF_filename=data_path * "aSF_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tstep_$(saveat)_tol_$(reltol).txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF)
    end

    τRange = 0.0:saveat:τF |> collect
    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tstep_$(saveat)_tol_$(reltol).txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end
    println("Self-force file saved to: " * SF_filename)
    println("ODE saved to: " * ODE_filename)
end

end