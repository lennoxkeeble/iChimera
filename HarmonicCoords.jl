#=

    In this module we write functions to perform coordinate transformations between Boyer-Lindquist (BL) and Harmonic coordinates as per Sec. IIID in arXiv:1109.0572v2. This transformation is necessary in the kludge scheme introduced therein since
    the post-Minkowskian self-force, upon which the scheme is based, is computed from multipole moments expressed in harmonics coordinates.

=#

module HarmonicCoords
using LinearAlgebra
using StaticArrays
import ..Kerr.KerrMetric: g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ

"""
# Common Arguments in this module
- `r::Float64`: Boyer-Lindquist radial coordinate.
- `θ::Float64`: Boyer-Lindquist polar coordinate.
- `ϕ::Float64`: Boyer-Lindquist azimuthal coordinate.
- `xBL::AbstractVector{Float64}`: Boyer-Lindquist coordinates, xBL = [r, θ, ϕ].
- `vBL::AbstractVector{Float64}`: Boyer-Lindquist velocity wrt coordinate time, vBL = [dr_dt, dθ_dt, dϕ_dt].
- `aBL::AbstractVector{Float64}`: Boyer-Lindquist acceleration wrt coordinate time, aBL = [d^2r_dt^2, d^2θ_dt^2, d^2ϕ_dt^2].
- `xH::AbstractVector{Float64}`: Harmonic coordinates, xH = [x, y, z].
- `vH::AbstractVector{Float64}`: Harmonic velocity wrt coordinate time, vH = [dx_dt, dy_dt, dz_dt].
- `aH::AbstractVector{Float64}`: Harmonic acceleration wrt coordinate time, aH = [d^2x_dt^2, d^2y_dt^2, d^2z_dt^2].
- `rH::Float64`: rH = sqrt(xH^2 + yH^2 + zH^2).
- `a::Float64`: Kerr black hole spin parameter.

# Notes
- The vast majority of the longer equatinos here we computed in a MMA notebook and don't directly come from arXiv:1109.0572v2. They are long, but their derivation is straightforwad since one simply uses differentiates the forward transformations
in Eqs. 67-70 and the inverse transformations in 71-76 in order to compute the jacobian and hessians of the forward and inverse transformations.
"""

# define useful functions
otimes(a::AbstractVector{Float64}, b::AbstractVector{Float64}) = [a[i] * b[j] for i in eachindex(a), j in eachindex(b)]
otimes(a::AbstractVector{Float64}) = [a[i] * a[j] for i in eachindex(a), j in eachindex(a)]
norm2_3d(u::AbstractVector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
norm_3d(u::AbstractVector{Float64}) = sqrt(norm2_3d(u))

# define r±
rplus(a::Float64) = 1.0 + sqrt(1.0 - a^2)
rminus(a::Float64) = 1.0 - sqrt(1.0 - a^2)

# define functions used in coordinate transformations, where r is in BL coordinates
Ω(r::Float64, a::Float64) = tan(a * log((r - rminus(a)) / (r - rplus(a))) / (2.0 * sqrt(1.0 - a^2)))   # Eq. 76
Φ(r::Float64, a::Float64) = π/2 - atan(((r - 1.0) / a + Ω(r, a)), (1.0 - (r - 1.0) * Ω(r, a) / a))    # Eq. 75
∂Φ_∂r(r::Float64, a::Float64) = a / ((a^2 + (1.0 - r)^2) * (a^2 + r * (r - 2.0)))
∂2Φ_∂rr(r::Float64, a::Float64) = 2.0a * (1.0 - r) * (-1.0 / ((a^2 + (1.0 - r)^2)^2) + 1.0 / ((a^2 + r * (r - 2.0))^2))

# transforms a set of BL coordinates to harmonic coordinates where x = xBL = [r, θ, ϕ]
function xBLtoH!(xH::AbstractVector{Float64}, xBL::AbstractVector{Float64}, a::Float64)
    xH[1] = sqrt((xBL[1] - 1.0)^2 + a^2) * sin(xBL[2]) * cos(xBL[3] - Φ(xBL[1], a))   # Eq. 68
    xH[2] = sqrt((xBL[1] - 1.0)^2 + a^2) * sin(xBL[2]) * sin(xBL[3] - Φ(xBL[1], a))   # Eq. 69
    xH[3] = (xBL[1] - 1.0) * cos(xBL[2])    # Eq. 70.0
end

# transforms a set of BL coordinates to harmonic coordinates where x = xBL = [r, θ, ϕ]
function xBLtoH(xBL::AbstractVector{Float64}, a::Float64)
    xh = sqrt((xBL[1] - 1.0)^2 + a^2) * sin(xBL[2]) * cos(xBL[3] - Φ(xBL[1], a))   # Eq. 68
    yh = sqrt((xBL[1] - 1.0)^2 + a^2) * sin(xBL[2]) * sin(xBL[3] - Φ(xBL[1], a))   # Eq. 69
    zh = (xBL[1] - 1.0) * cos(xBL[2])    # Eq. 70.0
    return [xh, yh, zh]  
end

# transforms a set of harmonic coordinates to BL where xH = [xH, yH, zH]
function xHtoBL(xH::AbstractVector{Float64}, a::Float64)
    rH = norm_3d(xH)   # Eq. 74
    rBL = 1.0 + sqrt((rH^2 - a^2 + sqrt((rH^2 - a^2)^2 + 4.0 * (a^2) * (xH[3]^2))) / 2.0)    # Eq. 72
    θ = acos(xH[3] / (rBL - 1.0))    # Eq. 73
    ϕ = Φ(rBL, a) + atan(xH[2], xH[1])  # Eq. 71
    return [rBL, θ, ϕ]
end

ρ(xH::AbstractVector{Float64}, a::Float64) = sqrt((norm2_3d(xH) - a^2 + sqrt((norm2_3d(xH) - a^2)^2 + 4.0 * (a * xH[3])^2)) / 2.0)   # Eq. B2

# Jacobian

# J^{BL}_{H}
∂r_∂xH(xH::AbstractVector{Float64}, a::Float64) = ρ(xH, a) * xH[1] / (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH))
∂r_∂yH(xH::AbstractVector{Float64}, a::Float64) = ρ(xH, a) * xH[2] / (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH))
∂r_∂zH(xH::AbstractVector{Float64}, a::Float64) = (ρ(xH, a)^2 + a^2) * xH[3] / (ρ(xH, a) * (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH)))
∂r_∂rH(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂r_∂xH(xH, a), ∂r_∂yH(xH, a), ∂r_∂zH(xH, a)]


∂θ_∂xH(xH::AbstractVector{Float64}, a::Float64) = xH[1] * xH[3] / (sqrt(ρ(xH, a)^2 - xH[3]^2) * (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH)))
∂θ_∂yH(xH::AbstractVector{Float64}, a::Float64) = xH[2] * xH[3] / (sqrt(ρ(xH, a)^2 - xH[3]^2) * (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH)))
∂θ_∂zH(xH::AbstractVector{Float64}, a::Float64) = (a^2.0 * (xH[3]^2 - ρ(xH, a)^2) + ρ(xH, a)^2.0 * (-2.0 * ρ(xH, a)^2 + norm2_3d(xH) + xH[3]^2)) / (ρ(xH, a)^2.0 * sqrt(ρ(xH, a)^2 - xH[3]^2) * (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH)))
∂θ_∂rH(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂θ_∂xH(xH, a), ∂θ_∂yH(xH, a), ∂θ_∂zH(xH, a)]

∂ϕ_∂xH(xH::AbstractVector{Float64}, a::Float64) = ρ(xH, a) * xH[1] * ∂Φ_∂r(ρ(xH, a) + 1.0, a) / (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH)) - xH[2] / (norm2_3d(xH) - xH[3]^2)
∂ϕ_∂yH(xH::AbstractVector{Float64}, a::Float64) = ρ(xH, a) * xH[2] * ∂Φ_∂r(ρ(xH, a) + 1.0, a) / (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH)) + xH[1] / (norm2_3d(xH) - xH[3]^2)
∂ϕ_∂zH(xH::AbstractVector{Float64}, a::Float64) = xH[3] * (a^2 + ρ(xH, a)^2) * ∂Φ_∂r(ρ(xH, a) + 1.0, a) / (ρ(xH, a) * (a^2 + 2.0 * ρ(xH, a)^2 - norm2_3d(xH)))
∂ϕ_∂rH(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂ϕ_∂xH(xH, a), ∂ϕ_∂yH(xH, a), ∂ϕ_∂zH(xH, a)]

jBLH(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂r_∂xH(xH, a) ∂r_∂yH(xH, a) ∂r_∂zH(xH, a); ∂θ_∂xH(xH, a) ∂θ_∂yH(xH, a) ∂θ_∂zH(xH, a); ∂ϕ_∂xH(xH, a) ∂ϕ_∂yH(xH, a) ∂ϕ_∂zH(xH, a)]

∂rBL_∂xH(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂r_∂xH(xH, a), ∂θ_∂xH(xH, a), ∂ϕ_∂xH(xH, a)]
∂rBL_∂yH(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂r_∂yH(xH, a), ∂θ_∂yH(xH, a), ∂ϕ_∂yH(xH, a)]
∂rBL_∂zH(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂r_∂zH(xH, a), ∂θ_∂zH(xH, a), ∂ϕ_∂zH(xH, a)]


# J^{H}_{BL}
∂xH_∂r(xH::AbstractVector{Float64}, a::Float64) = xH[1] * ρ(xH, a) / (a^2 + ρ(xH, a)^2) + xH[2] * ∂Φ_∂r(ρ(xH, a) + 1.0, a)
∂xH_∂θ(xH::AbstractVector{Float64}, a::Float64) = xH[1] * xH[3] / sqrt(ρ(xH, a)^2 - xH[3]^2)
∂xH_∂ϕ(xH::AbstractVector{Float64}, a::Float64) = -xH[2]
∂xH_∂xBL(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂xH_∂r(xH, a), ∂xH_∂θ(xH, a), ∂xH_∂ϕ(xH, a)]


∂yH_∂r(xH::AbstractVector{Float64}, a::Float64) = xH[2] * ρ(xH, a) / (a^2 + ρ(xH, a)^2) - xH[1] * ∂Φ_∂r(ρ(xH, a) + 1.0, a)
∂yH_∂θ(xH::AbstractVector{Float64}, a::Float64) = xH[2] * xH[3] / sqrt(ρ(xH, a)^2 - xH[3]^2)
∂yH_∂ϕ(xH::AbstractVector{Float64}, a::Float64) = xH[1] 
∂yH_∂xBL(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂yH_∂r(xH, a), ∂yH_∂θ(xH, a), ∂yH_∂ϕ(xH, a)]

∂zH_∂r(xH::AbstractVector{Float64}, a::Float64) = xH[3] / ρ(xH, a)
∂zH_∂θ(xH::AbstractVector{Float64}, a::Float64) = -sqrt(ρ(xH, a)^2 - xH[3]^2)
∂zH_∂ϕ(xH::AbstractVector{Float64}, a::Float64) = 0.0
∂zH_∂xBL(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂zH_∂r(xH, a), ∂zH_∂θ(xH, a), ∂zH_∂ϕ(xH, a)]

∂rH_∂r(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂xH_∂r(xH, a), ∂yH_∂r(xH, a), ∂zH_∂r(xH, a)]
∂rH_∂θ(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂xH_∂θ(xH, a), ∂yH_∂θ(xH, a), ∂zH_∂θ(xH, a)]
∂rH_∂ϕ(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂xH_∂ϕ(xH, a), ∂yH_∂ϕ(xH, a), ∂zH_∂ϕ(xH, a)]


jHBL(xH::AbstractVector{Float64}, a::Float64) = @SArray [∂xH_∂r(xH, a) ∂xH_∂θ(xH, a) ∂xH_∂ϕ(xH, a); ∂yH_∂r(xH, a) ∂yH_∂θ(xH, a) ∂yH_∂ϕ(xH, a); ∂zH_∂r(xH, a) ∂zH_∂θ(xH, a) ∂zH_∂ϕ(xH, a)]

# Hessians
# BL -> H

## rH = norm_3d(xH)
∂r_∂ij(xH::AbstractVector{Float64}, rH::Float64, a::Float64) = @SArray [((a - rH)*(a + rH)*(a^2 - rH^2 + 3*xH[1]^2)*ρ(xH, a) + 2*(2*a^2 - 2*rH^2 + xH[1]^2)*ρ(xH, a)^3 + 4*ρ(xH, a)^5)/(a^2 - rH^2 + 2*ρ(xH, a)^2)^3 (xH[1]*xH[2]*ρ(xH, a)*(3*a^2 - 3*rH^2 + 2*ρ(xH, a)^2))/(a^2 - rH^2 + 2*ρ(xH, a)^2)^3 (xH[1]*xH[3]*(a^4 - 3*rH^2*ρ(xH, a)^2 + 2*ρ(xH, a)^4 + a^2*(-rH^2 + ρ(xH, a)^2)))/(ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3);
(xH[1]*xH[2]*ρ(xH, a)*(3*a^2 - 3*rH^2 + 2*ρ(xH, a)^2))/(a^2 - rH^2 + 2*ρ(xH, a)^2)^3 ((a - rH)*(a + rH)*(a^2 - rH^2 + 3*xH[2]^2)*ρ(xH, a) + 2*(2*a^2 - 2*rH^2 + xH[2]^2)*ρ(xH, a)^3 + 4*ρ(xH, a)^5)/(a^2 - rH^2 + 2*ρ(xH, a)^2)^3 (xH[2]*xH[3]*(a^4 - 3*rH^2*ρ(xH, a)^2 + 2*ρ(xH, a)^4 + a^2*(-rH^2 + ρ(xH, a)^2)))/(ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3);
(xH[1]*xH[3]*(a^4 - 3*rH^2*ρ(xH, a)^2 + 2*ρ(xH, a)^4 + a^2*(-rH^2 + ρ(xH, a)^2)))/(ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3) (xH[2]*xH[3]*(a^4 - 3*rH^2*ρ(xH, a)^2 + 2*ρ(xH, a)^4 + a^2*(-rH^2 + ρ(xH, a)^2)))/(ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3) ((a^2 + ρ(xH, a)^2)*(a^2*(-a^2 + rH^2)*xH[3]^2 + ((a^2 - rH^2)^2 - 3*(a^2 + rH^2)*xH[3]^2)*ρ(xH, a)^2 + 2*(2*a^2 - 2*rH^2 + xH[3]^2)*ρ(xH, a)^4 + 4*ρ(xH, a)^6))/ (ρ(xH, a)^3*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3)]

∂θ_∂ij(xH::AbstractVector{Float64}, rH::Float64, a::Float64) = @SArray [(xH[3]*((a - rH)*(a + rH)*(a^2 - rH^2 + 2*xH[1]^2)*xH[3]^2 - (a - rH)*(a + rH)*(a^2 - rH^2 + xH[1]^2 - 4*xH[3]^2)*ρ(xH, a)^2 + 2*(-2*a^2 + 2*rH^2 + xH[1]^2 + 2*xH[3]^2)*ρ(xH, a)^4 - 4*ρ(xH, a)^6))/((xH[3] - ρ(xH, a))*(xH[3] + ρ(xH, a))*sqrt(-xH[3]^2 + ρ(xH, a)^2)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3) -((xH[1]*xH[2]*xH[3]*(2*ρ(xH, a)^4 + a^2*(2*xH[3]^2 - ρ(xH, a)^2) + rH^2*(-2*xH[3]^2 + ρ(xH, a)^2)))/((-xH[3]^2 + ρ(xH, a)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3)) (xH[1]*(2*rH^2*xH[3]^4 + rH^2*(rH - xH[3])*(rH + xH[3])*ρ(xH, a)^2 - 2*(2*rH^2 + xH[3]^2)*ρ(xH, a)^4 + 4*ρ(xH, a)^6 + a^4*(-xH[3]^2 + ρ(xH, a)^2) + a^2*(2*xH[3]^4 - 5*xH[3]^2*ρ(xH, a)^2 + 4*ρ(xH, a)^4 + rH^2*(xH[3]^2 - 2*ρ(xH, a)^2))))/((-xH[3]^2 + ρ(xH, a)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3);
-((xH[1]*xH[2]*xH[3]*(2*ρ(xH, a)^4 + a^2*(2*xH[3]^2 - ρ(xH, a)^2) + rH^2*(-2*xH[3]^2 + ρ(xH, a)^2)))/((-xH[3]^2 + ρ(xH, a)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3)) (xH[3]*((a - rH)*(a + rH)*(a^2 - rH^2 + 2*xH[2]^2)*xH[3]^2 - (a - rH)*(a + rH)*(a^2 - rH^2 + xH[2]^2 - 4*xH[3]^2)*ρ(xH, a)^2 + 2*(-2*a^2 + 2*rH^2 + xH[2]^2 + 2*xH[3]^2)*ρ(xH, a)^4 - 4*ρ(xH, a)^6))/((xH[3] - ρ(xH, a))*(xH[3] + ρ(xH, a))*sqrt(-xH[3]^2 + ρ(xH, a)^2)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3) (xH[2]*(2*rH^2*xH[3]^4 + rH^2*(rH - xH[3])*(rH + xH[3])*ρ(xH, a)^2 - 2*(2*rH^2 + xH[3]^2)*ρ(xH, a)^4 + 4*ρ(xH, a)^6 + a^4*(-xH[3]^2 + ρ(xH, a)^2) + a^2*(2*xH[3]^4 - 5*xH[3]^2*ρ(xH, a)^2 + 4*ρ(xH, a)^4 + rH^2*(xH[3]^2 - 2*ρ(xH, a)^2))))/((-xH[3]^2 + ρ(xH, a)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3);
(xH[1]*(2*rH^2*xH[3]^4 + rH^2*(rH - xH[3])*(rH + xH[3])*ρ(xH, a)^2 - 2*(2*rH^2 + xH[3]^2)*ρ(xH, a)^4 + 4*ρ(xH, a)^6 + a^4*(-xH[3]^2 + ρ(xH, a)^2) + a^2*(2*xH[3]^4 - 5*xH[3]^2*ρ(xH, a)^2 + 4*ρ(xH, a)^4 + rH^2*(xH[3]^2 - 2*ρ(xH, a)^2))))/((-xH[3]^2 + ρ(xH, a)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3) (xH[2]*(2*rH^2*xH[3]^4 + rH^2*(rH - xH[3])*(rH + xH[3])*ρ(xH, a)^2 - 2*(2*rH^2 + xH[3]^2)*ρ(xH, a)^4 + 4*ρ(xH, a)^6 + a^4*(-xH[3]^2 + ρ(xH, a)^2) + a^2*(2*xH[3]^4 - 5*xH[3]^2*ρ(xH, a)^2 + 4*ρ(xH, a)^4 + rH^2*(xH[3]^2 - 2*ρ(xH, a)^2))))/((-xH[3]^2 + ρ(xH, a)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3) (xH[3]*(2*a^6*(xH[3]^2 - ρ(xH, a)^2)^2 + a^4*(xH[3] - ρ(xH, a))*(xH[3] + ρ(xH, a))*(8*xH[3]^2*ρ(xH, a)^2 - 9*ρ(xH, a)^4 + rH^2*(-2*xH[3]^2 + 3*ρ(xH, a)^2)) + ρ(xH, a)^4*(rH^6 - 6*xH[3]^2*ρ(xH, a)^4 + 4*ρ(xH, a)^6 - rH^4*(xH[3]^2 + 3*ρ(xH, a)^2) + rH^2*(2*xH[3]^4 + 3*xH[3]^2*ρ(xH, a)^2)) + a^2*(-(rH^4*xH[3]^2*ρ(xH, a)^2) + 6*xH[3]^4*ρ(xH, a)^4 - 19*xH[3]^2*ρ(xH, a)^6 + 12*ρ(xH, a)^8 + rH^2*(8*xH[3]^2*ρ(xH, a)^4 - 6*ρ(xH, a)^6))))/(ρ(xH, a)^4*(-xH[3]^2 + ρ(xH, a)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3)]

∂ϕ_∂ij(xH::AbstractVector{Float64}, rH::Float64, a::Float64) = @SArray [(2*xH[1]*xH[2])/(xH[1]^2 + xH[2]^2)^2 + (ρ(xH, a)*(((a - rH)*(a + rH)*(a^2 - rH^2 + 3*xH[1]^2) + 2*(2*a^2 - 2*rH^2 + xH[1]^2)*ρ(xH, a)^2 + 4*ρ(xH, a)^4)*∂Φ_∂r(ρ(xH, a) + 1.0, a) + xH[1]^2*ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(a^2 - rH^2 + 2*ρ(xH, a)^2)^3 (-xH[1]^2 + xH[2]^2)/(xH[1]^2 + xH[2]^2)^2 + (xH[1]*xH[2]*ρ(xH, a)*((3*a^2 - 3*rH^2 + 2*ρ(xH, a)^2)*∂Φ_∂r(ρ(xH, a) + 1.0, a) + ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(a^2 - rH^2 + 2*ρ(xH, a)^2)^3 (xH[1]*xH[3]*((a^4 - 3*rH^2*ρ(xH, a)^2 + 2*ρ(xH, a)^4 + a^2*(-rH^2 + ρ(xH, a)^2))*∂Φ_∂r(ρ(xH, a) + 1.0, a) + ρ(xH, a)*(a^2 + ρ(xH, a)^2)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3);
(-xH[1]^2 + xH[2]^2)/(xH[1]^2 + xH[2]^2)^2 + (xH[1]*xH[2]*ρ(xH, a)*((3*a^2 - 3*rH^2 + 2*ρ(xH, a)^2)*∂Φ_∂r(ρ(xH, a) + 1.0, a) + ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(a^2 - rH^2 + 2*ρ(xH, a)^2)^3 (-2*xH[1]*xH[2])/(xH[1]^2 + xH[2]^2)^2 + (ρ(xH, a)*(((a - rH)*(a + rH)*(a^2 - rH^2 + 3*xH[2]^2) + 2*(2*a^2 - 2*rH^2 + xH[2]^2)*ρ(xH, a)^2 + 4*ρ(xH, a)^4)*∂Φ_∂r(ρ(xH, a) + 1.0, a) + xH[2]^2*ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(a^2 - rH^2 + 2*ρ(xH, a)^2)^3 (xH[2]*xH[3]*((a^4 - 3*rH^2*ρ(xH, a)^2 + 2*ρ(xH, a)^4 + a^2*(-rH^2 + ρ(xH, a)^2))*∂Φ_∂r(ρ(xH, a) + 1.0, a) + ρ(xH, a)*(a^2 + ρ(xH, a)^2)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3);
(xH[1]*xH[3]*((a^4 - 3*rH^2*ρ(xH, a)^2 + 2*ρ(xH, a)^4 + a^2*(-rH^2 + ρ(xH, a)^2))*∂Φ_∂r(ρ(xH, a) + 1.0, a) + ρ(xH, a)*(a^2 + ρ(xH, a)^2)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3) (xH[2]*xH[3]*((a^4 - 3*rH^2*ρ(xH, a)^2 + 2*ρ(xH, a)^4 + a^2*(-rH^2 + ρ(xH, a)^2))*∂Φ_∂r(ρ(xH, a) + 1.0, a) + ρ(xH, a)*(a^2 + ρ(xH, a)^2)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(ρ(xH, a)*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3) ((a^2 + ρ(xH, a)^2)*((a^2*(-a^2 + rH^2)*xH[3]^2 + ((a^2 - rH^2)^2 - 3*(a^2 + rH^2)*xH[3]^2)*ρ(xH, a)^2 + 2*(2*a^2 - 2*rH^2 + xH[3]^2)*ρ(xH, a)^4 + 4*ρ(xH, a)^6)*∂Φ_∂r(ρ(xH, a) + 1.0, a) + xH[3]^2*ρ(xH, a)*(a^2 + ρ(xH, a)^2)*(a^2 - rH^2 + 2*ρ(xH, a)^2)*∂2Φ_∂rr(ρ(xH, a) + 1.0, a)))/(ρ(xH, a)^3*(a^2 - rH^2 + 2*ρ(xH, a)^2)^3)]

# outputs 2D array where [j, k] = ∂²xⁱ/∂xʲ∂xᵏ
HessBLH(xH::AbstractVector{Float64}, rH::Float64, a::Float64, i::Int) = i==1 ? ∂r_∂ij(xH, rH, a) : i==2 ? ∂θ_∂ij(xH, rH, a) : i==3 ? ∂ϕ_∂ij(xH, rH, a) : throw(DomainError(i, "i should be in the range 1 ≤ i ≤ 3"))

# H -> BL
∂x_∂ij(xH::AbstractVector{Float64}, a::Float64) = @SArray [(a^2*xH[1])/(a^2 + (-ρ(xH, a))^2)^2 + (2*(ρ(xH, a))*xH[2]*∂Φ_∂r(ρ(xH, a) + 1.0, a))/(a^2 + (-ρ(xH, a))^2) - xH[1]*∂Φ_∂r(ρ(xH, a) + 1.0, a)^2 + xH[2]*∂2Φ_∂rr(ρ(xH, a) + 1.0, a) ((ρ(xH, a))*xH[1]*xH[3] + (a^2 + (-ρ(xH, a))^2)*xH[2]*xH[3]*∂Φ_∂r(ρ(xH, a) + 1.0, a))/((a^2 + (-ρ(xH, a))^2)*sqrt((-ρ(xH, a))^2 - xH[3]^2)) ((-ρ(xH, a))*xH[2])/(a^2 + (-ρ(xH, a))^2) + xH[1]*∂Φ_∂r(ρ(xH, a) + 1.0, a);
((ρ(xH, a))*xH[1]*xH[3] + (a^2 + (-ρ(xH, a))^2)*xH[2]*xH[3]*∂Φ_∂r(ρ(xH, a) + 1.0, a))/((a^2 + (-ρ(xH, a))^2)*sqrt((-ρ(xH, a))^2 - xH[3]^2)) -xH[1] -((xH[2]*xH[3])/sqrt((-ρ(xH, a))^2 - xH[3]^2));
((-ρ(xH, a))*xH[2])/(a^2 + (-ρ(xH, a))^2) + xH[1]*∂Φ_∂r(ρ(xH, a) + 1.0, a) -((xH[2]*xH[3])/sqrt((-ρ(xH, a))^2 - xH[3]^2)) -xH[1]]

∂y_∂ij(xH::AbstractVector{Float64}, a::Float64) = @SArray [(a^2*xH[2])/(a^2 + (-ρ(xH, a))^2)^2 + (2*(-ρ(xH, a))*xH[1]*∂Φ_∂r(ρ(xH, a) + 1.0, a))/(a^2 + (-ρ(xH, a))^2) - xH[2]*∂Φ_∂r(ρ(xH, a) + 1.0, a)^2 - xH[1]*∂2Φ_∂rr(ρ(xH, a) + 1.0, a) -(((-ρ(xH, a))*xH[2]*xH[3] + (a^2 + (-ρ(xH, a))^2)*xH[1]*xH[3]*∂Φ_∂r(ρ(xH, a) + 1.0, a))/((a^2 + (-ρ(xH, a))^2)*sqrt((-ρ(xH, a))^2 - xH[3]^2))) ((ρ(xH, a))*xH[1])/(a^2 + (-ρ(xH, a))^2) + xH[2]*∂Φ_∂r(ρ(xH, a) + 1.0, a);
-(((-ρ(xH, a))*xH[2]*xH[3] + (a^2 + (-ρ(xH, a))^2)*xH[1]*xH[3]*∂Φ_∂r(ρ(xH, a) + 1.0, a))/((a^2 + (-ρ(xH, a))^2)*sqrt((-ρ(xH, a))^2 - xH[3]^2))) -xH[2] (xH[1]*xH[3])/sqrt((-ρ(xH, a))^2 - xH[3]^2);
((ρ(xH, a))*xH[1])/(a^2 + (-ρ(xH, a))^2) + xH[2]*∂Φ_∂r(ρ(xH, a) + 1.0, a) (xH[1]*xH[3])/sqrt((-ρ(xH, a))^2 - xH[3]^2) -xH[2]]

∂z_∂ij(xH::AbstractVector{Float64}, a::Float64) = @SArray [0.0 -sqrt(1.0 - xH[3]^2/(-ρ(xH, a))^2) 0.0;
-sqrt(1.0 - xH[3]^2/(-ρ(xH, a))^2) -xH[3] 0.0;
0.0 0.0 0.0]

# outputs 2D array where [j, k] = ∂²xⁱ/∂xʲ∂xᵏ
HessHBL(xH::AbstractVector{Float64}, rH::Float64, a::Float64, i::Int) = i==1 ? ∂x_∂ij(xH, a) : i==2 ? ∂y_∂ij(xH, a) : i==3 ? ∂z_∂ij(xH, a) : throw(DomainError(i, "i should be in the range 1 ≤ i ≤ 3"))

∂ᵢr∂ⱼr(xH::AbstractVector{Float64}, a::Float64) = otimes(∂r_∂rH(xH, a))
∂ᵢθ∂ⱼθ(xH::AbstractVector{Float64}, a::Float64) = otimes(∂θ_∂rH(xH, a))
∂ᵢϕ∂ⱼϕ(xH::AbstractVector{Float64}, a::Float64) = otimes(∂ϕ_∂rH(xH, a))

∂ij_∂r(xH::AbstractVector{Float64}, a::Float64) = otimes(∂rH_∂r(xH, a))
∂ij_∂θ(xH::AbstractVector{Float64}, a::Float64) = otimes(∂rH_∂θ(xH, a))
∂ij_∂ϕ(xH::AbstractVector{Float64}, a::Float64) = otimes(∂rH_∂ϕ(xH, a))

# covariant metric in harmonic coordinates: r denotes spacial indices
g_tt_H(xH::AbstractVector{Float64}, a::Float64) = g_tt(xHtoBL(xH, a)..., a)
g_tr_H(xH::AbstractVector{Float64}, a::Float64) = g_tϕ(xHtoBL(xH, a)..., a) * ∂ϕ_∂rH(xH, a)
g_rr_H(xH::AbstractVector{Float64}, a::Float64) = g_rr(xHtoBL(xH, a)..., a) * (∂ᵢr∂ⱼr(xH, a)) + g_θθ(xHtoBL(xH, a)..., a) * (∂ᵢθ∂ⱼθ(xH, a)) + g_ϕϕ(xHtoBL(xH, a)..., a) * (∂ᵢϕ∂ⱼϕ(xH, a))   # Eq. B18

@views function g_μν_H(xH::AbstractVector{Float64}, a::Float64)
    gg = zeros(4, 4)
    xBL = HarmonicCoords.xHtoBL(xH, a)
    gg[1, 1] = g_tt(xBL[1], xBL[2], xBL[3], a)
    gg[1, 2] = g_tϕ(xBL[1], xBL[2], xBL[3], a) * ∂ϕ_∂xH(xH, a); gg[2, 1] = gg[1, 2];
    gg[1, 3] = g_tϕ(xBL[1], xBL[2], xBL[3], a) * ∂ϕ_∂yH(xH, a); gg[3, 1] = gg[1, 3];
    gg[1, 4] = g_tϕ(xBL[1], xBL[2], xBL[3], a) * ∂ϕ_∂zH(xH, a); gg[4, 1] = gg[1, 4];
    gg[2:4, 2:4] = g_rr_H(xH, a)
    return gg
end

# contravariant metric in harmonic coordinates: r denotes spacial indices
gTT_H(xH::AbstractVector{Float64}, a::Float64) = gTT(xHtoBL(xH, a)..., a)
gTR_H(xH::AbstractVector{Float64}, a::Float64) = gTΦ(xHtoBL(xH, a)..., a) * ∂rH_∂ϕ(xH, a)
gRR_H(xH::AbstractVector{Float64}, a::Float64) = gRR(xHtoBL(xH, a)..., a) * (∂ij_∂r(xH, a)) + gThTh(xHtoBL(xH, a)..., a) * (∂ij_∂θ(xH, a)) + gΦΦ(xHtoBL(xH, a)..., a) * (∂ij_∂ϕ(xH, a))    # Eq. B18

@views function gμν_H(xH::AbstractVector{Float64}, a::Float64)
    gg = zeros(4, 4)
    xBL = HarmonicCoords.xHtoBL(xH, a)
    gg[1, 1] = gTT(xBL[1], xBL[2], xBL[3], a)
    gg[1, 2] = gTΦ(xBL[1], xBL[2], xBL[3], a) * ∂xH_∂ϕ(xH, a); gg[2, 1] = gg[1, 2];
    gg[1, 3] = gTΦ(xBL[1], xBL[2], xBL[3], a) * ∂yH_∂ϕ(xH, a); gg[3, 1] = gg[1, 3];
    gg[1, 4] = gTΦ(xBL[1], xBL[2], xBL[3], a) * ∂zH_∂ϕ(xH, a); gg[4, 1] = gg[1, 4];
    gg[2:4, 2:4] = gRR_H(xH, a)
    return gg
end

# transfrom BL velocities to harmonic
vBLtoH(xH::AbstractVector{Float64}, vBL::AbstractVector{Float64}, a::Float64) = ∂rH_∂r(xH, a) * vBL[1] .+ ∂rH_∂θ(xH, a) * vBL[2] .+ ∂rH_∂ϕ(xH, a) * vBL[3]   # Eq. 78

# transfrom BL velocities to harmonic
function vBLtoH!(vH::AbstractVector{Float64}, xH::AbstractVector{Float64}, vBL::AbstractVector{Float64}, a::Float64)
    vH[1] = vBL[1] * ∂xH_∂r(xH, a) + vBL[2] * ∂xH_∂θ(xH, a) + vBL[3] * ∂xH_∂ϕ(xH, a)
    vH[2] = vBL[1] * ∂yH_∂r(xH, a) + vBL[2] * ∂yH_∂θ(xH, a) + vBL[3] * ∂yH_∂ϕ(xH, a)
    vH[3] = vBL[1] * ∂zH_∂r(xH, a) + vBL[2] * ∂zH_∂θ(xH, a) + vBL[3] * ∂zH_∂ϕ(xH, a)
end

# transfrom harmonic velocities to BL
vHtoBL(xH::AbstractVector{Float64}, vH::AbstractVector{Float64}, a::Float64) = ∂rBL_∂xH(xH, a) * vH[1] .+ ∂rBL_∂yH(xH, a) * vH[2] .+ ∂rBL_∂zH(xH, a) * vH[3]   # Eq. 78

# transfrom BL accelerations to harmonic
function aBLtoH!(aH::AbstractVector{Float64}, xH::AbstractVector{Float64}, vBL::AbstractVector{Float64}, aBL::AbstractVector{Float64}, a::Float64)   # Eq. 79
    HessXHtoBL = HessHBL(xH, norm_3d(xH), a, 1) 
    HessYHtoBL = HessHBL(xH, norm_3d(xH), a, 2) 
    HessZHtoBL = HessHBL(xH, norm_3d(xH), a, 3)
    aH[1] = aBL[1] * ∂xH_∂r(xH, a) + aBL[2] * ∂xH_∂θ(xH, a) + aBL[3] * ∂xH_∂ϕ(xH, a) + HessXHtoBL[1, 1] * vBL[1]^2 + HessXHtoBL[2, 2] * vBL[2]^2 + HessXHtoBL[3, 3] * vBL[3]^2 + 2.0 * HessXHtoBL[1, 2] * vBL[1] * vBL[2] + 2.0 * HessXHtoBL[1, 3] * vBL[1] * vBL[3] + 2.0 * HessXHtoBL[2, 3] * vBL[2] * vBL[3]
    aH[2] = aBL[1] * ∂yH_∂r(xH, a) + aBL[2] * ∂yH_∂θ(xH, a) + aBL[3] * ∂yH_∂ϕ(xH, a) + HessYHtoBL[1, 1] * vBL[1]^2 + HessYHtoBL[2, 2] * vBL[2]^2 + HessYHtoBL[3, 3] * vBL[3]^2 + 2.0 * HessYHtoBL[1, 2] * vBL[1] * vBL[2] + 2.0 * HessYHtoBL[1, 3] * vBL[1] * vBL[3] + 2.0 * HessYHtoBL[2, 3] * vBL[2] * vBL[3]
    aH[3] = aBL[1] * ∂zH_∂r(xH, a) + aBL[2] * ∂zH_∂θ(xH, a) + aBL[3] * ∂zH_∂ϕ(xH, a) + HessZHtoBL[1, 1] * vBL[1]^2 + HessZHtoBL[2, 2] * vBL[2]^2 + HessZHtoBL[3, 3] * vBL[3]^2 + 2.0 * HessZHtoBL[1, 2] * vBL[1] * vBL[2] + 2.0 * HessZHtoBL[1, 3] * vBL[1] * vBL[3] + 2.0 * HessZHtoBL[2, 3] * vBL[2] * vBL[3]
end

# transfrom BL accelerations to harmonic
function aBLtoH(xH::AbstractVector{Float64}, vBL::AbstractVector{Float64}, aBL::AbstractVector{Float64}, a::Float64)   # Eq. 79
    HessXHtoBL = HessHBL(xH, norm_3d(xH), a, 1) 
    HessYHtoBL = HessHBL(xH, norm_3d(xH), a, 2) 
    HessZHtoBL = HessHBL(xH, norm_3d(xH), a, 3) 
    return ∂rH_∂r(xH, a) * aBL[1] + ∂rH_∂θ(xH, a) * aBL[2] + ∂rH_∂ϕ(xH, a) * aBL[3] + [HessXHtoBL[1, 1],  HessYHtoBL[1, 1], HessZHtoBL[1, 1]] * vBL[1]^2 + [HessXHtoBL[2, 2],  HessYHtoBL[2, 2], HessZHtoBL[2, 2]] * vBL[2]^2 + [HessXHtoBL[3, 3],  HessYHtoBL[3, 3], HessZHtoBL[3, 3]] * vBL[3]^2 + 2.0 * [HessXHtoBL[1, 2],  HessYHtoBL[1, 2], HessZHtoBL[1, 2]] * vBL[1] * vBL[2] + 2.0 * [HessXHtoBL[1, 3],  HessYHtoBL[1, 3], HessZHtoBL[1, 3]] * vBL[1] * vBL[3] + 2.0 * [HessXHtoBL[2, 3],  HessYHtoBL[2, 3], HessZHtoBL[2, 3]] * vBL[2] * vBL[3]
end

# transfrom harmonic accelerations to BL
function aHtoBL(xH::AbstractVector{Float64}, vH::AbstractVector{Float64}, aH::AbstractVector{Float64}, a::Float64)   # Eq. 79
    Hess_rBLtoH = HessBLH(xH, norm_3d(xH), a, 1) 
    Hess_θBLtoH = HessBLH(xH, norm_3d(xH), a, 2) 
    Hess_ϕBLtoH = HessBLH(xH, norm_3d(xH), a, 3) 
    return ∂rBL_∂xH(xH, a) * aH[1] + ∂rBL_∂yH(xH, a) * aH[2] + ∂rBL_∂zH(xH, a) * aH[3] + [Hess_rBLtoH[1, 1],  Hess_θBLtoH[1, 1], Hess_ϕBLtoH[1, 1]] * vH[1]^2 + [Hess_rBLtoH[2, 2],  Hess_θBLtoH[2, 2], Hess_ϕBLtoH[2, 2]] * vH[2]^2 + [Hess_rBLtoH[3, 3],  Hess_θBLtoH[3, 3], Hess_ϕBLtoH[3, 3]] * vH[3]^2 + 2.0 * [Hess_rBLtoH[1, 2],  Hess_θBLtoH[1, 2], Hess_ϕBLtoH[1, 2]] * vH[1] * vH[2] + 2.0 * [Hess_rBLtoH[1, 3],  Hess_θBLtoH[1, 3], Hess_ϕBLtoH[1, 3]] * vH[1] * vH[3] + 2.0 * [Hess_rBLtoH[2, 3],  Hess_θBLtoH[2, 3], Hess_ϕBLtoH[2, 3]] * vH[2] * vH[3]
end

# transfrom harmonic accelerations to BL
function aHtoBL!(aBL::AbstractVector{Float64}, xH::AbstractVector{Float64}, vH::AbstractVector{Float64}, aH::AbstractVector{Float64}, a::Float64)
    Hess_rBLtoH = HessBLH(xH, norm_3d(xH), a, 1) 
    Hess_θBLtoH = HessBLH(xH, norm_3d(xH), a, 2) 
    Hess_ϕBLtoH = HessBLH(xH, norm_3d(xH), a, 3)
    aBL[1] = aH[1] * ∂r_∂xH(xH, a) + aH[2] * ∂r_∂yH(xH, a) + aH[3] * ∂r_∂zH(xH, a) + Hess_rBLtoH[1, 1] * vH[1]^2 + Hess_rBLtoH[2, 2] * vH[2]^2 + Hess_rBLtoH[3, 3] * vH[3]^2 + 2.0 * Hess_rBLtoH[1, 2] * vH[1] * vH[2] + 2.0 * Hess_rBLtoH[1, 3] * vH[1] * vH[3] + 2.0 * Hess_rBLtoH[2, 3] * vH[2] * vH[3]
    aBL[2] = aH[1] * ∂θ_∂xH(xH, a) + aH[2] * ∂θ_∂yH(xH, a) + aH[3] * ∂θ_∂zH(xH, a) + Hess_θBLtoH[1, 1] * vH[1]^2 + Hess_θBLtoH[2, 2] * vH[2]^2 + Hess_θBLtoH[3, 3] * vH[3]^2 + 2.0 * Hess_θBLtoH[1, 2] * vH[1] * vH[2] + 2.0 * Hess_θBLtoH[1, 3] * vH[1] * vH[3] + 2.0 * Hess_θBLtoH[2, 3] * vH[2] * vH[3]
    aBL[3] = aH[1] * ∂ϕ_∂xH(xH, a) + aH[2] * ∂ϕ_∂yH(xH, a) + aH[3] * ∂ϕ_∂zH(xH, a) + Hess_ϕBLtoH[1, 1] * vH[1]^2 + Hess_ϕBLtoH[2, 2] * vH[2]^2 + Hess_ϕBLtoH[3, 3] * vH[3]^2 + 2.0 * Hess_ϕBLtoH[1, 2] * vH[1] * vH[2] + 2.0 * Hess_ϕBLtoH[1, 3] * vH[1] * vH[3] + 2.0 * Hess_ϕBLtoH[2, 3] * vH[2] * vH[3]
end

end