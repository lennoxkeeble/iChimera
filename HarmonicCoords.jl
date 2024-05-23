# we write covariant vectors with underscores (e.g., for BL coordinates x^BL = xBL x_μ = x_BL)
module HarmonicCoords
using LinearAlgebra
using StaticArrays

# define useful functions
otimes(a::Vector, b::Vector) = [a[i] * b[j] for i in eachindex(a), j in eachindex(b)]
otimes(a::Vector) = [a[i] * a[j] for i in eachindex(a), j in eachindex(a)]
norm2_3d(u::Vector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
norm_3d(u::Vector{Float64}) = sqrt(norm2_3d(u))

# define r±
rplus(a::Float64, M::Float64) = M + sqrt(M^2 - a^2)
rminus(a::Float64, M::Float64) = M - sqrt(M^2 - a^2)

# define functions used in coordinate transformations, where r is in BL coordinates
Ω(r:: Float64, a::Float64, M::Float64) = tan(a * log((r - rminus(a, M)) / (r - rplus(a, M))) / (2.0 * sqrt(M^2 - a^2)))   # Eq. 76
Φ(r::Float64, a::Float64, M::Float64) = π/2 - atan(((r - M) / a + Ω(r, a, M)), (1.0 - (r - M) * Ω(r, a, M) / a))    # Eq. 75
∂Φ_∂r(r::Float64, a::Float64, M::Float64) = a * M^2 / ((a^2 + (M - r)^2) * (a^2 + r * (r - 2M)))
∂2Φ_∂rr(r::Float64, a::Float64, M::Float64) = 2.0a * (M - r) * (-1.0 / ((a^2 + (M - r)^2)^2) + 1.0 / ((a^2 + r * (r - 2M))^2))

# transforms a set of BL coordinates to harmonic coordinates where x = xBL = [r, θ, ϕ]
function xBLtoH(xBL::Vector{Float64}, a::Float64, M::Float64)
    xh = sqrt((xBL[1] - M)^2 + a^2) * sin(xBL[2]) * cos(xBL[3] - Φ(xBL[1], a, M))   # Eq. 68
    yh = sqrt((xBL[1] - M)^2 + a^2) * sin(xBL[2]) * sin(xBL[3] - Φ(xBL[1], a, M))   # Eq. 69
    zh = (xBL[1] - M) * cos(xBL[2])    # Eq. 70.0
    return [xh, yh, zh]  
end

# transforms a set of harmonic coordinates to BL where x = rH= [x, y, z]
function xHtoBL(xH::Vector{Float64}, a::Float64, M::Float64)
    rH = norm_3d(xH)   # Eq. 74
    rBL = M + sqrt((rH^2 - a^2 + sqrt((rH^2 - a^2)^2 + 4.0 * (a^2) * (xH[3]^2))) / 2.0)    # Eq. 72
    θ = acos(xH[3] / (rBL - M))    # Eq. 73
    ϕ = Φ(rBL, a, M) + atan(xH[2], xH[1])  # Eq. 71
    return [rBL, θ, ϕ]
end

ρ(xH::Vector{Float64}, a::Float64, M::Float64) = sqrt((norm2_3d(xH) - a^2 + sqrt((norm2_3d(xH) - a^2)^2 + 4.0 * (a * xH[3])^2)) / 2.0)   # Eq. B2

# Jacobian

# J^{BL}_{H}
∂r_∂xH(xH::Vector{Float64}, a::Float64, M::Float64) = ρ(xH, a, M) * xH[1] / (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH))
∂r_∂yH(xH::Vector{Float64}, a::Float64, M::Float64) = ρ(xH, a, M) * xH[2] / (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH))
∂r_∂zH(xH::Vector{Float64}, a::Float64, M::Float64) = (ρ(xH, a, M)^2 + a^2) * xH[3] / (ρ(xH, a, M) * (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH)))
∂r_∂rH(xH::Vector{Float64}, a::Float64, M::Float64) = [∂r_∂xH(xH, a, M), ∂r_∂yH(xH, a, M), ∂r_∂zH(xH, a, M)]


∂θ_∂xH(xH::Vector{Float64}, a::Float64, M::Float64) = xH[1] * xH[3] / (sqrt(ρ(xH, a, M)^2 - xH[3]^2) * (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH)))
∂θ_∂yH(xH::Vector{Float64}, a::Float64, M::Float64) = xH[2] * xH[3] / (sqrt(ρ(xH, a, M)^2 - xH[3]^2) * (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH)))
∂θ_∂zH(xH::Vector{Float64}, a::Float64, M::Float64) = (a^2.0 * (xH[3]^2 - ρ(xH, a, M)^2) + ρ(xH, a, M)^2.0 * (-2.0 * ρ(xH, a, M)^2 + norm2_3d(xH) + xH[3]^2)) / (ρ(xH, a, M)^2.0 * sqrt(ρ(xH, a, M)^2 - xH[3]^2) * (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH)))
∂θ_∂rH(xH::Vector{Float64}, a::Float64, M::Float64) = [∂θ_∂xH(xH, a, M), ∂θ_∂yH(xH, a, M), ∂θ_∂zH(xH, a, M)]

∂ϕ_∂xH(xH::Vector{Float64}, a::Float64, M::Float64) = ρ(xH, a, M) * xH[1] * ∂Φ_∂r(ρ(xH, a, M) + M, a, M) / (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH)) - xH[2] / (norm2_3d(xH) - xH[3]^2)
∂ϕ_∂yH(xH::Vector{Float64}, a::Float64, M::Float64) = ρ(xH, a, M) * xH[2] * ∂Φ_∂r(ρ(xH, a, M) + M, a, M) / (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH)) + xH[1] / (norm2_3d(xH) - xH[3]^2)
∂ϕ_∂zH(xH::Vector{Float64}, a::Float64, M::Float64) = xH[3] * (a^2 + ρ(xH, a, M)^2) * ∂Φ_∂r(ρ(xH, a, M) + M, a, M) / (ρ(xH, a, M) * (a^2 + 2.0 * ρ(xH, a, M)^2 - norm2_3d(xH)))
∂ϕ_∂rH(xH::Vector{Float64}, a::Float64, M::Float64) = [∂ϕ_∂xH(xH, a, M), ∂ϕ_∂yH(xH, a, M), ∂ϕ_∂zH(xH, a, M)]

jBLH(xH::Vector{Float64}, a::Float64, M::Float64) = [∂r_∂xH(xH, a, M) ∂r_∂yH(xH, a, M) ∂r_∂zH(xH, a, M); ∂θ_∂xH(xH, a, M) ∂θ_∂yH(xH, a, M) ∂θ_∂zH(xH, a, M); ∂ϕ_∂xH(xH, a, M) ∂ϕ_∂yH(xH, a, M) ∂ϕ_∂zH(xH, a, M)]

∂rBL_∂xH(xH::Vector{Float64}, a::Float64, M::Float64) = [∂r_∂xH(xH, a, M), ∂θ_∂xH(xH, a, M), ∂ϕ_∂xH(xH, a, M)]
∂rBL_∂yH(xH::Vector{Float64}, a::Float64, M::Float64) = [∂r_∂yH(xH, a, M), ∂θ_∂yH(xH, a, M), ∂ϕ_∂yH(xH, a, M)]
∂rBL_∂zH(xH::Vector{Float64}, a::Float64, M::Float64) = [∂r_∂zH(xH, a, M), ∂θ_∂zH(xH, a, M), ∂ϕ_∂zH(xH, a, M)]


# J^{H}_{BL}
∂xH_∂r(xH::Vector{Float64}, a::Float64, M::Float64) = xH[1] * ρ(xH, a, M) / (a^2 + ρ(xH, a, M)^2) + xH[2] * ∂Φ_∂r(ρ(xH, a, M) + M, a, M)
∂xH_∂θ(xH::Vector{Float64}, a::Float64, M::Float64) = xH[1] * xH[3] / sqrt(ρ(xH, a, M)^2 - xH[3]^2)
∂xH_∂ϕ(xH::Vector{Float64}, a::Float64, M::Float64) = -xH[2]
∂xH_∂xBL(xH::Vector{Float64}, a::Float64, M::Float64) = [∂xH_∂r(xH, a, M), ∂xH_∂θ(xH, a, M), ∂xH_∂ϕ(xH, a, M)]


∂yH_∂r(xH::Vector{Float64}, a::Float64, M::Float64) = xH[2] * ρ(xH, a, M) / (a^2 + ρ(xH, a, M)^2) - xH[1] * ∂Φ_∂r(ρ(xH, a, M) + M, a, M)
∂yH_∂θ(xH::Vector{Float64}, a::Float64, M::Float64) = xH[2] * xH[3] / sqrt(ρ(xH, a, M)^2 - xH[3]^2)
∂yH_∂ϕ(xH::Vector{Float64}, a::Float64, M::Float64) = xH[1] 
∂yH_∂xBL(xH::Vector{Float64}, a::Float64, M::Float64) = [∂yH_∂r(xH, a, M), ∂yH_∂θ(xH, a, M), ∂yH_∂ϕ(xH, a, M)]

∂zH_∂r(xH::Vector{Float64}, a::Float64, M::Float64) = xH[3] / ρ(xH, a, M)
∂zH_∂θ(xH::Vector{Float64}, a::Float64, M::Float64) = -sqrt(ρ(xH, a, M)^2 - xH[3]^2)
∂zH_∂ϕ(xH::Vector{Float64}, a::Float64, M::Float64) = 0.0
∂zH_∂xBL(xH::Vector{Float64}, a::Float64, M::Float64) = [∂zH_∂r(xH, a, M), ∂zH_∂θ(xH, a, M), ∂zH_∂ϕ(xH, a, M)]

∂rH_∂r(xH::Vector{Float64}, a::Float64, M::Float64) = [∂xH_∂r(xH, a, M), ∂yH_∂r(xH, a, M), ∂zH_∂r(xH, a, M)]
∂rH_∂θ(xH::Vector{Float64}, a::Float64, M::Float64) = [∂xH_∂θ(xH, a, M), ∂yH_∂θ(xH, a, M), ∂zH_∂θ(xH, a, M)]
∂rH_∂ϕ(xH::Vector{Float64}, a::Float64, M::Float64) = [∂xH_∂ϕ(xH, a, M), ∂yH_∂ϕ(xH, a, M), ∂zH_∂ϕ(xH, a, M)]


jHBL(xH::Vector{Float64}, a::Float64, M::Float64) = [∂xH_∂r(xH, a, M) ∂xH_∂θ(xH, a, M) ∂xH_∂ϕ(xH, a, M); ∂yH_∂r(xH, a, M) ∂yH_∂θ(xH, a, M) ∂yH_∂ϕ(xH, a, M); ∂zH_∂r(xH, a, M) ∂zH_∂θ(xH, a, M) ∂zH_∂ϕ(xH, a, M)]



# Hessians
# BL -> H

## rH = norm_3d(xH)
∂r_∂ij(xH::Vector{Float64}, rH::Float64, a::Float64, M::Float64) = [((a - rH)*(a + rH)*(a^2 - rH^2 + 3*xH[1]^2)*ρ(xH, a, M) + 2*(2*a^2 - 2*rH^2 + xH[1]^2)*ρ(xH, a, M)^3 + 4*ρ(xH, a, M)^5)/(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3 (xH[1]*xH[2]*ρ(xH, a, M)*(3*a^2 - 3*rH^2 + 2*ρ(xH, a, M)^2))/(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3 (xH[1]*xH[3]*(a^4 - 3*rH^2*ρ(xH, a, M)^2 + 2*ρ(xH, a, M)^4 + a^2*(-rH^2 + ρ(xH, a, M)^2)))/(ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3);
(xH[1]*xH[2]*ρ(xH, a, M)*(3*a^2 - 3*rH^2 + 2*ρ(xH, a, M)^2))/(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3 ((a - rH)*(a + rH)*(a^2 - rH^2 + 3*xH[2]^2)*ρ(xH, a, M) + 2*(2*a^2 - 2*rH^2 + xH[2]^2)*ρ(xH, a, M)^3 + 4*ρ(xH, a, M)^5)/(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3 (xH[2]*xH[3]*(a^4 - 3*rH^2*ρ(xH, a, M)^2 + 2*ρ(xH, a, M)^4 + a^2*(-rH^2 + ρ(xH, a, M)^2)))/(ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3);
(xH[1]*xH[3]*(a^4 - 3*rH^2*ρ(xH, a, M)^2 + 2*ρ(xH, a, M)^4 + a^2*(-rH^2 + ρ(xH, a, M)^2)))/(ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3) (xH[2]*xH[3]*(a^4 - 3*rH^2*ρ(xH, a, M)^2 + 2*ρ(xH, a, M)^4 + a^2*(-rH^2 + ρ(xH, a, M)^2)))/(ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3) ((a^2 + ρ(xH, a, M)^2)*(a^2*(-a^2 + rH^2)*xH[3]^2 + ((a^2 - rH^2)^2 - 3*(a^2 + rH^2)*xH[3]^2)*ρ(xH, a, M)^2 + 2*(2*a^2 - 2*rH^2 + xH[3]^2)*ρ(xH, a, M)^4 + 4*ρ(xH, a, M)^6))/ (ρ(xH, a, M)^3*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3)]

∂θ_∂ij(xH::Vector{Float64}, rH::Float64, a::Float64, M::Float64) = [(xH[3]*((a - rH)*(a + rH)*(a^2 - rH^2 + 2*xH[1]^2)*xH[3]^2 - (a - rH)*(a + rH)*(a^2 - rH^2 + xH[1]^2 - 4*xH[3]^2)*ρ(xH, a, M)^2 + 2*(-2*a^2 + 2*rH^2 + xH[1]^2 + 2*xH[3]^2)*ρ(xH, a, M)^4 - 4*ρ(xH, a, M)^6))/((xH[3] - ρ(xH, a, M))*(xH[3] + ρ(xH, a, M))*sqrt(-xH[3]^2 + ρ(xH, a, M)^2)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3) -((xH[1]*xH[2]*xH[3]*(2*ρ(xH, a, M)^4 + a^2*(2*xH[3]^2 - ρ(xH, a, M)^2) + rH^2*(-2*xH[3]^2 + ρ(xH, a, M)^2)))/((-xH[3]^2 + ρ(xH, a, M)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3)) (xH[1]*(2*rH^2*xH[3]^4 + rH^2*(rH - xH[3])*(rH + xH[3])*ρ(xH, a, M)^2 - 2*(2*rH^2 + xH[3]^2)*ρ(xH, a, M)^4 + 4*ρ(xH, a, M)^6 + a^4*(-xH[3]^2 + ρ(xH, a, M)^2) + a^2*(2*xH[3]^4 - 5*xH[3]^2*ρ(xH, a, M)^2 + 4*ρ(xH, a, M)^4 + rH^2*(xH[3]^2 - 2*ρ(xH, a, M)^2))))/((-xH[3]^2 + ρ(xH, a, M)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3);
-((xH[1]*xH[2]*xH[3]*(2*ρ(xH, a, M)^4 + a^2*(2*xH[3]^2 - ρ(xH, a, M)^2) + rH^2*(-2*xH[3]^2 + ρ(xH, a, M)^2)))/((-xH[3]^2 + ρ(xH, a, M)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3)) (xH[3]*((a - rH)*(a + rH)*(a^2 - rH^2 + 2*xH[2]^2)*xH[3]^2 - (a - rH)*(a + rH)*(a^2 - rH^2 + xH[2]^2 - 4*xH[3]^2)*ρ(xH, a, M)^2 + 2*(-2*a^2 + 2*rH^2 + xH[2]^2 + 2*xH[3]^2)*ρ(xH, a, M)^4 - 4*ρ(xH, a, M)^6))/((xH[3] - ρ(xH, a, M))*(xH[3] + ρ(xH, a, M))*sqrt(-xH[3]^2 + ρ(xH, a, M)^2)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3) (xH[2]*(2*rH^2*xH[3]^4 + rH^2*(rH - xH[3])*(rH + xH[3])*ρ(xH, a, M)^2 - 2*(2*rH^2 + xH[3]^2)*ρ(xH, a, M)^4 + 4*ρ(xH, a, M)^6 + a^4*(-xH[3]^2 + ρ(xH, a, M)^2) + a^2*(2*xH[3]^4 - 5*xH[3]^2*ρ(xH, a, M)^2 + 4*ρ(xH, a, M)^4 + rH^2*(xH[3]^2 - 2*ρ(xH, a, M)^2))))/((-xH[3]^2 + ρ(xH, a, M)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3);
(xH[1]*(2*rH^2*xH[3]^4 + rH^2*(rH - xH[3])*(rH + xH[3])*ρ(xH, a, M)^2 - 2*(2*rH^2 + xH[3]^2)*ρ(xH, a, M)^4 + 4*ρ(xH, a, M)^6 + a^4*(-xH[3]^2 + ρ(xH, a, M)^2) + a^2*(2*xH[3]^4 - 5*xH[3]^2*ρ(xH, a, M)^2 + 4*ρ(xH, a, M)^4 + rH^2*(xH[3]^2 - 2*ρ(xH, a, M)^2))))/((-xH[3]^2 + ρ(xH, a, M)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3) (xH[2]*(2*rH^2*xH[3]^4 + rH^2*(rH - xH[3])*(rH + xH[3])*ρ(xH, a, M)^2 - 2*(2*rH^2 + xH[3]^2)*ρ(xH, a, M)^4 + 4*ρ(xH, a, M)^6 + a^4*(-xH[3]^2 + ρ(xH, a, M)^2) + a^2*(2*xH[3]^4 - 5*xH[3]^2*ρ(xH, a, M)^2 + 4*ρ(xH, a, M)^4 + rH^2*(xH[3]^2 - 2*ρ(xH, a, M)^2))))/((-xH[3]^2 + ρ(xH, a, M)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3) (xH[3]*(2*a^6*(xH[3]^2 - ρ(xH, a, M)^2)^2 + a^4*(xH[3] - ρ(xH, a, M))*(xH[3] + ρ(xH, a, M))*(8*xH[3]^2*ρ(xH, a, M)^2 - 9*ρ(xH, a, M)^4 + rH^2*(-2*xH[3]^2 + 3*ρ(xH, a, M)^2)) + ρ(xH, a, M)^4*(rH^6 - 6*xH[3]^2*ρ(xH, a, M)^4 + 4*ρ(xH, a, M)^6 - rH^4*(xH[3]^2 + 3*ρ(xH, a, M)^2) + rH^2*(2*xH[3]^4 + 3*xH[3]^2*ρ(xH, a, M)^2)) + a^2*(-(rH^4*xH[3]^2*ρ(xH, a, M)^2) + 6*xH[3]^4*ρ(xH, a, M)^4 - 19*xH[3]^2*ρ(xH, a, M)^6 + 12*ρ(xH, a, M)^8 + rH^2*(8*xH[3]^2*ρ(xH, a, M)^4 - 6*ρ(xH, a, M)^6))))/(ρ(xH, a, M)^4*(-xH[3]^2 + ρ(xH, a, M)^2)^1.5*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3)]

∂ϕ_∂ij(xH::Vector{Float64}, rH::Float64, a::Float64, M::Float64) = [(2*xH[1]*xH[2])/(xH[1]^2 + xH[2]^2)^2 + (ρ(xH, a, M)*(((a - rH)*(a + rH)*(a^2 - rH^2 + 3*xH[1]^2) + 2*(2*a^2 - 2*rH^2 + xH[1]^2)*ρ(xH, a, M)^2 + 4*ρ(xH, a, M)^4)*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + xH[1]^2*ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3 (-xH[1]^2 + xH[2]^2)/(xH[1]^2 + xH[2]^2)^2 + (xH[1]*xH[2]*ρ(xH, a, M)*((3*a^2 - 3*rH^2 + 2*ρ(xH, a, M)^2)*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3 (xH[1]*xH[3]*((a^4 - 3*rH^2*ρ(xH, a, M)^2 + 2*ρ(xH, a, M)^4 + a^2*(-rH^2 + ρ(xH, a, M)^2))*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + ρ(xH, a, M)*(a^2 + ρ(xH, a, M)^2)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3);
(-xH[1]^2 + xH[2]^2)/(xH[1]^2 + xH[2]^2)^2 + (xH[1]*xH[2]*ρ(xH, a, M)*((3*a^2 - 3*rH^2 + 2*ρ(xH, a, M)^2)*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3 (-2*xH[1]*xH[2])/(xH[1]^2 + xH[2]^2)^2 + (ρ(xH, a, M)*(((a - rH)*(a + rH)*(a^2 - rH^2 + 3*xH[2]^2) + 2*(2*a^2 - 2*rH^2 + xH[2]^2)*ρ(xH, a, M)^2 + 4*ρ(xH, a, M)^4)*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + xH[2]^2*ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3 (xH[2]*xH[3]*((a^4 - 3*rH^2*ρ(xH, a, M)^2 + 2*ρ(xH, a, M)^4 + a^2*(-rH^2 + ρ(xH, a, M)^2))*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + ρ(xH, a, M)*(a^2 + ρ(xH, a, M)^2)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3);
(xH[1]*xH[3]*((a^4 - 3*rH^2*ρ(xH, a, M)^2 + 2*ρ(xH, a, M)^4 + a^2*(-rH^2 + ρ(xH, a, M)^2))*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + ρ(xH, a, M)*(a^2 + ρ(xH, a, M)^2)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3) (xH[2]*xH[3]*((a^4 - 3*rH^2*ρ(xH, a, M)^2 + 2*ρ(xH, a, M)^4 + a^2*(-rH^2 + ρ(xH, a, M)^2))*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + ρ(xH, a, M)*(a^2 + ρ(xH, a, M)^2)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(ρ(xH, a, M)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3) ((a^2 + ρ(xH, a, M)^2)*((a^2*(-a^2 + rH^2)*xH[3]^2 + ((a^2 - rH^2)^2 - 3*(a^2 + rH^2)*xH[3]^2)*ρ(xH, a, M)^2 + 2*(2*a^2 - 2*rH^2 + xH[3]^2)*ρ(xH, a, M)^4 + 4*ρ(xH, a, M)^6)*∂Φ_∂r(ρ(xH, a, M) + M, a, M) + xH[3]^2*ρ(xH, a, M)*(a^2 + ρ(xH, a, M)^2)*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M)))/(ρ(xH, a, M)^3*(a^2 - rH^2 + 2*ρ(xH, a, M)^2)^3)]

# outputs 2D array where [j, k] = ∂²xⁱ/∂xʲ∂xᵏ
HessBLH(xH::Vector{Float64}, rH::Float64, a::Float64, M::Float64, i::Int) = i==1 ? ∂r_∂ij(xH, rH, a, M) : i==2 ? ∂θ_∂ij(xH, rH, a, M) : i==3 ? ∂ϕ_∂ij(xH, rH, a, M) : throw(DomainError(i, "i should be in the range 1 ≤ i ≤ 3"))

# H -> BL
∂x_∂ij(xH::Vector{Float64}, a::Float64, M::Float64) = [(a^2*xH[1])/(a^2 + (-ρ(xH, a, M))^2)^2 + (2*(ρ(xH, a, M))*xH[2]*∂Φ_∂r(ρ(xH, a, M) + M, a, M))/(a^2 + (-ρ(xH, a, M))^2) - xH[1]*∂Φ_∂r(ρ(xH, a, M) + M, a, M)^2 + xH[2]*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M) ((ρ(xH, a, M))*xH[1]*xH[3] + (a^2 + (-ρ(xH, a, M))^2)*xH[2]*xH[3]*∂Φ_∂r(ρ(xH, a, M) + M, a, M))/((a^2 + (-ρ(xH, a, M))^2)*sqrt((-ρ(xH, a, M))^2 - xH[3]^2)) ((-ρ(xH, a, M))*xH[2])/(a^2 + (-ρ(xH, a, M))^2) + xH[1]*∂Φ_∂r(ρ(xH, a, M) + M, a, M);
((ρ(xH, a, M))*xH[1]*xH[3] + (a^2 + (-ρ(xH, a, M))^2)*xH[2]*xH[3]*∂Φ_∂r(ρ(xH, a, M) + M, a, M))/((a^2 + (-ρ(xH, a, M))^2)*sqrt((-ρ(xH, a, M))^2 - xH[3]^2)) -xH[1] -((xH[2]*xH[3])/sqrt((-ρ(xH, a, M))^2 - xH[3]^2));
((-ρ(xH, a, M))*xH[2])/(a^2 + (-ρ(xH, a, M))^2) + xH[1]*∂Φ_∂r(ρ(xH, a, M) + M, a, M) -((xH[2]*xH[3])/sqrt((-ρ(xH, a, M))^2 - xH[3]^2)) -xH[1]]

∂y_∂ij(xH::Vector{Float64}, a::Float64, M::Float64) = [(a^2*xH[2])/(a^2 + (-ρ(xH, a, M))^2)^2 + (2*(-ρ(xH, a, M))*xH[1]*∂Φ_∂r(ρ(xH, a, M) + M, a, M))/(a^2 + (-ρ(xH, a, M))^2) - xH[2]*∂Φ_∂r(ρ(xH, a, M) + M, a, M)^2 - xH[1]*∂2Φ_∂rr(ρ(xH, a, M) + M, a, M) -(((-ρ(xH, a, M))*xH[2]*xH[3] + (a^2 + (-ρ(xH, a, M))^2)*xH[1]*xH[3]*∂Φ_∂r(ρ(xH, a, M) + M, a, M))/((a^2 + (-ρ(xH, a, M))^2)*sqrt((-ρ(xH, a, M))^2 - xH[3]^2))) ((ρ(xH, a, M))*xH[1])/(a^2 + (-ρ(xH, a, M))^2) + xH[2]*∂Φ_∂r(ρ(xH, a, M) + M, a, M);
-(((-ρ(xH, a, M))*xH[2]*xH[3] + (a^2 + (-ρ(xH, a, M))^2)*xH[1]*xH[3]*∂Φ_∂r(ρ(xH, a, M) + M, a, M))/((a^2 + (-ρ(xH, a, M))^2)*sqrt((-ρ(xH, a, M))^2 - xH[3]^2))) -xH[2] (xH[1]*xH[3])/sqrt((-ρ(xH, a, M))^2 - xH[3]^2);
((ρ(xH, a, M))*xH[1])/(a^2 + (-ρ(xH, a, M))^2) + xH[2]*∂Φ_∂r(ρ(xH, a, M) + M, a, M) (xH[1]*xH[3])/sqrt((-ρ(xH, a, M))^2 - xH[3]^2) -xH[2]]

∂z_∂ij(xH::Vector{Float64}, a::Float64, M::Float64) = [0.0 -sqrt(1.0 - xH[3]^2/(-ρ(xH, a, M))^2) 0.0;
-sqrt(1.0 - xH[3]^2/(-ρ(xH, a, M))^2) -xH[3] 0.0;
0.0 0.0 0.0]

# outputs 2D array where [j, k] = ∂²xⁱ/∂xʲ∂xᵏ
HessHBL(xH::Vector{Float64}, rH::Float64, a::Float64, M::Float64, i::Int) = i==1 ? ∂x_∂ij(xH, a, M) : i==2 ? ∂y_∂ij(xH, a, M) : i==3 ? ∂z_∂ij(xH, a, M) : throw(DomainError(i, "i should be in the range 1 ≤ i ≤ 3"))


## TO-DO FIND BETTER NOTATION HERE SINCE THESE ARE PRODUCTS OF FIRST DERIVATIVES, NOT SECOND DERIVATIVES
∂ᵢr∂ⱼr(xH::Vector{Float64}, a::Float64, M::Float64) = otimes(∂r_∂rH(xH, a, M))
∂ᵢθ∂ⱼθ(xH::Vector{Float64}, a::Float64, M::Float64) = otimes(∂θ_∂rH(xH, a, M))
∂ᵢϕ∂ⱼϕ(xH::Vector{Float64}, a::Float64, M::Float64) = otimes(∂ϕ_∂rH(xH, a, M))

∂ij_∂r(xH::Vector{Float64}, a::Float64, M::Float64) = otimes(∂rH_∂r(xH, a, M))
∂ij_∂θ(xH::Vector{Float64}, a::Float64, M::Float64) = otimes(∂rH_∂θ(xH, a, M))
∂ij_∂ϕ(xH::Vector{Float64}, a::Float64, M::Float64) = otimes(∂rH_∂ϕ(xH, a, M))

# covariant metric in harmonic coordinates: r denotes spacial indices
g_tt_H(xH::Vector{Float64}, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_tt(0., xHtoBL(xH, a, M)..., a, M)
g_tr_H(xH::Vector{Float64}, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_tϕ(0., xHtoBL(xH, a, M)..., a, M) * ∂ϕ_∂rH(xH, a, M)
g_rr_H(xH::Vector{Float64}, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = g_rr(0., xHtoBL(xH, a, M)..., a, M) * (∂ᵢr∂ⱼr(xH, a, M)) + g_θθ(0., xHtoBL(xH, a, M)..., a, M) * (∂ᵢθ∂ⱼθ(xH, a, M)) + g_ϕϕ(0., xHtoBL(xH, a, M)..., a, M) * (∂ᵢϕ∂ⱼϕ(xH, a, M))   # Eq. B18
g_μν_H(xH::Vector{Float64}, a::Float64, M::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function) = hcat([g_tt_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ), g_tr_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)...], vcat(transpose(g_tr_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)), g_rr_H(xH, a, M, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ)))

# contravariant (note that we need to therefore input the contravariant metric component)
gTT_H(xH::Vector{Float64}, a::Float64, M::Float64, g_TT::Function, g_TΦ::Function, g_RR::Function, g_ThTh::Function, g_ΦΦ::Function) = g_TT(0., xHtoBL(xH, a, M)..., a, M)
gTR_H(xH::Vector{Float64}, a::Float64, M::Float64, g_TT::Function, g_TΦ::Function, g_RR::Function, g_ThTh::Function, g_ΦΦ::Function) = g_TΦ(0., xHtoBL(xH, a, M)..., a, M) * ∂rH_∂ϕ(xH, a, M)
gRR_H(xH::Vector{Float64}, a::Float64, M::Float64, g_TT::Function, g_TΦ::Function, g_RR::Function, g_ThTh::Function, g_ΦΦ::Function) = g_RR(0., xHtoBL(xH, a, M)..., a, M) * (∂ij_∂r(xH, a, M)) + g_ThTh(0., xHtoBL(xH, a, M)..., a, M) * (∂ij_∂θ(xH, a, M)) + g_ΦΦ(0., xHtoBL(xH, a, M)..., a, M) * (∂ij_∂ϕ(xH, a, M))    # Eq. B18
gμν_H(xH::Vector{Float64}, a::Float64, M::Float64, g_TT::Function, g_TΦ::Function, g_RR::Function, g_ThTh::Function, g_ΦΦ::Function) = hcat([gTT_H(xH, a, M, g_TT, g_TΦ, g_RR, g_ThTh, g_ΦΦ), gTR_H(xH, a, M, g_TT, g_TΦ, g_RR, g_ThTh, g_ΦΦ)...], vcat(transpose(gTR_H(xH, a, M, g_TT, g_TΦ, g_RR, g_ThTh, g_ΦΦ)), gRR_H(xH, a, M, g_TT, g_TΦ, g_RR, g_ThTh, g_ΦΦ)))

# transfrom velocities and accelerations to harmonic coordinates
vBLtoH(xH::Vector{Float64}, vBL::Vector{Float64}, a::Float64, M::Float64) = ∂rH_∂r(xH, a, M) * vBL[1] .+ ∂rH_∂θ(xH, a, M) * vBL[2] .+ ∂rH_∂ϕ(xH, a, M) * vBL[3]   # Eq. 78
vHtoBL(xH::Vector{Float64}, vH::Vector{Float64}, a::Float64, M::Float64) = ∂rBL_∂xH(xH, a, M) * vH[1] .+ ∂rBL_∂yH(xH, a, M) * vH[2] .+ ∂rBL_∂zH(xH, a, M) * vH[3]   # Eq. 78


function aBLtoH(xH::Vector{Float64}, vBL::Vector{Float64}, aBL::Vector{Float64}, a::Float64, M::Float64)   # Eq. 79
    HessXHtoBL = HessHBL(xH, norm_3d(xH), a, M, 1) 
    HessYHtoBL = HessHBL(xH, norm_3d(xH), a, M, 2) 
    HessZHtoBL = HessHBL(xH, norm_3d(xH), a, M, 3) 
    return ∂rH_∂r(xH, a, M) * aBL[1] + ∂rH_∂θ(xH, a, M) * aBL[2] + ∂rH_∂ϕ(xH, a, M) * aBL[3] + [HessXHtoBL[1, 1],  HessYHtoBL[1, 1], HessZHtoBL[1, 1]] * vBL[1]^2 + [HessXHtoBL[2, 2],  HessYHtoBL[2, 2], HessZHtoBL[2, 2]] * vBL[2]^2 + [HessXHtoBL[3, 3],  HessYHtoBL[3, 3], HessZHtoBL[3, 3]] * vBL[3]^2 + 2.0 * [HessXHtoBL[1, 2],  HessYHtoBL[1, 2], HessZHtoBL[1, 2]] * vBL[1] * vBL[2] + 2.0 * [HessXHtoBL[1, 3],  HessYHtoBL[1, 3], HessZHtoBL[1, 3]] * vBL[1] * vBL[3] + 2.0 * [HessXHtoBL[2, 3],  HessYHtoBL[2, 3], HessZHtoBL[2, 3]] * vBL[2] * vBL[3]
end

function aHtoBL(xH::Vector{Float64}, vH::Vector{Float64}, aH::Vector{Float64}, a::Float64, M::Float64)   # Eq. 79
    Hess_rBLtoH = HessBLH(xH, norm_3d(xH), a, M, 1) 
    Hess_θBLtoH = HessBLH(xH, norm_3d(xH), a, M, 2) 
    Hess_ϕBLtoH = HessBLH(xH, norm_3d(xH), a, M, 3) 
    return ∂rBL_∂xH(xH, a, M) * aH[1] + ∂rBL_∂yH(xH, a, M) * aH[2] + ∂rBL_∂zH(xH, a, M) * aH[3] + [Hess_rBLtoH[1, 1],  Hess_θBLtoH[1, 1], Hess_ϕBLtoH[1, 1]] * vH[1]^2 + [Hess_rBLtoH[2, 2],  Hess_θBLtoH[2, 2], Hess_ϕBLtoH[2, 2]] * vH[2]^2 + [Hess_rBLtoH[3, 3],  Hess_θBLtoH[3, 3], Hess_ϕBLtoH[3, 3]] * vH[3]^2 + 2.0 * [Hess_rBLtoH[1, 2],  Hess_θBLtoH[1, 2], Hess_ϕBLtoH[1, 2]] * vH[1] * vH[2] + 2.0 * [Hess_rBLtoH[1, 3],  Hess_θBLtoH[1, 3], Hess_ϕBLtoH[1, 3]] * vH[1] * vH[3] + 2.0 * [Hess_rBLtoH[2, 3],  Hess_θBLtoH[2, 3], Hess_ϕBLtoH[2, 3]] * vH[2] * vH[3]
end

end