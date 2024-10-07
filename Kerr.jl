module Kerr

"""
# Common Arguments in this module
- `r::Float64`: Boyer-Lindquist radial coordinate.
- `θ::Float64`: Boyer-Lindquist polar coordinate.
- `ϕ::Float64`: Boyer-Lindquist azimuthal coordinate.
- `a::Float64`: Kerr black hole spin parameter, 0 < a < 1.
- `p::Float64`: semi-latus rectum of the orbit (defined by, e.g., Eq. 23).
- `e::Float64`: eccentricity of the orbit (defined by, e.g., Eq. 23).
- `θmin::Float64`: minimum polar angle of the orbit.
- `E::Float64`: energy per unit mass of the test particle moving along the geodesic (Eq. 14).
- `L::Float64`: axial (i.e., z-component of the) angular momentum per unit mass of the test particle moving along the geodesic (Eq. 15).
- `C::Float64`: Carter constant of the orbit---note that this C is what is commonly referred to as 'Q' elsewhere (Eq. 17).
- `Q::Float64`: Alternative definition of the Carter constant (Eq. 16).
- `ra::Float64`: apastron of the orbit (furtherst radial turning point, Eq. 22).
- `rp::Float64`: periastron of the orbit (closest radial turning point, Eq. 22).
"""

module KerrMetric
using LinearAlgebra
using StaticArrays

# define inner/outer horizons
rplus(a::Float64)::Float64 = 1.0 + sqrt(1.0 - a^2)
rminus(a::Float64)::Float64 = 1.0 - sqrt(1.0 - a^2)

# covariant metric components
Δ(r::Float64, a::Float64)::Float64 = r^2 - 2.0 * r + a^2
Σ(r::Float64, θ::Float64, a::Float64)::Float64 = r^2 + (a * cos(θ))^2
@inline g_tt(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = - (1.0 - 2.0 * r / Σ(r, θ, a))
@inline g_tϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -2.0 * a * r * (sin(θ)^2) / Σ(r, θ, a)
@inline g_rr(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = Σ(r, θ, a) / Δ(r, a)
@inline g_θθ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = Σ(r, θ, a)
@inline g_ϕϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = (r^2 + a^2 + 2.0 * (a^2) * r * (sin(θ)^2) / Σ(r, θ, a)) * sin(θ)^2
g(r::Float64, θ::Float64, ϕ::Float64, a::Float64) = @SMatrix [g_tt(r, θ, ϕ, a) 0.0 0.0 g_tϕ(r, θ, ϕ, a); 0.0 g_rr(r, θ, ϕ, a) 0.0 0.0; 0.0 0.0 g_θθ(r, θ, ϕ, a) 0.0; g_tϕ(r, θ, ϕ, a) 0.0 0.0 g_ϕϕ(r, θ, ϕ, a)]
g_μν(r::Float64, θ::Float64, ϕ::Float64, a::Float64, μ::Int, ν::Int)::Float64 = (μ==1) && (ν==1) ? g_tt(r, θ, ϕ, a) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? g_tϕ(r, θ, ϕ, a) : μ==2 && ν==2 ? g_rr(r, θ, ϕ, a) : μ==3 && ν==3 ? g_θθ(r, θ, ϕ, a) : μ==4 & ν==4 ? g_ϕϕ(r, θ, ϕ, a) : 0.0

# contravariant components
@inline gTT(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = (-1.0 + 2.0 * r * (a^2 + r^2) / ((a^2 + r^2) * Σ(r, θ, a) + 2.0a^2 * r * sin(θ)^2))^(-1.0)
@inline gTΦ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -2.0a * r / (2.0a^2 * r * sin(θ)^2 - (a^2 + r^2) * (2.0 * r - Σ(r, θ, a)))
@inline gRR(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = Δ(r, a) / Σ(r, θ, a)
@inline gThTh(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = 1.0 / Σ(r, θ, a)
@inline gΦΦ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = (2.0 * r - Σ(r, θ, a)) / (sin(θ)^2 * (a^2 * r * cos(2.0θ) + r * (a^2 + 2.0r^2) - (a^2 + r^2) * Σ(r, θ, a)))
ginv(r::Float64, θ::Float64, ϕ::Float64, a::Float64) = @SMatrix [gTT(r, θ, ϕ, a) 0.0 0.0 gTΦ(r, θ, ϕ, a); 0.0 gRR(r, θ, ϕ, a) 0.0 0.0; 0.0 0.0 gThTh(r, θ, ϕ, a) 0.0; gTΦ(r, θ, ϕ, a) 0.0 0.0 gΦΦ(r, θ, ϕ, a)]
ginvμν(r::Float64, θ::Float64, ϕ::Float64, a::Float64, μ::Int, ν::Int)::Float64 = (μ==1) && (ν==1) ? gTT(r, θ, ϕ, a) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? gTΦ(r, θ, ϕ, a) : μ==2 && ν==2 ? gRR(r, θ, ϕ, a) : μ==3 && ν==3 ? gThTh(r, θ, ϕ, a) : μ==4 & ν==4 ? gΦΦ(r, θ, ϕ, a) : 0.0

# two-rank killing tensor components - Eq. 12 of (arXiv:1109.0572v2)
@inline ξ_tt(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = ((-2.0*a^2*r*sin(θ)^2 + (a^2 + r^2)*(2.0*r - Σ(r,θ,a)))^2 + r^2*Δ(r,a)*(2.0*r - Σ(r,θ,a))*Σ(r,θ,a))/(Δ(r,a)*Σ(r,θ,a)^2)
@inline ξ_tϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = (a*sin(θ)^2*(-4*r^2*(a^2 + r^2)^2 + 4*a^2*r*sin(θ)^2* (-(a^2*r*sin(θ)^2) + (a^2 + r^2)*(2.0*r - Σ(r,θ,a))) + 2.0*r*(2.0*(a^2 + r^2)^2 - r^2*Δ(r,a))*Σ(r,θ,a) - (a^2 + r^2)^2*Σ(r,θ,a)^2))/(Δ(r,a)*Σ(r,θ,a)^2)
@inline ξ_rr(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = ((r^2 - Σ(r,θ,a))*Σ(r,θ,a))/Δ(r,a)
@inline ξ_θθ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = r^2*Σ(r,θ,a)
@inline ξ_ϕϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = (r^2*sin(θ)^2*Δ(r,a)*Σ(r,θ,a)* (2.0*a^2*r*sin(θ)^2 + (a^2 + r^2)*Σ(r,θ,a)) + sin(θ)^4*(a*r*(a^2 + 2.0*r^2) + a^3*r*cos(2.0*θ) - a*(a^2 + r^2)*Σ(r,θ,a))^2)/(Δ(r,a)*Σ(r,θ,a)^2)
@inline ξ(r::Float64, θ::Float64, ϕ::Float64, a::Float64) = @SMatrix [ξ_tt(r, θ, ϕ, a) 0.0 0.0 ξ_tϕ(r, θ, ϕ, a); 0.0 ξ_rr(r, θ, ϕ, a) 0.0 0.0; 0.0 0.0 ξ_θθ(r, θ, ϕ, a) 0.0; ξ_tϕ(r, θ, ϕ, a) 0.0 0.0 ξ_ϕϕ(r, θ, ϕ, a)]
@inline ξ_μν(r::Float64, θ::Float64, ϕ::Float64, a::Float64, μ::Int, ν::Int)::Float64 = (μ==1) && (ν==1) ? ξ_tt(r, θ, ϕ, a) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? ξ_tϕ(r, θ, ϕ, a) : μ==2 && ν==2 ? ξ_rr(r, θ, ϕ, a) : μ==3 && ν==3 ? ξ_θθ(r, θ, ϕ, a) : μ==4 & ν==4 ? ξ_ϕϕ(r, θ, ϕ, a) : 0.0

# Christoffel symbols
@inline Γttr(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -4.0 * (a^2 + r^2) * (-r^2 + (a^2 * cos(θ)^2)) / ((a^2 + r * (r - 2.0)) * (a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
@inline Γttθ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -4.0a^2 * r * sin(2.0θ) / ((a^2 + 2.0r^2 + a^2* cos(2.0θ))^2)
@inline Γtrϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = 4.0a *(-r^2 * (a^2 + 3r^2) + a^2 * (a - r) * (a + r) * cos(θ)^2) * sin(θ)^2 / ((a^2 + r * (r - 2.0)) * (a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
@inline Γtθϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = 8.0a^3 * r * cos(θ) * sin(θ)^3 / ((a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γtμν(r::Float64, θ::Float64, ϕ::Float64, a::Float64, μ::Int, ν::Int)::Float64 = ((μ==1) && (ν==2)) || ((μ==2) && (ν==1)) ? Γttr(r, θ, ϕ, a) : ((μ==1) && (ν==3)) || ((μ==3) && (ν==1)) ? Γttθ(r, θ, ϕ, a) : ((μ==2) && (ν==4)) || ((μ==4) && (ν==2)) ? Γtrϕ(r, θ, ϕ, a) : ((μ==3) && (ν==4)) || ((μ==4) && (ν==3)) ? Γtθϕ(r, θ, ϕ, a) : 0.0

@inline Γrtt(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -(a^2 + r * (r - 2.0)) * (-r^2 + a^2 * cos(θ)^2) / ((r^2 + a^2 * cos(θ)^2)^3)
@inline Γrtϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = a * (a^2 + r * (r - 2.0)) * (-r^2 + a^2 * cos(θ)^2) * sin(θ)^2 / ((r^2 + a^2 * cos(θ)^2)^3)
@inline Γrrr(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = ((1.0 - r) / (a^2 - 2.0 * r + r^2)) + (r / (r^2 + a^2 * cos(θ)^2))
@inline Γrrθ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 =  -a^2 * cos(θ) * sin(θ) / (r^2 + a^2 * cos(θ)^2)
@inline Γrθθ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -r * (a^2 + r * (r - 2.0)) / (r^2 + a^2 * cos(θ)^2)
@inline Γrϕϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = - (a^2 + r * (r - 2.0)) * sin(θ)^2 * (r * (r^2 + a^2 * cos(θ)^2)^2 + a^2 * (-r + a * cos(θ)) * (r + a * cos(θ)) * sin(θ)^2) / ((r^2 + a^2 * cos(θ)^2)^3)
Γrμν(r::Float64, θ::Float64, ϕ::Float64, a::Float64, μ::Int, ν::Int)::Float64 = (μ==1) && (ν==1)  ? Γrtt(r, θ, ϕ, a) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? Γrtϕ(r, θ, ϕ, a) : (μ==2) && (ν==2) ? Γrrr(r, θ, ϕ, a) : ((μ==2) && (ν==3)) || ((μ==3) && (ν==2)) ? Γrrθ(r, θ, ϕ, a) : (μ==3) && (ν==3) ? Γrθθ(r, θ, ϕ, a) : (μ==4) && (ν==4) ? Γrϕϕ(r, θ, ϕ, a) : 0.0

@inline Γθtt(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -2.0a^2 * r * cos(θ) * sin(θ) / ((r^2 + a^2 * cos(θ)^2)^3)
@inline Γθtϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = a * r * (a^2 + r^2) * sin(2.0θ) / ((r^2 + a^2 * cos(θ)^2)^3)
@inline Γθrr(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = a^2 * cos(θ) * sin(θ) / ((a^2 + r * (r - 2.0)) * (r^2 + a^2 * cos(θ)^2))
@inline Γθrθ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = r / (r^2 + a^2 * cos(θ)^2)
@inline Γθθθ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -a^2 * cos(θ) * sin(θ) / (r^2 + a^2 * cos(θ)^2)
@inline Γθϕϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -cos(θ) * sin(θ) * ((a^2 + r^2) * (r^2 + a^2 * cos(θ)^2)^2 + a^2 * r * (3a^2 + 4.0r^2 + a^2 * cos(2.0θ)) * sin(θ)^2) / ((r^2 + a^2 * cos(θ)^2)^3)
Γθμν(r::Float64, θ::Float64, ϕ::Float64, a::Float64, μ::Int, ν::Int)::Float64 = (μ==1) && (ν==1)  ? Γθtt(r, θ, ϕ, a) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? Γθtϕ(r, θ, ϕ, a) : (μ==2) && (ν==2) ? Γθrr(r, θ, ϕ, a) : ((μ==2) && (ν==3)) || ((μ==3) && (ν==2)) ? Γθrθ(r, θ, ϕ, a) : (μ==3) && (ν==3) ? Γθθθ(r, θ, ϕ, a) : (μ==4) && (ν==4) ? Γθϕϕ(r, θ, ϕ, a) : 0.0

@inline Γϕtr(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = 4.0a * (r^2 - a^2 * cos(θ)^2) / ((a^2 + r * (r - 2.0)) * (a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
@inline Γϕtθ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = -8.0a * r * cot(θ) / ((a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
@inline Γϕrϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = (4.0r * (r^2 + a^2 * cos(θ)^2) * (r * (r - 2.0) + a^2 * cos(θ)^2) + 4.0a^2 * (-r + a * cos(θ)) * (r + a * cos(θ)) * sin(θ)^2) / ((a^2 + r * (r - 2.0)) * (a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
@inline Γϕθϕ(r::Float64, θ::Float64, ϕ::Float64, a::Float64)::Float64 = cot(θ) + 4.0a^2 * r* sin(2.0θ) / ((a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γϕμν(r::Float64, θ::Float64, ϕ::Float64, a::Float64, μ::Int, ν::Int)::Float64 = ((μ==1) && (ν==2)) || ((μ==2) && (ν==1)) ? Γϕtr(r, θ, ϕ, a) : ((μ==1) && (ν==3)) || ((μ==3) && (ν==1)) ? Γϕtθ(r, θ, ϕ, a) : ((μ==2) && (ν==4)) || ((μ==4) && (ν==2)) ? Γϕrϕ(r, θ, ϕ, a) : ((μ==3) && (ν==4)) || ((μ==4) && (ν==3)) ? Γϕθϕ(r, θ, ϕ, a) : 0.0

Γαμν(r::Float64, θ::Float64, ϕ::Float64, a::Float64, α::Int, μ::Int, ν::Int)::Float64 = α==1 ? Γtμν(r, θ, ϕ, a, μ, ν) : α==2 ? Γrμν(r, θ, ϕ, a, μ, ν) : α==3 ? Γθμν(r, θ, ϕ, a, μ, ν) : α==4 ? Γϕμν(r, θ, ϕ, a, μ, ν) : throw(DomainError(α, "α should be in the range 1 ≤ α ≤ 4"))

end

# this module solves the second order geodesic equation. These functions are not being actively maintained, and are not used in the main code. They are kept here for potential future use.
module GeodesicEquation
import ..KerrMetric: Γtrϕ, Γttr, Γtrϕ, Γttr, Γttθ, Γtθϕ, Γrrr, Γrrθ, Γrtt, Γrtϕ, Γrθθ, Γrϕϕ, Γθrr, Γθrθ, Γθtt, Γθtϕ, Γθθθ, Γθϕϕ, Γϕrϕ, Γϕtr, Γϕtθ, Γϕθϕ
using ..Kerr
using ...ConstantsOfMotion
using DifferentialEquations
using HDF5
using StaticArrays

# expressions for dt/dτ and dϕ/dτ from Lagrangian
tdot(r::Float64, θ::Float64, ϕ::Float64, a::Float64, EE::Float64, LL::Float64, g_ϕϕ::Function, g_tϕ::Function, g_tt::Function) = (EE * g_ϕϕ(r, θ, ϕ, a) + LL * g_tϕ(r, θ, ϕ, a)) / (g_tϕ(r, θ, ϕ, a)^2 - g_tt(r, θ, ϕ, a) * g_ϕϕ(r, θ, ϕ, a))   # Eq. 5.9
ϕdot(r::Float64, θ::Float64, ϕ::Float64, a::Float64, EE::Float64, LL::Float64, g_ϕϕ::Function, g_tϕ::Function, g_tt::Function) = - (EE * g_tϕ(r, θ, ϕ, a) + LL * g_tt(r, θ, ϕ, a)) / (g_tϕ(r, θ, ϕ, a)^2 - g_tt(r, θ, ϕ, a) * g_ϕϕ(r, θ, ϕ, a))   # Eq. 5.10

# initial conditions for bound kerr orbits starting in equatorial plane
function boundKerr_ics(a::Float64, EEi::Float64, LLi::Float64, ri::Float64, θmin::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function)
    ti = 0.0
    ϕi = 0.0
    xi = @SArray [ti, ri, θmin, ϕi]
    uti = tdot(xi..., a, EEi, LLi, g_ϕϕ, g_tϕ, g_tt)
    uri = 0.0
    uϕi = ϕdot(xi..., a, EEi, LLi, g_ϕϕ, g_tϕ, g_tt)
    uθmin2 = (-1 - g_rr(xi..., a) * uri^2 - g_tt(xi..., a) * uti^2 - 2.0 * g_tϕ(xi..., a) * uti * uϕi - g_ϕϕ(xi..., a) * uϕi^2) / g_θθ(xi..., a)    # Eq. 5.11
    uθmin = abs(uθmin2) <= 1e-14 ? 0. : sqrt(uθmin2)   # replace solutions close to zero by zero exactly
    uxi = @SArray[uti, uri, uθmin, uϕi]
    return [uxi, xi]
end

# geodesic equations
tddot(r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64) = -2.0 * (rdot * (ϕdot * Γtrϕ(r, θ, ϕ, a) + tdot * Γttr(r, θ, ϕ, a)) + θdot * (tdot * Γttθ(r, θ, ϕ, a) + ϕdot * Γtθϕ(r, θ, ϕ, a)))
rddot(r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64) = -(rdot * (rdot * Γrrr(r, θ, ϕ, a) + 2.0 * θdot * Γrrθ(r, θ, ϕ, a)) + tdot * (tdot * Γrtt(r, θ, ϕ, a) + 2.0 * ϕdot * Γrtϕ(r, θ, ϕ, a)) + θdot^2 * Γrθθ(r, θ, ϕ, a) + ϕdot^2 * Γrϕϕ(r, θ, ϕ, a))
θddot(r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64) = -(rdot * (rdot * Γθrr(r, θ, ϕ, a) + 2.0 * θdot * Γθrθ(r, θ, ϕ, a)) + tdot * (tdot * Γθtt(r, θ, ϕ, a) + 2.0 * ϕdot * Γθtϕ(r, θ, ϕ, a)) + θdot^2 * Γθθθ(r, θ, ϕ, a) + ϕdot^2 * Γθϕϕ(r, θ, ϕ, a))
ϕddot(r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64) = -2.0 * (rdot * (ϕdot * Γϕrϕ(r, θ, ϕ, a) + tdot * Γϕtr(r, θ, ϕ, a)) + θdot * (tdot * Γϕtθ(r, θ, ϕ, a) + ϕdot * Γϕθϕ(r, θ, ϕ, a)))
xμddot(μ::Int, t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64) = μ==1 ? tddot(r, θ, ϕ, tdot, rdot, θdot, ϕdot, a) : μ==2 ? rddot(r, θ, ϕ, tdot, rdot, θdot, ϕdot, a) : μ==3 ? θddot(r, θ, ϕ, tdot, rdot, θdot, ϕdot, a) : ϕddot(r, θ, ϕ, tdot, rdot, θdot, ϕdot, a)

# equation for ODE solver
function geodesicEq(du, u, params, t)
    @SArray [tddot(u..., du..., params...), rddot(u..., du..., params...), θddot(u..., du..., params...), ϕddot(u..., du..., params...)]
end

# computes trajectory in Kerr characterized by a, p, e, θmin (M=1, μ=1)
function compute_kerr_geodesic(a::Float64, p::Float64, e::Float64, θmin::Float64, τmax::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-16, abstol::Float64=1e-16, saveat::Float64=0.5; data_path::String="Results/")
    # orbital parameters

    # define periastron and apastron
    rp = p / (1 + e);   # Eq. 6.1
    ra = p / (1 - e);   # Eq. 6.1

    # calculate integrals of motion from orbital parameters
    E, L, Q, C = ConstantsOfMotion.compute_ELC(a, p, e, θmin)   # dimensionless constants

    # initial conditions for Kerr geodesic trajectory
    ri = ra; τspan = (0.0, τmax); params = @SArray [a];
    τ = 0:saveat:τmax |> collect

    ics = GeodesicEquation.boundKerr_ics(a, E, L, ri, θmin, KerrMetric.g_tt,  KerrMetric.g_tϕ,  KerrMetric.g_rr, KerrMetric.g_θθ, KerrMetric.g_ϕϕ);
    prob = SecondOrderODEProblem(GeodesicEquation.geodesicEq, ics..., τspan, params);
    sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=τ);
 
    # deconstruct solution
    tdot = sol[1, :];
    rdot = sol[2, :];
    θdot = sol[3, :];
    ϕdot = sol[4, :];
    t = sol[5, :];
    r = sol[6, :];
    θ = sol[7, :];
    ϕ = sol[8, :];

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt τ)
    tddot = GeodesicEquation.tddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    rddot = GeodesicEquation.rddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    θddot = GeodesicEquation.θddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    ϕddot = GeodesicEquation.ϕddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);

    # save trajectory- rows are: τ, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([τ, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot]))
    mkpath(data_path)
    ODE_filename=data_path * "ODE_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_tstep_$(saveat)_T_$(τmax)_tol_$(reltol).txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end
    println("ODE saved to: " * ODE_filename)
end

end
end