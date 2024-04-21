module Kerr

module KerrMetric
using LinearAlgebra
using StaticArrays

# define inner/outer horizons
rplus(a::Float64, M::Float64) = M + sqrt(M^2 - a^2)
rminus(a::Float64, M::Float64) = M - sqrt(M^2 - a^2)

# covariant metric components
Δ(r::Float64, a::Float64, M::Float64) = r^2 - 2.0M * r + a^2
Σ(r::Float64, θ::Float64, a::Float64) = r^2 + (a * cos(θ))^2
g_tt(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = - (1.0 - 2.0M * r / Σ(r, θ, a))
g_tϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -2.0M * a * r * (sin(θ)^2) / Σ(r, θ, a)
g_rr(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = Σ(r, θ, a) / Δ(r, a, M)
g_θθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = Σ(r, θ, a)
g_ϕϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (r^2 + a^2 + 2M * (a^2) * r * (sin(θ)^2) / Σ(r, θ, a)) * sin(θ)^2
g(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = @SMatrix [g_tt(t, r, θ, ϕ, a, M) 0.0 0.0 g_tϕ(t, r, θ, ϕ, a, M); 0.0 g_rr(t, r, θ, ϕ, a, M) 0.0 0.0; 0.0 0.0 g_θθ(t, r, θ, ϕ, a, M) 0.0; g_tϕ(t, r, θ, ϕ, a, M) 0.0 0.0 g_ϕϕ(t, r, θ, ϕ, a, M)]
g_μν(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, μ::Int, ν::Int) = (μ==1) && (ν==1) ? g_tt(t, r, θ, ϕ, a, M) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? g_tϕ(t, r, θ, ϕ, a, M) : μ==2 && ν==2 ? g_rr(t, r, θ, ϕ, a, M) : μ==3 && ν==3 ? g_θθ(t, r, θ, ϕ, a, M) : μ==4 & ν==4 ? g_ϕϕ(t, r, θ, ϕ, a, M) : 0.0

# contravariant components
gTT(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (-1.0 + 2.0M * r * (a^2 + r^2) / ((a^2 + r^2) * Σ(r, θ, a) + 2.0a^2 * M * r * sin(θ)^2))^(-1.0)
gTΦ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -2.0a * M * r / (2.0a^2 * M * r * sin(θ)^2 - (a^2 + r^2) * (2.0M * r - Σ(r, θ, a)))
gRR(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = Δ(r, a, M) / Σ(r, θ, a)
gThTh(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = 1.0 / Σ(r, θ, a)
gΦΦ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (2.0M * r - Σ(r, θ, a)) / (sin(θ)^2 * (a^2 * M * r * cos(2.0θ) + M * r * (a^2 + 2.0r^2) - (a^2 + r^2) * Σ(r, θ, a)))
ginv(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = @SMatrix [gTT(t, r, θ, ϕ, a, M) 0.0 0.0 gTΦ(t, r, θ, ϕ, a, M); 0.0 gRR(t, r, θ, ϕ, a, M) 0.0 0.0; 0.0 0.0 gThTh(t, r, θ, ϕ, a, M) 0.0; gTΦ(t, r, θ, ϕ, a, M) 0.0 0.0 gΦΦ(t, r, θ, ϕ, a, M)]
ginvμν(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, μ::Int, ν::Int) = (μ==1) && (ν==1) ? gTT(t, r, θ, ϕ, a, M) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? gTΦ(t, r, θ, ϕ, a, M) : μ==2 && ν==2 ? gRR(t, r, θ, ϕ, a, M) : μ==3 && ν==3 ? gThTh(t, r, θ, ϕ, a, M) : μ==4 & ν==4 ? gΦΦ(t, r, θ, ϕ, a, M) : 0.0

# two-rank killing tensor components
ξ_tt(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = ((-2*a^2*M*r*sin(θ)^2 + (a^2 + r^2)*(2*M*r - Σ(r,θ,a)))^2 + r^2*Δ(r,a,M)*(2*M*r - Σ(r,θ,a))*Σ(r,θ,a))/(Δ(r,a,M)*Σ(r,θ,a)^2)
ξ_tϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (a*sin(θ)^2*(-4*M^2*r^2*(a^2 + r^2)^2 + 4*a^2*M*r*sin(θ)^2* (-(a^2*M*r*sin(θ)^2) + (a^2 + r^2)*(2*M*r - Σ(r,θ,a))) + 2*M*r*(2*(a^2 + r^2)^2 - r^2*Δ(r,a,M))*Σ(r,θ,a) - (a^2 + r^2)^2*Σ(r,θ,a)^2))/(Δ(r,a,M)*Σ(r,θ,a)^2)
ξ_rr(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = ((r^2 - Σ(r,θ,a))*Σ(r,θ,a))/Δ(r,a,M)
ξ_θθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = r^2*Σ(r,θ,a)
ξ_ϕϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (r^2*sin(θ)^2*Δ(r,a,M)*Σ(r,θ,a)* (2*a^2*M*r*sin(θ)^2 + (a^2 + r^2)*Σ(r,θ,a)) + sin(θ)^4*(a*M*r*(a^2 + 2*r^2) + a^3*M*r*cos(2*θ) - a*(a^2 + r^2)*Σ(r,θ,a))^2)/(Δ(r,a,M)*Σ(r,θ,a)^2)
ξ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = @SMatrix [ξ_tt(t, r, θ, ϕ, a, M) 0.0 0.0 ξ_tϕ(t, r, θ, ϕ, a, M); 0.0 ξ_rr(t, r, θ, ϕ, a, M) 0.0 0.0; 0.0 0.0 ξ_θθ(t, r, θ, ϕ, a, M) 0.0; ξ_tϕ(t, r, θ, ϕ, a, M) 0.0 0.0 ξ_ϕϕ(t, r, θ, ϕ, a, M)]
ξ_μν(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, μ::Int, ν::Int) = (μ==1) && (ν==1) ? ξ_tt(t, r, θ, ϕ, a, M) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? ξ_tϕ(t, r, θ, ϕ, a, M) : μ==2 && ν==2 ? ξ_rr(t, r, θ, ϕ, a, M) : μ==3 && ν==3 ? ξ_θθ(t, r, θ, ϕ, a, M) : μ==4 & ν==4 ? ξ_ϕϕ(t, r, θ, ϕ, a, M) : 0.0

# Christoffel symbols
Γttr(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -4.0M * (a^2 + r^2) * (-r^2 + (a^2 * cos(θ)^2)) / ((a^2 + r * (r - 2.0M)) * (a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γttθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -4.0a^2 * M * r * sin(2.0θ) / ((a^2 + 2.0r^2 + a^2* cos(2.0θ))^2)
Γtrϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = 4.0a * M *(-r^2 * (a^2 + 3r^2) + a^2 * (a - r) * (a + r) * cos(θ)^2) * sin(θ)^2 / ((a^2 + r * (r - 2.0M)) * (a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γtθϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = 8.0a^3 * M * r * cos(θ) * sin(θ)^3 / ((a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γtμν(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, μ::Int, ν::Int) = ((μ==1) && (ν==2)) || ((μ==2) && (ν==1)) ? Γttr(t, r, θ, ϕ, a, M) : ((μ==1) && (ν==3)) || ((μ==3) && (ν==1)) ? Γttθ(t, r, θ, ϕ, a, M) : ((μ==2) && (ν==4)) || ((μ==4) && (ν==2)) ? Γtrϕ(t, r, θ, ϕ, a, M) : ((μ==3) && (ν==4)) || ((μ==4) && (ν==3)) ? Γtθϕ(t, r, θ, ϕ, a, M) : 0.0

Γrtt(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -M * (a^2 + r * (r - 2.0M)) * (-r^2 + a^2 * cos(θ)^2) / ((r^2 + a^2 * cos(θ)^2)^3)
Γrtϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = a * M * (a^2 + r * (r - 2.0M)) * (-r^2 + a^2 * cos(θ)^2) * sin(θ)^2 / ((r^2 + a^2 * cos(θ)^2)^3)
Γrrr(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = ((M - r) / (a^2 - 2.0M * r + r^2)) + (r / (r^2 + a^2 * cos(θ)^2))
Γrrθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) =  -a^2 * cos(θ) * sin(θ) / (r^2 + a^2 * cos(θ)^2)
Γrθθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -r * (a^2 + r * (r - 2.0M)) / (r^2 + a^2 * cos(θ)^2)
Γrϕϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = - (a^2 + r * (r - 2.0M)) * sin(θ)^2 * (r * (r^2 + a^2 * cos(θ)^2)^2 + a^2 * M * (-r + a * cos(θ)) * (r + a * cos(θ)) * sin(θ)^2) / ((r^2 + a^2 * cos(θ)^2)^3)
Γrμν(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, μ::Int, ν::Int) = (μ==1) && (ν==1)  ? Γrtt(t, r, θ, ϕ, a, M) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? Γrtϕ(t, r, θ, ϕ, a, M) : (μ==2) && (ν==2) ? Γrrr(t, r, θ, ϕ, a, M) : ((μ==2) && (ν==3)) || ((μ==3) && (ν==2)) ? Γrrθ(t, r, θ, ϕ, a, M) : (μ==3) && (ν==3) ? Γrθθ(t, r, θ, ϕ, a, M) : (μ==4) && (ν==4) ? Γrϕϕ(t, r, θ, ϕ, a, M) : 0.0

Γθtt(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -2.0a^2 * M * r * cos(θ) * sin(θ) / ((r^2 + a^2 * cos(θ)^2)^3)
Γθtϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = a * M * r * (a^2 + r^2) * sin(2.0θ) / ((r^2 + a^2 * cos(θ)^2)^3)
Γθrr(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = a^2 * cos(θ) * sin(θ) / ((a^2 + r * (r - 2.0M)) * (r^2 + a^2 * cos(θ)^2))
Γθrθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = r / (r^2 + a^2 * cos(θ)^2)
Γθθθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -a^2 * cos(θ) * sin(θ) / (r^2 + a^2 * cos(θ)^2)
Γθϕϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -cos(θ) * sin(θ) * ((a^2 + r^2) * (r^2 + a^2 * cos(θ)^2)^2 + a^2 * M * r * (3a^2 + 4.0r^2 + a^2 * cos(2.0θ)) * sin(θ)^2) / ((r^2 + a^2 * cos(θ)^2)^3)
Γθμν(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, μ::Int, ν::Int) = (μ==1) && (ν==1)  ? Γθtt(t, r, θ, ϕ, a, M) : ((μ==1) && (ν==4)) || ((μ==4) && (ν==1)) ? Γθtϕ(t, r, θ, ϕ, a, M) : (μ==2) && (ν==2) ? Γθrr(t, r, θ, ϕ, a, M) : ((μ==2) && (ν==3)) || ((μ==3) && (ν==2)) ? Γθrθ(t, r, θ, ϕ, a, M) : (μ==3) && (ν==3) ? Γθθθ(t, r, θ, ϕ, a, M) : (μ==4) && (ν==4) ? Γθϕϕ(t, r, θ, ϕ, a, M) : 0.0

Γϕtr(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = 4.0a * M * (r^2 - a^2 * cos(θ)^2) / ((a^2 + r * (r - 2.0M)) * (a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γϕtθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = -8.0a * M * r * cot(θ) / ((a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γϕrϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (4.0r * (r^2 + a^2 * cos(θ)^2) * (r * (r - 2.0M) + a^2 * cos(θ)^2) + 4.0a^2 * M * (-r + a * cos(θ)) * (r + a * cos(θ)) * sin(θ)^2) / ((a^2 + r * (r - 2.0M)) * (a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γϕθϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = cot(θ) + 4.0a^2 * M * r* sin(2.0θ) / ((a^2 + 2.0r^2 + a^2 * cos(2.0θ))^2)
Γϕμν(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, μ::Int, ν::Int) = ((μ==1) && (ν==2)) || ((μ==2) && (ν==1)) ? Γϕtr(t, r, θ, ϕ, a, M) : ((μ==1) && (ν==3)) || ((μ==3) && (ν==1)) ? Γϕtθ(t, r, θ, ϕ, a, M) : ((μ==2) && (ν==4)) || ((μ==4) && (ν==2)) ? Γϕrϕ(t, r, θ, ϕ, a, M) : ((μ==3) && (ν==4)) || ((μ==4) && (ν==3)) ? Γϕθϕ(t, r, θ, ϕ, a, M) : 0.0

Γαμν(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, α::Int, μ::Int, ν::Int) = α==1 ? Γtμν(t, r, θ, ϕ, a, M, μ, ν) : α==2 ? Γrμν(t, r, θ, ϕ, a, M, μ, ν) : α==3 ? Γθμν(t, r, θ, ϕ, a, M, μ, ν) : α==4 ? Γϕμν(t, r, θ, ϕ, a, M, μ, ν) : throw(DomainError(α, "α should be in the range 1 ≤ α ≤ 4"))

end 

module ConstantsOfMotion
using LinearAlgebra
using Symbolics
using QuadGK
using Elliptic
using PolynomialRoots
using ..Kerr
using GSL

# calculate dimensionless E, L, Q, as per Schmidt (arXiv:gr-qc/0202090)
function SchmidtELQ(a::Float64, p::Float64, e::Float64, θi::Float64)
    # define turning points rp, ra
    rp = p / (1 + e)
    ra  = p / (1 - e)

    Δ(r) = r^2 - 2r + a^2
    zm = cos(θi) # z_{-}
    
    # define functions f, g, h, and d as in Eqs. 2.27 - 2.30
    f(r) = r^4 + (a^2) * (r * (r + 2) + (zm^2) * Δ(r))
    g(r) = 2a*r
    h(r) = r * (r-2) + ((zm^2) / (1 - zm^2)) * Δ(r)
    d(r) = (r^2 + (a^2) * zm^2) * Δ(r)

    # define derivatives of f, g, h, and d as in Eqs. 2.32 - 2.35
    fp(r) = 4r^3 + (2a^2) * ((1 + zm^2) * r + (1 - zm^2))
    gp(r) = 2a
    hp(r) = (2 * (r - 1)) / (1 - zm^2)
    dp(r) = 2 * (2r - 3) * r^2 + (2a^2) * ((1 + zm^2) * r - zm^2)

    # define symbols ε, η, κ, ρ, σ in Eqs. 2.30, 2.31 (current implementation only for e ≠ 0)
    if 0.0 < e < 1.0
        ε = det([d(ra) g(ra); d(rp) g(rp)])
        η = det([f(ra) g(ra); f(rp) g(rp)])
        κ = det([d(ra) h(ra); d(rp) h(rp)])
        ρ = det([f(ra) h(ra); f(rp) h(rp)])
        σ = det([g(ra) h(ra); g(rp) h(rp)])
    elseif e == 0.0
        ε = det([d(ra) g(ra); dp(rp) gp(rp)])
        η = det([f(ra) g(ra); fp(rp) gp(rp)])
        κ = det([d(ra) h(ra); dp(rp) hp(rp)])
        ρ = det([f(ra) h(ra); fp(rp) hp(rp)])
        σ = det([g(ra) h(ra); gp(rp) hp(rp)])
    end
    En = sqrt((κ*ρ + 2ε*σ - 2*sqrt(σ * (-η * κ^2 + ε * κ * ρ + σ *ε^2))) / (ρ^2 + 4η * σ)) # Eq. 2.32

    L = sqrt((ε - η * En^2) / σ) # Eq. 2.34

    Q = (zm^2) * ((a^2) * (1 - En^2) + (L^2) / (1 - zm^2)) # Eq. 2.25

    return En, L, Q
end

# calculates dimensionless kerr fundamental frequencies wrt proper time and the conversion factor to boyer-lindquist coordinate time (Schmidt)
function SchmidtKerrFreqs(a::Float64, p::Float64, e::Float64, θi::Float64)
    # constants of motion
    En, L, Q = SchmidtELQ(a, p, e, θi)

    zm = cos(θi)
    zp = sqrt(((1)/((2a^2) * (1 - En^2))) * ((a^2) * (1 - En^2) + L^2 + Q + sqrt((4a^2) * (-1 + En^2) * Q + ((-a^2) * (-1 + En^2) + L^2 + Q)^2)))
    k = (zm^2)/(zp^2)

    # define functions J, H, G, F for ra(p, ed, θi, a)ial integra(p, el, θi, a) computation
    J(χ) = (1 - En^2) * (1 - e^2) + 2 * (1 - En^2 - (1 - e^2) / p) * (1 + e * cos(χ)) + (((a^2) * ((-1 + e^2)^2) * Q)/(p^4)) * (1 + e * cos(χ))^2   # Eq. B.11
    H(χ) = 1 - (2 / p) * (1 + e * cos(χ)) + ((a^2) / (p^2)) * (1 + e * cos(χ))^2   # Eq. 2.10
    G(χ) = L - ((2 * (L - a * En)) / p) * (1 + e * cos(χ))   # Eq. 2.11
    F(χ) = En + ((a^2) * En / (p^2)) * (1 + e * cos(χ))^2 - (2a * (L - a * En) / (p^3)) * (1 + e * cos(χ))^3

    X, errX = quadgk(χ -> 1/sqrt(J(χ)), 0, π)   # Eq. 2.6
    Y, errY = quadgk(χ -> (p^2)/(((1 + e * cos(χ))^2) * sqrt(J(χ))), 0, π)   # Eq. 2.7
    Z, errZ = quadgk(χ -> G(χ) / (H(χ) * sqrt(J(χ))), 0, π)   # Eq. 2.8
    W, errW = quadgk(χ -> (F(χ) * p^2)/(((1 + e * cos(χ))^2) * H(χ) * sqrt(J(χ))), 0, π) 

    β = sqrt((1 - En^2) * a^2)

    EK = Elliptic.K(k)
    EE = Elliptic.E(k)
    EPi = Elliptic.Pi(zm^2, π/2, k)
     
    # calculate frequencies
    Λ = (Y + (a^2) * (zp^2) * X) * EK - (a^2) * (zp^2) * X * EE   # Eq. 2.5
    ωr = (π * p * EK) / ((1 - e^2) * Λ)   # Eq. 2.2
    ωθ = (π * β * zp * X) / (2Λ)   # Eq. 2.3
    ωϕ = (1 / Λ) * ((Z - L * X) * EK + L * X * EPi)   # Eq. 2.4
    Γ = (1 / Λ) * (EK * W + (zp^2) * (EK - EE) * (a^2) * En * X)

    return [ωr, ωθ, ωϕ, Γ]
end


# define functions used in mappings between (E, L, Q), and (p, e, θmin), as per Sopuerta, Yunes (arXiv:1109.0572v2) in Appendix E

# coefficients of polynomial in E, L (Eq. E3)
αI(a::Float64, M::Float64, rI::Float64, zm::Float64) = (rI^2 + a^2) * (rI^2 + a^2 * zm) + 2.0M * rI * a^2 * (1.0 - zm)    # Eq. E4
βI(a::Float64, M::Float64, rI::Float64, zm::Float64) = - 2.0M * rI * a    # Eq. E5
γI(a::Float64, M::Float64, rI::Float64, zm::Float64) = -(1.0 / (1.0 - zm)) * (rI^2 + a^2 * zm - 2.0 * M * rI)    # Eq. E6
λI(a::Float64, M::Float64, rI::Float64, zm::Float64) = -(rI^2 + a^2 * zm) * (rI^2 - 2.0M * rI + a^2)    # Eq. E7

# for circular orbits
α2(a::Float64, M::Float64, r0::Float64, zm::Float64) = 2.0r0 * (r0^2 + a^2) - a^2 * (r0 - M) * (1.0 - zm)    # Eq. E8
β2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -a * M    # Eq. E9
γ2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -(r0 - M) / (1.0 - zm)    # Eq. E10
λ2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -r0 * (r0^2 - 2.0M * r0 + a^2) - (r0 - M) * (r0^2 + a^2 * zm)    # Eq. E11

# define [*, *] operation in Eq. E3
commute(Πa::Float64, Πp::Float64, Ωa::Float64, Ωp::Float64) = Πa * Ωp - Πp * Ωa

# compute prograde constants of motion - note that their "C" is Schmidt's "Q", and their "Q" is the "alternative definition"
function ELQ(a::Float64, p::Float64, e::Float64, θmin::Float64, M::Float64)
    zm = cos(θmin)^2
    if e==0.0
        r0 = p * M
        α1 = αI(a, M, r0, zm)
        α2 = ConstantsOfMotion.α2(a, M, r0, zm)
        β1 = βI(a, M, r0, zm)
        β2 = ConstantsOfMotion.β2(a, M, r0, zm)
        γ1 = γI(a, M, r0, zm)
        γ2 = ConstantsOfMotion.γ2(a, M, r0, zm)
        λ1 = λI(a, M, r0, zm)
        λ2 = ConstantsOfMotion.λ2(a, M, r0, zm)
    else
        rp = p * M / (1 + e)
        ra = p * M / (1 - e)
        α1 = αI(a, M, ra, zm)
        α2 = αI(a, M, rp, zm)
        β1 = βI(a, M, ra, zm)
        β2 = βI(a, M, rp, zm)
        γ1 = γI(a, M, ra, zm)
        γ2 = γI(a, M, rp, zm)
        λ1 = λI(a, M, ra, zm)
        λ2 = λI(a, M, rp, zm)
    end
    
    # write out coefficients of Eq. E12 in the form ax^2 + bx + c
    aa = (commute(α1, α2, γ1, γ2)^2 + 4.0 * commute(α1, α2, β1, β2) * commute(γ1, γ2, β1, β2))
    b = 2.0 * (commute(α1, α2, γ1, γ2) * commute(λ1, λ2, γ1, γ2) + 2.0 * commute(γ1, γ2, β1, β2) * commute(λ1, λ2, β1, β2))
    c = commute(λ1, λ2, γ1, γ2)^2

    # prograge energy (Eq. E12) - retrograde is other root
    Ep = sqrt((-b - sqrt(b^2 - 4.0aa * c))/ 2.0aa)
    # prograde z-component of angular momentum (Eq. E14) - retrograde is negative root
    Lp = sqrt((commute(α1, α2, β1, β2) * Ep^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))

    C = zm * (Lp^2 / (1.0 - zm) + a^2 * (1.0 - Ep^2))    # Eq. E2
    Q = C + (Lp - a * Ep)^2    # Eq. 17
    return Ep, Lp, Q, C
end

s = [(-1, -1), (-1, 1), (1, -1), (1, 1)]    # sign pairs (s₁, s₂) in Eq. E32


# compute p, e, θ from (a, E, L, Q, C)
function peθ_gsl(a::Float64, E::Float64, L::Float64, Q::Float64, C::Float64, M::Float64)
    # define coefficients of radial quartic (Eq. E24)
    a0 = a^2 * C / (1.0 - E^2)
    a1 = - 2.0M * Q / (1.0 - E^2)
    a2 =  (a^2 * (1.0 - E^2) + L^2 + C) / (1.0 - E^2)
    a3 = - 2.0M / (1.0 - E^2)

    δ = -3.0 * a3^2 / 8.0 + a2
    τ = a3^3 / 8.0 - a2 * a3 / 2.0 + a1
    ε =  -3.0 * a3^4 /256.0 + a2 * a3^2 / 16.0 - a1 * a3 / 4.0 + a0

    b_0 = δ^3 / 2.0 - δ * ε / 2.0 - τ^2 / 8.0
    b_1 = 2.0 * δ^2 - ε
    b_2 = 5.0δ/2.0

    # solve depressed cubic (Eq. E27)
    root_1 = Cdouble[0]; root_2 = Cdouble[0]; root_3 = Cdouble[0];
    GSL.poly_solve_cubic(b_2, b_1, b_0, root_1, root_2, root_3)
    y1 = root_1[1]

    # solve radial quartic (Eq. E24)
    δplus2y1 = δ + 2.0y1;
    Threeδplus2y1 = 3.0δ + 2.0y1;
    TwoδdivSqrt = 2.0τ / sqrt(δplus2y1);

    S2FactorS1_plus1 = -(Threeδplus2y1 + TwoδdivSqrt)
    S2FactorS1_minus1 = -(Threeδplus2y1 - TwoδdivSqrt)

    rOne = -0.25 * a3 + 0.5 * (sqrt(δplus2y1) + sqrt(S2FactorS1_plus1))
    rTwo = -0.25 * a3 + 0.5 * (sqrt(δplus2y1) - sqrt(S2FactorS1_plus1))
    rThree= -0.25 * a3 + 0.5 * (-sqrt(δplus2y1) + sqrt(S2FactorS1_minus1))
    rFour= -0.25 * a3 + 0.5 * (-sqrt(δplus2y1) - sqrt(S2FactorS1_minus1))

    # r₄ < r₃ < rₚ < rₐ
    r = sort([rOne, rTwo, rThree, rFour])
    p = 2.0 * r[3] * r[4] / (M * (r[3] + r[4]))    # Eq. 23
    e = (r[4] - r[3]) / (r[3] + r[4])   # Eq. 23

    ## now calculate θmin
    # coefficients of polynomial in Eq. E33
    c0 = C / (a^2 * (1.0 - E^2))
    c1 = -1.0 - (L^2 + C) / (a^2 * (1.0 - E^2))

    θmin = acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))
    return p, e, θmin
end

# compute p, e, θ from (a, E, L, Q, C)
function peθ(a::Float64, E::Float64, L::Float64, Q::Float64, C::Float64, M::Float64)
    # define coefficients of radial quartic (Eq. E24)
    a0 = a^2 * C / (1.0 - E^2)
    a1 = - 2.0M * Q / (1.0 - E^2)
    a2 =  (a^2 * (1.0 - E^2) + L^2 + C) / (1.0 - E^2)
    a3 = - 2.0M / (1.0 - E^2)

    δ = -3.0 * a3^2 / 8.0 + a2
    τ = a3^3 / 8.0 - a2 * a3 / 2.0 + a1
    ε =  -3.0 * a3^4 /256.0 + a2 * a3^2 / 16.0 - a1 * a3 / 4.0 + a0
    yroots = roots([δ^3 / 2.0 - δ * ε / 2.0 - τ^2 / 8.0, 2.0 * δ^2 - ε, 5.0δ/2.0, 1])    # find roots of cubic polynomial Eq. E27

    # choose one of the real roots (at least one is guaranteed)
    if imag.(yroots[1])==0
        y1 = real.(yroots[1])
    elseif imag.(yroots[2])==0
        y1 = real.(yroots[2])
    else
        y1 = real.(yroots[3])
    end

    # return a3

    r = zeros(4)
    @inbounds Threads.@threads for i=1:4
        # r[i] = -a3/4.0 + (1.0/2.0) * (s[i][1] * sqrt(δ + 2.0y1) + s[i][2] * sqrt(-(3.0δ + 2.0y1 + s[i][1] * 2.0τ / sqrt(δ + 2.0y1))))
        r[i] = -a3/4.0 + (1.0/2.0) * (s[i][1] * sqrt(δ + 2.0y1) + s[i][2] * sqrt(-(3.0δ + 2.0y1 + s[i][1] * 2.0τ / sqrt(δ + 2.0y1))))
    end

    # r₄ < r₃ < rₚ < rₐ
    r = sort(r)
    p = 2.0 * r[3] * r[4] / (M * (r[3] + r[4]))    # Eq. 23
    e = (r[4] - r[3]) / (r[3] + r[4])   # Eq. 23

    ## now calculate θmin
    # coefficients of polynomial in Eq. E33
    c0 = C / (a^2 * (1.0 - E^2))
    c1 = -1.0 - (L^2 + C) / (a^2 * (1.0 - E^2))

    θmin = acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))
    return p, e, θmin
end

function KerrFreqs(a::Float64, p::Float64, e::Float64, θmin::Float64, E::Float64, L::Float64, Q::Float64, C::Float64, rplus::Float64, rminus::Float64, M::Float64)
    zm = cos(θmin)^2
    zp = C / (a^2 * (1.0-E^2) * zm)    # Eq. E23
    ra=p * M / (1.0 - e); rp=p * M / (1.0 + e);
    A = M / (1.0 - E^2) - (ra + rp) / 2.0    # Eq. E20
    B = a^2 * C / ((1.0 - E^2) * ra * rp)    # Eq. E21
    r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19

    kr = sqrt((ra-rp) * (r3-r4) / ((ra-r3) * (rp-r4)))    # Eq. F5
    kθ = sqrt(zm/zp)    # Eq. F5

    K_kr = Elliptic.K(kr^2)
    K_kθ = Elliptic.K(kθ^2)
    E_kr = Elliptic.E(kr^2)
    E_kθ = Elliptic.E(kθ^2)

    hr = (ra-rp) / (ra-r3)
    hp = (ra-rp) * (r3-rplus) / ((ra-r3) * (rp-rplus)); hm = (ra-rp) * (r3-rminus) / ((ra-r3) * (rp-rminus))

    Πhr = Elliptic.Pi(hr, π/2, kr^2); Πhp = Elliptic.Pi(hp, π/2, kr^2); Πhm = Elliptic.Pi(hm, π/2, kr^2)
    Πzm = Elliptic.Pi(zm, π/2, kθ^2); Πzp = Elliptic.Pi(-zp, π/2, kθ^2); 

    γr = π * sqrt((1.0-E^2) * (ra-r3) * (rp-r4)) / (2.0K_kr)    # Eq. F3
    γθ = π * a * sqrt((1.0-E^2)*zp)/(2.0K_kθ)    # Eq. F4
    γϕ = 2.0a * γr / (π * (rplus - rminus) * sqrt((1.0-E^2) * (ra-r3)*(rp-r4))) * ((2.0M*E*rplus-a*L) / (r3-rplus) * (K_kr - (rp-r3)/(rp-rplus) * Πhp) - 
        (2.0M*E*rminus-a*L) / (r3-rminus) * (K_kr - (rp-r3)/(rp-rminus) * Πhm)) + 2.0 * L * γθ / (π * a * sqrt((1.0-E^2)*zp)) * Πzm   # Eq. F8
    γt = 4.0M^2 * E + 2.0a * E * sqrt(zp) / (π * sqrt(1.0-E^2)) * (K_kθ-E_kθ) * γθ + 2.0γr / (π * sqrt((1.0-E^2) * (ra-r3) * (rp-r4))) * (
        0.5E * ((r3 * (ra+rp+r3) - ra * rp) * K_kr + (rp-r3) * (ra+rp+r3+r4) * Πhr + (ra-r3) * (rp-r4) * E_kr) + 2.0M * E * (r3 * K_kr + (rp-r3) * Πhr)+
        2.0M / (rplus-rminus) * (((4.0M^2 * E-a*L) * rplus - 2.0M * a^2 * E)/(r3-rplus) * (K_kr - (rp-r3)/(rp-rplus) * Πhp) - 
        ((4.0M^2 * E-a*L) * rminus - 2.0M * a^2 * E)/(r3-rminus) * (K_kr - (rp-r3)/(rp-rminus) * Πhm)))

    return [γr, γθ, γϕ, γt]
end
end

module KerrGeodesics
import ..KerrMetric: Γtrϕ, Γttr, Γtrϕ, Γttr, Γttθ, Γtθϕ, Γrrr, Γrrθ, Γrtt, Γrtϕ, Γrθθ, Γrϕϕ, Γθrr, Γθrθ, Γθtt, Γθtϕ, Γθθθ, Γθϕϕ, Γϕrϕ, Γϕtr, Γϕtθ, Γϕθϕ
using ..KerrMetric
using ..ConstantsOfMotion
using DifferentialEquations
using DelimitedFiles
using StaticArrays

# expressions for dt/dτ and dϕ/dτ from Lagrangian
tdot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, EE::Float64, LL::Float64, g_ϕϕ::Function, g_tϕ::Function, g_tt::Function) = (EE * g_ϕϕ(t, r, θ, ϕ, a, M) + LL * g_tϕ(t, r, θ, ϕ, a, M)) / (g_tϕ(t, r, θ, ϕ, a, M)^2 - g_tt(t, r, θ, ϕ, a, M) * g_ϕϕ(t, r, θ, ϕ, a, M))   # Eq. 5.9
ϕdot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, EE::Float64, LL::Float64, g_ϕϕ::Function, g_tϕ::Function, g_tt::Function) = - (EE * g_tϕ(t, r, θ, ϕ, a, M) + LL * g_tt(t, r, θ, ϕ, a, M)) / (g_tϕ(t, r, θ, ϕ, a, M)^2 - g_tt(t, r, θ, ϕ, a, M) * g_ϕϕ(t, r, θ, ϕ, a, M))   # Eq. 5.10

# initial conditions for bound kerr orbits starting in equatorial plane
function boundKerr_ics(a::Float64, M::Float64, EEi::Float64, LLi::Float64, ri::Float64, θi::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function)
    ti = 0.0
    ϕi = 0.0
    xi = @SArray [ti, ri, θi, ϕi]
    uti = tdot(xi..., a, M, EEi, LLi, g_ϕϕ, g_tϕ, g_tt)
    uri = 0.0
    uϕi = ϕdot(xi..., a, M, EEi, LLi, g_ϕϕ, g_tϕ, g_tt)
    uθi² = (-1 - g_rr(xi..., a, M) * uri^2 - g_tt(xi..., a, M) * uti^2 - 2.0 * g_tϕ(xi..., a, M) * uti * uϕi - g_ϕϕ(xi..., a, M) * uϕi^2) / g_θθ(xi..., a, M)    # Eq. 5.11
    uθi = abs(uθi²) <= 1e-14 ? 0. : sqrt(uθi²)   # replace solutions close to zero by zero exactly
    uxi = @SArray[uti, uri, uθi, uϕi]
    return [uxi, xi]
end

# geodesic equations
tddot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = -2.0 * (rdot * (ϕdot * Γtrϕ(t, r, θ, ϕ, a, M) + tdot * Γttr(t, r, θ, ϕ, a, M)) + θdot * (tdot * Γttθ(t, r, θ, ϕ, a, M) + ϕdot * Γtθϕ(t, r, θ, ϕ, a, M)))
rddot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = -(rdot * (rdot * Γrrr(t, r, θ, ϕ, a, M) + 2.0 * θdot * Γrrθ(t, r, θ, ϕ, a, M)) + tdot * (tdot * Γrtt(t, r, θ, ϕ, a, M) + 2.0 * ϕdot * Γrtϕ(t, r, θ, ϕ, a, M)) + θdot^2 * Γrθθ(t, r, θ, ϕ, a, M) + ϕdot^2 * Γrϕϕ(t, r, θ, ϕ, a, M))
θddot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = -(rdot * (rdot * Γθrr(t, r, θ, ϕ, a, M) + 2.0 * θdot * Γθrθ(t, r, θ, ϕ, a, M)) + tdot * (tdot * Γθtt(t, r, θ, ϕ, a, M) + 2.0 * ϕdot * Γθtϕ(t, r, θ, ϕ, a, M)) + θdot^2 * Γθθθ(t, r, θ, ϕ, a, M) + ϕdot^2 * Γθϕϕ(t, r, θ, ϕ, a, M))
ϕddot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = -2.0 * (rdot * (ϕdot * Γϕrϕ(t, r, θ, ϕ, a, M) + tdot * Γϕtr(t, r, θ, ϕ, a, M)) + θdot * (tdot * Γϕtθ(t, r, θ, ϕ, a, M) + ϕdot * Γϕθϕ(t, r, θ, ϕ, a, M)))
xμddot(μ::Int, t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = μ==1 ? tddot(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, a, M) : μ==2 ? rddot(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, a, M) : μ==3 ? θddot(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, a, M) : ϕddot(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, a, M)

# equation for ODE solver
function geodesicEq(du, u, params, t)
    @SArray [tddot(u..., du..., params...), rddot(u..., du..., params...), θddot(u..., du..., params...), ϕddot(u..., du..., params...)]
end

# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
function compute_kerr_geodesic(a::Float64, p::Float64, e::Float64, θi::Float64, τmax::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-16, abstol::Float64=1e-16, saveat::Float64=0.5; data_path::String="Results/")
    # orbital parameters
    M = 1.0;

    # define periastron and apastron
    rp = p * M / (1 + e);   # Eq. 6.1
    ra = p * M / (1 - e);   # Eq. 6.1

    # calculate integrals of motion from orbital parameters
    E, L, Q = ConstantsOfMotion.ELQ(a, p, e, θi, M)   # dimensionless constants

    # initial conditions for Kerr geodesic trajectory
    ri = ra; τspan = (0.0, τmax); params = @SArray [a, M];
    τ = 0:saveat:τmax |> collect

    ics = KerrGeodesics.boundKerr_ics(a, M, E, L, ri, θi, KerrMetric.g_tt,  KerrMetric.g_tϕ,  KerrMetric.g_rr, KerrMetric.g_θθ, KerrMetric.g_ϕϕ);
    prob = SecondOrderODEProblem(KerrGeodesics.geodesicEq, ics..., τspan, params);
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
    tddot = KerrGeodesics.tddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    rddot = KerrGeodesics.rddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    θddot = KerrGeodesics.θddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    ϕddot = KerrGeodesics.ϕddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);

    # save trajectory- rows are: τ, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([τ, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot]))
    mkpath(data_path)
    ODE_filename=data_path * "ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(τmax)_tol_$(reltol).txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end
    println("ODE saved to: " * ODE_filename)
end

end
end