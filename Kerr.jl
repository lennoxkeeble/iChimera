module Kerr

module KerrMetric
using LinearAlgebra
using StaticArrays

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
ξ_tt(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (r^2 * Δ(r, a, M) * (2.0M * r - Σ(r, θ, a)) * Σ(r, θ, a) + 2.0 * ((a^2 + r^2) * (2.0M * r - Σ(r, θ, a)) - 2.0a^2 * M * r * sin(θ)^2)^2) / (Δ(r, a, M) * Σ(r, θ, a)^2)
ξ_tϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (1.0 / (Δ(r, a, M) * Σ(r, θ, a)^2)) * (-2.0a * (4.0 * M^2 * r^2 * (a^2 + r^2)^2 + M * r * (-4.0 * (a^2 + r^2)^2 + r^2 * Δ(r, a, M)) * Σ(r, θ, a) + (a^2 + r^2)^2 * Σ(r, θ, a)^2) * sin(θ)^2 + 8.0 * a^3 * M * r * (a^2 + r^2) * (2.0M * r - Σ(r, θ, a)) * sin(θ)^4 - 8.0 * a^5 * M^2 * r^2 * sin(θ)^6)
ξ_rr(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (r^2 - 2.0 * Σ(r, θ, a)) * Σ(r, θ, a) / Δ(r, a, M)
ξ_θθ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = r^2 * Σ(r, θ, a)
ξ_ϕϕ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = (1.0 / (Δ(r, a, M) * Σ(r, θ, a)^2)) * (2.0 * (a * M * r * (a^2 + 2.0 * r^2) - a * (a^2 + r^2) * Σ(r, θ, a) + a^3 * M * r * cos(2.0θ))^2 * sin(θ)^4 + r^2 * Δ(r, a, M) * Σ(r, θ, a) * sin(θ)^2 * ((a^2 + r^2) * Σ(r, θ, a) + 2.0 * a^2 * M * r * sin(θ)^2))
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

function ELQ(a::Float64, p::Float64, e::Float64, θi::Float64)
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

# calculates dimensionless kerr fundamental frequencies wrt proper time and the conversion factor to boyer-lindquist coordinate time
function KerrFreqs(a::Float64, p::Float64, e::Float64, θi::Float64)
    # constants of motion
    En = En(a, p, e, θi)
    L = L(a, p, e, θi)
    Q = Q(a, p, e, θi)

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

end

module KerrGeodesics
import ..KerrMetric: Γtrϕ, Γttr, Γtrϕ, Γttr, Γttθ, Γtθϕ, Γrrr, Γrrθ, Γrtt, Γrtϕ, Γrθθ, Γrϕϕ, Γθrr, Γθrθ, Γθtt, Γθtϕ, Γθθθ, Γθϕϕ, Γϕrϕ, Γϕtr, Γϕtθ, Γϕθϕ
using ..KerrMetric
using ..ConstantsOfMotion
using DifferentialEquations
using DelimitedFiles

# expressions for dt/dτ and dϕ/dτ from Lagrangian
tdot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, EE::Float64, LL::Float64, g_ϕϕ::Function, g_tϕ::Function, g_tt::Function) = (EE * g_ϕϕ(t, r, θ, ϕ, a, M) + LL * g_tϕ(t, r, θ, ϕ, a, M)) / (g_tϕ(t, r, θ, ϕ, a, M)^2 - g_tt(t, r, θ, ϕ, a, M) * g_ϕϕ(t, r, θ, ϕ, a, M))   # Eq. 5.9
ϕdot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, EE::Float64, LL::Float64, g_ϕϕ::Function, g_tϕ::Function, g_tt::Function) = - (EE * g_tϕ(t, r, θ, ϕ, a, M) + LL * g_tt(t, r, θ, ϕ, a, M)) / (g_tϕ(t, r, θ, ϕ, a, M)^2 - g_tt(t, r, θ, ϕ, a, M) * g_ϕϕ(t, r, θ, ϕ, a, M))   # Eq. 5.10

# initial conditions for bound kerr orbits starting in equatorial plane
function boundKerr_ics(a::Float64, M::Float64, m::Float64, EEi::Float64, LLi::Float64, ri::Float64, θi::Float64, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function)
    ti = 0.0
    ϕi = 0.0
    xi = [ti, ri, θi, ϕi]
    uti = tdot(xi..., a, M, EEi, LLi, g_ϕϕ, g_tϕ, g_tt)
    uri = 0.0
    uϕi = ϕdot(xi..., a, M, EEi, LLi, g_ϕϕ, g_tϕ, g_tt)
    uθi² = (-1 - g_rr(xi..., a, M) * uri^2 - g_tt(xi..., a, M) * uti^2 - 2.0 * g_tϕ(xi..., a, M) * uti * uϕi - g_ϕϕ(xi..., a, M) * uϕi^2) / g_θθ(xi..., a, M)    # Eq. 5.11
    uθi = abs(uθi²) <= 1e-14 ? 0. : sqrt(uθi²)   # replace solutions close to zero by zero exactly
    uxi = [uti, uri, uθi, uϕi]
    return [uxi, xi]
end

# geodesic equations
tddot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = -2.0 * (rdot * (ϕdot * Γtrϕ(t, r, θ, ϕ, a, M) + tdot * Γttr(t, r, θ, ϕ, a, M)) + θdot * (tdot * Γttθ(t, r, θ, ϕ, a, M) + ϕdot * Γtθϕ(t, r, θ, ϕ, a, M)))
rddot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = -(rdot * (rdot * Γrrr(t, r, θ, ϕ, a, M) + 2.0 * θdot * Γrrθ(t, r, θ, ϕ, a, M)) + tdot * (tdot * Γrtt(t, r, θ, ϕ, a, M) + 2.0 * ϕdot * Γrtϕ(t, r, θ, ϕ, a, M)) + θdot^2 * Γrθθ(t, r, θ, ϕ, a, M) + ϕdot^2 * Γrϕϕ(t, r, θ, ϕ, a, M))
θddot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = -(rdot * (rdot * Γθrr(t, r, θ, ϕ, a, M) + 2.0 * θdot * Γθrθ(t, r, θ, ϕ, a, M)) + tdot * (tdot * Γθtt(t, r, θ, ϕ, a, M) + 2.0 * ϕdot * Γθtϕ(t, r, θ, ϕ, a, M)) + θdot^2 * Γθθθ(t, r, θ, ϕ, a, M) + ϕdot^2 * Γθϕϕ(t, r, θ, ϕ, a, M))
ϕddot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = -2.0 * (rdot * (ϕdot * Γϕrϕ(t, r, θ, ϕ, a, M) + tdot * Γϕtr(t, r, θ, ϕ, a, M)) + θdot * (tdot * Γϕtθ(t, r, θ, ϕ, a, M) + ϕdot * Γϕθϕ(t, r, θ, ϕ, a, M)))
xμddot(μ::Int, t::Float64, r::Float64, θ::Float64, ϕ::Float64, tdot::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, a::Float64, M::Float64) = μ==1 ? tddot(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, a, M) : μ==2 ? rddot(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, a, M) : μ==3 ? θddot(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, a, M) : ϕddot(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, a, M)

# equation for ODE solver
function geodesicEq!(ddu, du, u, params, t)
    ddu[1] = tddot(u..., du..., params...)
    ddu[2] = rddot(u..., du..., params...)
    ddu[3] = θddot(u..., du..., params...)
    ddu[4] = ϕddot(u..., du..., params...)
end

# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
function compute_kerr_geodesic(a::Float64, p::Float64, e::Float64, θi::Float64, τmax::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-16, abstol::Float64=1e-16, saveat::Float64=0.5; data_path::String="Results/")
    # orbital parameters
    M = 1.0; m = 1.0;

    # define periastron and apastron
    rp = p * M / (1 + e);   # Eq. 6.1
    ra = p * M / (1 - e);   # Eq. 6.1

    # calculate integrals of motion from orbital parameters
    E, L, Q = ConstantsOfMotion.ELQ(a, p, e, θi)   # dimensionless constants

    # initial conditions for Kerr geodesic trajectory
    ri = ra; θi = π/2; τspan = (0.0, τmax); params = [a, M];

    ics = KerrGeodesics.boundKerr_ics(a, M, m, E, L, ri, θi, KerrMetric.g_tt,  KerrMetric.g_tϕ,  KerrMetric.g_rr, KerrMetric.g_θθ, KerrMetric.g_ϕϕ);
    prob = SecondOrderODEProblem(KerrGeodesics.geodesicEq!, ics..., τspan, params);
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat);

    # deconstruct solution
    τ = 0:saveat:τmax |> collect
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