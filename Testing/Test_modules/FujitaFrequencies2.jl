#=

    In this module we compute the fundamental frequencies in Mino time, as in https://arxiv.org/abs/0906.1420, being careful to carry explicity carry out the decompositions in Eq. 7 so that we can incorporate
    special cases of circular and/or equatorial orbits, where T_r = const, Φ_r = const in the former case, and T_θ = const, Φ_θ = const in the latter

=#

module FujitaFrequencies2
using Elliptic

Δ(r::Float64, a::Float64, M::Float64) = r^2 - 2.0M * r +a^2
P(r::Float64, a::Float64, E::Float64, L::Float64) = E * (r^2 + a^2) - a * L   # below Eq. 1
Tr(r::Float64, a::Float64, E::Float64, L::Float64, M::Float64) = (r^2 + a^2) * P(r, a, E, L) / Δ(r, a, M)   # below Eq. 1
Φr(r::Float64, a::Float64, E::Float64, L::Float64, M::Float64) = a * P(r, a, E, L) / Δ(r, a, M)   # below Eq. 1

# above Eq. 17
function γ_tθ(a::Float64, θmin::Float64, E::Float64, L::Float64, zp::Float64, E_kθ::Float64, K_kθ::Float64, Πzm::Float64)
    if θmin==π/2
        γ = -a^2 * E
    else
        γ = -((a^2 * E * (zp * E_kθ + (1.0 - zp) * K_kθ))/K_kθ)
    end
end

# Eq. 17
function γ_ϕθ(a::Float64, θmin::Float64, E::Float64, L::Float64, zp::Float64, E_kθ::Float64, K_kθ::Float64, Πzm::Float64)
    if θmin==π/2
        γ = L
    else
        γ = (L * Πzm) / K_kθ
    end
end

# Eq. 17: γtr ≡ Γ - γtθ - a * L
function γ_tr(a::Float64, p::Float64, e::Float64, E::Float64, L::Float64, zp::Float64, rplus::Float64, rminus::Float64, r1::Float64, r2::Float64, r3::Float64, r4::Float64, M::Float64, E_kr::Float64, E_kθ::Float64, 
    K_kr::Float64, K_kθ::Float64, Πhm::Float64, Πhp::Float64, Πhr::Float64, Πzm::Float64)
    if e==0.0
        γ = Tr(p/M, a, E, L, M)
    else
        γ = (2*a^2*E - 2*a*L + 8*E*M^2 - E*r1*r2 + 4*E*M*r3 + E*r1*r3 + E*r2*r3 + E*r3^2 - (4*M*(2*a^2*E*M + a*L*r3 - 4*E*M^2*r3))/((r3 - rminus)*(r3 - rplus)) + 
            (E*(r1 - r3)*(r2 - r4)*E_kr)/K_kr - (4*M*(r2 - r3)*(2*a^2*E*M + a*L*rminus - 4*E*M^2*rminus)*Πhm)/((r2 - rminus)*(-r3 + rminus)*(rminus - rplus)*K_kr) - 
            (4*M*(r2 - r3)*(2*a^2*E*M + a*L*rplus - 4*E*M^2*rplus)*Πhp)/((r2 - rplus)*(-r3 + rplus)*(-rminus + rplus)*K_kr) + 
            (E*(r2 - r3)*(4*M + r1 + r2 + r3 + r4)*Πhr)/K_kr)/2.
    end
end

# Eq. 17: γϕr ≡ γϕ - γϕθ + a * E
function γ_ϕr(a::Float64, p::Float64, e::Float64, E::Float64, L::Float64, zp::Float64, rplus::Float64, rminus::Float64, r1::Float64, r2::Float64, r3::Float64, r4::Float64, M::Float64, E_kr::Float64, E_kθ::Float64, 
    K_kr::Float64, K_kθ::Float64, Πhm::Float64, Πhp::Float64, Πhr::Float64, Πzm::Float64)
    if e==0.0
        γ = Φr(p/M, a, E, L, M)
    else
        γ = (a*((r2 - rminus)*(r2 - rplus)*(rminus - rplus)*(a*L + E*r3*(-2*M - r3 + rminus) + E*(r3 - rminus)*rplus)*K_kr + 
            (r2 - r3)*((a*L - 2*E*M*rminus)*(r2 - rplus)*(-r3 + rplus)*Πhm + (r2 - rminus)*(r3 - rminus)*(a*L - 2*E*M*rplus)*Πhp)))/
            ((r3 - rminus)*(-r2 + rminus)*(r2 - rplus)*(r3 - rplus)*(rminus - rplus)*K_kr)
    end
end

γr(e::Float64, E::Float64, r1::Float64, r2::Float64, r3::Float64, r4::Float64, K_kr::Float64) = e==0.0 ? 1e50 : π * sqrt((1.0-E^2) * (r1-r3) * (r2-r4)) / (2K_kr)    # Eq. 15
γθ(θmin::Float64, L::Float64, ε0::Float64, zp::Float64, K_kθ::Float64) = θmin==π/2 ? 1e50 : π * L * sqrt(ε0 * zp) / (2.0K_kθ)   # Eq. 15


# compute the fundamental frequencies in Mino time
function compute_frequencies(a::Float64, p::Float64, e::Float64, θmin::Float64, E::Float64, L::Float64, C::Float64, rplus::Float64, rminus::Float64, M::Float64)
    # define roots of R(r) in Eq. 10
    r1=p * M / (1.0 - e); r2=p * M / (1.0 + e)    # Eq. 11
    AplusB=2.0 * M / (1.0 - E^2) - (r1 + r2)    # Eq. 11
    AtimesB=a^2 * C / ((1.0 - E^2) * r1 * r2)    # Eq. 11
    r3 = 0.5 * (AplusB + sqrt(AplusB^2 - 4 * AtimesB))    # Eq.11
    r4 = AtimesB / r3    # Eq. 11

    ε0=a^2 * (1.0 - E^2) / (L^2)
    zm = cos(θmin)^2; zp = C / (ε0 * zm * L^2)

    kr = sqrt((r1-r2) * (r3-r4) / ((r1-r3) * (r2-r4)))    # Eq. 13
    kθ = sqrt(zm/zp)    # Eq. 13

    K_kr = Elliptic.K(kr^2)
    K_kθ = Elliptic.K(kθ^2)
    E_kr = Elliptic.E(kr^2)
    E_kθ = Elliptic.E(kθ^2)

    hr = (r1-r2) / (r1-r3)
    hp = (r1-r2) * (r3-rplus) / ((r1-r3) * (r2-rplus)); hm = (r1-r2) * (r3-rminus) / ((r1-r3) * (r2-rminus))

    Πhr = Elliptic.Pi(hr, π/2, kr^2); Πhp = Elliptic.Pi(hp, π/2, kr^2); Πhm = Elliptic.Pi(hm, π/2, kr^2)
    Πzm = Elliptic.Pi(zm, π/2, kθ^2); Πzp = Elliptic.Pi(zp, π/2, kθ^2); 

    γr = FujitaFrequencies2.γr(e, E, r1, r2, r3, r4, K_kr)
    γθ = FujitaFrequencies2.γθ(θmin, L, ε0, zp, K_kθ)
    Γ = FujitaFrequencies2.γ_tr(a, p, e, E, L, zp, rplus, rminus, r1, r2, r3, r4, M, E_kr, E_kθ, K_kr, K_kθ, Πhm, Πhp, Πhr, Πzm) + γ_tθ(a, θmin, E, L, zp, E_kθ, K_kθ, Πzm) + a * L
    γϕ = FujitaFrequencies2.γ_ϕr(a, p, e, E, L, zp, rplus, rminus, r1, r2, r3, r4, M, E_kr, E_kθ, K_kr, K_kθ, Πhm, Πhp, Πhr, Πzm) + γ_ϕθ(a, θmin, E, L, zp, E_kθ, K_kθ, Πzm) - a * E
    return [γr, γθ, γϕ, Γ]
end

end
