module FujitaFrequencies
using Elliptic
# computing the fundamental frequencies in Mino time, as in https://arxiv.org/abs/0906.1420
function compute_frequencies(a::Float64, p::Float64, e::Float64, θmin::Float64, E::Float64, L::Float64, C::Float64, rp::Float64, rm::Float64, M::Float64)
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
    hp = (r1-r2) * (r3-rp) / ((r1-r3) * (r2-rp)); hm = (r1-r2) * (r3-rm) / ((r1-r3) * (r2-rm))

    Πhr = Elliptic.Pi(hr, π/2, kr^2); Πhp = Elliptic.Pi(hp, π/2, kr^2); Πhm = Elliptic.Pi(hm, π/2, kr^2)
    Πzm = Elliptic.Pi(zm, π/2, kθ^2); Πzp = Elliptic.Pi(zp, π/2, kθ^2); 

    γr = π * sqrt((1.0-E^2) * (r1-r3) * (r2-r4)) / (2K_kr)    # Eq. 15
    γθ = π * L * sqrt(ε0 * zp) / (2.0K_kθ)
    Γ = 4.0M^2 * E + (2.0a^2 * E * zp * γθ) * (K_kθ-E_kθ) / (π * L * sqrt(ε0 * zp)) + 2.0γr / (π * sqrt((1.0-E^2) * (r1-r3) * (r2-r4))) *
        (0.5*E*((r3 * (r1+r2+r3)-r1*r2) * K_kr + (r2-r3) * (r1+r2+r3+r4) * Πhr + (r1-r3) * (r2-r4) * E_kr) + 2.0M * E * 
        (r3 * K_kr + (r2-r3) * Πhr) + 2.0M / (rp-rm) * ((rp * (4.0M^2 * E-a * L) - 2.0M * a^2 * E) * (K_kr-(r2-r3) * Πhp / (r2-rp))/(r3-rp) -
        (rm * (4.0M^2 * E-a * L) - 2.0M * a^2 * E) * (K_kr-(r2-r3) * Πhm / (r2-rm))/(r3-rm)))
    γϕ = 2.0 * γθ * Πzm / (π * sqrt(ε0 * zp)) + 2.0a * γr / (π*(rp-rm) * sqrt((1.0-E^2) * (r1-r3) * (r2-r4))) * 
        ((2.0M*E*rp-a*L) / (r3-rp) * (K_kr-(r2-r3) * Πhp / (r2-rp))-(2.0M*E*rm-a*L) / (r3-rm) * (K_kr-(r2-r3) * Πhm / (r2-rm)))

    return [γr, γθ, γϕ, Γ]
end
end

