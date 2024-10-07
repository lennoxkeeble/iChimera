#=

    In this module we provide functions to map between from (E, L, Q) to (p, e, θmin) and vice versa, as well as to calculate the fundamental frequencies. The functions are based on the works of Schmidt (arXiv:gr-qc/0202090),
    Sopuerta, Yunes (arXiv:1109.0572v2). Throughout, Eq. X will refer to expressions in Sopuerta, Yunes (arXiv:1109.0572v2). Note that we also provide code for the mapping (E, L, Q) -> (p, e, θmin) as per Hughes, 2024 (arXiv:2401.09577) in the
    file "HughesOrbitalConstants.jl". We provide code to map from the inclincation angles ι (Eq. 25) and I/xI (Eq. 1.2-1.3 arXiv:2401.09577v2) to θmin in the file "InclinationMappings.jl".

=#

module ConstantsOfMotion
using LinearAlgebra
using QuadGK
using Elliptic
using PolynomialRoots
using GSL

"""
# Common Arguments in this module
- `a::Float64`: Kerr black hole spin parameter, 0 < a < 1.
- `p::Float64`: semi-latus rectum of the orbit (defined by, e.g., Eq. 23).
- `e::Float64`: eccentricity of the orbit (defined by, e.g., Eq. 23).
- `θmin::Float64`: minimum polar angle of the orbit (radians).
- `E::Float64`: energy per unit mass of the test particle moving along the geodesic (Eq. 14).
- `L::Float64`: axial (i.e., z-component of the) angular momentum per unit mass of the test particle moving along the geodesic (Eq. 15).
- `C::Float64`: Carter constant---note that this C is what is commonly referred to as 'Q' elsewhere (Eq. 17).
- `Q::Float64`: Alternative definition of the Carter constant (Eq. 16).
- `ra::Float64`: apastron of the orbit (furtherst radial turning point, Eq. 22).
- `rp::Float64`: periastron of the orbit (closest radial turning point, Eq. 22).
- `zp::Float64`: root the the theta geodesic equation (Eq. 91-92)
- `zm::Float64`: root the the theta geodesic equation (Eq. 91-92)
"""

# define inner/outer horizons
rplus(a::Float64)::Float64 = 1.0 + sqrt(1.0 - a^2)
rminus(a::Float64)::Float64 = 1.0 - sqrt(1.0 - a^2)

# define functions used in mappings between (E, L, Q), and (p, e, θmin), as per Sopuerta, Yunes (arXiv:1109.0572v2) in Appendix E
# coefficients of polynomial in E, L (Eq. E3)
@inline αI(a::Float64, rI::Float64, zm::Float64)::Float64 = (rI^2 + a^2) * (rI^2 + a^2 * zm) + 2.0 * rI * a^2 * (1.0 - zm)    # Eq. E4
@inline βI(a::Float64, rI::Float64, zm::Float64)::Float64 = - 2.0 * rI * a    # Eq. E5
@inline γI(a::Float64, rI::Float64, zm::Float64)::Float64 = -(1.0 / (1.0 - zm)) * (rI^2 + a^2 * zm - 2.0 * rI)    # Eq. E6
@inline λI(a::Float64, rI::Float64, zm::Float64)::Float64 = -(rI^2 + a^2 * zm) * (rI^2 - 2.0 * rI + a^2)    # Eq. E7

# for circular orbits
@inline α2(a::Float64, r0::Float64, zm::Float64)::Float64 = 2.0r0 * (r0^2 + a^2) - a^2 * (r0 - 1.0) * (1.0 - zm)    # Eq. E8
@inline β2(a::Float64, r0::Float64, zm::Float64)::Float64 = -a    # Eq. E9
@inline γ2(a::Float64, r0::Float64, zm::Float64)::Float64 = -(r0 - 1.0) / (1.0 - zm)    # Eq. E10
@inline λ2(a::Float64, r0::Float64, zm::Float64)::Float64 = -r0 * (r0^2 - 2.0 * r0 + a^2) - (r0 - 1.0) * (r0^2 + a^2 * zm)    # Eq. E11

# define [*, *] operation in Eq. E3
@inline commute(Πa::Float64, Πp::Float64, Ωa::Float64, Ωp::Float64)::Float64 = Πa * Ωp - Πp * Ωa

# compute prograde constants of motion - note that their "C" is Schmidt's "Q", and their "Q" is the "alternative definition" (Eqs. 16-17)
function compute_ELC(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64)    
    zm = cos(θmin)^2
    if e==0.0
        r0 = p
        α1 = αI(a, r0, zm)
        α2 = ConstantsOfMotion.α2(a, r0, zm)
        β1 = βI(a, r0, zm)
        β2 = ConstantsOfMotion.β2(a, r0, zm)
        γ1 = γI(a, r0, zm)
        γ2 = ConstantsOfMotion.γ2(a, r0, zm)
        λ1 = λI(a, r0, zm)
        λ2 = ConstantsOfMotion.λ2(a, r0, zm)
    else
        rp = p / (1 + e)
        ra = p / (1 - e)
        α1 = αI(a, ra, zm)
        α2 = αI(a, rp, zm)
        β1 = βI(a, ra, zm)
        β2 = βI(a, rp, zm)
        γ1 = γI(a, ra, zm)
        γ2 = γI(a, rp, zm)
        λ1 = λI(a, ra, zm)
        λ2 = λI(a, rp, zm)
    end
    
    # write out coefficients of Eq. E12 in the form ax^2 + bx + c
    aa = (commute(α1, α2, γ1, γ2)^2 + 4.0 * commute(α1, α2, β1, β2) * commute(γ1, γ2, β1, β2))
    b = 2.0 * (commute(α1, α2, γ1, γ2) * commute(λ1, λ2, γ1, γ2) + 2.0 * commute(γ1, γ2, β1, β2) * commute(λ1, λ2, β1, β2))
    c = commute(λ1, λ2, γ1, γ2)^2

    # prograde
    if sign_Lz>0
        # prograge energy (Eq. E12) - retrograde is other root
        E = sqrt((-b - sqrt(b^2 - 4.0aa * c))/ 2.0aa)
        # prograde z-component of angular momentum (Eq. E14) - retrograde is negative root
        L = sqrt((commute(α1, α2, β1, β2) * E^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))
    else
        # retrograde
        E = sqrt((-b + sqrt(b^2 - 4.0aa * c))/ 2.0aa)
        L = -sqrt((commute(α1, α2, β1, β2) * E^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))
    end

    if θmin==0.0
        C = 0.0
    else
        C = zm * (L^2 / (1.0 - zm) + a^2 * (1.0 - E^2))    # Eq. E2
    end

    Q = C + (L - a * E)^2    # Eq. 17
    
    return E, L, Q, C
end

# compute p, e, θ from (a, E, L, Q, C)  as per Sopuerta, Yunes (arXiv:1109.0572v2) in Appendix E using GSL root solver
s = [(-1, -1), (-1, 1), (1, -1), (1, 1)]    # sign pairs (s₁, s₂) in Eq. E32
function compute_p_e_θmin(a::Float64, E::Float64, L::Float64, Q::Float64, C::Float64)
    # define coefficients of radial quartic (Eq. E24)
    a0 = a^2 * C / (1.0 - E^2)
    a1 = - 2.0 * Q / (1.0 - E^2)
    a2 =  (a^2 * (1.0 - E^2) + L^2 + C) / (1.0 - E^2)
    a3 = - 2.0 / (1.0 - E^2)

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
    p = 2.0 * r[3] * r[4] / (r[3] + r[4])    # Eq. 23
    e = (r[4] - r[3]) / (r[3] + r[4])   # Eq. 23

    ## now calculate θmin
    # coefficients of polynomial in Eq. E33
    c0 = C / (a^2 * (1.0 - E^2))
    c1 = -1.0 - (L^2 + C) / (a^2 * (1.0 - E^2))

    θmin = acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))
    # iota = acos(Lp/sqrt(Lp^2+C))
    return p, e, θmin
end

# compute p, e, θ from (a, E, L, Q, C)  as per Sopuerta, Yunes (arXiv:1109.0572v2) in Appendix E using julia Roots
function compute_p_e_θmin_julia_roots(a::Float64, E::Float64, L::Float64, Q::Float64, C::Float64)
    # define coefficients of radial quartic (Eq. E24)
    a0 = a^2 * C / (1.0 - E^2)
    a1 = - 2.0 * Q / (1.0 - E^2)
    a2 =  (a^2 * (1.0 - E^2) + L^2 + C) / (1.0 - E^2)
    a3 = - 2.0 / (1.0 - E^2)

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
    @inbounds for i=1:4
        # r[i] = -a3/4.0 + (1.0/2.0) * (s[i][1] * sqrt(δ + 2.0y1) + s[i][2] * sqrt(-(3.0δ + 2.0y1 + s[i][1] * 2.0τ / sqrt(δ + 2.0y1))))
        r[i] = -a3/4.0 + (1.0/2.0) * (s[i][1] * sqrt(δ + 2.0y1) + s[i][2] * sqrt(-(3.0δ + 2.0y1 + s[i][1] * 2.0τ / sqrt(δ + 2.0y1))))
    end

    # r₄ < r₃ < rₚ < rₐ
    r = sort(r)
    p = 2.0 * r[3] * r[4] / (r[3] + r[4])    # Eq. 23
    e = (r[4] - r[3]) / (r[3] + r[4])   # Eq. 23

    ## now calculate θmin
    # coefficients of polynomial in Eq. E33
    c0 = C / (a^2 * (1.0 - E^2))
    c1 = -1.0 - (L^2 + C) / (a^2 * (1.0 - E^2))

    θmin = acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))
    return p, e, θmin
end

# compute Mino time frequencies as per Sopuerta, Yunes (arXiv:1109.0572v2) in Appendix E
function KerrFreqs(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64)
    E, L, Q, C = compute_ELC(a, p, e, θmin, sign_Lz)
    rplus = ConstantsOfMotion.rplus(a)
    rminus = ConstantsOfMotion.rminus(a)
    zm = cos(θmin)^2
    zp = C / (a^2 * (1.0-E^2) * zm)    # Eq. E23
    ra=p / (1.0 - e); rp=p / (1.0 + e);
    A = 1.0 / (1.0 - E^2) - (ra + rp) / 2.0    # Eq. E20
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
    γϕ = 2.0a * γr / (π * (rplus - rminus) * sqrt((1.0-E^2) * (ra-r3)*(rp-r4))) * ((2.0*E*rplus-a*L) / (r3-rplus) * (K_kr - (rp-r3)/(rp-rplus) * Πhp) - 
        (2.0*E*rminus-a*L) / (r3-rminus) * (K_kr - (rp-r3)/(rp-rminus) * Πhm)) + 2.0 * L * γθ / (π * a * sqrt((1.0-E^2)*zp)) * Πzm   # Eq. F8
    γt = 4.0 * E + 2.0a * E * sqrt(zp) / (π * sqrt(1.0-E^2)) * (K_kθ-E_kθ) * γθ + 2.0γr / (π * sqrt((1.0-E^2) * (ra-r3) * (rp-r4))) * (
        0.5E * ((r3 * (ra+rp+r3) - ra * rp) * K_kr + (rp-r3) * (ra+rp+r3+r4) * Πhr + (ra-r3) * (rp-r4) * E_kr) + 2.0 * E * (r3 * K_kr + (rp-r3) * Πhr)+
        2.0 / (rplus-rminus) * (((4.0 * E-a*L) * rplus - 2.0 * a^2 * E)/(r3-rplus) * (K_kr - (rp-r3)/(rp-rplus) * Πhp) - 
        ((4.0 * E-a*L) * rminus - 2.0 * a^2 * E)/(r3-rminus) * (K_kr - (rp-r3)/(rp-rminus) * Πhm)))

    # special cases in which the frequencies are infinite
    if e == 0.0 && θmin == π/2   # circular equatorial
        γr = 1e12; γθ =1e12;
    elseif e == 0.0   # circular non-equatorial
        γr = 1e12;
    elseif θmin == π/2   # non-circular equatorial
        γθ = 1e12;
    end

    return [γr, γθ, γϕ, γt]
end

function KerrFreqs(a::Float64, p::Float64, e::Float64, θmin::Float64, E::Float64, L::Float64, Q::Float64, C::Float64, rplus::Float64, rminus::Float64)
    zm = cos(θmin)^2
    zp = C / (a^2 * (1.0-E^2) * zm)    # Eq. E23
    ra=p / (1.0 - e); rp=p / (1.0 + e);
    A = 1.0 / (1.0 - E^2) - (ra + rp) / 2.0    # Eq. E20
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
    γϕ = 2.0a * γr / (π * (rplus - rminus) * sqrt((1.0-E^2) * (ra-r3)*(rp-r4))) * ((2.0*E*rplus-a*L) / (r3-rplus) * (K_kr - (rp-r3)/(rp-rplus) * Πhp) - 
        (2.0*E*rminus-a*L) / (r3-rminus) * (K_kr - (rp-r3)/(rp-rminus) * Πhm)) + 2.0 * L * γθ / (π * a * sqrt((1.0-E^2)*zp)) * Πzm   # Eq. F8
    γt = 4.0 * E + 2.0a * E * sqrt(zp) / (π * sqrt(1.0-E^2)) * (K_kθ-E_kθ) * γθ + 2.0γr / (π * sqrt((1.0-E^2) * (ra-r3) * (rp-r4))) * (
        0.5E * ((r3 * (ra+rp+r3) - ra * rp) * K_kr + (rp-r3) * (ra+rp+r3+r4) * Πhr + (ra-r3) * (rp-r4) * E_kr) + 2.0 * E * (r3 * K_kr + (rp-r3) * Πhr)+
        2.0 / (rplus-rminus) * (((4.0 * E-a*L) * rplus - 2.0 * a^2 * E)/(r3-rplus) * (K_kr - (rp-r3)/(rp-rplus) * Πhp) - 
        ((4.0 * E-a*L) * rminus - 2.0 * a^2 * E)/(r3-rminus) * (K_kr - (rp-r3)/(rp-rminus) * Πhm)))

    # special cases in which the frequencies are infinite
    if e == 0.0 && θmin == π/2   # circular equatorial
        γr = 1e12; γθ =1e12;
    elseif e == 0.0   # circular non-equatorial
        γr = 1e12;
    elseif θmin == π/2   # non-circular equatorial
        γθ = 1e12;
    end

    return [γr, γθ, γϕ, γt]
end

# calculate dimensionless E, L, Q, as per Schmidt (arXiv:gr-qc/0202090)
function SchmidtELQ(a::Float64, p::Float64, e::Float64, θmin::Float64)
    # define turning points rp, ra
    rp = p / (1 + e)
    ra  = p / (1 - e)

    Δ(r) = r^2 - 2r + a^2
    zm = cos(θmin) # z_{-}
    
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
    En = sqrt((κ*ρ + 2ε*σ - 2.0*sqrt(σ * (-η * κ^2 + ε * κ * ρ + σ *ε^2))) / (ρ^2 + 4η * σ)) # Eq. 2.32

    L = sqrt((ε - η * En^2) / σ) # Eq. 2.34

    Q = (zm^2) * ((a^2) * (1 - En^2) + (L^2) / (1 - zm^2)) # Eq. 2.25

    return En, L, Q
end

# calculates dimensionless kerr fundamental frequencies wrt proper time and the conversion factor to boyer-lindquist as per Schmidt (arXiv:gr-qc/0202090)
function SchmidtKerrFreqs(a::Float64, p::Float64, e::Float64, θmin::Float64)
    # constants of motion
    En, L, Q = SchmidtELQ(a, p, e, θmin)

    zm = cos(θmin)
    zp = sqrt(((1)/((2a^2) * (1 - En^2))) * ((a^2) * (1 - En^2) + L^2 + Q + sqrt((4a^2) * (-1 + En^2) * Q + ((-a^2) * (-1 + En^2) + L^2 + Q)^2)))
    k = (zm^2)/(zp^2)

    # define functions J, H, G, F for ra(p, ed, θmin, a)ial integra(p, el, θmin, a) computation
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