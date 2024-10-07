#=

    In this module we use the method outlined in Hughes 2024 (arXiv:2401.09577v2) for converting (E, L, Q) -> (p, e, θmin).

=#

module HughesOrbitalConstants

"""
# Common Arguments in this module
- `a::Float64`: Kerr black hole spin parameter, 0 < a < 1.
- `p::Float64`: semi-latus rectum of the orbit.
- `e::Float64`: eccentricity of the orbit.
- `θmin::Float64`: minimum polar angle of the orbit.
- `E::Float64`: energy per unit mass of the test particle moving along the geodesic.
- `L::Float64`: axial (i.e., z-component of the) angular momentum per unit mass of the test particle moving along the geodesic.
- `Q::Float64`: Carter constant.
- `ra::Float64`: apastron of the orbit (furtherst radial turning point.
- `rp::Float64`: periastron of the orbit (closest radial turning point.
"""


# semi-latus rectum and eccentricity
p(ra::Float64, rp::Float64)::Float64 = 2.0ra * rp / (ra + rp)
e(ra::Float64, rp::Float64)::Float64 = (ra - rp) / (ra + rp)

# auxiliary terms in Eq. 2.7
aux1(a::Float64, E::Float64, L::Float64, Q::Float64)::Float64 = Q + L^2 - a^2 * (1.0 - E^2)
aux2(a::Float64, E::Float64, L::Float64)::Float64 = 4a^2 * L^2 * (1.0 - E^2)

# xI ≡ cos(I) = sgn(Lz)sin(θmin) Eq. 1.3; 
xI(a::Float64, E::Float64, L::Float64, Q::Float64)::Float64 = Q == 0.0 ? sign(L) : sqrt(2) * L / sqrt(aux1(a, E, L, Q) + sqrt((aux1(a, E, L, Q))^2 + aux2(a, E, L)))   # Eq. 2.7

θmin(xI::Float64, L::Float64)::Float64 = asin(sign(L) * xI)   # inverting Eq. 1.3

function pe_equatorial(a::Float64, E::Float64, L::Float64)
    E2minus1 = E^2 - 1.0

    A0 = 2.0 * (a * E - L)^2 / E2minus1   # Eq. 3.6
    A1 = (a^2 * E2minus1 - L^2) / E2minus1   # Eq. 3.5
    A2 = 2.0 / E2minus1   # Eq. 3.4

    Q = (A2^2 - 3.0A1) / 9.0   # Eq. 3.7
    sqrtQ = sqrt(Q)
    R = (2.0A2^3 - 9.0A2 * A1 + 27.0A0) / 54.0   # Eq. 3.8
    θ = acos(R / (sqrtQ^3))   # Eq. 3.9

    ra = -2.0sqrtQ * cos((θ + 2π) / 3.0) - A2 / 3.0   # Eq. 3.10
    rp = -2.0sqrtQ * cos((θ - 2π) / 3.0) - A2 / 3.0   # Eq. 3.11
    r3 = -2.0sqrtQ * cos(θ / 3.0) - A2 / 3.0   # Eq. 3.12

    return ra, rp, r3
end

function pe_generic(a::Float64, E::Float64, L::Float64, Q::Float64)
    E2minus1 = E^2 - 1.0
    A0 = -a^2 * Q / E2minus1   # Eq. 4.5
    A1 = 2.0 * (Q + (a * E - L)^2) / E2minus1   # Eq. 4.4
    A2 = (a^2 * E2minus1 - L^2 - Q) / E2minus1  # Eq. 4.3
    A3 = 2.0 / E2minus1   # Eq. 4.2

    B0 = A0 - A1 * A3 / 4.0 + A2 * A3^2 / 16.0 - 3.0A3^4 / 256.0   # Eq. 4.8
    B1 = A1 - A2 * A3 / 2.0 + A3^3 / 8.0   # Eq. 4.7
    B2 = A2 - 3.0A3^2 / 8.0   # Eq. 4.6

    C0 = -B1^2 / 64.0   # Eq. 4.10
    C1 = B2^2 / 16.0 - B0 / 4.0   # Eq. 4.10
    C2 = B2 / 2.0   # Eq. 4.10

    Qrc = (C2^2 - 3.0C1) / 9.0   # Eq. 4.11
    sqrtQrc = sqrt(Qrc)
    Rrc = (2.0C2^3 - 9.0 * C2 * C1 + 27.0 * C0) / 54.0   # Eq. 4.12
    θrc = acos(Rrc / sqrtQrc^3)   # Eq. 4.13

    zrc1 = -2.0 * sqrtQrc * cos((θrc + 2π) / 3.0) - C2 / 3.0   # Eq. 4.14
    zrc2 = -2.0 * sqrtQrc * cos((θrc - 2π) / 3.0) - C2 / 3.0   # Eq. 4.15
    zrc3 = -2.0 * sqrtQrc * cos(θrc / 3.0) - C2 / 3.0   # Eq. 4.16

    sqrtzrc1 = sqrt(zrc1)
    aux = sqrt(zrc2 + zrc3 - 2.0 * sign(B1) * sqrt(zrc2 * zrc3))   # auxiliary square root term in Eqs. 4.17 - 4.20

    ra = -1.0 / (2.0 * E2minus1) + sqrtzrc1 + aux
    rp = -1.0 / (2.0 * E2minus1) + sqrtzrc1 - aux
    r3 = -1.0 / (2.0 * E2minus1) - sqrtzrc1 + aux
    r4 = -1.0 / (2.0 * E2minus1) - sqrtzrc1 - aux

    return ra, rp, r3, r4
end

function compute_p_e_θmin(a::Float64, E::Float64, L::Float64, Q::Float64, C::Float64)
    # equatorial
    if C==0
        ra, rp, r3 = pe_equatorial(a, E, L)
        r4 = 0.0
    else
        ra, rp, r3, r4 = pe_generic(a, E, L, C)
    end

    return p(ra, rp), e(ra, rp), θmin(xI(a, E, L, C), L)
end

end