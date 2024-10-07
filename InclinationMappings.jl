#=

    In this module we provide mappings from the common inclination angles x_{I} (see e.g., Eqs. 1.2-1.3 of arXiv:2401.09577v2) and ι (see e.g., Eq. 25 of arXiv:1109.0572v2) to θmin, which is the angle used in most functions written in this package. Any
    equation references without arxiv numbers are from arXiv:2401.09577v2.

=#

module InclinationMappings
using Roots
using ..ConstantsOfMotion

"""
# Common Arguments in this module
- `a::Float64`: Kerr black hole spin parameter, 0 < a < 1.
- `p::Float64`: semi-latus rectum of the orbit (defined by, e.g., Eq. 23).
- `e::Float64`: eccentricity of the orbit (defined by, e.g., Eq. 23).
- `θmin::Float64`: minimum polar angle of the orbit (radians).
- `ι::Float64`: inclination of the orbit (degrees) as defined in Eq. 25.
- `I::Float64`: inclination of the orbit (degrees) as defined in Eq. 1.2 of arXiv:2401.09577v2.
- `xI::Float64`: inclination defined by sign(Lz)*sin(θmin) as defined in Eq. 1.3 of arXiv:2401.09577v2.
- `E::Float64`: energy per unit mass of the test particle moving along the geodesic (Eq. 14).
- `L::Float64`: axial (i.e., z-component of the) angular momentum per unit mass of the test particle moving along the geodesic (Eq. 15).
- `sign_Lz::Int64`: sign of the z-component of the angular momentum (Lz) of the orbiting particle.
- `C::Float64`: Carter constant of the orbit---note that this C is what is commonly referred to as 'Q' elsewhere (Eq. 17).
- `Q::Float64`: Alternative definition of the Carter constant (Eq. 16).
"""

# compute ι from cos(ι) = L_{z} / sqrt(L_{z}^2 + C)
function compute_iota(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64)::Float64
    E, L, Q, C = ConstantsOfMotion.compute_ELC(a, p, e, θmin, sign_Lz)
    return acos(L / sqrt(L^2 + C))
end

# compute θmin corresponding to iota---note iota is in degrees and θmin will be in radians
function iota_to_theta_min(a::Float64, p::Float64, e::Float64, ι::Float64)
    ι = deg2rad(ι)
    if ι < 0. || ι > π
        throw(DomainError("ι must be in the range [0, 180]"))
    else
        sign_Lz = ι < π/2 ? +1 : -1;
        iota_theta(θmin::Float64)::Float64 = compute_iota(a, p, e, θmin, sign_Lz) - ι
        θmin = find_zeros(iota_theta, 0.001, π/2-0.001)
    end
    return length(θmin) == 1 ? θmin[1] : throw(DomainError("Congratulations! This has never happened before. You managed to break the code! There were $(length(θmin)) roots found for θmin"))
end

# Eq. 1.2 of arXiv:2401.09577v2---note that I is in degrees and θmin will be in radians
function I_to_theta_min(I::Float64, sign_Lz::Int64)::Float64
    I = deg2rad(I)
    if I < 0. || I > π
        throw(DomainError("I must be in the range [0, 180]"))
    end
    return sign_Lz * (π/2 - I)
end

# Eq. 1.3 of arXiv:2401.09577v2
function xI_to_theta_min(xI::Float64, sign_Lz::Int64)::Float64
    if abs(xI) > 1
        throw(DomainError("abs(xI) must be less than unity"))
    end
    return asin(xI * sign_Lz)
end

end