#=

    In this module we write functions which compute the radiaction reaction potentials and their derivatives. See Eqs. 44-45 in arXiv:1109.0572v2. The sums in these expressions (and their respective derivatives) have been expanded
    and copied from a mathematica notebook.

=#

module RRPotentials

"""
# Common Arguments in this module
- `xH::AbstractVector{Float64}`: Harmonic coordinates, xH = [x, y, z].
- `Mij5::AbstractMatrix{Float64}`: The fifth derivative of the mass quadrupole, defined in Eq. 48.
- `Mij6::AbstractMatrix{Float64}`: The sixth derivative of the mass quadrupole.
- `Mij7::AbstractMatrix{Float64}`: The seventh derivative of the mass quadrupole.
- `Mij8::AbstractMatrix{Float64}`: The eighth derivative of the mass quadrupole.
- `Mijk7::AbstractMatrix{Float64}`: The seventh derivative of the mass octupole, defined in Eq. 48.
- `Mijk8::AbstractMatrix{Float64}`: The eighth derivative of the mass octupole.
- `Sij5::AbstractMatrix{Float64}`: The fifth derivative of the current quadrupole, defined in Eq. 49.
- `Sij6::AbstractMatrix{Float64}`: The sixth derivative of the current quadrupole.

# Notes
- The derivatives of the multipole moments are computed numerically, either using finite differences or Fourier-fits.
"""

δ(x::Int, y::Int)::Int = x == y ? 1 : 0

const levi_civita_table = Dict(
    (1, 2, 3) => 1,
    (2, 3, 1) => 1,
    (3, 1, 2) => 1,
    (3, 2, 1) => -1,
    (2, 1, 3) => -1,
    (1, 3, 2) => -1
)

function εijk(i::Int, j::Int, k::Int)::Int
    return get(levi_civita_table, (i, j, k), 0)
end

function Vrr(xH::AbstractArray, Mij5::AbstractArray, Mij7::AbstractArray, Mijk7::AbstractArray)    # Eq. 44
    return (-(xH[1]^2*Mij5[1,1]) - xH[1]*(xH[2]*(Mij5[1,2] + Mij5[2,1]) + xH[3]*(Mij5[1,3] + Mij5[3,1])) - 
    xH[2]*(xH[2]*Mij5[2,2] + xH[3]*(Mij5[2,3] + Mij5[3,2])) - xH[3]^2*Mij5[3,3])/5. - 
    ((xH[1]^2 + xH[2]^2 + xH[3]^2)*(xH[1]^2*Mij7[1,1] + xH[2]^2*Mij7[2,2] + 
    xH[1]*(xH[2]*(Mij7[1,2] + Mij7[2,1]) + xH[3]*(Mij7[1,3] + Mij7[3,1])) + xH[2]*xH[3]*(Mij7[2,3] + Mij7[3,2]) + 
    xH[3]^2*Mij7[3,3]))/70. + (xH[1]^3*Mijk7[1,1,1] + xH[2]^3*Mijk7[2,2,2] + 
    xH[1]^2*(xH[2]*(Mijk7[1,1,2] + Mijk7[1,2,1] + Mijk7[2,1,1]) + xH[3]*(Mijk7[1,1,3] + Mijk7[1,3,1] + Mijk7[3,1,1])) + 
    xH[2]^2*xH[3]*(Mijk7[2,2,3] + Mijk7[2,3,2] + Mijk7[3,2,2]) + 
    xH[1]*(xH[2]^2*(Mijk7[1,2,2] + Mijk7[2,1,2] + Mijk7[2,2,1]) + 
    xH[2]*xH[3]*(Mijk7[1,2,3] + Mijk7[1,3,2] + Mijk7[2,1,3] + Mijk7[2,3,1] + Mijk7[3,1,2] + Mijk7[3,2,1]) + 
    xH[3]^2*(Mijk7[1,3,3] + Mijk7[3,1,3] + Mijk7[3,3,1])) + xH[2]*xH[3]^2*(Mijk7[2,3,3] + Mijk7[3,2,3] + Mijk7[3,3,2]) + 
    xH[3]^3*Mijk7[3,3,3])/189.
end

function ∂Vrr_∂t(xH::AbstractArray, Mij6::AbstractArray, Mij8::AbstractArray, Mijk8::AbstractArray)    # Eq. 7.25
    return (-(xH[1]^2*Mij6[1,1]) - xH[1]*(xH[2]*(Mij6[1,2] + Mij6[2,1]) + xH[3]*(Mij6[1,3] + Mij6[3,1])) - 
    xH[2]*(xH[2]*Mij6[2,2] + xH[3]*(Mij6[2,3] + Mij6[3,2])) - xH[3]^2*Mij6[3,3])/5. - 
    ((xH[1]^2 + xH[2]^2 + xH[3]^2)*(xH[1]^2*Mij8[1,1] + xH[2]^2*Mij8[2,2] + 
    xH[1]*(xH[2]*(Mij8[1,2] + Mij8[2,1]) + xH[3]*(Mij8[1,3] + Mij8[3,1])) + xH[2]*xH[3]*(Mij8[2,3] + Mij8[3,2]) + 
    xH[3]^2*Mij8[3,3]))/70. + (xH[1]^3*Mijk8[1,1,1] + xH[2]^3*Mijk8[2,2,2] + 
    xH[1]^2*(xH[2]*(Mijk8[1,1,2] + Mijk8[1,2,1] + Mijk8[2,1,1]) + xH[3]*(Mijk8[1,1,3] + Mijk8[1,3,1] + Mijk8[3,1,1])) + 
    xH[2]^2*xH[3]*(Mijk8[2,2,3] + Mijk8[2,3,2] + Mijk8[3,2,2]) + 
    xH[1]*(xH[2]^2*(Mijk8[1,2,2] + Mijk8[2,1,2] + Mijk8[2,2,1]) + 
    xH[2]*xH[3]*(Mijk8[1,2,3] + Mijk8[1,3,2] + Mijk8[2,1,3] + Mijk8[2,3,1] + Mijk8[3,1,2] + Mijk8[3,2,1]) + 
    xH[3]^2*(Mijk8[1,3,3] + Mijk8[3,1,3] + Mijk8[3,3,1])) + xH[2]*xH[3]^2*(Mijk8[2,3,3] + Mijk8[3,2,3] + Mijk8[3,3,2]) + 
    xH[3]^3*Mijk8[3,3,3])/189.
end

function ∂Vrr_∂a(xH::AbstractArray, Mij5::AbstractArray, Mij7::AbstractArray, Mijk7::AbstractArray, a::Int)    # Eq. 7.30
    return (-(δ(1,a)*(2*xH[1]*Mij5[1,1] + xH[2]*(Mij5[1,2] + Mij5[2,1]) + xH[3]*(Mij5[1,3] + Mij5[3,1]))) - 
    δ(2,a)*(xH[1]*(Mij5[1,2] + Mij5[2,1]) + 2*xH[2]*Mij5[2,2] + xH[3]*(Mij5[2,3] + Mij5[3,2])) - 
    δ(3,a)*(xH[1]*(Mij5[1,3] + Mij5[3,1]) + xH[2]*(Mij5[2,3] + Mij5[3,2]) + 2*xH[3]*Mij5[3,3]))/5. - 
    ((δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3])*
    (xH[1]^2*Mij7[1,1] + xH[2]^2*Mij7[2,2] + xH[1]*
    (xH[2]*(Mij7[1,2] + Mij7[2,1]) + xH[3]*(Mij7[1,3] + Mij7[3,1])) + xH[2]*xH[3]*(Mij7[2,3] + Mij7[3,2]) + 
    xH[3]^2*Mij7[3,3]))/35. - ((xH[1]^2 + xH[2]^2 + xH[3]^2)*
    (δ(1,a)*(2*xH[1]*Mij7[1,1] + xH[2]*(Mij7[1,2] + Mij7[2,1]) + xH[3]*(Mij7[1,3] + Mij7[3,1])) + 
    δ(2,a)*(xH[1]*(Mij7[1,2] + Mij7[2,1]) + 2*xH[2]*Mij7[2,2] + xH[3]*(Mij7[2,3] + Mij7[3,2])) + 
    δ(3,a)*(xH[1]*(Mij7[1,3] + Mij7[3,1]) + xH[2]*(Mij7[2,3] + Mij7[3,2]) + 2*xH[3]*Mij7[3,3])))/70. + 
    (δ(1,a)*(3*xH[1]^2*Mijk7[1,1,1] + xH[2]^2*(Mijk7[1,2,2] + Mijk7[2,1,2] + Mijk7[2,2,1]) + 
    2*xH[1]*(xH[2]*(Mijk7[1,1,2] + Mijk7[1,2,1] + Mijk7[2,1,1]) + xH[3]*(Mijk7[1,1,3] + Mijk7[1,3,1] + Mijk7[3,1,1])) + 
    xH[2]*xH[3]*(Mijk7[1,2,3] + Mijk7[1,3,2] + Mijk7[2,1,3] + Mijk7[2,3,1] + Mijk7[3,1,2] + Mijk7[3,2,1]) + 
    xH[3]^2*(Mijk7[1,3,3] + Mijk7[3,1,3] + Mijk7[3,3,1])) + 
    δ(2,a)*(xH[1]^2*(Mijk7[1,1,2] + Mijk7[1,2,1] + Mijk7[2,1,1]) + 3*xH[2]^2*Mijk7[2,2,2] + 
    xH[1]*(2*xH[2]*(Mijk7[1,2,2] + Mijk7[2,1,2] + Mijk7[2,2,1]) + 
    xH[3]*(Mijk7[1,2,3] + Mijk7[1,3,2] + Mijk7[2,1,3] + Mijk7[2,3,1] + Mijk7[3,1,2] + Mijk7[3,2,1])) + 
    2*xH[2]*xH[3]*(Mijk7[2,2,3] + Mijk7[2,3,2] + Mijk7[3,2,2]) + xH[3]^2*(Mijk7[2,3,3] + Mijk7[3,2,3] + Mijk7[3,3,2])) + 
    δ(3,a)*(xH[1]^2*(Mijk7[1,1,3] + Mijk7[1,3,1] + Mijk7[3,1,1]) + 
    xH[2]^2*(Mijk7[2,2,3] + Mijk7[2,3,2] + Mijk7[3,2,2]) + 
    xH[1]*(xH[2]*(Mijk7[1,2,3] + Mijk7[1,3,2] + Mijk7[2,1,3] + Mijk7[2,3,1] + Mijk7[3,1,2] + Mijk7[3,2,1]) + 
    2*xH[3]*(Mijk7[1,3,3] + Mijk7[3,1,3] + Mijk7[3,3,1])) + 2*xH[2]*xH[3]*(Mijk7[2,3,3] + Mijk7[3,2,3] + Mijk7[3,3,2]) + 
    3*xH[3]^2*Mijk7[3,3,3]))/189.
end

function Virr(xH::AbstractArray, Mij6::AbstractArray, Sij5::AbstractArray)   # Eq. 45
    V = [0., 0., 0.]  
    @inbounds for i=1:3
        V[i] = (-4*(εijk(i,2,1)*xH[2] + εijk(i,3,1)*xH[3])*(xH[1]*Sij5[1,1] + xH[2]*Sij5[1,2] + xH[3]*Sij5[1,3]) - 4*xH[1]*(εijk(i,1,2)*xH[1] + εijk(i,3,2)*xH[3])*Sij5[2,1] - 
        4*xH[2]*(εijk(i,1,2)*xH[1] + εijk(i,3,2)*xH[3])*Sij5[2,2] - 4*xH[3]*(εijk(i,1,2)*xH[1] + εijk(i,3,2)*xH[3])*Sij5[2,3] - 4*xH[1]*(εijk(i,1,3)*xH[1] + εijk(i,2,3)*xH[2])*Sij5[3,1] - 
        4*xH[2]*(εijk(i,1,3)*xH[1] + εijk(i,2,3)*xH[2])*Sij5[3,2] - 4*(εijk(i,1,3)*xH[1] + εijk(i,2,3)*xH[2])*xH[3]*Sij5[3,3])/45. + 
        ((xH[1]^2*xH[i] - ((xH[1]^2 + xH[2]^2 + xH[3]^2)*(2*δ(1,i)*xH[1] + xH[i]))/5.)*Mij6[1,1] + 
        (-0.2*((δ(2,i)*xH[1] + δ(1,i)*xH[2])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[1]*xH[2]*xH[i])*Mij6[1,2] + 
        (-0.2*((δ(3,i)*xH[1] + δ(1,i)*xH[3])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[1]*xH[3]*xH[i])*Mij6[1,3] + 
        (-0.2*((δ(2,i)*xH[1] + δ(1,i)*xH[2])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[1]*xH[2]*xH[i])*Mij6[2,1] + 
        (xH[2]^2*xH[i] - ((xH[1]^2 + xH[2]^2 + xH[3]^2)*(2*δ(2,i)*xH[2] + xH[i]))/5.)*Mij6[2,2] + 
        (-0.2*((δ(3,i)*xH[2] + δ(2,i)*xH[3])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[2]*xH[3]*xH[i])*Mij6[2,3] + 
        (-0.2*((δ(3,i)*xH[1] + δ(1,i)*xH[3])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[1]*xH[3]*xH[i])*Mij6[3,1] + 
        (-0.2*((δ(3,i)*xH[2] + δ(2,i)*xH[3])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[2]*xH[3]*xH[i])*Mij6[3,2] + 
        (xH[3]^2*xH[i] - ((xH[1]^2 + xH[2]^2 + xH[3]^2)*(2*δ(3,i)*xH[3] + xH[i]))/5.)*Mij6[3,3])/21.
    end
    return V
end

function ∂Virr_∂t(xH::AbstractArray, Mij7::AbstractArray, Sij6::AbstractArray, i::Int)   # Eq. 7.26
    return (-4*(εijk(i,2,1)*xH[2] + εijk(i,3,1)*xH[3])*(xH[1]*Sij6[1,1] + xH[2]*Sij6[1,2] + xH[3]*Sij6[1,3]) - 4*xH[1]*(εijk(i,1,2)*xH[1] + εijk(i,3,2)*xH[3])*Sij6[2,1] - 
    4*xH[2]*(εijk(i,1,2)*xH[1] + εijk(i,3,2)*xH[3])*Sij6[2,2] - 4*xH[3]*(εijk(i,1,2)*xH[1] + εijk(i,3,2)*xH[3])*Sij6[2,3] - 4*xH[1]*(εijk(i,1,3)*xH[1] + εijk(i,2,3)*xH[2])*Sij6[3,1] - 
    4*xH[2]*(εijk(i,1,3)*xH[1] + εijk(i,2,3)*xH[2])*Sij6[3,2] - 4*(εijk(i,1,3)*xH[1] + εijk(i,2,3)*xH[2])*xH[3]*Sij6[3,3])/45. + 
    ((xH[1]^2*xH[i] - ((xH[1]^2 + xH[2]^2 + xH[3]^2)*(2*δ(1,i)*xH[1] + xH[i]))/5.)*Mij7[1,1] + 
    (-0.2*((δ(2,i)*xH[1] + δ(1,i)*xH[2])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[1]*xH[2]*xH[i])*Mij7[1,2] + 
    (-0.2*((δ(3,i)*xH[1] + δ(1,i)*xH[3])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[1]*xH[3]*xH[i])*Mij7[1,3] + 
    (-0.2*((δ(2,i)*xH[1] + δ(1,i)*xH[2])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[1]*xH[2]*xH[i])*Mij7[2,1] + 
    (xH[2]^2*xH[i] - ((xH[1]^2 + xH[2]^2 + xH[3]^2)*(2*δ(2,i)*xH[2] + xH[i]))/5.)*Mij7[2,2] + 
    (-0.2*((δ(3,i)*xH[2] + δ(2,i)*xH[3])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[2]*xH[3]*xH[i])*Mij7[2,3] + 
    (-0.2*((δ(3,i)*xH[1] + δ(1,i)*xH[3])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[1]*xH[3]*xH[i])*Mij7[3,1] + 
    (-0.2*((δ(3,i)*xH[2] + δ(2,i)*xH[3])*(xH[1]^2 + xH[2]^2 + xH[3]^2)) + xH[2]*xH[3]*xH[i])*Mij7[3,2] + 
    (xH[3]^2*xH[i] - ((xH[1]^2 + xH[2]^2 + xH[3]^2)*(2*δ(3,i)*xH[3] + xH[i]))/5.)*Mij7[3,3])/21.
end

function ∂Virr_∂a(xH::AbstractArray, Mij6::AbstractArray, Sij5::AbstractArray, i::Int, a::Int)   # Eq. 45
    return (-4*(δ(1,a)*(2*xH[1]*(εijk(i,1,2)*Sij5[2,1] + εijk(i,1,3)*Sij5[3,1]) + 
    xH[2]*(εijk(i,2,1)*Sij5[1,1] + εijk(i,1,2)*Sij5[2,2] + εijk(i,2,3)*Sij5[3,1] + εijk(i,1,3)*Sij5[3,2]) + 
    xH[3]*(εijk(i,3,1)*Sij5[1,1] + εijk(i,3,2)*Sij5[2,1] + εijk(i,1,2)*Sij5[2,3] + εijk(i,1,3)*Sij5[3,3])) + 
    δ(3,a)*(2*xH[3]*(εijk(i,3,1)*Sij5[1,3] + εijk(i,3,2)*Sij5[2,3]) + 
    xH[1]*(εijk(i,3,1)*Sij5[1,1] + εijk(i,3,2)*Sij5[2,1] + εijk(i,1,2)*Sij5[2,3] + εijk(i,1,3)*Sij5[3,3]) + 
    xH[2]*(εijk(i,3,1)*Sij5[1,2] + εijk(i,2,1)*Sij5[1,3] + εijk(i,3,2)*Sij5[2,2] + εijk(i,2,3)*Sij5[3,3])) + 
    δ(2,a)*(xH[1]*(εijk(i,2,1)*Sij5[1,1] + εijk(i,1,2)*Sij5[2,2] + εijk(i,2,3)*Sij5[3,1] + εijk(i,1,3)*Sij5[3,2]) + 
    2*xH[2]*(εijk(i,2,1)*Sij5[1,2] + εijk(i,2,3)*Sij5[3,2]) + 
    xH[3]*(εijk(i,3,1)*Sij5[1,2] + εijk(i,2,1)*Sij5[1,3] + εijk(i,3,2)*Sij5[2,2] + εijk(i,2,3)*Sij5[3,3]))))/45. + 
    ((δ(i,a)*xH[1]^2 - ((2*δ(1,i)*δ(1,a) + δ(i,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + 2*δ(1,a)*xH[1]*xH[i] - 
    (2*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3])*(2*δ(1,i)*xH[1] + xH[i]))/5.)*Mij6[1,1] + 
    (δ(i,a)*xH[1]*xH[2] - (2*(δ(2,i)*xH[1] + δ(1,i)*xH[2])*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3]))/
    5. - ((δ(1,a)*δ(2,i) + δ(1,i)*δ(2,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + δ(2,a)*xH[1]*xH[i] + 
    δ(1,a)*xH[2]*xH[i])*Mij6[1,2] + (δ(i,a)*xH[1]*xH[3] - 
    (2*(δ(3,i)*xH[1] + δ(1,i)*xH[3])*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3]))/5. - 
    ((δ(1,a)*δ(3,i) + δ(1,i)*δ(3,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + δ(3,a)*xH[1]*xH[i] + 
    δ(1,a)*xH[3]*xH[i])*Mij6[1,3] + (δ(i,a)*xH[1]*xH[2] - 
    (2*(δ(2,i)*xH[1] + δ(1,i)*xH[2])*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3]))/5. - 
    ((δ(1,a)*δ(2,i) + δ(1,i)*δ(2,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + δ(2,a)*xH[1]*xH[i] + 
    δ(1,a)*xH[2]*xH[i])*Mij6[2,1] + (δ(i,a)*xH[2]^2 - 
    ((2*δ(2,i)*δ(2,a) + δ(i,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + 2*δ(2,a)*xH[2]*xH[i] - 
    (2*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3])*(2*δ(2,i)*xH[2] + xH[i]))/5.)*Mij6[2,2] + 
    (δ(i,a)*xH[2]*xH[3] - (2*(δ(3,i)*xH[2] + δ(2,i)*xH[3])*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3]))/
    5. - ((δ(2,a)*δ(3,i) + δ(2,i)*δ(3,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + δ(3,a)*xH[2]*xH[i] + 
    δ(2,a)*xH[3]*xH[i])*Mij6[2,3] + (δ(i,a)*xH[1]*xH[3] - 
    (2*(δ(3,i)*xH[1] + δ(1,i)*xH[3])*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3]))/5. - 
    ((δ(1,a)*δ(3,i) + δ(1,i)*δ(3,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + δ(3,a)*xH[1]*xH[i] + 
    δ(1,a)*xH[3]*xH[i])*Mij6[3,1] + (δ(i,a)*xH[2]*xH[3] - 
    (2*(δ(3,i)*xH[2] + δ(2,i)*xH[3])*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3]))/5. - 
    ((δ(2,a)*δ(3,i) + δ(2,i)*δ(3,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + δ(3,a)*xH[2]*xH[i] + 
    δ(2,a)*xH[3]*xH[i])*Mij6[3,2] + (δ(i,a)*xH[3]^2 - 
    ((2*δ(3,i)*δ(3,a) + δ(i,a))*(xH[1]^2 + xH[2]^2 + xH[3]^2))/5. + 2*δ(3,a)*xH[3]*xH[i] - 
    (2*(δ(1,a)*xH[1] + δ(2,a)*xH[2] + δ(3,a)*xH[3])*(2*δ(3,i)*xH[3] + xH[i]))/5.)*Mij6[3,3])/21.
end

end