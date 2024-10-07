#=

    This module contains code which evolves the first order geodesic equations in Kerr (i.e., the separated Hamilton Jacobi equations) in coordinate time. The particular formulation of the equations follows the notation in Sec. IVA in Sopuerta,
    Yunes, 2011 (arXiv:1109.0572v2), which, in turn, is derived from Drasco & Hughes, 2004 (arXiv:astro-ph/0308479v3). Throughout, Eq. X will refer to expressions in the former reference. Note that throughout, we assume M=1.

=#

module BLTimeGeodesics
using ..Kerr
using ..ConstantsOfMotion
using StaticArrays
using DifferentialEquations
using HDF5

"""
# Common Arguments in this module
- `r::Float64`: Boyer-Lindquist radial coordinate.
- `θ::Float64`: Boyer-Lindquist polar coordinate.
- `ϕ::Float64`: Boyer-Lindquist azimuthal coordinate.
- `drdt::Float64`: Coordinate-time first derivative of the radial coordinate.
- `dθdt::Float64`: Coordinate-time first derivative of the polar coordinate.
- `dϕdt::Float64`: Coordinate-time first derivative of the azimuthal coordinate.
- `psi (ψ)::Float64`: angle variable associated with r, defined in Eq. 89, which is used to evolve the geodesic ODEs.
- `chi (χ)::Float64`: angle variable associated with θ, defined in Eq. 89, which is used to evolve the geodesic ODEs.
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
- `p3::Float64`: transformation one of the roots of the radial function R(r) (Eq. 90), e.g., p3 = r3 * (1 - e) / M.
- `p4::Float64`: transformation one of the roots of the radial function R(r) (Eq. 90), e.g., p4 = r4 * (1 - e) / M.
- `zp::Float64`: root the the theta geodesic equation (Eq. 91-92)
- `zm::Float64`: root the the theta geodesic equation (Eq. 91-92)
"""

Δ(r::Float64, a::Float64)::Float64 = r^2 - 2.0 * r + a^2

# define second derivatives of BL coordinates wrt t from the second-order geodesic equations
@inline dr2_dt2(r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64)::Float64 = -(drdt)^2 * Kerr.KerrMetric.Γrrr(r, θ, ϕ, a) - 2.0*drdt*dθdt*Kerr.KerrMetric.Γrrθ(r, θ, ϕ, a)-Kerr.KerrMetric.Γrtt(r, θ, ϕ, a)-
dϕdt*(2.0Kerr.KerrMetric.Γrtϕ(r, θ, ϕ, a) + dϕdt*Kerr.KerrMetric.Γrϕϕ(r, θ, ϕ, a))+2.0drdt^2*(dϕdt * Kerr.KerrMetric.Γtrϕ(r, θ, ϕ, a)+Kerr.KerrMetric.Γttr(r, θ, ϕ, a)) + dθdt*(-dθdt*Kerr.KerrMetric.Γrθθ(r, θ, ϕ, a)+
2.0drdt*(Kerr.KerrMetric.Γttθ(r, θ, ϕ, a)+dϕdt*Kerr.KerrMetric.Γtθϕ(r, θ, ϕ, a)))


@inline dθ2_dt2(r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64)::Float64 = 2.0drdt * dθdt * dϕdt * Kerr.KerrMetric.Γtrϕ(r, θ, ϕ, a)+2.0drdt * dθdt * Kerr.KerrMetric.Γttr(r, θ, ϕ, a)+
2.0dθdt^2*Kerr.KerrMetric.Γttθ(r, θ, ϕ, a)+2.0dθdt^2*dϕdt * Kerr.KerrMetric.Γtθϕ(r, θ, ϕ, a)-drdt^2*Kerr.KerrMetric.Γθrr(r, θ, ϕ, a)-2.0drdt*dθdt*Kerr.KerrMetric.Γθrθ(r, θ, ϕ, a)-Kerr.KerrMetric.Γθtt(r, θ, ϕ, a)-
2.0dϕdt*Kerr.KerrMetric.Γθtϕ(r, θ, ϕ, a)-dθdt^2*Kerr.KerrMetric.Γθθθ(r, θ, ϕ, a)-dϕdt^2*Kerr.KerrMetric.Γθϕϕ(r, θ, ϕ, a)

@inline dϕ2_dt2(r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64)::Float64 = 2.0*(drdt*dϕdt^2*Kerr.KerrMetric.Γtrϕ(r, θ, ϕ, a)+
drdt*dϕdt*Kerr.KerrMetric.Γttr(r, θ, ϕ, a)+dθdt*dϕdt*Kerr.KerrMetric.Γttθ(r, θ, ϕ, a)+dθdt*dϕdt^2*Kerr.KerrMetric.Γtθϕ(r, θ, ϕ, a)-drdt*dϕdt*Kerr.KerrMetric.Γϕrϕ(r, θ, ϕ, a)-drdt*Kerr.KerrMetric.Γϕtr(r, θ, ϕ, a)-
dθdt*Kerr.KerrMetric.Γϕtθ(r, θ, ϕ, a)-dθdt*dϕdt*Kerr.KerrMetric.Γϕθϕ(r, θ, ϕ, a))

# conversion between angle ψ and r (Eq. 89)
@inline r(psi::Float64, p::Float64, e::Float64)::Float64 = p / (1.0 + e * cos(psi))
@inline dr_dpsi(psi::Float64, p::Float64, e::Float64)::Float64 = p * e * sin(psi)/ ((1.0 + e * cos(psi))^2)
@inline dr_dt(dpsi_dt::Float64, psi::Float64, p::Float64, e::Float64)::Float64 = dr_dpsi(psi, p, e) * dpsi_dt

# conversion between angle χ and z=cos(θ)^2 (Eq. 89, 92)
@inline z(χ::Float64, θmin::Float64)::Float64 = cos(θmin)^2 * cos(χ)^2
@inline dθ_dchi(χ::Float64, θ::Float64, θmin::Float64)::Float64 = cos(θmin)^2 * sin(2.0χ) / sin(2.0θ)
@inline dθ_dt(dchi_dt::Float64, χ::Float64, θ::Float64, θmin::Float64)::Float64 = dθ_dchi(χ, θ, θmin) * dchi_dt

"""
Function to compute dt/dτ (Eq. 28)
# Arguments
- `v::AbstractVector{Float64}`: spatial velocity vector in BL coordinates wrt t, e.g., v = [drdt, dθdt, dϕdt].
"""
@inline function Γ(r::Float64, θ::Float64, ϕ::Float64, v::AbstractVector{Float64}, a::Float64)::Float64
    one_over_Γ = -Kerr.KerrMetric.g_tt(r, θ, ϕ, a) - 2.0 * Kerr.KerrMetric.g_tϕ(r, θ, ϕ, a) * v[3] - Kerr.KerrMetric.g_rr(r, θ, ϕ, a) * v[1]^2 - Kerr.KerrMetric.g_θθ(r, θ, ϕ, a) * v[2]^2 -
        Kerr.KerrMetric.g_ϕϕ(r, θ, ϕ, a) * v[3]^2    
    return sqrt(1.0/one_over_Γ)
end

# computes Ψ, defined in Eq. 96, which is an input in the geodesic equations (Eqs. 93-95)
@inline function Ψ(psi::Float64, a::Float64, p::Float64, e::Float64, E::Float64, L::Float64)::Float64
    r = BLTimeGeodesics.r(psi, p, e)
    Δ = BLTimeGeodesics.Δ(r, a)
    return (((r^2 + a^2)^2) / Δ - a^2) * E - 2.0 * a * r * L / Δ
end

# compute time derivatives of angle variables ψ, χ, ϕ (Eqs. 93-95) 
@inline function psi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)::Float64
    return  sqrt((1.0 - E^2) * (p * (1.0 - e) - p3 * (1.0 + e * cos(psi))) * (p * (1.0 + e) - p4 * (1.0 + e * cos(psi)))) / ((1.0 - e^2) * (Ψ(psi, a, p, e, E, L) + a^2 * E * z(chi, θmin)))
end

@inline function chi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)::Float64
    return sqrt(a^2 * (1.0 - E^2) * (zp - zm * cos(chi)^2)) / (Ψ(psi, a, p, e, E, L) + a^2 * E * z(chi, θmin))
end

@inline function phi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)::Float64
    r = BLTimeGeodesics.r(psi, p, e)
    Δ = BLTimeGeodesics.Δ(r, a)
    return  (2.0 * a * r * E / Δ + (1.0 / (1.0 - z(chi, θmin)) - a^2 / Δ) * L) / (Ψ(psi, a, p, e, E, L) + a^2 * E * z(chi, θmin))
end

# repeat these functions specifically for ODE solver - u = [ψ, χ, ϕ], p (= params) = [a, E, L, p, e, θmin, p3, p4, zp, zm]
@inline function psi_dot(u::SVector{3, Float64}, p::SVector{10, Float64})::Float64
    return  sqrt((1.0 - p[2]^2) * (p[4] * (1.0 - p[5]) - p[7] * (1.0 + p[5] * cos(u[1]))) * (p[4] * (1.0 + p[5]) - p[8] * (1.0 + p[5] * cos(u[1])))) / ((1.0 - p[5]^2) * (Ψ(u[1], p[1], p[4], p[5], p[2], p[3]) + p[1]^2 * p[2] * z(u[2], p[6])))
end

@inline function chi_dot(u::SVector{3, Float64}, p::SVector{10, Float64})::Float64
    return sqrt(p[1]^2 * (1.0 - p[2]^2) * (p[9] - p[10] * cos(u[2])^2)) / (Ψ(u[1], p[1], p[4], p[5], p[2], p[3]) + p[1]^2 * p[2] * z(u[2], p[6]))
end

@inline function phi_dot(u::SVector{3, Float64}, p::SVector{10, Float64})::Float64
    r = BLTimeGeodesics.r(u[1], p[4], p[5])
    Δ = BLTimeGeodesics.Δ(r, p[1])
    return  (2.0 * p[1] * r * p[2] / Δ + (1.0 / (1.0 - z(u[2], p[6])) - p[1]^2 / Δ) * p[3]) / (Ψ(u[1], p[1], p[4], p[5], p[2], p[3]) + p[1]^2 * p[2] * z(u[2], p[6]))
end

# compute constants required as input for the ODE solver in order to evaluate the expressions for dψ/dt, dχ/dt, dϕ/dt
function compute_ODE_params(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64)::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}
    # calculate integrals of motion from orbital parameters
    E, L, Q, C = ConstantsOfMotion.compute_ELC(a, p, e, θmin, sign_Lz)

    # periastron and apastron
    rp = p / (1 + e)
    ra = p / (1 - e)

    # compute roots of radial function R(r)
    zm = cos(θmin)^2
    zp = C / (a^2 * (1.0-E^2) * zm)    # Eq. E23
    A = 1.0 / (1.0 - E^2) - (ra + rp) / 2.0    # Eq. E20
    B = a^2 * C / ((1.0 - E^2) * ra * rp)    # Eq. E21
    r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
    p3 = r3 * (1.0 - e); p4 = r4 * (1.0 + e);  # Above Eq. 96

    return E, L, Q, C, ra, p3, p4, zp, zm
end

# initial conditions for the angle variables in the geodesic equations---for circular orbits set ψ = 0. Due to axisymmetry, we can set ϕ = 0.
function HJ_ics(psi_i::Float64, chi_i::Float64)
    return @SArray [psi_i, chi_i, 0.0]
end

# equation for ODE solver
@inline function HJ_Eqns(u, params, t)
    @SArray [psi_dot(u, params), chi_dot(u, params), phi_dot(u, params)]
end

# equation for ODE solver for circular orbits
@inline function HJ_Eqns_circular(u, params, t)
    @SArray [0.0, chi_dot(u, params), phi_dot(u, params)]
end

# equation for ODE solver for equatorial orbits
@inline function HJ_Eqns_equatorial(u, params, t)
    @SArray [psi_dot(u, params), 0.0, phi_dot(u, params)]
end

"""
    compute_kerr_geodesic(...args)

Computes geodesic trajectory in Kerr characterized by a, p, e, θmin (M=1, μ=1) in coordinate time

# Arguments
- `nPoints::Int64`: Number of points in the geodesic (i.e., the resolution).
- `specify_params::Bool`: If true, the constants of motion (E, L, C) are specified by the user.
- `tmax::Float64`: Maximum coordinate time (in units of M) for the geodesic evolution.
- `Δti::Float64`: Initial time step for the geodesic evolution.
- `reltol::Float64`: Relative tolerance for the ODE solver.
- `abstol::Float64`: Absolute tolerance for the ODE solver.
- `ics::SVector{3, Float64}`: Initial conditions for the geodesic trajectory. Default is ψ=0.0, χ=0.0, ϕ=0.0.
- `data_path::String`: Path to save the ODE solution.
- `save_to_file::Bool`: If true, the ODE solution is saved to a file, otherwise the solution is returned.

# Returns
- `nothing` if save_to_file=true, otherwise returns the geodesic trajectory and its first and second derivatives as a tuple.

# Notes
- We have chosen to specify the number of points in the geodesic and the total time range, but we could have also chosen to specify the time step and total time range, or the time step and the number of points. In order to be good physicists, then,
we state that the follow key assumption of our implementation--- Assumption 1: the user can convert between these different ways of specifying the resolution and the time range of the geodesic!
- There are a few key word arguments set to zero which the user can specify if they wish. The rationale here is that one might want to solve the geodesic equation many times iteratively, pausing in between to do some calculations. This allows
one to ``pause'' the geodesic and return at some desired point. However, if the computations one performs between such geodesic computations are faster than ~1ms, then it is probably wiser to not reuse this function iteratively because it exits the
geodesic solver after each computation, which is slower than using DifferentialEquation's integrator interface, which allows one to dynamically step through the integation. For an example of this in the context of this module and package, see the
file 'MinoFDMInspiral.jl'.
- We have chosen the RK4 integrator with auto switching. We have tested other integrator methods and found that the solution does not change much.
"""
function compute_kerr_geodesic(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, nPoints::Int64, specify_params::Bool, tmax::Float64=50.0, Δti::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10;
    ics::SVector{3, Float64}=SA[0.0, 0.0, 0.0], E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0, zm::Float64=0.0,
    data_path::String="Results/", save_to_file::Bool=false)

    if !specify_params
        E, L, Q, C, ra, p3, p4, zp, zm = compute_ODE_params(a, p, e, θmin, sign_Lz)
    end

    params = @SArray [a, E, L, p, e, θmin, p3, p4, zp, zm]

    tspan = (0.0, tmax); saveat_t = range(start=tspan[1], length=nPoints, stop=tspan[2])

    prob = e == 0.0 ? ODEProblem(HJ_Eqns_circular, ics, tspan, params) : ODEProblem(HJ_Eqns, ics, tspan, params);
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);

    # deconstruct solution
    t = sol.t;
    psi = sol[1, :];
    chi = mod.(sol[2, :], 2π);
    ϕ = sol[3, :];

    # convert to BL coordinates
    psi_dot = zeros(nPoints); chi_dot = zeros(nPoints); ϕ_dot = zeros(nPoints);
    r = zeros(nPoints); θ = zeros(nPoints); r_dot = zeros(nPoints); θ_dot = zeros(nPoints);
    r_ddot = zeros(nPoints); θ_ddot = zeros(nPoints); ϕ_ddot = zeros(nPoints); dt_dτ = zeros(nPoints);

    @inbounds for i in eachindex(t)
        # compute time derivatives
        psi_dot[i] = BLTimeGeodesics.psi_dot(psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        chi_dot[i] = BLTimeGeodesics.chi_dot(psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        ϕ_dot[i] = BLTimeGeodesics.phi_dot(psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)

        # compute BL coordinates t, r, θ and their time derivatives
        r[i] = BLTimeGeodesics.r(psi[i], p, e)
        θ[i] = acos((π/2<chi[i]<1.5π) ? -sqrt(BLTimeGeodesics.z(chi[i], θmin)) : sqrt(BLTimeGeodesics.z(chi[i], θmin)))
        r_dot[i] = dr_dt(psi_dot[i], psi[i], p, e);
        θ_dot[i] = dθ_dt(chi_dot[i], chi[i], θ[i], θmin);
        v = [r_dot[i], θ_dot[i], ϕ_dot[i]];
        dt_dτ[i] = Γ(r[i], θ[i], ϕ[i], v, a)

        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        r_ddot[i] = BLTimeGeodesics.dr2_dt2(r[i], θ[i], ϕ[i], r_dot[i], θ_dot[i], ϕ_dot[i], a)
        θ_ddot[i] = BLTimeGeodesics.dθ2_dt2(r[i], θ[i], ϕ[i], r_dot[i], θ_dot[i], ϕ_dot[i], a)
        ϕ_ddot[i] = BLTimeGeodesics.dϕ2_dt2(r[i], θ[i], ϕ[i], r_dot[i], θ_dot[i], ϕ_dot[i], a)
    end

    if save_to_file
        mkpath(data_path)
        sign_Lz > 0 ? type = "prograde" : type = "retrograde"
        ODE_filename=data_path * "Kerr_geo_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(nPoints)_T_$(tmax)_rtol_$(reltol)_atol_$(abstol).h5"
        h5open(ODE_filename, "w") do file
            file["t"] = t
            file["r"] = r
            file["theta"] = θ
            file["phi"] = ϕ
            file["r_dot"] = r_dot
            file["theta_dot"] = θ_dot
            file["phi_dot"] = ϕ_dot
            file["r_ddot"] = r_ddot
            file["theta_ddot"] = θ_ddot
            file["phi_ddot"] = ϕ_ddot
            file["Gamma"] = dt_dτ
            file["psi"] = psi
            file["chi"] = chi
        end
        println("File created: " * ODE_filename)
    else
        return t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi
    end
end

# ODE equations for evolution into the past
function HJ_Eqns_past(u, params, t)
    @SArray [-psi_dot(u, params), -chi_dot(u, params), -phi_dot(u, params)]
end

function HJ_Eqns_circular_past(u, params, t)
    @SArray [0.0, -chi_dot(u, params), -phi_dot(u, params)]
end

# computes trajectory in Kerr characterized by a, p, e, θmin (M=1, μ=1) into the past. Note that we leave the input tmax > 0.
function compute_kerr_geodesic_past(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, nPoints::Int64, specify_params::Bool, tmax::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10;
    ics::SVector{3, Float64}=SA[0.0, 0.0, 0.0], E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0, zm::Float64=0.0,
    data_path::String="Results/", save_to_file::Bool=false)
    
    if !specify_params
        E, L, Q, C, ra, p3, p4, zp, zm = compute_ODE_params(a, p, e, θmin, sign_Lz)
    end

    params = @SArray [a, E, L, p, e, θmin, p3, p4, zp, zm]

    tspan = (0.0, tmax); saveat_t = range(start=tspan[1], length=nPoints, stop=tspan[2])
    prob = e == 0.0 ? ODEProblem(HJ_Eqns_circular_past, ics, tspan, params) : ODEProblem(HJ_Eqns_past, ics, tspan, params);
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);
 
    # deconstruct solution
    t = -sol.t;
    psi = sol[1, :];
    chi = mod.(sol[2, :], 2π);
    ϕ = sol[3, :];

    # convert to BL coordinates
    psi_dot = zeros(nPoints); chi_dot = zeros(nPoints); ϕ_dot = zeros(nPoints);
    r = zeros(nPoints); θ = zeros(nPoints); r_dot = zeros(nPoints); θ_dot = zeros(nPoints);
    r_ddot = zeros(nPoints); θ_ddot = zeros(nPoints); ϕ_ddot = zeros(nPoints); dt_dτ = zeros(nPoints);

    @inbounds for i in eachindex(t)
        # compute time derivatives
        psi_dot[i] = BLTimeGeodesics.psi_dot(psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        chi_dot[i] = BLTimeGeodesics.chi_dot(psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        ϕ_dot[i] = BLTimeGeodesics.phi_dot(psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)

        # compute BL coordinates t, r, θ and their time derivatives
        r[i] = BLTimeGeodesics.r(psi[i], p, e)
        θ[i] = acos((π/2<chi[i]<1.5π) ? -sqrt(BLTimeGeodesics.z(chi[i], θmin)) : sqrt(BLTimeGeodesics.z(chi[i], θmin)))
        r_dot[i] = dr_dt(psi_dot[i], psi[i], p, e);
        θ_dot[i] = dθ_dt(chi_dot[i], chi[i], θ[i], θmin);
        v = [r_dot[i], θ_dot[i], ϕ_dot[i]];
        dt_dτ[i] = Γ(r[i], θ[i], ϕ[i], v, a)

        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        r_ddot[i] = BLTimeGeodesics.dr2_dt2(r[i], θ[i], ϕ[i], r_dot[i], θ_dot[i], ϕ_dot[i], a)
        θ_ddot[i] = BLTimeGeodesics.dθ2_dt2(r[i], θ[i], ϕ[i], r_dot[i], θ_dot[i], ϕ_dot[i], a)
        ϕ_ddot[i] = BLTimeGeodesics.dϕ2_dt2(r[i], θ[i], ϕ[i], r_dot[i], θ_dot[i], ϕ_dot[i], a)
    end

    # reverse so time increases with successive columns
    reverse!(t); reverse!(r); reverse!(θ); reverse!(ϕ); reverse!(r_dot); reverse!(θ_dot); reverse!(ϕ_dot);
    reverse!(r_ddot); reverse!(θ_ddot); reverse!(ϕ_ddot); reverse!(dt_dτ); reverse!(psi); reverse!(chi);

    if save_to_file
        mkpath(data_path)
        sign_Lz > 0 ? type = "prograde" : type = "retrograde"
        ODE_filename=data_path * "Kerr_geo_past_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(nPoints)_T_$(tmax)_rtol_$(reltol)_atol_$(abstol).h5"
        h5open(ODE_filename, "w") do file
            file["t"] = t
            file["r"] = r
            file["theta"] = θ
            file["phi"] = ϕ
            file["r_dot"] = r_dot
            file["theta_dot"] = θ_dot
            file["phi_dot"] = ϕ_dot
            file["r_ddot"] = r_ddot
            file["theta_ddot"] = θ_ddot
            file["phi_ddot"] = ϕ_ddot
            file["Gamma"] = dt_dτ
            file["psi"] = psi
            file["chi"] = chi
        end

        println("File created: " * ODE_filename)
    else
        return t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi
    end
end


"""
    compute_kerr_geodesic_past_and_future(...args)

Computes geodesic trajectory in Kerr characterized by a, p, e, θmin (M=1, μ=1) into the future and past of some point with given initial conditions

# Arguments
- `total_num_points::Int64`: Odd number of points in the entire geodesic (i.e., the resolution).
- `total_time_range::Float64`: total coordinate time elapsed throughout the entire geodesic evolution.
- `Δti::Float64`: Initial time step for the geodesic evolution.
- `ics::SVector{3, Float64}`: Initial conditions for the geodesic trajectory. Default is ψ=0.0, χ=0.0, ϕ=0.0.
- `data_path::String`: Path to save the ODE solution.
- `save_to_file::Bool`: If true, the ODE solution is saved to a file, otherwise the solution is returned.

# Returns
- `nothing` if save_to_file=true, otherwise returns the geodesic trajectory and its first and second derivatives as a tuple.
"""
# computes trajectory in Kerr characterized by a, p, e, θmin (M=1, μ=1)
function compute_kerr_geodesic_past_and_future(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, specify_params::Bool, total_num_points::Int64, total_time_range::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10;
    ics::SVector{3, Float64}=SA[0.0, 0.0, 0.0], E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0, zm::Float64=0.0, data_path::String="Results/", save_to_file::Bool=false)
    if total_num_points % 2 == 0
        throw(ArgumentError("total_num_points must be an odd number"))
    end

    # number of points and time range for future/past parts of geodesic. We choose to include the initial condition in the future geodesic, so we must add 1 to the total number of points for the past geodesic since we will delete the initial condition.
    nPointsPast = total_num_points÷2 + 1
    nPointsFuture = total_num_points÷2 + 1
    tmax = total_time_range/2.0;

    # future part of geodesic
    t_f, r_f, θ_f, ϕ_f, r_dot_f, θ_dot_f, ϕ_dot_f, r_ddot_f, θ_ddot_f, ϕ_ddot_f, Γ_f, psi_f, chi_f = BLTimeGeodesics.compute_kerr_geodesic(a, p, e, θmin, sign_Lz, nPointsFuture, specify_params, tmax, Δti, reltol, abstol; 
    ics = ics, E, L, Q, C, ra, p3, p4, zp, zm, save_to_file = false)

    # past part of geodesic
    t_p, r_p, θ_p, ϕ_p, r_dot_p, θ_dot_p, ϕ_dot_p, r_ddot_p, θ_ddot_p, ϕ_ddot_p, Γ_p, psi_p, chi_p = BLTimeGeodesics.compute_kerr_geodesic_past(a, p, e, θmin, sign_Lz, nPointsPast, specify_params, tmax, Δti, reltol, abstol;
    ics = ics, E, L, Q, C, ra, p3, p4, zp, zm, save_to_file = false)

    # combine
    t = [t_p[1:nPointsPast-1]; t_f]
    r = [r_p[1:nPointsPast-1]; r_f]
    θ = [θ_p[1:nPointsPast-1]; θ_f]
    ϕ = [ϕ_p[1:nPointsPast-1]; ϕ_f]
    r_dot = [r_dot_p[1:nPointsPast-1]; r_dot_f]
    θ_dot = [θ_dot_p[1:nPointsPast-1]; θ_dot_f]
    ϕ_dot = [ϕ_dot_p[1:nPointsPast-1]; ϕ_dot_f]
    r_ddot = [r_ddot_p[1:nPointsPast-1]; r_ddot_f]
    θ_ddot = [θ_ddot_p[1:nPointsPast-1]; θ_ddot_f]
    ϕ_ddot = [ϕ_ddot_p[1:nPointsPast-1]; ϕ_ddot_f]
    dt_dτ = [Γ_p[1:nPointsPast-1]; Γ_f]
    psi = [psi_p[1:nPointsPast-1]; psi_f]
    chi = [chi_p[1:nPointsPast-1]; chi_f]

    if save_to_file
        mkpath(data_path)
        sign_Lz > 0 ? type = "prograde" : type = "retrograde"
        ODE_filename=data_path * "Kerr_geo_past_future_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(total_num_points)_T_$(total_time_range)_rtol_$(reltol)_atol_$(abstol).h5"
        h5open(ODE_filename, "w") do file
            file["t"] = t
            file["r"] = r
            file["theta"] = θ
            file["phi"] = ϕ
            file["r_dot"] = r_dot
            file["theta_dot"] = θ_dot
            file["phi_dot"] = ϕ_dot
            file["r_ddot"] = r_ddot
            file["theta_ddot"] = θ_ddot
            file["phi_ddot"] = ϕ_ddot
            file["Gamma"] = dt_dτ
            file["psi"] = psi
            file["chi"] = chi
        end

        println("File created: " * ODE_filename)
    else
        return t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi
    end
end

# load geodesic trajectory from file
function load_kerr_geodesic_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, nPoints::Int64, tmax::Float64, reltol::Float64, abstol::Float64, data_path::String)
    sign_Lz > 0 ? type = "prograde" : type = "retrograde"
    ODE_filename=data_path * "Kerr_geo_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(nPoints)_T_$(tmax)_rtol_$(reltol)_atol_$(abstol).h5"
    h5f = h5open(ODE_filename, "r")
    t = h5f["t"][:]
    r = h5f["r"][:]
    θ = h5f["theta"][:]
    ϕ = h5f["phi"][:]
    close(h5f)
    return t, r, θ, ϕ
end

# load full solution from file
function load_full_kerr_geodesic(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, nPoints::Int64, tmax::Float64, reltol::Float64, abstol::Float64, data_path::String)
    sign_Lz > 0 ? type = "prograde" : type = "retrograde"
    ODE_filename=data_path * "Kerr_geo_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(nPoints)_T_$(tmax)_rtol_$(reltol)_atol_$(abstol).h5"
    h5f = h5open(ODE_filename, "r")
    t = h5f["t"][:]
    r = h5f["r"][:]
    θ = h5f["theta"][:]
    ϕ = h5f["phi"][:]
    r_dot = h5f["r_dot"][:]
    θ_dot = h5f["theta_dot"][:]
    ϕ_dot = h5f["phi_dot"][:]
    r_ddot = h5f["r_ddot"][:]
    θ_ddot = h5f["theta_ddot"][:]
    ϕ_ddot = h5f["phi_ddot"][:]
    dt_dτ = h5f["Gamma"][:]
    psi = h5f["psi"][:]
    chi = h5f["chi"][:]
    close(h5f)
    return t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi
end

end