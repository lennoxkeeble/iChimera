#=

    This module contains code which evolves the first order geodesic equations in Kerr (i.e., the separated Hamilton Jacobi equations) in Mino time, λ. The particular formulation of the equations follows the notation in Sec. IVA in Sopuerta,
    Yunes, 2011 (arXiv:1109.0572v2), which, in turn, is derived from Drasco & Hughes, 2004 (arXiv:astro-ph/0308479v3). Throughout, Eq. X will refer to expressions in the former reference. Note that throughout, we assume M=1.

=#

module MinoTimeGeodesics
using ..Kerr
using ..ConstantsOfMotion
import ..BLTimeGeodesics: Δ, dr2_dt2, dθ2_dt2, dϕ2_dt2, r, dr_dpsi, z, dθ_dchi, Γ, Ψ, compute_ODE_params
using StaticArrays
using DifferentialEquations
using HDF5

"""
# Common Arguments in this module
- `λ::Float64`: Mino time.
- `t::Float64`: coordinate-time.
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
- `zp::Float64`: root the the theta geodesic equation (Eq. 91-92)
- `zm::Float64`: root the the theta geodesic equation (Eq. 91-92)
- `p4::Float64`: transformation one of the roots of the radial function R(r) (Eq. 90), e.g., p4 = r4 * (1 - e) / M.
"""

# functions to compute derivatives of BL coordinates wrt λ
dr_dλ(dpsi_dλ::Float64, psi::Float64, p::Float64, e::Float64) = dr_dpsi(psi, p, e) * dpsi_dλ
dθ_dλ(dchi_dλ::Float64, χ::Float64, θ::Float64, θmin::Float64) = dθ_dchi(χ, θ, θmin) * dchi_dλ

# derivatives of angle variables ψ, χ, ϕ wrt λ
@inline function dt_dλ(t::Float64, psi::Float64, chi::Float64, ϕ::Float64, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    r = MinoTimeGeodesics.r(psi, p, e)
    Δ = MinoTimeGeodesics.Δ(r, a)
    return E * ((r^2+a^2)^2 / Δ - a^2 * (1.0-z(chi, θmin))) + a*L*(1.0-(r^2+a^2)/Δ)
end

@inline function dψ_dλ(t::Float64, psi::Float64, chi::Float64, ϕ::Float64, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    return sqrt((1.0 - E^2) * (p * (1.0 - e) - p3 * (1.0 + e * cos(psi))) * (p * (1.0 + e) - p4 * (1.0 + e * cos(psi)))) / (1.0 - e^2)
end

@inline function dχ_dλ(t::Float64, psi::Float64, chi::Float64, ϕ::Float64, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    return sqrt(a^2 * (1.0 - E^2) * (zp - zm * cos(chi)^2))
end

@inline function dϕ_dλ(t::Float64, psi::Float64, chi::Float64, ϕ::Float64, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    r = MinoTimeGeodesics.r(psi, p, e)
    Δ = MinoTimeGeodesics.Δ(r, a)
    return  (2.0 * a * r * E / Δ + (1.0 / (1.0 - z(chi, θmin)) - a^2 / Δ) * L)
end

# initial conditions for bound kerr orbits starting in equatorial plane
function Mino_ics(t0::Float64, ri::Float64, p::Float64, e::Float64)
    psi_i = e==0 ? 0.0 : π    # Eq. 89 --- for circular orbits set ψ = 0, otherwise we start the orbit at apastron
    chi_i = 0.0    # Eq. 89 - since we start orbit at θ = θmin
    ϕi = 0.0    # by axisymmetry can start orbit at ϕ = 0
    return @SArray [t0, psi_i, chi_i, ϕi]
end


# initial conditions for t and the angle variables in the geodesic equations---for circular orbits set ψ = 0. Due to stationarity and axisymmetry, we can set ϕ = 0, t = 0.
function Mino_ics(psi_i::Float64, chi_i::Float64)
    return @SArray [0.0, psi_i, chi_i, 0.0]
end

# equation for ODE solver
function HJ_Eqns(u, params, λ)
    @SArray [dt_dλ(u..., params...), dψ_dλ(u..., params...), dχ_dλ(u..., params...), dϕ_dλ(u..., params...)]
end

function HJ_Eqns_circular(u, params, t)
    @SArray [dt_dλ(u..., params...), 0.0, dχ_dλ(u..., params...), dϕ_dλ(u..., params...)]
end

"""
    compute_kerr_geodesic(...args)

Computes geodesic trajectory in Kerr characterized by a, p, e, θmin (M=1, μ=1) in Minot time

# Arguments
- `nPoints::Int64`: Number of points in the geodesic (i.e., the resolution).
- `specify_params::Bool`: If true, the constants of motion (E, L, C) are specified by the user.
- `λmax::Float64`: Maximum Mino time (in units of M) for the geodesic evolution.
- `Δλi::Float64`: Initial time step for the geodesic evolution.
- `reltol::Float64`: Relative tolerance for the ODE solver.
- `abstol::Float64`: Absolute tolerance for the ODE solver.
- `ics::SVector{4, Float64}`: Initial conditions for the geodesic trajectory. Default is t=0.0, ψ=0.0, χ=0.0, ϕ=0.0.
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
function compute_kerr_geodesic(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, nPoints::Int64, specify_params::Bool, λmax::Float64=3000.0, Δλi::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10;
    ics::SVector{4, Float64}=SA[0.0, 0.0, 0.0, 0.0], E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0, zm::Float64=0.0,
    data_path::String="Results/", save_to_file::Bool=false)

    if !specify_params
        E, L, Q, C, ra, p3, p4, zp, zm = compute_ODE_params(a, p, e, θmin, sign_Lz)
    end

    params = @SArray [a, E, L, p, e, θmin, p3, p4, zp, zm]


    # initial conditions for Kerr geodesic trajectory
    λspan = (0.0, λmax); saveat_λ = range(start=λspan[1], length=nPoints, stop=λspan[2])

    prob = e == 0.0 ? ODEProblem(HJ_Eqns_circular, ics, λspan, params) : ODEProblem(HJ_Eqns, ics, λspan, params);
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δλi, reltol = reltol, abstol = abstol, saveat=saveat_λ);
 
    # deconstruct solution
    λ = sol.t;
    t = sol[1, :];
    psi = sol[2, :];
    chi = mod.(sol[3, :], 2π);
    ϕ = sol[4, :];

    # convert to BL coordinates
    dt_dλ = zeros(nPoints); dψ_dλ = zeros(nPoints); dχ_dλ = zeros(nPoints); dϕ_dλ = zeros(nPoints); dϕ_dt = zeros(nPoints);
    r = zeros(nPoints); θ = zeros(nPoints); dr_dt = zeros(nPoints); dθ_dt = zeros(nPoints);
    d2r_dt2 = zeros(nPoints); d2θ_dt2 = zeros(nPoints); d2ϕ_dt2 = zeros(nPoints); dt_dτ = zeros(nPoints);

    @inbounds for i in eachindex(t)
        # compute time derivatives (wrt λ)
        dt_dλ[i] = MinoTimeGeodesics.dt_dλ(t[i], psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        dψ_dλ[i] = MinoTimeGeodesics.dψ_dλ(t[i], psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        dχ_dλ[i] = MinoTimeGeodesics.dχ_dλ(t[i], psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        dϕ_dλ[i] = MinoTimeGeodesics.dϕ_dλ(t[i], psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)

        # compute BL coordinates r, θ and their time derivatives (wrt λ)
        r[i] = MinoTimeGeodesics.r(psi[i], p, e)
        θ[i] = acos((π/2<chi[i]<1.5π) ? -sqrt(MinoTimeGeodesics.z(chi[i], θmin)) : sqrt(MinoTimeGeodesics.z(chi[i], θmin)))

        dr_dλ = MinoTimeGeodesics.dr_dλ(dψ_dλ[i], psi[i], p, e);
        dθ_dλ = MinoTimeGeodesics.dθ_dλ(dχ_dλ[i], chi[i], θ[i], θmin);

        # compute derivatives wrt t
        dr_dt[i] = dr_dλ / dt_dλ[i]
        dθ_dt[i] = dθ_dλ / dt_dλ[i] 
        dϕ_dt[i] = dϕ_dλ[i] / dt_dλ[i] 

        # compute gamma factor
        v = [dr_dt[i], dθ_dt[i], dϕ_dt[i]] # v=dxi/dt
        dt_dτ[i] = MinoTimeGeodesics.Γ(r[i], θ[i], ϕ[i], v, a)
        
        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        d2r_dt2[i] = MinoTimeGeodesics.dr2_dt2(r[i], θ[i], ϕ[i], dr_dt[i], dθ_dt[i], dϕ_dt[i], a)
        d2θ_dt2[i] = MinoTimeGeodesics.dθ2_dt2(r[i], θ[i], ϕ[i], dr_dt[i], dθ_dt[i], dϕ_dt[i], a)
        d2ϕ_dt2[i] = MinoTimeGeodesics.dϕ2_dt2(r[i], θ[i], ϕ[i], dr_dt[i], dθ_dt[i], dϕ_dt[i], a)
    end
    if save_to_file
        mkpath(data_path)
        sign_Lz > 0 ? type = "prograde" : type = "retrograde"
        ODE_filename=data_path * "Kerr_geo_Mino_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(nPoints)_Λ_$(λmax)_rtol_$(reltol)_atol_$(abstol).h5"
        h5open(ODE_filename, "w") do file
            file["lambda"] = λ
            file["t"] = t
            file["r"] = r
            file["theta"] = θ
            file["phi"] = ϕ
            file["r_dot"] = dr_dt
            file["theta_dot"] = dθ_dt
            file["phi_dot"] = dϕ_dt
            file["r_ddot"] = d2r_dt2
            file["theta_ddot"] = d2θ_dt2
            file["phi_ddot"] = d2ϕ_dt2
            file["Gamma"] = dt_dτ
            file["psi"] = psi
            file["chi"] = chi
            file["dt_dlambda"] = dt_dλ
        end
        println("File created: " * ODE_filename)
    else
        return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, psi, chi, dt_dλ
    end
end

# ODE equations for evolution into the past
function HJ_Eqns_past(u, params, λ)
    @SArray [-dt_dλ(u..., params...), -dψ_dλ(u..., params...), -dχ_dλ(u..., params...), -dϕ_dλ(u..., params...)]
end

function HJ_Eqns_circular_past(u, params, t)
    @SArray [-dt_dλ(u..., params...), 0.0, -dχ_dλ(u..., params...), -dϕ_dλ(u..., params...)]
end

# computes trajectory in Kerr characterized by a, p, e, θmin (M=1, μ=1) into the past in Mino time. Note that we leave the input λmax > 0.
function compute_kerr_geodesic_past(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, nPoints::Int64, specify_params::Bool, λmax::Float64=3000.0, Δλi::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10;
    ics::SVector{4, Float64}=SA[0.0, 0.0, 0.0, 0.0], E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0, zm::Float64=0.0,
    data_path::String="Results/", save_to_file::Bool=true)

    if !specify_params
        E, L, Q, C, ra, p3, p4, zp, zm = compute_ODE_params(a, p, e, θmin, sign_Lz)
    end

    params = @SArray [a, E, L, p, e, θmin, p3, p4, zp, zm]

    # initial conditions for Kerr geodesic trajectory
    λspan = (0.0, λmax); saveat_λ = range(start=λspan[1], length=nPoints, stop=λspan[2])

    prob = e == 0.0 ? ODEProblem(HJ_Eqns_circular_past, ics, λspan, params) : ODEProblem(HJ_Eqns_past, ics, λspan, params);
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δλi, reltol = reltol, abstol = abstol, saveat=saveat_λ);
 
    # deconstruct solution
    λ = -sol.t;
    t = sol[1, :];
    psi = sol[2, :];
    chi = mod.(sol[3, :], 2π);
    ϕ = sol[4, :];

    # convert to BL coordinates
    dt_dλ = zeros(nPoints); dψ_dλ = zeros(nPoints); dχ_dλ = zeros(nPoints); dϕ_dλ = zeros(nPoints); dϕ_dt = zeros(nPoints);
    r = zeros(nPoints); θ = zeros(nPoints); dr_dt = zeros(nPoints); dθ_dt = zeros(nPoints);
    d2r_dt2 = zeros(nPoints); d2θ_dt2 = zeros(nPoints); d2ϕ_dt2 = zeros(nPoints); dt_dτ = zeros(nPoints);

    @inbounds for i in eachindex(t)
        # compute time derivatives (wrt λ)
        dt_dλ[i] = MinoTimeGeodesics.dt_dλ(t[i], psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        dψ_dλ[i] = MinoTimeGeodesics.dψ_dλ(t[i], psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        dχ_dλ[i] = MinoTimeGeodesics.dχ_dλ(t[i], psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)
        dϕ_dλ[i] = MinoTimeGeodesics.dϕ_dλ(t[i], psi[i], chi[i], ϕ[i], a, E, L, p, e, θmin, p3, p4, zp, zm)

        # compute BL coordinates r, θ and their time derivatives (wrt λ)
        r[i] = MinoTimeGeodesics.r(psi[i], p, e)
        θ[i] = acos((π/2<chi[i]<1.5π) ? -sqrt(MinoTimeGeodesics.z(chi[i], θmin)) : sqrt(MinoTimeGeodesics.z(chi[i], θmin)))

        dr_dλ = MinoTimeGeodesics.dr_dλ(dψ_dλ[i], psi[i], p, e);
        dθ_dλ = MinoTimeGeodesics.dθ_dλ(dχ_dλ[i], chi[i], θ[i], θmin);

        # compute derivatives wrt t
        dr_dt[i] = dr_dλ / dt_dλ[i]
        dθ_dt[i] = dθ_dλ / dt_dλ[i] 
        dϕ_dt[i] = dϕ_dλ[i] / dt_dλ[i] 

        # compute gamma factor
        v = [dr_dt[i], dθ_dt[i], dϕ_dt[i]] # v=dxi/dt
        dt_dτ[i] = MinoTimeGeodesics.Γ(r[i], θ[i], ϕ[i], v, a)
        
        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        d2r_dt2[i] = MinoTimeGeodesics.dr2_dt2(r[i], θ[i], ϕ[i], dr_dt[i], dθ_dt[i], dϕ_dt[i], a)
        d2θ_dt2[i] = MinoTimeGeodesics.dθ2_dt2(r[i], θ[i], ϕ[i], dr_dt[i], dθ_dt[i], dϕ_dt[i], a)
        d2ϕ_dt2[i] = MinoTimeGeodesics.dϕ2_dt2(r[i], θ[i], ϕ[i], dr_dt[i], dθ_dt[i], dϕ_dt[i], a)
    end

    # reverse so time increases with successive columns
    reverse!(λ); reverse!(t); reverse!(r); reverse!(θ); reverse!(ϕ); reverse!(dr_dt); reverse!(dθ_dt); reverse!(dϕ_dt);
    reverse!(d2r_dt2); reverse!(d2θ_dt2); reverse!(d2ϕ_dt2); reverse!(dt_dλ); reverse!(psi); reverse!(chi); reverse!(dt_dτ);

    if save_to_file
        mkpath(data_path)
        sign_Lz > 0 ? type = "prograde" : type = "retrograde"
        ODE_filename=data_path * "Kerr_geo_Mino_past_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(nPoints)_Λ_$(λmax)_rtol_$(reltol)_atol_$(abstol).h5"
        h5open(ODE_filename, "w") do file
            file["lambda"] = λ
            file["t"] = t
            file["r"] = r
            file["theta"] = θ
            file["phi"] = ϕ
            file["r_dot"] = dr_dt
            file["theta_dot"] = dθ_dt
            file["phi_dot"] = dϕ_dt
            file["r_ddot"] = d2r_dt2
            file["theta_ddot"] = d2θ_dt2
            file["phi_ddot"] = d2ϕ_dt2
            file["Gamma"] = dt_dτ
            file["psi"] = psi
            file["chi"] = chi
            file["dt_dlambda"] = dt_dλ
        end
        println("File created: " * ODE_filename)
    else
        return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, psi, chi, dt_dλ
    end
end


"""
    compute_kerr_geodesic_past_and_future(...args)

Computes geodesic trajectory in Kerr characterized by a, p, e, θmin (M=1, μ=1) into the future and past of some point with given initial conditions

# Arguments
- `total_num_points::Int64`: Odd number of points in the entire geodesic (i.e., the resolution).
- `total_time_range::Float64`: total Mino time elapsed throughout the entire geodesic evolution.
- `Δλi::Float64`: Initial time step for the geodesic evolution.
- `ics::SVector{3, Float64}`: Initial conditions for the geodesic trajectory. Default is ψ=0.0, χ=0.0, ϕ=0.0.
- `data_path::String`: Path to save the ODE solution.
- `save_to_file::Bool`: If true, the ODE solution is saved to a file, otherwise the solution is returned.

# Returns
- `nothing` if save_to_file=true, otherwise returns the geodesic trajectory and its first and second derivatives as a tuple.
"""
function compute_kerr_geodesic_past_and_future(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, specify_params::Bool, total_num_points::Int64, total_time_range::Float64=3000.0, Δλi::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10;
    ics::SVector{4, Float64} = SA[0.0, 0.0, 0.0, 0.0], E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0, zm::Float64=0.0, data_path::String="Results/", save_to_file::Bool=false)
    if total_num_points % 2 == 0
        throw(ArgumentError("total_num_points must be an odd number"))
    end

    # number of points and time range for future/past parts of geodesic. We choose to include the initial condition in the future geodesic, so we must add 1 to the total number of points for the past geodesic since we will delete the initial condition.
    nPointsPast = total_num_points÷2 + 1
    nPointsFuture = total_num_points÷2 + 1
    λmax = total_time_range/2.0;

    # future part of geodesic
    λ_f, t_f, r_f, θ_f, ϕ_f, dr_dt_f, dθ_dt_f, dϕ_dt_f, d2r_dt2_f, d2θ_dt2_f, d2ϕ_dt2_f, Γ_f, psi_f, chi_f, dt_dλ_f = MinoTimeGeodesics.compute_kerr_geodesic(a, p, e, θmin, sign_Lz, nPointsFuture, specify_params, λmax, Δλi, reltol, abstol; 
    ics = ics, E, L, Q, C, ra, p3, p4, zp, zm, save_to_file =false)

    # past part of geodesic
    λ_p, t_p, r_p, θ_p, ϕ_p, dr_dt_p, dθ_dt_p, dϕ_dt_p, d2r_dt2_p, d2θ_dt2_p, d2ϕ_dt2_p, Γ_p, psi_p, chi_p, dt_dλ_p = MinoTimeGeodesics.compute_kerr_geodesic_past(a, p, e, θmin, sign_Lz, nPointsPast, specify_params, λmax, Δλi, reltol, abstol;
    ics = ics, E, L, Q, C, ra, p3, p4, zp, zm, save_to_file =false)

    # combine
    λ = [λ_p[1:nPointsPast-1]; λ_f]
    t = [t_p[1:nPointsPast-1]; t_f]
    r = [r_p[1:nPointsPast-1]; r_f]
    θ = [θ_p[1:nPointsPast-1]; θ_f]
    ϕ = [ϕ_p[1:nPointsPast-1]; ϕ_f]
    dr_dt = [dr_dt_p[1:nPointsPast-1]; dr_dt_f]
    dθ_dt = [dθ_dt_p[1:nPointsPast-1]; dθ_dt_f]
    dϕ_dt = [dϕ_dt_p[1:nPointsPast-1]; dϕ_dt_f]
    d2r_dt2 = [d2r_dt2_p[1:nPointsPast-1]; d2r_dt2_f]
    d2θ_dt2 = [d2θ_dt2_p[1:nPointsPast-1]; d2θ_dt2_f]
    d2ϕ_dt2 = [d2ϕ_dt2_p[1:nPointsPast-1]; d2ϕ_dt2_f]
    dt_dτ = [Γ_p[1:nPointsPast-1]; Γ_f]
    psi = [psi_p[1:nPointsPast-1]; psi_f]
    chi = [chi_p[1:nPointsPast-1]; chi_f]
    dt_dλ = [dt_dλ_p[1:nPointsPast-1]; dt_dλ_f]

    if save_to_file
        mkpath(data_path)
        sign_Lz > 0 ? type = "prograde" : type = "retrograde"
        ODE_filename=data_path * "Kerr_geo_past_future_Mino_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(total_num_points)_Λ_$(total_time_range)_rtol_$(reltol)_atol_$(abstol).h5"
        h5open(ODE_filename, "w") do file
            file["lambda"] = λ
            file["t"] = t
            file["r"] = r
            file["theta"] = θ
            file["phi"] = ϕ
            file["r_dot"] = dr_dt
            file["theta_dot"] = dθ_dt
            file["phi_dot"] = dϕ_dt
            file["r_ddot"] = d2r_dt2
            file["theta_ddot"] = d2θ_dt2
            file["phi_ddot"] = d2ϕ_dt2
            file["Gamma"] = dt_dτ
            file["psi"] = psi
            file["chi"] = chi
            file["dt_dlambda"] = dt_dλ
        end

        println("File created: " * ODE_filename)
    else
        return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, psi, chi, dt_dλ
    end
end

# load geodesic trajectory from file
function load_kerr_geodesic_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, nPoints::Int64, λmax::Float64, reltol::Float64, abstol::Float64, data_path::String)
    sign_Lz > 0 ? type = "prograde" : type = "retrograde"
    ODE_filename=data_path * "Kerr_geo_Mino_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(nPoints)_Λ_$(λmax)_rtol_$(reltol)_atol_$(abstol).h5"
    h5f = h5open(ODE_filename, "r")
    λ = h5f["lambda"][:]
    r = h5f["r"][:]
    θ = h5f["theta"][:]
    ϕ = h5f["phi"][:]
    close(h5f)
    return λ, r, θ, ϕ
end

# load full solution from file
function load_full_kerr_geodesic(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, nPoints::Int64, λmax::Float64, reltol::Float64, abstol::Float64, data_path::String)
    sign_Lz > 0 ? type = "prograde" : type = "retrograde"
    ODE_filename=data_path * "Kerr_geo_Mino_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_"*type*"_nPoints_$(nPoints)_Λ_$(λmax)_rtol_$(reltol)_atol_$(abstol).h5"
    h5f = h5open(ODE_filename, "r")
    λ = h5f["lambda"][:]
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
    dt_dλ = h5f["dt_dlambda"][:]
    close(h5f)
    return λ, t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi, dt_dλ
end

end