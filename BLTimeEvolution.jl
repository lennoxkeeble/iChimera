#=

In this module we evolve the separated Hamilton Jacobi equation with respect to Boyer-Lindquist time t. See Sec. IVA in Sopuerta, Yunes 2011 (arXiv:1109.0572v2) for further details.

=#

module BLTimeEvolution
using ..Kerr, StaticArrays, DifferentialEquations, DelimitedFiles

Δ(r::Float64, a::Float64, M::Float64) = r^2 - 2.0M * r + a^2

dr2_dt2(t::Float64, r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64, M::Float64) = -(drdt)^2 * Kerr.KerrMetric.Γrrr(t, r, θ, ϕ, a, M) - 2.0*drdt*dθdt*Kerr.KerrMetric.Γrrθ(t, r, θ, ϕ, a, M)-Kerr.KerrMetric.Γrtt(t, r, θ, ϕ, a, M)-
dϕdt*(2.0Kerr.KerrMetric.Γrtϕ(t, r, θ, ϕ, a, M) + dϕdt*Kerr.KerrMetric.Γrϕϕ(t, r, θ, ϕ, a, M))+2.0drdt^2*(dϕdt * Kerr.KerrMetric.Γtrϕ(t, r, θ, ϕ, a, M)+Kerr.KerrMetric.Γttr(t, r, θ, ϕ, a, M)) + dθdt*(-dθdt*Kerr.KerrMetric.Γrθθ(t, r, θ, ϕ, a, M)+
2.0drdt*(Kerr.KerrMetric.Γttθ(t, r, θ, ϕ, a, M)+dϕdt*Kerr.KerrMetric.Γtθϕ(t, r, θ, ϕ, a, M)))


dθ2_dt2(t::Float64, r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64, M::Float64) = 2.0drdt * dθdt * dϕdt * Kerr.KerrMetric.Γtrϕ(t, r, θ, ϕ, a, M)+2.0drdt * dθdt * Kerr.KerrMetric.Γttr(t, r, θ, ϕ, a, M)+
2.0dθdt^2*Kerr.KerrMetric.Γttθ(t, r, θ, ϕ, a, M)+2.0dθdt^2*dϕdt * Kerr.KerrMetric.Γtθϕ(t, r, θ, ϕ, a, M)-drdt^2*Kerr.KerrMetric.Γθrr(t, r, θ, ϕ, a, M)-2.0drdt*dθdt*Kerr.KerrMetric.Γθrθ(t, r, θ, ϕ, a, M)-Kerr.KerrMetric.Γθtt(t, r, θ, ϕ, a, M)-
2.0dϕdt*Kerr.KerrMetric.Γθtϕ(t, r, θ, ϕ, a, M)-dθdt^2*Kerr.KerrMetric.Γθθθ(t, r, θ, ϕ, a, M)-dϕdt^2*Kerr.KerrMetric.Γθϕϕ(t, r, θ, ϕ, a, M)

dϕ2_dt2(t::Float64, r::Float64, θ::Float64, ϕ::Float64, drdt::Float64, dθdt::Float64, dϕdt::Float64, a::Float64, M::Float64) = 2.0*(drdt*dϕdt^2*Kerr.KerrMetric.Γtrϕ(t, r, θ, ϕ, a, M)+
drdt*dϕdt*Kerr.KerrMetric.Γttr(t, r, θ, ϕ, a, M)+dθdt*dϕdt*Kerr.KerrMetric.Γttθ(t, r, θ, ϕ, a, M)+dθdt*dϕdt^2*Kerr.KerrMetric.Γtθϕ(t, r, θ, ϕ, a, M)-drdt*dϕdt*Kerr.KerrMetric.Γϕrϕ(t, r, θ, ϕ, a, M)-drdt*Kerr.KerrMetric.Γϕtr(t, r, θ, ϕ, a, M)-
dθdt*Kerr.KerrMetric.Γϕtθ(t, r, θ, ϕ, a, M)-dθdt*dϕdt*Kerr.KerrMetric.Γϕθϕ(t, r, θ, ϕ, a, M))

r(psi::Float64, p::Float64, e::Float64, M::Float64) = p * M / (1.0 + e * cos(psi))    # Eq. 89
dr_dpsi(psi::Float64, p::Float64, e::Float64, M::Float64) = p * M * e * sin(psi)/ ((1.0 + e * cos(psi))^2)
dr_dt(dpsi_dt::Float64, psi::Float64, p::Float64, e::Float64, M::Float64) = dr_dpsi(psi, p, e, M) * dpsi_dt

z(χ::Float64, θmin::Float64) = cos(θmin)^2 * cos(χ)^2    # Eq. 89
dθ_dchi(χ::Float64, θ::Float64, θmin::Float64) = cos(θmin)^2 * sin(2.0χ) / sin(2.0θ)
dθ_dt(dchi_dt::Float64, χ::Float64, θ::Float64, θmin::Float64) = dθ_dchi(χ, θ, θmin) * dchi_dt

# function to compute dt/dτ (Eq. 28)
function Γ(t::Float64, r::Float64, θ::Float64, ϕ::Float64, v::Vector{Float64}, a::Float64, M::Float64)
    one_over_Γ = -Kerr.KerrMetric.g_tt(t, r, θ, ϕ, a, M)
    @inbounds for i=1:3
        one_over_Γ += -2.0Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 1, i+1) * v[i]    # i+1 since g_μν is indexed with μ=1,2,3,4
        @inbounds for j=1:3
            one_over_Γ += -Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, i+1, j+1) * v[i] * v[j]
        end
    end
    return sqrt(1.0/one_over_Γ)
end

function Ψ(psi::Float64, a::Float64, p::Float64, e::Float64, M::Float64, E::Float64, L::Float64)
    r = BLTimeEvolution.r(psi, p, e, M)
    Δ = BLTimeEvolution.Δ(r, a, M)
    return (((r^2 + a^2)^2) / Δ - a^2) * E - 2.0M * a * r * L / Δ
end

function psi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    return M * sqrt((1.0 - E^2) * (p * (1.0 - e) - p3 * (1.0 + e * cos(psi))) * (p * (1.0 + e) - p4 * (1.0 + e * cos(psi)))) / ((1.0 - e^2) * (Ψ(psi, a, p, e, M, E, L) + a^2 * E * z(chi, θmin)))
end

function chi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    return sqrt(a^2 * (1.0 - E^2) * (zp - zm * cos(chi)^2)) / (Ψ(psi, a, p, e, M, E, L) + a^2 * E * z(chi, θmin))
end

function phi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    r = BLTimeEvolution.r(psi, p, e, M)
    Δ = BLTimeEvolution.Δ(r, a, M)
    return  (2.0M * a * r * E / Δ + (1.0 / (1.0 - z(chi, θmin)) - a^2 / Δ) * L) / (Ψ(psi, a, p, e, M, E, L) + a^2 * E * z(chi, θmin))
end

# initial conditions for bound kerr orbits starting in equatorial plane
function HJ_ics(ri::Float64, p::Float64, e::Float64, M::Float64)
    psi_i = e==0 ? 0.0 : π    # Eq. 89 --- for circular orbits set ψ = 0, otherwise we start the orbit at apastron
    chi_i = 0.0    # Eq. 89 - since we start orbit at θ = θmin
    ϕi = 0.0    # by axisymmetry can start orbit at ϕ = 0
    return @SArray [psi_i, chi_i, ϕi]
end

function compute_ODE_params(a::Float64, p::Float64, e::Float64, θi::Float64)
    # orbital parameters
    M = 1.0;

    # calculate integrals of motion from orbital parameters
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)

    # define periastron and apastron
    rp = p * M / (1 + e);   # Eq. 6.1
    ra = p * M / (1 - e);   # Eq. 6.1

    # compute roots of radial function R(r)
    zm = cos(θi)^2
    zp = C / (a^2 * (1.0-E^2) * zm)    # Eq. E23
    ra=p * M / (1.0 - e); rp=p * M / (1.0 + e);
    A = M / (1.0 - E^2) - (ra + rp) / 2.0    # Eq. E20
    B = a^2 * C / ((1.0 - E^2) * ra * rp)    # Eq. E21
    r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
    p3 = r3 * (1.0 - e) / M; p4 = r4 * (1.0 + e) / M    # Above Eq. 96

    # array of params for ODE solver
    return E, L, Q, C, ra, p3, p4, zp, zm
end

# equation for ODE solver
function HJ_Eqns(u, params, t)
    @SArray [psi_dot(u..., params...), chi_dot(u..., params...), phi_dot(u..., params...)]
end

function HJ_Eqns_circular(u, params, t)
    @SArray [0.0, chi_dot(u..., params...), phi_dot(u..., params...)]
end

# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
function compute_kerr_geodesic(a::Float64, p::Float64, e::Float64, θi::Float64, nPoints::Int64, specify_ics::Bool, specify_params::Bool,
    tmax::Float64=50.0, Δti::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10; ics::SVector{3, Float64}=SA[0.0, 0.0, 0.0],
    E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0,
    zm::Float64=0.0, data_path::String="Results/", save_to_file::Bool=false)

    M=1.;
    if !specify_params
        E, L, Q, C, ra, p3, p4, zp, zm = compute_ODE_params(a, p, e, θi)
    end
    params = @SArray [a, M, E, L, p, e, θi, p3, p4, zp, zm]

    if !specify_ics
        # initial conditions for Kerr geodesic trajectory
        ri = ra; # start at apastron
        ics = HJ_ics(ri, p, e, M);
    end

    tspan = (0.0, tmax); saveat_t = range(start=tspan[1], length=nPoints, stop=tspan[2])

    prob = e == 0.0 ? ODEProblem(HJ_Eqns_circular, ics, tspan, params) : ODEProblem(HJ_Eqns, ics, tspan, params);
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);
 
    # deconstruct solution
    t = sol.t;
    psi = sol[1, :];
    chi = mod.(sol[2, :], 2π);
    ϕ = sol[3, :];

    # compute time derivatives
    psi_dot = BLTimeEvolution.psi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    chi_dot = BLTimeEvolution.chi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    ϕ_dot = BLTimeEvolution.phi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)

    # compute BL coordinates t, r, θ and their time derivatives
    r = BLTimeEvolution.r.(psi, p, e, M)
    θ = [acos((π/2<chi[i]<1.5π) ? -sqrt(BLTimeEvolution.z(chi[i], θi)) : sqrt(BLTimeEvolution.z(chi[i], θi))) for i in eachindex(chi)]

    r_dot = dr_dt.(psi_dot, psi, p, e, M);
    θ_dot = dθ_dt.(chi_dot, chi, θ, θi);
    v = [[r_dot[i], θ_dot[i], ϕ_dot[i]] for i in eachindex(t)];
    dt_dτ = @. Γ(t, r, θ, ϕ, v, a, M)

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    r_ddot = BLTimeEvolution.dr2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)
    θ_ddot = BLTimeEvolution.dθ2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)
    ϕ_ddot = BLTimeEvolution.dϕ2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)

    if save_to_file
        # save trajectory- rows are: t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
        sol = [reshape(t, 1, nPoints); reshape(r, 1, nPoints); reshape(θ, 1, nPoints); reshape(ϕ, 1, nPoints);
                reshape(r_dot, 1, nPoints); reshape(θ_dot, 1, nPoints); reshape(ϕ_dot, 1, nPoints); reshape(r_ddot, 1, nPoints);
                reshape(θ_ddot, 1, nPoints); reshape(ϕ_ddot, 1, nPoints); reshape(dt_dτ, 1, nPoints); reshape(psi, 1, nPoints); reshape(chi, 1, nPoints)]
        
        mkpath(data_path)
        ODE_filename=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(diff(saveat_t)[1])_T_$(tmax)_tol_$(reltol).txt"
        
        open(ODE_filename, "w") do io
            writedlm(io, sol)
        end

        println("ODE saved to: " * ODE_filename)
    else
        return t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi
    end
end

### evolution into the past ###
# equation for ODE solver
function HJ_Eqns_past(u, params, t)
    @SArray [-psi_dot(u..., params...), -chi_dot(u..., params...), -phi_dot(u..., params...)]
end

function HJ_Eqns_circular_past(u, params, t)
    @SArray [-0.0, -chi_dot(u..., params...), -phi_dot(u..., params...)]
end

# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
function compute_kerr_geodesic_past(a::Float64, p::Float64, e::Float64, θi::Float64, nPoints::Int64, specify_ics::Bool, specify_params::Bool,
    tmax::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10; ics::SVector{3, Float64}=SA[0.0, 0.0, 0.0],
    E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0,
    zm::Float64=0.0, data_path::String="Results/", save_to_file::Bool=false)
    M=1.;

    if !specify_params
        E, L, Q, C, ra, p3, p4, zp, zm = compute_ODE_params(a, p, e, θi)
    end

    params = @SArray [a, M, E, L, p, e, θi, p3, p4, zp, zm]

    if !specify_ics
        # initial conditions for Kerr geodesic trajectory
        ra = p * M / (1.0 - e); ri = ra; 
        ics = HJ_ics(ri, p, e, M);
    end

    tspan = (0.0, tmax); saveat_t = range(start=tspan[1], length=nPoints, stop=tspan[2])
    prob = e == 0.0 ? ODEProblem(HJ_Eqns_circular_past, ics, tspan, params) : ODEProblem(HJ_Eqns_past, ics, tspan, params);
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);
 
    # deconstruct solution
    t = -sol.t;
    psi = sol[1, :];
    chi = mod.(sol[2, :], 2π);
    ϕ = sol[3, :];

    # compute time derivatives
    psi_dot = BLTimeEvolution.psi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    chi_dot = BLTimeEvolution.chi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    ϕ_dot = BLTimeEvolution.phi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)

    # compute BL coordinates t, r, θ and their time derivatives
    r = BLTimeEvolution.r.(psi, p, e, M)
    θ = [acos((π/2<chi[i]<1.5π) ? -sqrt(BLTimeEvolution.z(chi[i], θi)) : sqrt(BLTimeEvolution.z(chi[i], θi))) for i in eachindex(chi)]

    r_dot = dr_dt.(psi_dot, psi, p, e, M);
    θ_dot = dθ_dt.(chi_dot, chi, θ, θi);
    v = [[r_dot[i], θ_dot[i], ϕ_dot[i]] for i in eachindex(t)];
    dt_dτ = @. Γ(t, r, θ, ϕ, v, a, M)

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    r_ddot = BLTimeEvolution.dr2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)
    θ_ddot = BLTimeEvolution.dθ2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)
    ϕ_ddot = BLTimeEvolution.dϕ2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)

    # reverse so time increases with successive columns
    reverse!(t); reverse!(r); reverse!(θ); reverse!(ϕ); reverse!(r_dot); reverse!(θ_dot); reverse!(ϕ_dot);
    reverse!(r_ddot); reverse!(θ_ddot); reverse!(ϕ_ddot); reverse!(dt_dτ); reverse!(psi); reverse!(chi);

    if save_to_file
        # save trajectory- rows are: t, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
        sol = [reshape(t, 1, nPoints); reshape(r, 1, nPoints); reshape(θ, 1, nPoints); reshape(ϕ, 1, nPoints);
            reshape(r_dot, 1, nPoints); reshape(θ_dot, 1, nPoints); reshape(ϕ_dot, 1, nPoints); reshape(r_ddot, 1, nPoints);
            reshape(θ_ddot, 1, nPoints); reshape(ϕ_ddot, 1, nPoints); reshape(dt_dτ, 1, nPoints); reshape(psi, 1, nPoints); reshape(chi, 1, nPoints)]
            
        mkpath(data_path)
        ODE_filename=data_path * "HJ_ODE_past_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(reltol).txt"
        open(ODE_filename, "w") do io
            writedlm(io, sol)
        end
        println("ODE saved to: " * ODE_filename)
    else
        # # remove t=0 data which will be duplicated
        # pop!(t); pop!(r); pop!(θ); pop!(ϕ); pop!(r_dot); pop!(θ_dot); pop!(ϕ_dot);
        # pop!(r_ddot); pop!(θ_ddot); pop!(ϕ_ddot); pop!(dt_dτ); pop!(psi); pop!(chi);
        return t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi
    end
end

# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
@views function compute_kerr_geodesic_past_and_future(ics::SVector{3, Float64}, a::Float64, p::Float64, e::Float64, θmin::Float64, 
    specify_params::Bool, total_num_points::Int64, total_time_range::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10;
    E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0,
    zm::Float64=0.0, data_path::String="Results/", save_to_file::Bool=false)
    use_custom_ics = true;
    nPoints = total_num_points÷2 + mod(total_num_points, 2);
    tmax = total_time_range/2.0;

    # future part of geodesic
    t_f, r_f, θ_f, ϕ_f, r_dot_f, θ_dot_f, ϕ_dot_f, r_ddot_f, θ_ddot_f, ϕ_ddot_f, Γ_f, psi_f, chi_f = BLTimeEvolution.compute_kerr_geodesic(a, p, e, θmin, nPoints, use_custom_ics, specify_params, tmax, Δti, reltol, abstol; 
    ics = ics, E, L, Q, C, ra, p3, p4, zp, zm, save_to_file = false)

    # past part of geodesic
    t_p, r_p, θ_p, ϕ_p, r_dot_p, θ_dot_p, ϕ_dot_p, r_ddot_p, θ_ddot_p, ϕ_ddot_p, Γ_p, psi_p, chi_p = BLTimeEvolution.compute_kerr_geodesic_past(a, p, e, θmin, nPoints, use_custom_ics, specify_params, tmax, Δti, reltol, abstol;
    ics = ics, E, L, Q, C, ra, p3, p4, zp, zm, save_to_file = false)

    if save_to_file
        sol = [reshape([t_p[1:nPoints-1]; t_f], 1, total_num_points); reshape([r_p[1:nPoints-1]; r_f], 1, total_num_points); reshape([θ_p[1:nPoints-1]; θ_f], 1, total_num_points); reshape([ϕ_p[1:nPoints-1]; ϕ_f], 1, total_num_points);
            reshape([r_dot_p[1:nPoints-1]; r_dot_f], 1, total_num_points); reshape([θ_dot_p[1:nPoints-1]; θ_dot_f], 1, total_num_points); reshape([ϕ_dot_p[1:nPoints-1]; ϕ_dot_f], 1, total_num_points); reshape([r_ddot_p[1:nPoints-1]; r_ddot_f], 1, total_num_points);
            reshape([θ_ddot_p[1:nPoints-1]; θ_ddot_f], 1, total_num_points); reshape([ϕ_ddot_p[1:nPoints-1]; ϕ_ddot_f], 1, total_num_points); reshape([dt_dτ_p[1:nPoints-1]; dt_dτ_f], 1, total_num_points); reshape([psi_p[1:nPoints-1]; psi_f], 1, total_num_points); reshape([chi_p[1:nPoints-1]; chi_f], 1, total_num_points)]

        mkpath(data_path)
        ODE_filename=data_path * "HJ_ODE_past_future_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(reltol).txt"
        open(ODE_filename, "w") do io
            writedlm(io, sol)
        end
        println("ODE saved to: " * ODE_filename)
    else
        return [t_p[1:nPoints-1]; t_f], [r_p[1:nPoints-1]; r_f], [θ_p[1:nPoints-1]; θ_f], [ϕ_p[1:nPoints-1]; ϕ_f], [r_dot_p[1:nPoints-1]; r_dot_f], [θ_dot_p[1:nPoints-1]; θ_dot_f], [ϕ_dot_p[1:nPoints-1]; ϕ_dot_f], [r_ddot_p[1:nPoints-1]; r_ddot_f],
        [θ_ddot_p[1:nPoints-1]; θ_ddot_f], [ϕ_ddot_p[1:nPoints-1]; ϕ_ddot_f], [Γ_p[1:nPoints-1]; Γ_f], [psi_p[1:nPoints-1]; psi_f], [chi_p[1:nPoints-1]; chi_f]
    end
end

end