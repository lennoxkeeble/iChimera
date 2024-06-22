#=

In this module we evolve the separated Hamilton Jacobi equation with respect to Mino time λ. See Sec. III in Drasco & Hughes 2004 (arXiv:astro-ph/0308479v3) 
for further details.

=#

include("/home/lkeeble/GRSuite/Testing/BL_time_coordinate_derivs/d2x.jl");
include("/home/lkeeble/GRSuite/Testing/BL_time_coordinate_derivs/d3x.jl");
include("/home/lkeeble/GRSuite/Testing/BL_time_coordinate_derivs/d4x.jl");
include("/home/lkeeble/GRSuite/Testing/BL_time_coordinate_derivs/d5x.jl");
include("/home/lkeeble/GRSuite/Testing/BL_time_coordinate_derivs/d6x.jl");
include("/home/lkeeble/GRSuite/Testing/Test_modules/MinoTimeBLTimeDerivs.jl");

module MinoDerivs
using ..Deriv2, ..Deriv3, ..Deriv4, ..Deriv5, ..Deriv6, ..MinoTimeDerivs

function compute_mino_derivs!(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, d4x::AbstractArray, d5x::AbstractArray, 
    d6x::AbstractArray, dλdt::AbstractArray, d2λdt::AbstractArray, d3λdt::AbstractArray, d4λdt::AbstractArray, d5λdt::AbstractArray, d6λdt::AbstractArray, 
    a::Float64, E::Float64, L::Float64)
    @inbounds Threads.@threads for i in eachindex(x[1])
        d2x[i] = [Deriv2.d2r_dt(dx[i], x[i], a), Deriv2.d2θ_dt(dx[i], x[i], a), Deriv2.d2ϕ_dt(dx[i], x[i], a)]
        d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
        d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
        d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        
        dλdt[i] = MinoTimeDerivs.dλ_dt(x[i], a, M, E, L)
        d2λdt[i] = MinoTimeDerivs.d2λ_dt(dx[i], x[i], a, M, E, L)
        d3λdt[i] = MinoTimeDerivs.d3λ_dt(d2x[i], dx[i], x[i], a, M, E, L)
        d4λdt[i] = MinoTimeDerivs.d4λ_dt(d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
        d5λdt[i] = MinoTimeDerivs.d5λ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
        d6λdt[i] = MinoTimeDerivs.d6λ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
    end
end
end

module MinoEvolution
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
dr_dλ(dpsi_dλ::Float64, psi::Float64, p::Float64, e::Float64, M::Float64) = dr_dpsi(psi, p, e, M) * dpsi_dλ

z(χ::Float64, θmin::Float64) = cos(θmin)^2 * cos(χ)^2    # Eq. 89
dθ_dchi(χ::Float64, θ::Float64, θmin::Float64) = cos(θmin)^2 * sin(2.0χ) / sin(2.0θ)
dθ_dλ(dchi_dλ::Float64, χ::Float64, θ::Float64, θmin::Float64) = dθ_dchi(χ, θ, θmin) * dchi_dλ

# function to compute dt/dτ (Eq. 28)
function Γ(λ::Float64, r::Float64, θ::Float64, ϕ::Float64, v::Vector{Float64}, a::Float64, M::Float64)
    one_over_Γ = -Kerr.KerrMetric.g_tt(λ, r, θ, ϕ, a, M)
    @inbounds for i=1:3
        one_over_Γ += -2.0Kerr.KerrMetric.g_μν(λ, r, θ, ϕ, a, M, 1, i+1) * v[i]    # i+1 since g_μν is indexed with μ=1,2,3,4
        @inbounds for j=1:3
            one_over_Γ += -Kerr.KerrMetric.g_μν(λ, r, θ, ϕ, a, M, i+1, j+1) * v[i] * v[j]
        end
    end
    return sqrt(1.0/one_over_Γ)
end

function Ψ(psi::Float64, a::Float64, p::Float64, e::Float64, M::Float64, E::Float64, L::Float64)
    r = MinoEvolution.r(psi, p, e, M)
    Δ = MinoEvolution.Δ(r, a, M)
    return (((r^2 + a^2)^2) / Δ - a^2) * E - 2.0M * a * r * L / Δ
end

function dt_dλ(t::Float64, psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    r = MinoEvolution.r(psi, p, e, M)
    Δ = MinoEvolution.Δ(r, a, M)
    return E * ((r^2+a^2)^2 / Δ - a^2 * (1.0-z(chi, θmin))) + a*L*(1.0-(r^2+a^2)/Δ)
end

function dψ_dλ(t::Float64, psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    return M * sqrt((1.0 - E^2) * (p * (1.0 - e) - p3 * (1.0 + e * cos(psi))) * (p * (1.0 + e) - p4 * (1.0 + e * cos(psi)))) / (1.0 - e^2)
end

function dχ_dλ(t::Float64, psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    return sqrt(a^2 * (1.0 - E^2) * (zp - zm * cos(chi)^2))
end

function dϕ_dλ(t::Float64, psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    r = MinoEvolution.r(psi, p, e, M)
    Δ = MinoEvolution.Δ(r, a, M)
    return  (2.0M * a * r * E / Δ + (1.0 / (1.0 - z(chi, θmin)) - a^2 / Δ) * L)
end

# initial conditions for bound kerr orbits starting in equatorial plane
function Mino_ics(t0::Float64, ri::Float64, p::Float64, e::Float64, M::Float64)
    psi_i = e==0 ? 0.0 : acos((p * M / ri - 1.0) / e)    # Eq. 89
    chi_i = 0.0    # Eq. 89 - since we start orbit at θ = θmin
    ϕi = 0.0    # by axisymmetry can start orbit at ϕ = 0
    return @SArray [t0, psi_i, chi_i, ϕi]
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
function HJ_Eqns(u, params, λ)
    @SArray [dt_dλ(u..., params...), dψ_dλ(u..., params...), dχ_dλ(u..., params...), dϕ_dλ(u..., params...)]
end

function HJ_Eqns_circular(u, params, t)
    @SArray [dt_dλ(u..., params...), 0.0, dχ_dλ(u..., params...), dϕ_dλ(u..., params...)]
end

# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
function compute_kerr_geodesic(a::Float64, p::Float64, e::Float64, θi::Float64, nPoints::Int64, specify_ics::Bool, specify_params::Bool,
    λmax::Float64=3000.0, Δλi::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10; ics::SVector{4, Float64}=SA[0.0, 0.0, 0.0, 0.0],
    E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0,
    zm::Float64=0.0, data_path::String="Results/", save_to_file::Bool=true, inspiral::Bool=false)
    # orbital parameters
    M = 1.0;

    if !specify_params
        E, L, Q, C, ra, p3, p4, zp, zm = compute_ODE_params(a, p, e, θi)
    end

    params = @SArray [a, M, E, L, p, e, θi, p3, p4, zp, zm]

    if !specify_ics
        # initial conditions for Kerr geodesic trajectory
        ri = ra; # start at apastron
        t0 = 0.0;
        ics = Mino_ics(t0, ri, p, e, M);
    end

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

    # compute time derivatives (wrt λ)
    dt_dλ = MinoEvolution.dt_dλ.(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dψ_dλ = MinoEvolution.dψ_dλ.(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dχ_dλ = MinoEvolution.dχ_dλ.(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dϕ_dλ = MinoEvolution.dϕ_dλ.(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)

    # compute BL coordinates r, θ and their time derivatives (wrt λ)
    r = MinoEvolution.r.(psi, p, e, M)
    θ = [acos((π/2<chi[i]<1.5π) ? -sqrt(MinoEvolution.z(chi[i], θi)) : sqrt(MinoEvolution.z(chi[i], θi))) for i in eachindex(chi)]

    dr_dλ = MinoEvolution.dr_dλ.(dψ_dλ, psi, p, e, M);
    dθ_dλ = MinoEvolution.dθ_dλ.(dχ_dλ, chi, θ, θi);

    # compute derivatives wrt t
    dr_dt = @. dr_dλ / dt_dλ
    dθ_dt = @. dθ_dλ / dt_dλ 
    dϕ_dt = @. dϕ_dλ / dt_dλ 

    # compute derivatives wrt τ
    v = [[dr_dt[i], dθ_dt[i], dϕ_dt[i]] for i in eachindex(λ)]; # v=dxi/dt

    dt_dτ = @. MinoEvolution.Γ(t, r, θ, ϕ, v, a, M)
    
    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    d2r_dt2 = MinoEvolution.dr2_dt2.(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
    d2θ_dt2 = MinoEvolution.dθ2_dt2.(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
    d2ϕ_dt2 = MinoEvolution.dϕ2_dt2.(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)

    if inspiral
        return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, psi, chi, dt_dλ
    else
        # save trajectory- rows are: λ, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
        sol = [reshape(λ, 1, nPoints); reshape(t, 1, nPoints); reshape(r, 1, nPoints); reshape(θ, 1, nPoints); reshape(ϕ, 1, nPoints);
                reshape(dr_dt, 1, nPoints); reshape(dθ_dt, 1, nPoints); reshape(dϕ_dt, 1, nPoints); reshape(d2r_dt2, 1, nPoints);
                reshape(d2θ_dt2, 1, nPoints); reshape(d2ϕ_dt2, 1, nPoints); reshape(dt_dτ, 1, nPoints); reshape(psi, 1, nPoints);
                reshape(chi, 1, nPoints); reshape(dt_dλ, 1, nPoints)]
        
        if save_to_file
            mkpath(data_path)
            ODE_filename=data_path * "Mino_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_λstep_$(diff(saveat_λ)[1])_λmax_$(λmax)_tol_$(reltol).txt"
            open(ODE_filename, "w") do io
                writedlm(io, sol)
            end
            println("ODE saved to: " * ODE_filename)
        else
            return sol
        end
    end
end

### evolution into the past ###
# equation for ODE solver
function HJ_Eqns_past(u, params, λ)
    @SArray [-dt_dλ(u..., params...), -dψ_dλ(u..., params...), -dχ_dλ(u..., params...), -dϕ_dλ(u..., params...)]
end

function HJ_Eqns_circular_past(u, params, t)
    @SArray [-dt_dλ(u..., params...), 0.0, -dχ_dλ(u..., params...), -dϕ_dλ(u..., params...)]
end

# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
function compute_kerr_geodesic_past(a::Float64, p::Float64, e::Float64, θi::Float64, nPoints::Int64, specify_ics::Bool, specify_params::Bool,
    λmax::Float64=3000.0, Δλi::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10; ics::SVector{4, Float64}=SA[0.0, 0.0, 0.0, 0.0],
    E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0,
    zm::Float64=0.0, data_path::String="Results/", save_to_file::Bool=true, inspiral::Bool=false)
    # orbital parameters
    M = 1.0;

    if !specify_params
        E, L, Q, C, ra, p3, p4, zp, zm = compute_ODE_params(a, p, e, θi)
    end

    params = @SArray [a, M, E, L, p, e, θi, p3, p4, zp, zm]

    if !specify_ics
        # initial conditions for Kerr geodesic trajectory
        ri = ra; # start at apastron
        t0 = 0.0;
        ics = Mino_ics(t0, ri, p, e, M);
    end

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

    # compute time derivatives (wrt λ)
    dt_dλ = MinoEvolution.dt_dλ.(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dψ_dλ = MinoEvolution.dψ_dλ.(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dχ_dλ = MinoEvolution.dχ_dλ.(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dϕ_dλ = MinoEvolution.dϕ_dλ.(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)

    # compute BL coordinates r, θ and their time derivatives (wrt λ)
    r = MinoEvolution.r.(psi, p, e, M)
    θ = [acos((π/2<chi[i]<1.5π) ? -sqrt(MinoEvolution.z(chi[i], θi)) : sqrt(MinoEvolution.z(chi[i], θi))) for i in eachindex(chi)]

    dr_dλ = MinoEvolution.dr_dλ.(dψ_dλ, psi, p, e, M);
    dθ_dλ = MinoEvolution.dθ_dλ.(dχ_dλ, chi, θ, θi);

    # compute derivatives wrt t
    dr_dt = @. dr_dλ / dt_dλ
    dθ_dt = @. dθ_dλ / dt_dλ 
    dϕ_dt = @. dϕ_dλ / dt_dλ 

    # compute derivatives wrt τ
    v = [[dr_dt[i], dθ_dt[i], dϕ_dt[i]] for i in eachindex(λ)]; # v=dxi/dt

    dt_dτ = @. MinoEvolution.Γ(t, r, θ, ϕ, v, a, M)
    
    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    d2r_dt2 = MinoEvolution.dr2_dt2.(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
    d2θ_dt2 = MinoEvolution.dθ2_dt2.(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
    d2ϕ_dt2 = MinoEvolution.dϕ2_dt2.(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)

    # reverse so time increases with successive columns
    reverse!(λ); reverse!(t); reverse!(r); reverse!(θ); reverse!(ϕ); reverse!(dr_dt); reverse!(dθ_dt); reverse!(dϕ_dt);
    reverse!(d2r_dt2); reverse!(d2θ_dt2); reverse!(d2ϕ_dt2); reverse!(dt_dλ); reverse!(psi); reverse!(chi); reverse!(dt_dτ)

    if inspiral
        # remove t=0 data which will be duplicated
        pop!(λ); pop!(t); pop!(r); pop!(θ); pop!(ϕ); pop!(dr_dt); pop!(dθ_dt); pop!(dϕ_dt);
        pop!(d2r_dt2); pop!(d2θ_dt2); pop!(d2ϕ_dt2); pop!(dt_dλ); pop!(psi); pop!(chi); pop!(dt_dτ)
        return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, psi, chi, dt_dλ
    else
        # save trajectory- rows are: λ, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
        sol = [reshape(λ, 1, nPoints); reshape(t, 1, nPoints); reshape(r, 1, nPoints); reshape(θ, 1, nPoints); reshape(ϕ, 1, nPoints);
                reshape(dr_dt, 1, nPoints); reshape(dθ_dt, 1, nPoints); reshape(dϕ_dt, 1, nPoints); reshape(d2r_dt2, 1, nPoints);
                reshape(d2θ_dt2, 1, nPoints); reshape(d2ϕ_dt2, 1, nPoints); reshape(dt_dτ, 1, nPoints); reshape(psi, 1, nPoints); 
                reshape(chi, 1, nPoints); reshape(dt_dλ, 1, nPoints)]
        # reverse so time increases with successive columns
        reverse!(sol; dims=2)

        if save_to_file
            mkpath(data_path)
            ODE_filename=data_path * "Mino_ODE_past_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_λstep_$(diff(saveat_λ)[1])_λmax_$(λmax)_tol_$(reltol).txt"
            open(ODE_filename, "w") do io
                writedlm(io, sol)
            end
            println("ODE saved to: " * ODE_filename)
        else
            return sol
        end
    end
end


# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
function compute_kerr_geodesic_past_and_future(ics::SVector{4, Float64}, a::Float64, p::Float64, e::Float64, θmin::Float64,
    specify_params::Bool, total_num_points::Int64, total_time_range::Float64=3000.0, Δλi::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10;
    E::Float64=0.0, L::Float64=0.0, Q::Float64=0.0, C::Float64=0.0, ra::Float64=0.0, p3::Float64=0.0, p4::Float64=0.0, zp::Float64=0.0,
    zm::Float64=0.0, data_path::String="Results/", inspiral::Bool=false)
    save_to_file = false;
    use_custom_ics = true;
    nPoints = total_num_points÷2 + mod(total_num_points, 2);
    λmax = total_time_range/2.0;
    if inspiral
        # future part of geodesic
        λ_f, t_f, r_f, θ_f, ϕ_f, r_dot_f, θ_dot_f, ϕ_dot_f, r_ddot_f, θ_ddot_f, ϕ_ddot_f, Γ_f, psi_f, chi_f, dt_dλ_f = MinoEvolution.compute_kerr_geodesic(a, p, e, θmin, nPoints, use_custom_ics, specify_params, λmax, Δλi, reltol, abstol; 
        ics = ics, E, L, Q, C, ra, p3, p4, zp, zm, save_to_file = save_to_file, inspiral=true)

        # past part of geodesic
        λ_p, t_p, r_p, θ_p, ϕ_p, r_dot_p, θ_dot_p, ϕ_dot_p, r_ddot_p, θ_ddot_p, ϕ_ddot_p, Γ_p, psi_p, chi_p, dt_dλ_p = MinoEvolution.compute_kerr_geodesic_past(a, p, e, θmin, nPoints, use_custom_ics, specify_params, λmax, Δλi, reltol, abstol;
        ics = ics, E, L, Q, C, ra, p3, p4, zp, zm, save_to_file = save_to_file, inspiral=true)

        # merge
        return [λ_p; λ_f], [t_p; t_f], [r_p; r_f], [θ_p; θ_f], [ϕ_p; ϕ_f], [r_dot_p; r_dot_f], [θ_dot_p; θ_dot_f], [ϕ_dot_p; ϕ_dot_f],
        [r_ddot_p; r_ddot_f], [θ_ddot_p; θ_ddot_f], [ϕ_ddot_p; ϕ_ddot_f], [Γ_p; Γ_f], [psi_p; psi_f], [chi_p; chi_f], [dt_dλ_p; dt_dλ_f]
    else
        geodesic_future = MinoEvolution.compute_kerr_geodesic(a, p, e, θmin, nPoints, use_custom_ics, specify_params, λmax, Δλi, reltol, abstol; ics = ics,
        E, L, Q, C, ra, p3, p4, zp, zm, save_to_file = save_to_file)
        geodesic_past = MinoEvolution.compute_kerr_geodesic_past(a, p, e, θmin, nPoints, use_custom_ics, specify_params, λmax, Δλi, reltol, abstol; ics = ics,
        E, L, Q, C, ra, p3, p4, zp, zm, save_to_file = save_to_file)
        return hcat(geodesic_past[:, 1:(nPoints-1)], geodesic_future)
    end
end

end