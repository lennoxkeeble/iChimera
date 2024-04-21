#=

In this module we evolve the separated Hamilton Jacobi equation with respect to Boyer-Lindquist time t. See Sec. IVA in Sopuerta, Yunes 2011 (arXiv:1109.0572v2) for further details.

=#

module HJEvolution
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
    r = HJEvolution.r(psi, p, e, M)
    Δ = HJEvolution.Δ(r, a, M)
    return (((r^2 + a^2)^2) / Δ - a^2) * E - 2.0M * a * r * L / Δ
end

function psi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    return M * sqrt((1.0 - E^2) * (p * (1.0 - e) - p3 * (1.0 + e * cos(psi))) * (p * (1.0 + e) - p4 * (1.0 + e * cos(psi)))) / ((1.0 - e^2) * (Ψ(psi, a, p, e, M, E, L) + a^2 * E * z(chi, θmin)))
end

function chi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    return sqrt(a^2 * (1.0 - E^2) * (zp - zm * cos(chi)^2)) / (Ψ(psi, a, p, e, M, E, L) + a^2 * E * z(chi, θmin))
end

function phi_dot(psi::Float64, chi::Float64, ϕ::Float64, a::Float64, M::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    r = HJEvolution.r(psi, p, e, M)
    Δ = HJEvolution.Δ(r, a, M)
    return  (2.0M * a * r * E / Δ + (1.0 / (1.0 - z(chi, θmin)) - a^2 / Δ) * L) / (Ψ(psi, a, p, e, M, E, L) + a^2 * E * z(chi, θmin))
end

# initial conditions for bound kerr orbits starting in equatorial plane
function HJ_ics(ri::Float64, p::Float64, e::Float64, M::Float64)
    psi_i = acos((p * M / ri - 1.0) / e)    # Eq. 89
    chi_i = 0.0    # Eq. 89 - since we start orbit at θ = θmin
    ϕi = 0.0    # by axisymmetry can start orbit at ϕ = 0
    return @SArray [psi_i, chi_i, ϕi]
end

# equation for ODE solver
function geodesicEq(u, params, t)
    @SArray [psi_dot(u..., params...), chi_dot(u..., params...), phi_dot(u..., params...)]
end

# computes trajectory in Kerr characterized by a, p, e, θi (M=1, μ=1)
function compute_kerr_geodesic(a::Float64, p::Float64, e::Float64, θi::Float64, nPoints::Int64, tmax::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-12, abstol::Float64=1e-10, saveat::Float64=0.5; data_path::String="Results/")
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
    params = @SArray [a, M, E, L, p, e, θi, p3, p4, zp, zm]

    # initial conditions for Kerr geodesic trajectory
    ri = ra; tspan = (0.0, tmax); saveat_t = range(start=tspan[1], length=nPoints, stop=tspan[2])

    ics = HJ_ics(ri, p, e, M);
    prob = ODEProblem(geodesicEq, ics, tspan, params);
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);
 
    # deconstruct solution
    t = sol.t;
    psi = sol[1, :];
    chi = mod.(sol[2, :], 2π);
    ϕ = sol[3, :];

    # compute time derivatives
    psi_dot = HJEvolution.psi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    chi_dot = HJEvolution.chi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    ϕ_dot = HJEvolution.phi_dot.(psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)

    # compute BL coordinates t, r, θ and their time derivatives
    r = HJEvolution.r.(psi, p, e, M)
    θ = [acos((π/2<chi[i]<1.5π) ? -sqrt(HJEvolution.z(chi[i], θi)) : sqrt(HJEvolution.z(chi[i], θi))) for i in eachindex(chi)]

    r_dot = dr_dt.(psi_dot, psi, p, e, M);
    θ_dot = dθ_dt.(chi_dot, chi, θ, θi);
    v = [[r_dot[i], θ_dot[i], ϕ_dot[i]] for i in eachindex(t)];
    dt_dτ = @. Γ(t, r, θ, ϕ, v, a, M)

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    r_ddot = HJEvolution.dr2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)
    θ_ddot = HJEvolution.dθ2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)
    ϕ_ddot = HJEvolution.dϕ2_dt2.(t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a, M)

    # save trajectory- rows are: t, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ]))
    mkpath(data_path)
    ODE_filename=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(reltol).txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end
    println("ODE saved to: " * ODE_filename)
end

end