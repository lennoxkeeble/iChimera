#=

    In this file we evolve the chimera inspiral using the finite difference approach without exiting the ODE solver. This is equivalent to the inspiral in `ChimeraInspiral.jl`, but this version, where one doesn't continously exit and enter the solver,
    is the first step in making the evolution faster. However, at present, the bottleneck preventing this from being faster is the evaluation of the functions for the analytic derivatives of the multipole moments.

=#

module MinoFDMInspiral
using LinearAlgebra
using Combinatorics
using StaticArrays
using HDF5
using DifferentialEquations
using ....Kerr
using ....ConstantsOfMotion
using ....MinoTimeGeodesics
using ....FourierFitGSL
using ....CircularNonEquatorial
import ....HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ....HarmonicCoords
using ....SelfAcceleration
using ....EstimateMultipoleDerivs
using ....SelfAcceleration
using ....SymmetricTensors
using ....EvolveConstants
using ....Waveform
using ....MultipoleFDM
using JLD2
using ...ChimeraInspiral
using DifferentialEquations
using Printf

"""
# Common Arguments in this module
- `tInspiral::Float64`: total coordinate time (in units of M) for the inspiral.
- `nPointsGeodesic::Int64`: number of points in each piecewise geodesic.
- `a::Float64`: Kerr black hole spin parameter 0 < a < 1.
- `p::Float64`: initial semi-latus rectum.
- `e::Float64`: initial eccentricity.
- `θmin::Float64`: initial inclination angle.
- `q::Float64`: mass ratio.
- `psi_0::Float64`: initial r angle variable.
- `chi_0::Float64`: initial θ angle variable.
- `phi_0::Float64`: initial ϕ.
- `h::Float64`: step size for finite differencing.
- `reltol::Float64`: relative tolerance for ODE solver.
- `abstol::Float64`: absolute tolerance for ODE solver.
- `data_path::String`: path to save data.
"""

# master function for computing the inspiral trajectory
function compute_inspiral(tInspiral::Float64, nPointsGeodesic::Int64, a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64=0.001, reltol::Float64=1e-14, abstol::Float64=1e-14; data_path::String="Data/")
    # create arrays for trajectory
    λ = Float64[]; t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[]; dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[]; d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[]; dt_dλ = Float64[];
    λ_temp = zeros(nPointsGeodesic); t_temp = zeros(nPointsGeodesic); r_temp = zeros(nPointsGeodesic); θ_temp = zeros(nPointsGeodesic); ϕ_temp = zeros(nPointsGeodesic); dt_dτ_temp = zeros(nPointsGeodesic); dr_dt_temp = zeros(nPointsGeodesic);
    dθ_dt_temp = zeros(nPointsGeodesic); dϕ_dt_temp = zeros(nPointsGeodesic); d2r_dt2_temp = zeros(nPointsGeodesic); d2θ_dt2_temp = zeros(nPointsGeodesic); d2ϕ_dt2_temp = zeros(nPointsGeodesic); dt_dλ_temp = zeros(nPointsGeodesic);

    Mij2_wf = [Float64[] for i=1:3, j=1:3];
    Mijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf = [Float64[] for i=1:3, j=1:3];
    Sijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];

    Sij2_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3];
    Mijk3_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3, k=1:3];
    Sijk3_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3, k=1:3, l=1:3];

    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    stencil_array_length = 11;   # set by Float64 of points in FDM stencil
    
    # initialize temporary data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()

    xBL_stencil = [zeros(3) for i in 1:stencil_array_length]; xBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    vBL_stencil = [zeros(3) for i in 1:stencil_array_length]; vBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    aBL_stencil = [zeros(3) for i in 1:stencil_array_length]; aBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    xH_stencil = [zeros(3) for i in 1:stencil_array_length];  xH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    vH_stencil = [zeros(3) for i in 1:stencil_array_length];  vH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    v_stencil = zeros(stencil_array_length);   v_wf = zeros(nPointsGeodesic);
    rH_stencil = zeros(stencil_array_length);  rH_wf = zeros(nPointsGeodesic);
    aH_stencil = [zeros(3) for i in 1:stencil_array_length];  aH_wf = [zeros(3) for i in 1:nPointsGeodesic];

    tt_stencil=zeros(stencil_array_length); rr_stencil=zeros(stencil_array_length); r_dot_stencil=zeros(stencil_array_length); r_ddot_stencil=zeros(stencil_array_length); θθ_stencil=zeros(stencil_array_length); 
    θ_dot_stencil=zeros(stencil_array_length); θ_ddot_stencil=zeros(stencil_array_length); ϕϕ_stencil=zeros(stencil_array_length); ϕ_dot_stencil=zeros(stencil_array_length); ϕ_ddot_stencil=zeros(stencil_array_length);
    
    # temporary arrays for multipole moments
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mijkl2_data = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    Sijk1_data= [Float64[] for i=1:3, j=1:3, k=1:3]

    # arrays for self-force computation
    Mij5 = zeros(3, 3)
    Mij6 = zeros(3, 3)
    Mij7 = zeros(3, 3)
    Mij8 = zeros(3, 3)
    Mijk7 = zeros(3, 3, 3)
    Mijk8 = zeros(3, 3, 3)
    Sij5 = zeros(3, 3)
    Sij6 = zeros(3, 3)
    aSF_BL_temp = zeros(4)
    aSF_H_temp = zeros(4)

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi, CCi = ConstantsOfMotion.compute_ELC(a, p, e, θmin, sign_Lz)   

    # store orbital params in arrays
    EE = ones(1) * EEi; 
    Edot = zeros(1);
    LL = ones(1) * LLi; 
    Ldot = zeros(1);
    CC = ones(1) * CCi;
    Cdot = zeros(1);
    QQ = ones(1) * QQi
    Qdot = zeros(1);
    pArray = ones(1) * p;
    ecc = ones(1) * e;
    θminArray = ones(1) * θmin;

    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    t_Fluxes = ones(1) * t0;
    rLSO = ChimeraInspiral.LSO_p(a)

    Δλi=h/10;    # initial time step for geodesic integration

    # intervals between self-force computations set by step size h and number of points in the geodesic
    compute_SF = h * (nPointsGeodesic - 1)

    # initialize ODE porblem
    E, L, Q, C, ra, p3, p4, zp, zm = MinoTimeGeodesics.compute_ODE_params(a, p, e, θmin, sign_Lz);

    params = @SArray [a, E, L, p, e, θmin, p3, p4, zp, zm];
    ics = @SArray[t0, psi_0, chi_0, phi_0];

    # initial conditions for Kerr geodesic trajectory
    λspan = (0.0, tInspiral);

    prob = e == 0.0 ? ODEProblem(MinoTimeGeodesics.HJ_Eqns_circular, ics, λspan, params) : ODEProblem(MinoTimeGeodesics.HJ_Eqns, ics, λspan, params);

    # initialize integrator
    integrator = init(prob, AutoTsit5(RK4()), adaptive=true, dt=Δλi, reltol = reltol, abstol = abstol)

    while integrator.u[1] < tInspiral
        # geodesic evolution
        evolve_inspiral!(integrator, h, compute_SF, λ_temp, t_temp, dt_dτ_temp, r_temp, dr_dt_temp, d2r_dt2_temp, θ_temp, dθ_dt_temp, d2θ_dt2_temp, ϕ_temp, dϕ_dt_temp, d2ϕ_dt2_temp, dt_dλ_temp)

        # store geodesic trajectory
        append!(λ, λ_temp); append!(t, t_temp); append!(r, r_temp); append!(θ, θ_temp); append!(ϕ, ϕ_temp); append!(dt_dτ, dt_dτ_temp); append!(dr_dt, dr_dt_temp); append!(dθ_dt, dθ_dt_temp);
        append!(dϕ_dt, dϕ_dt_temp); append!(d2r_dt2, d2r_dt2_temp); append!(d2θ_dt2, d2θ_dt2_temp); append!(d2ϕ_dt2, d2ϕ_dt2_temp); append!(dt_dλ, dt_dλ_temp);

        ###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θminArray); e_t = last(ecc);
        @views EstimateMultipoleDerivs.FiniteDifferences.compute_waveform_moments_and_derivs_Mino!(a, E_t, L_t, C_t, q, xBL_wf, vBL_wf, aBL_wf, xH_wf, rH_wf, vH_wf, aH_wf, v_wf, t_temp, r_temp,
        dr_dt_temp, d2r_dt2_temp, θ_temp, dθ_dt_temp, d2θ_dt2_temp, ϕ_temp, dϕ_dt_temp, d2ϕ_dt2_temp, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp, nPointsGeodesic, h)

        # store multipole data for waveforms — note that we only save the independent components
        @inbounds Threads.@threads for indices in SymmetricTensors.waveform_indices
            if length(indices)==2
                i1, i2 = indices
                append!(Mij2_wf[i1, i2], Mij2_data[i1, i2])
                append!(Sij2_wf[i1, i2], Sij2_wf_temp[i1, i2])
            elseif length(indices)==3
                i1, i2, i3 = indices
                append!(Mijk3_wf[i1, i2, i3], Mijk3_wf_temp[i1, i2, i3])
                append!(Sijk3_wf[i1, i2, i3], Sijk3_wf_temp[i1, i2, i3])
            else
                i1, i2, i3, i4 = indices
                append!(Mijkl4_wf[i1, i2, i3, i4], Mijkl4_wf_temp[i1, i2, i3, i4])
            end
        end

        # record time at which fluxes are computed
        append!(t_Fluxes, integrator.u[1])
        # compute self force

        # we must construct stencil for self-force computation. We wish to compute the self-force at the final point in a given piecewise geodesic. To get the best approximation, we place this point at the center of the stencil. So, we first take the last
        # 6 points in the piecewise geodesic and store this in the stencil. We then evole the trajectory for another 5 time steps and use these for the remaining 5 points in the stencil, placing the point of interest at the center. A crucial point here
        # is that the extra five points are "ficticious", meaning that we use them to place the final point in the geodesic at the center of the stencil, and one we have computed the necessary derivatives, we discard the extra points. Practically, this
        # is done by deep copying the integrator at the end of the most recent pieceiwse geodesic, and stepping this copied integrator forward to get the extra points. Once we compute the self force and the new constants of motion, we update these in the
        # original integrator and continue the evolution.
        compute_at=stencil_array_length÷2+1
        tt_stencil[1:compute_at]=t_temp[end-compute_at+1:end]
        rr_stencil[1:compute_at]=r_temp[end-compute_at+1:end]
        r_dot_stencil[1:compute_at]=dr_dt_temp[end-compute_at+1:end]
        r_ddot_stencil[1:compute_at]=d2r_dt2_temp[end-compute_at+1:end]
        θθ_stencil[1:compute_at]=θ_temp[end-compute_at+1:end]
        θ_dot_stencil[1:compute_at]=dθ_dt_temp[end-compute_at+1:end]
        θ_ddot_stencil[1:compute_at]=d2θ_dt2_temp[end-compute_at+1:end]
        ϕϕ_stencil[1:compute_at]=ϕ_temp[end-compute_at+1:end]
        ϕ_dot_stencil[1:compute_at]=dϕ_dt_temp[end-compute_at+1:end]
        ϕ_ddot_stencil[1:compute_at]=d2ϕ_dt2_temp[end-compute_at+1:end]

        compute_self_force!(integrator, C_t, aSF_H_temp, aSF_BL_temp, xBL_stencil, vBL_stencil, aBL_stencil, xH_stencil, rH_stencil, vH_stencil, aH_stencil, v_stencil, tt_stencil, rr_stencil, r_dot_stencil, r_ddot_stencil, θθ_stencil, 
            θ_dot_stencil, θ_ddot_stencil, ϕϕ_stencil, ϕ_dot_stencil, ϕ_ddot_stencil, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, q, h, compute_at, stencil_array_length);

        Δt = last(t) - t[end-nPointsGeodesic+1];
        EvolveConstants.Evolve_BL(Δt, a, last(r), last(θ), last(ϕ), last(dt_dτ), last(dr_dt), last(dθ_dt), last(dϕ_dt),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θminArray)

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)

        # update ODE params
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t / (1.0 - e_t); rp=p_t / (1.0 + e_t);
        A = 1.0 / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t); p4 = r4 * (1.0 + e_t)    # Above Eq. 96
        integrator.p = @SArray [a, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm];

        # print progress
        flush(stdout)
        print("Completion: $(round(100 * last(t)/tInspiral; digits=5))%   \r")
    end
    print("Completion: 100%   \r")

    # delete final "extra" energies and fluxes
    pop!(EE)
    pop!(LL)
    pop!(QQ)
    pop!(CC)
    pop!(pArray)
    pop!(ecc)
    pop!(θminArray)

    pop!(Edot)
    pop!(Ldot)
    pop!(Qdot)
    pop!(Cdot)
    pop!(t_Fluxes)

    # save data 
    mkpath(data_path)

    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm_turbo.jld2"
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    sol_filename=data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm_turbo.h5"
    h5open(sol_filename, "w") do file
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
        file["dt_dlambda"] = dt_dλ
        file["t_Fluxes"] = t_Fluxes
        file["Energy"] = EE
        file["AngularMomentum"] = LL
        file["CarterConstant"] = CC
        file["AltCarterConstant"] = QQ
        file["p"] = pArray
        file["eccentricity"] = ecc
        file["theta_min"] = θminArray
        file["Edot"] = Edot
        file["Ldot"] = Ldot
        file["Qdot"] = Qdot
        file["Cdot"] = Cdot
    end

    println("File created: " * sol_filename)
end

# evolve inspiral along one piecewise geodesic
function evolve_inspiral!(integrator, h::Number, Δλ::Number, λλ::Vector{<:Number}, tt::Vector{<:Number}, dt_dτ::Vector{<:Number}, rr::Vector{<:Number}, r_dot::Vector{<:Number}, r_ddot::Vector{<:Number}, θθ::Vector{<:Number}, θ_dot::Vector{<:Number}, θ_ddot::Vector{<:Number}, 
    ϕϕ::Vector{<:Number}, ϕ_dot::Vector{<:Number}, ϕ_ddot::Vector{<:Number}, dt_dλλ::Vector{<:Number})
    a, E, L, p, e, θmin, p3, p4, zp, zm = integrator.p
    track_num_steps = 0
    @inbounds for i = 1:length(λλ)
        track_num_steps += 1
        compute_BL_coords_traj!(integrator, i, λλ, tt, dt_dτ, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, dt_dλλ, a, E, L, p, e, θmin, p3, p4, zp, zm)
        step!(integrator, h, true)
    end
    if track_num_steps != length(λλ)
        throw(ArgumentError("Length of λλ array does not match the number ($(i)) of steps taken"))
    end
end

# takes the state of the integrator at a given step in the evolution and converts saves this states to the parent data arrays
function compute_BL_coords_traj!(integrator, i::Int, λλ::Vector{<:Number}, tt::Vector{<:Number}, dt_dτ::Vector{<:Number}, rr::Vector{<:Number}, r_dot::Vector{<:Number}, r_ddot::Vector{<:Number}, θθ::Vector{<:Number}, θ_dot::Vector{<:Number}, θ_ddot::Vector{<:Number}, 
    ϕϕ::Vector{<:Number}, ϕ_dot::Vector{<:Number}, ϕ_ddot::Vector{<:Number}, dt_dλλ::Vector{<:Number}, a::Number, E::Number, L::Number, p::Number, e::Number, θmin::Number, p3::Number, p4::Number, zp::Number, zm::Number)
    λ = integrator.t;
    t = integrator.u[1];
    psi = integrator.u[2];
    chi = mod(integrator.u[3], 2π);
    ϕ = integrator.u[4];

    # compute time derivatives (wrt λ)
    dt_dλ = MinoTimeGeodesics.dt_dλ(λ, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dψ_dλ = MinoTimeGeodesics.dψ_dλ(λ, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dχ_dλ = MinoTimeGeodesics.dχ_dλ(λ, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dϕ_dλ = MinoTimeGeodesics.dϕ_dλ(λ, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)

    # compute BL coordinates r, θ and their time derivatives (wrt λ)
    r = MinoTimeGeodesics.r(psi, p, e)
    θ = acos((π/2<chi<1.5π) ? -sqrt(MinoTimeGeodesics.z(chi, θmin)) : sqrt(MinoTimeGeodesics.z(chi, θmin)))

    dr_dλ = MinoTimeGeodesics.dr_dλ(dψ_dλ, psi, p, e);
    dθ_dλ = MinoTimeGeodesics.dθ_dλ(dχ_dλ, chi, θ, θmin);

    # compute derivatives wrt t
    dr_dt = dr_dλ / dt_dλ
    dθ_dt = dθ_dλ / dt_dλ 
    dϕ_dt = dϕ_dλ / dt_dλ 

    # compute derivatives wrt τ
    v = [dr_dt, dθ_dt, dϕ_dt]; # v=dxi/dt

    Γ = MinoTimeGeodesics.Γ(r, θ, ϕ, v, a)

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    d2r_dt2 = MinoTimeGeodesics.dr2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)
    d2θ_dt2 = MinoTimeGeodesics.dθ2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)
    d2ϕ_dt2 = MinoTimeGeodesics.dϕ2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)

    λλ[i] =  λ;
    tt[i] =  t; 
    dt_dτ[i] =  Γ; 
    rr[i] =  r; 
    r_dot[i] =  dr_dt; 
    r_ddot[i] =  d2r_dt2; 
    θθ[i] =  θ; 
    θ_dot[i] =  dθ_dt; 
    θ_ddot[i] =  d2θ_dt2; 
    ϕϕ[i] =  ϕ; 
    ϕ_dot[i] =  dϕ_dt;
    ϕ_ddot[i] =  d2ϕ_dt2;
    dt_dλλ[i] =  dt_dλ;
end

# specialized version of the above function to extract the state of the integrator and store it in the FDM stencils
function compute_BL_coords_SF!(integrator, i::Int, tt::Vector{<:Number}, rr::Vector{<:Number}, r_dot::Vector{<:Number}, r_ddot::Vector{<:Number}, θθ::Vector{<:Number}, θ_dot::Vector{<:Number}, θ_ddot::Vector{<:Number}, 
    ϕϕ::Vector{<:Number}, ϕ_dot::Vector{<:Number}, ϕ_ddot::Vector{<:Number}, a::Number, E::Number, L::Number, p::Number, e::Number, θmin::Number, p3::Number, p4::Number, zp::Number, zm::Number)
    λ = integrator.t;
    t = integrator.u[1];
    psi = integrator.u[2];
    chi = mod(integrator.u[3], 2π);
    ϕ = integrator.u[4];

    # compute time derivatives (wrt λ)
    dt_dλ = MinoTimeGeodesics.dt_dλ(λ, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dψ_dλ = MinoTimeGeodesics.dψ_dλ(λ, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dχ_dλ = MinoTimeGeodesics.dχ_dλ(λ, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dϕ_dλ = MinoTimeGeodesics.dϕ_dλ(λ, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)

    # compute BL coordinates r, θ and their time derivatives (wrt λ)
    r = MinoTimeGeodesics.r(psi, p, e)
    θ = acos((π/2<chi<1.5π) ? -sqrt(MinoTimeGeodesics.z(chi, θmin)) : sqrt(MinoTimeGeodesics.z(chi, θmin)))

    dr_dλ = MinoTimeGeodesics.dr_dλ(dψ_dλ, psi, p, e);
    dθ_dλ = MinoTimeGeodesics.dθ_dλ(dχ_dλ, chi, θ, θmin);

    # compute derivatives wrt t
    dr_dt = dr_dλ / dt_dλ
    dθ_dt = dθ_dλ / dt_dλ 
    dϕ_dt = dϕ_dλ / dt_dλ 

    # compute derivatives wrt τ
    v = [dr_dt, dθ_dt, dϕ_dt]; # v=dxi/dt

    Γ = MinoTimeGeodesics.Γ(r, θ, ϕ, v, a)

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    d2r_dt2 = MinoTimeGeodesics.dr2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)
    d2θ_dt2 = MinoTimeGeodesics.dθ2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)
    d2ϕ_dt2 = MinoTimeGeodesics.dϕ2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)


    tt[i] = t
    rr[i] = r 
    r_dot[i] = dr_dt 
    r_ddot[i] = d2r_dt2 
    θθ[i] = θ 
    θ_dot[i] = dθ_dt 
    θ_ddot[i] = d2θ_dt2 
    ϕϕ[i] = ϕ 
    ϕ_dot[i] = dϕ_dt
    ϕ_ddot[i] = d2ϕ_dt2 
end

# computes the self-force
function compute_self_force!(integrator, C::Number, aSF_H::Vector{<:Number}, aSF_BL::Vector{<:Number}, xBL_stencil::AbstractArray, vBL_stencil::AbstractArray, aBL_stencil::AbstractArray, xH_stencil::AbstractArray,
    rH_stencil::AbstractArray, vH_stencil::AbstractArray, aH_stencil::AbstractArray, v_stencil::AbstractArray, tt_stencil::Vector{<:Number}, rr_stencil::Vector{<:Number},
    r_dot_stencil::Vector{<:Number}, r_ddot_stencil::Vector{<:Number}, θθ_stencil::Vector{<:Number}, θ_dot_stencil::Vector{<:Number}, θ_ddot_stencil::Vector{<:Number}, ϕϕ_stencil::Vector{<:Number}, ϕ_dot_stencil::Vector{<:Number},
    ϕ_ddot_stencil::Vector{<:Number}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray,
    Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray,
    q::Number, h::Number, compute_at::Int, stencil_array_length::Int)
    
    copied_integrator = deepcopy(integrator)
    a, E, L, p, e, θmin, p3, p4, zp, zm = copied_integrator.p

    for i=compute_at+1:stencil_array_length
        # evolve geodesic
        compute_BL_coords_SF!(copied_integrator, i, tt_stencil, rr_stencil, r_dot_stencil, r_ddot_stencil, θθ_stencil, θ_dot_stencil, θ_ddot_stencil, ϕϕ_stencil, ϕ_dot_stencil, ϕ_ddot_stencil, a, E, L, p, e, θmin, p3, p4, zp, zm)
        step!(copied_integrator, h, true)
    end

    SelfAcceleration.FiniteDifferences.selfAcc_mino!(a, E, L, C, aSF_H, aSF_BL, xBL_stencil, vBL_stencil, aBL_stencil, xH_stencil,
        rH_stencil, vH_stencil, aH_stencil, v_stencil, tt_stencil, rr_stencil, r_dot_stencil, r_ddot_stencil, θθ_stencil, 
        θ_dot_stencil, θ_ddot_stencil, ϕϕ_stencil, ϕ_dot_stencil, ϕ_ddot_stencil, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, 
        Mijk2_data, Sij1_data, q, compute_at, h);

end

# functions to load the solution
function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    sol_filename=data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm_turbo.h5"
    h5f = h5open(sol_filename, "r")
    λ = h5f["lambda"][:]
    t = h5f["t"][:]
    r = h5f["r"][:]
    θ = h5f["theta"][:]
    ϕ = h5f["phi"][:]
    dr_dt = h5f["r_dot"][:]
    dθ_dt = h5f["theta_dot"][:]
    dϕ_dt = h5f["phi_dot"][:]
    d2r_dt2 = h5f["r_ddot"][:]
    d2θ_dt2 = h5f["theta_ddot"][:]
    d2ϕ_dt2 = h5f["phi_ddot"][:]
    dt_dτ = h5f["Gamma"][:]
    dt_dλ = h5f["dt_dlambda"][:]
    close(h5f)
    return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    sol_filename=data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm_turbo.h5"
    h5f = h5open(sol_filename, "r")
    t_Fluxes = h5f["t_Fluxes"][:]
    EE = h5f["Energy"][:]
    LL = h5f["AngularMomentum"][:]
    QQ = h5f["AltCarterConstant"][:]
    CC = h5f["CarterConstant"][:]
    pArray = h5f["p"][:]
    ecc = h5f["eccentricity"][:]
    θmin = h5f["theta_min"][:]
    Edot = h5f["Edot"][:]
    Ldot = h5f["Ldot"][:]
    Qdot = h5f["Qdot"][:]
    Cdot = h5f["Cdot"][:]
    close(h5f)
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm_turbo.jld2"
    waveform_data = load(waveform_filename)["data"]
    Mij2 = waveform_data["Mij2"]; SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij2);
    Mijk3 = waveform_data["Mijk3"]; SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3);
    Mijkl4 = waveform_data["Mijkl4"]; SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    Sij2 = waveform_data["Sij2"]; SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2);
    Sijk3 = waveform_data["Sijk3"]; SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);

    # compute h_{ij} tensor
    num_points = length(Mij2[1, 1]);
    hij = [zeros(num_points) for i=1:3, j=1:3];
    Waveform.hij!(hij, num_points, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    # project h_{ij} tensor
    h_plus, h_cross = Waveform.h_plus_cross(hij, Θ, Φ);
    return h_plus, h_cross
end

# useful for dummy runs (e.g., for resonances to estimate the duration of time needed by computing the time derivative of the fundamental frequencies)
function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    sol_filename=data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm_turbo.h5"
    rm(sol_filename)

    waveform_filename=data_path *  "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm_turbo.jld2"
    rm(waveform_filename)
end

end