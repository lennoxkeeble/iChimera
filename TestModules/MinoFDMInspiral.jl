#=
    In this file we evolve the chimera inspiral using FDM which allows one to avoid exiting the solver.

=#
module MinoFDMInspiral
using LinearAlgebra
using Combinatorics
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ....Kerr
using ....MinoTimeEvolution
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
using FileIO
using ...InspiralEvolution
using DifferentialEquations
# using SciMLBase

function compute_inspiral_HJE!(tOrbit::Float64, nPointsGeodesic::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64, 
    Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function,
    gRR::Function, gThTh::Function, gΦΦ::Function, h::Float64=0.15, reltol::Float64=1e-14, abstol::Float64=1e-14; data_path::String="Data/")


    # create arrays for trajectory
    λ = zeros(1); t = zeros(1); r = zeros(1); θ = zeros(1); ϕ = zeros(1); dt_dτ = zeros(1); dr_dt = zeros(1); dθ_dt = zeros(1); dϕ_dt = zeros(1); d2r_dt2 = zeros(1); d2θ_dt2 = zeros(1); d2ϕ_dt2 = zeros(1); dt_dλ = zeros(1);
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
    
    # initialize data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()

    xBL_stencil = [Float64[] for i in 1:stencil_array_length]; xBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    vBL_stencil = [Float64[] for i in 1:stencil_array_length]; vBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    aBL_stencil = [Float64[] for i in 1:stencil_array_length]; aBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    xH_stencil = [Float64[] for i in 1:stencil_array_length];  xH_wf = [Float64[] for i in 1:nPointsGeodesic];
    x_H_stencil = [Float64[] for i in 1:stencil_array_length]; x_H_wf = [Float64[] for i in 1:nPointsGeodesic];
    vH_stencil = [Float64[] for i in 1:stencil_array_length];  vH_wf = [Float64[] for i in 1:nPointsGeodesic];
    v_H_stencil = [Float64[] for i in 1:stencil_array_length]; v_H_wf = [Float64[] for i in 1:nPointsGeodesic];
    v_stencil = zeros(stencil_array_length);   v_wf = zeros(nPointsGeodesic);
    rH_stencil = zeros(stencil_array_length);  rH_wf = zeros(nPointsGeodesic);
    aH_stencil = [Float64[] for i in 1:stencil_array_length];  aH_wf = [Float64[] for i in 1:nPointsGeodesic];
    a_H_stencil = [Float64[] for i in 1:stencil_array_length]; a_H_wf = [Float64[] for i in 1:nPointsGeodesic];

    tt_stencil=zeros(stencil_array_length); rr_stencil=zeros(stencil_array_length); r_dot_stencil=zeros(stencil_array_length); r_ddot_stencil=zeros(stencil_array_length); θθ_stencil=zeros(stencil_array_length); 
    θ_dot_stencil=zeros(stencil_array_length); θ_ddot_stencil=zeros(stencil_array_length); ϕϕ_stencil=zeros(stencil_array_length); ϕ_dot_stencil=zeros(stencil_array_length); ϕ_ddot_stencil=zeros(stencil_array_length);
    
    # arrays for multipole moments
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

    # compute apastron
    ra = p * M / (1 - e);

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi, CCi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)   

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
    θmin = ones(1) * θi;

    # initial condition for Kerr geodesic trajectory
    λ0 = 0.0;
    t0 = 0.0
    t_Fluxes = ones(1) * t0;
    rLSO = InspiralEvolution.LSO_p(a, M)

    Δλi=h/10;    # initial time step for geodesic integration

    compute_SF = h * (nPointsGeodesic - 1)
    # initialize ODE porblem
    E, L, Q, C, ra, p3, p4, zp, zm = MinoTimeEvolution.compute_ODE_params(a, p, e, θi);

    params = @SArray [a, M, E, L, p, e, θi, p3, p4, zp, zm];
    ics = MinoTimeEvolution.Mino_ics(0.0, ra, p, e, M);

    # initial conditions for Kerr geodesic trajectory
    λspan = (0.0, tOrbit);

    prob = e == 0.0 ? ODEProblem(MinoTimeEvolution.HJ_Eqns_circular, ics, λspan, params) : ODEProblem(MinoTimeEvolution.HJ_Eqns, ics, λspan, params);

    # initialize integrator
    integrator = init(prob, AutoTsit5(RK4()), adaptive=true, dt=Δλi, reltol = reltol, abstol = abstol)

    # store initial conditions

    # store initial conditions
    compute_BL_coords_traj!(integrator, 1, λ, t, dt_dτ, r, dr_dt, d2r_dt2, θ, dθ_dt, d2θ_dt2, ϕ, dϕ_dt, d2ϕ_dt2, dt_dλ, a, M, E, L, p, e, θi, p3, p4, zp, zm)

    while integrator.u[1] < tOrbit
        # geodesic evolution
        evolve_inspiral!(integrator, h, compute_SF, λ_temp, t_temp, dt_dτ_temp, r_temp, dr_dt_temp, d2r_dt2_temp, θ_temp, dθ_dt_temp, d2θ_dt2_temp, ϕ_temp, dϕ_dt_temp, d2ϕ_dt2_temp, dt_dλ_temp)

        # store geodesic trajectory
        append!(λ, λ_temp); append!(t, t_temp); append!(r, r_temp); append!(θ, θ_temp); append!(ϕ, ϕ_temp); append!(dt_dτ, dt_dτ_temp); append!(dr_dt, dr_dt_temp); append!(dθ_dt, dθ_dt_temp);
        append!(dϕ_dt, dϕ_dt_temp); append!(d2r_dt2, d2r_dt2_temp); append!(d2θ_dt2, d2θ_dt2_temp); append!(d2ϕ_dt2, d2ϕ_dt2_temp); append!(dt_dλ, dt_dλ_temp);

        ###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θmin); e_t = last(ecc);
        @views EstimateMultipoleDerivs.FiniteDifferences.compute_waveform_moments_and_derivs_Mino!(a, E_t, L_t, C_t, m, M, xBL_wf, vBL_wf, aBL_wf, xH_wf, x_H_wf, rH_wf, vH_wf, v_H_wf, aH_wf, a_H_wf, v_wf, t_temp, r_temp,
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

        append!(t_Fluxes, integrator.u[1])
        # compute self force

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

        compute_self_force!(integrator, C_t, aSF_H_temp, aSF_BL_temp, xBL_stencil, vBL_stencil, aBL_stencil, xH_stencil, x_H_stencil,
            rH_stencil, vH_stencil, v_H_stencil, aH_stencil, a_H_stencil, v_stencil, tt_stencil, rr_stencil, r_dot_stencil, r_ddot_stencil, θθ_stencil, 
            θ_dot_stencil, θ_ddot_stencil, ϕϕ_stencil, ϕ_dot_stencil, ϕ_ddot_stencil, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, 
            Mijk2_data, Sij1_data, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, m, h, compute_at, stencil_array_length);

        Δt = last(t) - t[end-nPointsGeodesic+1];
        EvolveConstants.Evolve_BL(Δt, a, last(t), last(r), last(θ), last(ϕ), last(dt_dτ), last(dr_dt), last(dθ_dt), last(dϕ_dt),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPointsGeodesic)

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)

        # update ODE params
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t * M / (1.0 - e_t); rp=p_t * M / (1.0 + e_t);
        A = M / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t) / M; p4 = r4 * (1.0 + e_t) / M    # Above Eq. 96

        integrator.p = @SArray [a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm];
        flush(stdout)
        print("Completion: $(round(100 * last(t)/tOrbit; digits=5))%   \r")
    end
    print("Completion: 100%   \r")

    # delete final "extra" energies and fluxes
    pop!(EE)
    pop!(LL)
    pop!(QQ)
    pop!(CC)
    pop!(pArray)
    pop!(ecc)
    pop!(θmin)

    pop!(Edot)
    pop!(Ldot)
    pop!(Qdot)
    pop!(Cdot)
    pop!(t_Fluxes)

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end


    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end


    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_mino_fdm_turbo.jld2"
    waveform_dictionary = Dict{String, AbstractArray}("time" => t, "Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    # save params
    constants = (t_Fluxes, EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, h::Float64, reltol::Float64, data_path::String)
    # load ODE solution
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    sol = readdlm(ODE_filename)
    λ=sol[1,:]; t=sol[2,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt2=sol[9,:]; d2θ_dt2=sol[10,:]; d2ϕ_dt2=sol[11,:]; dt_dτ=sol[12,:]; dt_dλ=sol[13,:]
    return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, h::Float64, reltol::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    constants=readdlm(constants_filename)
    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    constants_derivs = readdlm(constants_derivs_filename)
    t_Fluxes, EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :], constants[8, :]
    Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, 
    h::Float64, reltol::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=data_path *  "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_mino_fdm_turbo.jld2"
    waveform_data = load(waveform_filename)["data"]
    t = waveform_data["time"]
    Mij2 = waveform_data["Mij2"]; SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij2);
    Mijk3 = waveform_data["Mijk3"]; SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3);
    Mijkl4 = waveform_data["Mijkl4"]; SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    Sij2 = waveform_data["Sij2"]; SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2);
    Sijk3 = waveform_data["Sijk3"]; SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);

    # compute h_{ij} tensor
    num_points = length(t);
    hij = [zeros(num_points) for i=1:3, j=1:3];
    Waveform.hij!(hij, num_points, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    # project h_{ij} tensor
    h_plus = Waveform.hplus(hij, Θ, Φ);
    h_cross = Waveform.hcross(hij, Θ, Φ);
    return t, h_plus, h_cross
end

function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, h::Float64, reltol::Float64, data_path::String)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    rm(SF_filename)

    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    rm(SF_filename)

    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    rm(ODE_filename)

    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_mino_fdm_turbo.jld2"
    rm(waveform_filename)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    rm(constants_filename)

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm_turbo.txt"
    rm(constants_derivs_filename)
end

function evolve_inspiral!(integrator, h::Number, Δλ::Number, λλ::Vector{<:Number}, tt::Vector{<:Number}, dt_dτ::Vector{<:Number}, rr::Vector{<:Number}, r_dot::Vector{<:Number}, r_ddot::Vector{<:Number}, θθ::Vector{<:Number}, θ_dot::Vector{<:Number}, θ_ddot::Vector{<:Number}, 
    ϕϕ::Vector{<:Number}, ϕ_dot::Vector{<:Number}, ϕ_ddot::Vector{<:Number}, dt_dλλ::Vector{<:Number})
    a, M, E, L, p, e, θi, p3, p4, zp, zm = integrator.p
    track_num_steps = 0
    @inbounds for i = 1:length(λλ)
        track_num_steps += 1
        step!(integrator, h, true)
        compute_BL_coords_traj!(integrator, i, λλ, tt, dt_dτ, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, dt_dλλ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    end
    if track_num_steps != length(λλ)
        throw(ArgumentError("Length of λλ array does not match the number ($(i)) of steps taken"))
    end
end

function compute_BL_coords_traj!(integrator, i::Int, λλ::Vector{<:Number}, tt::Vector{<:Number}, dt_dτ::Vector{<:Number}, rr::Vector{<:Number}, r_dot::Vector{<:Number}, r_ddot::Vector{<:Number}, θθ::Vector{<:Number}, θ_dot::Vector{<:Number}, θ_ddot::Vector{<:Number}, 
    ϕϕ::Vector{<:Number}, ϕ_dot::Vector{<:Number}, ϕ_ddot::Vector{<:Number}, dt_dλλ::Vector{<:Number}, a::Number, M::Number, E::Number, L::Number, p::Number, e::Number, θi::Number, p3::Number, p4::Number, zp::Number, zm::Number)
    λ = integrator.t;
    t = integrator.u[1];
    psi = integrator.u[2];
    chi = mod(integrator.u[3], 2π);
    ϕ = integrator.u[4];

    # compute time derivatives (wrt λ)
    dt_dλ = MinoTimeEvolution.dt_dλ(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dψ_dλ = MinoTimeEvolution.dψ_dλ(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dχ_dλ = MinoTimeEvolution.dχ_dλ(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dϕ_dλ = MinoTimeEvolution.dϕ_dλ(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)

    # compute BL coordinates r, θ and their time derivatives (wrt λ)
    r = MinoTimeEvolution.r(psi, p, e, M)
    θ = acos((π/2<chi<1.5π) ? -sqrt(MinoTimeEvolution.z(chi, θi)) : sqrt(MinoTimeEvolution.z(chi, θi)))

    dr_dλ = MinoTimeEvolution.dr_dλ(dψ_dλ, psi, p, e, M);
    dθ_dλ = MinoTimeEvolution.dθ_dλ(dχ_dλ, chi, θ, θi);

    # compute derivatives wrt t
    dr_dt = dr_dλ / dt_dλ
    dθ_dt = dθ_dλ / dt_dλ 
    dϕ_dt = dϕ_dλ / dt_dλ 

    # compute derivatives wrt τ
    v = [dr_dt, dθ_dt, dϕ_dt]; # v=dxi/dt

    Γ = MinoTimeEvolution.Γ(t, r, θ, ϕ, v, a, M)

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    d2r_dt2 = MinoTimeEvolution.dr2_dt2(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
    d2θ_dt2 = MinoTimeEvolution.dθ2_dt2(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
    d2ϕ_dt2 = MinoTimeEvolution.dϕ2_dt2(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)

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

function compute_BL_coords_SF!(integrator, i::Int, tt::Vector{<:Number}, rr::Vector{<:Number}, r_dot::Vector{<:Number}, r_ddot::Vector{<:Number}, θθ::Vector{<:Number}, θ_dot::Vector{<:Number}, θ_ddot::Vector{<:Number}, 
    ϕϕ::Vector{<:Number}, ϕ_dot::Vector{<:Number}, ϕ_ddot::Vector{<:Number}, a::Number, M::Number, E::Number, L::Number, p::Number, e::Number, θi::Number, p3::Number, p4::Number, zp::Number, zm::Number)
    λ = integrator.t;
    t = integrator.u[1];
    psi = integrator.u[2];
    chi = mod(integrator.u[3], 2π);
    ϕ = integrator.u[4];

    # compute time derivatives (wrt λ)
    dt_dλ = MinoTimeEvolution.dt_dλ(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dψ_dλ = MinoTimeEvolution.dψ_dλ(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dχ_dλ = MinoTimeEvolution.dχ_dλ(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    dϕ_dλ = MinoTimeEvolution.dϕ_dλ(λ, psi, chi, ϕ, a, M, E, L, p, e, θi, p3, p4, zp, zm)

    # compute BL coordinates r, θ and their time derivatives (wrt λ)
    r = MinoTimeEvolution.r(psi, p, e, M)
    θ = acos((π/2<chi<1.5π) ? -sqrt(MinoTimeEvolution.z(chi, θi)) : sqrt(MinoTimeEvolution.z(chi, θi)))

    dr_dλ = MinoTimeEvolution.dr_dλ(dψ_dλ, psi, p, e, M);
    dθ_dλ = MinoTimeEvolution.dθ_dλ(dχ_dλ, chi, θ, θi);

    # compute derivatives wrt t
    dr_dt = dr_dλ / dt_dλ
    dθ_dt = dθ_dλ / dt_dλ 
    dϕ_dt = dϕ_dλ / dt_dλ 

    # compute derivatives wrt τ
    v = [dr_dt, dθ_dt, dϕ_dt]; # v=dxi/dt

    Γ = MinoTimeEvolution.Γ(t, r, θ, ϕ, v, a, M)

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    d2r_dt2 = MinoTimeEvolution.dr2_dt2(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
    d2θ_dt2 = MinoTimeEvolution.dθ2_dt2(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)
    d2ϕ_dt2 = MinoTimeEvolution.dϕ2_dt2(t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a, M)


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

function compute_self_force!(integrator, C::Number, aSF_H::Vector{<:Number}, aSF_BL::Vector{<:Number}, xBL_stencil::AbstractArray, vBL_stencil::AbstractArray, aBL_stencil::AbstractArray, xH_stencil::AbstractArray, x_H_stencil::AbstractArray,
    rH_stencil::AbstractArray, vH_stencil::AbstractArray, v_H_stencil::AbstractArray, aH_stencil::AbstractArray, a_H_stencil::AbstractArray, v_stencil::AbstractArray, tt_stencil::Vector{<:Number}, rr_stencil::Vector{<:Number},
    r_dot_stencil::Vector{<:Number}, r_ddot_stencil::Vector{<:Number}, θθ_stencil::Vector{<:Number}, θ_dot_stencil::Vector{<:Number}, θ_ddot_stencil::Vector{<:Number}, ϕϕ_stencil::Vector{<:Number}, ϕ_dot_stencil::Vector{<:Number},
    ϕ_ddot_stencil::Vector{<:Number}, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray,
    Sij5::AbstractArray, Sij6::AbstractArray, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Sij1_data::AbstractArray, Γαμν::Function, g_μν::Function,
    g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, m::Number, h::Number, compute_at::Int, stencil_array_length::Int)
    
    copied_integrator = deepcopy(integrator)
    a, M, E, L, p, e, θi, p3, p4, zp, zm = copied_integrator.p

    for i=compute_at+1:stencil_array_length
        # evolve geodesic
        step!(copied_integrator, h, true)
        compute_BL_coords_SF!(copied_integrator, i, tt_stencil, rr_stencil, r_dot_stencil, r_ddot_stencil, θθ_stencil, θ_dot_stencil, θ_ddot_stencil, ϕϕ_stencil, ϕ_dot_stencil, ϕ_ddot_stencil, a, M, E, L, p, e, θi, p3, p4, zp, zm)
    end

    SelfAcceleration.FiniteDifferences.selfAcc_mino!(a, M, E, L, C, aSF_H, aSF_BL, xBL_stencil, vBL_stencil, aBL_stencil, xH_stencil, x_H_stencil,
        rH_stencil, vH_stencil, v_H_stencil, aH_stencil, a_H_stencil, v_stencil, tt_stencil, rr_stencil, r_dot_stencil, r_ddot_stencil, θθ_stencil, 
        θ_dot_stencil, θ_ddot_stencil, ϕϕ_stencil, ϕ_dot_stencil, ϕ_ddot_stencil, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, 
        Mijk2_data, Sij1_data, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, m, compute_at, h);

end
end