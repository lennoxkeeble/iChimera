#=

    In this module we compute an inspiral in Mino Time

=#

module MinoInspiral

using LinearAlgebra
using Combinatorics
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ..Kerr
using ..MinoTimeEvolution
using ..FourierFitGSL
using ..CircularNonEquatorial
import ..HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ..HarmonicCoords
using ..EstimateMultipoleDerivs
using ..SelfAcceleration
using ..EvolveConstants
using ..SymmetricTensors
using JLD2
using FileIO


Z_1(a::Float64, M::Float64) = 1 + (1 - a^2 / M^2)^(1/3) * ((1 + a / M)^(1/3) + (1 - a / M)^(1/3))
Z_2(a::Float64, M::Float64) = sqrt(3 * (a / M)^2 + Z_1(a, M)^2)
LSO_r(a::Float64, M::Float64) = M * (3 + Z_2(a, M) - sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # retrograde LSO
LSO_p(a::Float64, M::Float64) = M * (3 + Z_2(a, M) + sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # prograde LSO


function compute_inspiral!(t_range_factor::Float64, tOrbit::Float64, nPoints::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, nHarm::Int64, reltol::Float64=1e-12, abstol::Float64=1e-10; data_path::String="Data/")
    # create arrays for trajectory
    λ = Float64[]; t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[];
    Mij2_wf = [Float64[] for i=1:3, j=1:3];
    Mijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf = [Float64[] for i=1:3, j=1:3];
    Sijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    
    # initialize data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    fit_array_length = iseven(nPoints) ? nPoints+1 : nPoints
    xBL = [Float64[] for i in 1:fit_array_length]
    vBL = [Float64[] for i in 1:fit_array_length]
    aBL = [Float64[] for i in 1:fit_array_length]
    xH = [Float64[] for i in 1:fit_array_length]
    x_H = [Float64[] for i in 1:fit_array_length]
    vH = [Float64[] for i in 1:fit_array_length]
    v_H = [Float64[] for i in 1:fit_array_length]
    v = zeros(fit_array_length)
    rH = zeros(fit_array_length)
    aH = [Float64[] for i in 1:fit_array_length]
    a_H = [Float64[] for i in 1:fit_array_length]

    # arrays for waveform computation
    xBL_wf = [Float64[] for i in 1:nPoints]
    vBL_wf = [Float64[] for i in 1:nPoints]
    aBL_wf = [Float64[] for i in 1:nPoints]
    xH_wf = [Float64[] for i in 1:nPoints]
    x_H_wf = [Float64[] for i in 1:nPoints]
    vH_wf = [Float64[] for i in 1:nPoints]
    v_H_wf = [Float64[] for i in 1:nPoints]
    v_wf = zeros(nPoints)
    rH_wf = zeros(nPoints)
    aH_wf = [Float64[] for i in 1:nPoints]
    a_H_wf = [Float64[] for i in 1:nPoints]

    # arrays for multipole moments
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Mijkl2_data = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    Sijk1_data= [Float64[] for i=1:3, j=1:3, k=1:3]

    # "temporary" mulitpole arrays which contain the multipole data for a given piecewise geodesic
    Mij2_wf_temp = [Float64[] for i=1:3, j=1:3];
    Mijk3_wf_temp = [Float64[] for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf_temp = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf_temp = [Float64[] for i=1:3, j=1:3];
    Sijk3_wf_temp = [Float64[] for i=1:3, j=1:3, k=1:3];

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
    EE = ones(nPoints) * EEi; 
    Edot = zeros(nPoints-1);
    LL = ones(nPoints) * LLi; 
    Ldot = zeros(nPoints-1);
    CC = ones(nPoints) * CCi;
    Cdot = zeros(nPoints-1);
    QQ = ones(nPoints) * QQi
    Qdot = zeros(nPoints-1);
    pArray = ones(nPoints) * p;
    ecc = ones(nPoints) * e;
    θmin = ones(nPoints) * θi;

    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    λ0 = 0.0
    ics = MinoTimeEvolution.Mino_ics(t0, ra, p, e, M);

    rLSO = LSO_p(a, M)
    while tOrbit > t0
        # orbital parameters during current piecewise geodesic
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θmin); e_t = last(ecc);
        print("Completion: $(round(100 * t0/tOrbit; digits=5))%   \r")
        flush(stdout)   

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t * M / (1.0 - e_t); rp=p_t * M / (1.0 + e_t);
        A = M / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t) / M; p4 = r4 * (1.0 + e_t) / M    # Above Eq. 96
        # array of params for ODE solver
        params = @SArray [a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm]

        # compute fundamental frequencies in order to determine geodesic time range
        ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p_t, e_t, θmin_t, E_t, L_t, Q_t, C_t, rplus, rminus, M);    # Mino time frequencies
        ωr=ω[1]; ωθ=ω[2]; ωϕ=ω[3];   # BL time frequencies

        #  we want to perform each fit over a set of points which span a physical time range T_fit. In some cases, the frequencies are infinite, and we thus ignore them in our fitting procedure
        if e_t == 0.0 && θmin_t == π/2   # circular equatorial
            ωr = 1e50; ωθ =1e50;
            T_Fit = t_range_factor * minimum(@. 2π/ωϕ)
        elseif e_t == 0.0   # circular non-equatorial
            ωr = 1e50;
            T_Fit = t_range_factor * minimum(@. 2π/[ωθ, ωϕ])
        elseif θmin_t == π/2   # non-circular equatorial
            ωθ = 1e50;
            T_Fit = t_range_factor * minimum(@. 2π/[ωr, ωϕ])
        else   # generic case
            T_Fit = t_range_factor * minimum(@. 2π/ω[1:3])
        end

        saveat = T_Fit / (nPoints-1);    # the user specifies the number of points in each fit, i.e., the resolution, which determines at which points the interpolator should save data points

        # to compute the self force at a point, we must overshoot the solution into the future
        λF = λ0 + (nPoints-1) * saveat + (nPoints÷2) * saveat   # evolve geodesic up to λF
        total_num_points = nPoints+(nPoints÷2)   # total number of points in geodesic since we overshoot
        Δλi=saveat/10;    # initial time step for geodesic integration

        saveat_λ = range(λ0, λF, total_num_points) |> collect
        λspan=(λ0, λF)

        # stop when it reaches LSO
        condition(u, t , integrator) = u[1] - rLSO # Is zero when r = rLSO (to 5 d.p)
        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(condition, affect!)

        # numerically solve for geodesic motion
        prob = e == 0.0 ? ODEProblem(MinoTimeEvolution.HJ_Eqns_circular, ics, λspan, params) : ODEProblem(MinoTimeEvolution.HJ_Eqns, ics, λspan, params);
        
        # if e==0.0
        #     sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δλi, reltol = reltol, abstol = abstol, saveat=saveat_λ, callback = cb);
        # else
        #     sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δλi, reltol = reltol, abstol = abstol, saveat=saveat_λ);
        # end
      
        sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δλi, reltol = reltol, abstol = abstol, saveat=saveat_λ);

        # deconstruct solution
        λλ = sol.t;
        tt = sol[1, :];
        psi = sol[2, :];
        chi = mod.(sol[3, :], 2π);
        ϕϕ = sol[4, :];

        if (length(sol[1, :]) < total_num_points)
            println("Integration terminated at t = $(last(t))")
            println("total_num_points - len(sol) = $(total_num_points-length(sol[1,:]))")
            println("λ0 = $(λ0), λF = $(λF), total_num_points = $(total_num_points)\n")
            println("saveat_λ:")
            println(saveat_λ)
            println("\nsol.t:")
            println(sol.t)
            break
        elseif length(tt)>total_num_points
            λλ = sol.t[1:total_num_points];
            tt = sol[1, 1:total_num_points];
            psi = sol[2, 1:total_num_points];
            chi = mod.(sol[3, 1:total_num_points], 2π);
            ϕϕ = sol[4, 1:total_num_points];
        end
        
        # compute time derivatives (wrt λ)
        dt_dλ = MinoTimeEvolution.dt_dλ.(λλ, psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        dψ_dλ = MinoTimeEvolution.dψ_dλ.(λλ, psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        dχ_dλ = MinoTimeEvolution.dχ_dλ.(λλ, psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        dϕ_dλ = MinoTimeEvolution.dϕ_dλ.(λλ, psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)

        # compute BL coordinates t, r, θ and their time derivatives
        rr = MinoTimeEvolution.r.(psi, p_t, e_t, M)
        θθ = [acos((π/2<chi[i]<1.5π) ? -sqrt(MinoTimeEvolution.z(chi[i], θmin_t)) : sqrt(MinoTimeEvolution.z(chi[i], θmin_t))) for i in eachindex(chi)]

        dr_dλ = MinoTimeEvolution.dr_dλ.(dψ_dλ, psi, p_t, e_t, M);
        dθ_dλ = MinoTimeEvolution.dθ_dλ.(dχ_dλ, chi, θθ, θmin_t);
    
        # compute derivatives wrt t
        r_dot = @. dr_dλ / dt_dλ
        θ_dot = @. dθ_dλ / dt_dλ 
        ϕ_dot = @. dϕ_dλ / dt_dλ 

        # compute Γ factor
        v_spatial = [[r_dot[i], θ_dot[i], ϕ_dot[i]] for i in eachindex(λλ)]; # v_spatial=dxi/dt
        Γ = @. MinoTimeEvolution.Γ(tt, rr, θθ, ϕϕ, v_spatial, a, M)

        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        r_ddot = MinoTimeEvolution.dr2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)
        θ_ddot = MinoTimeEvolution.dθ2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)
        ϕ_ddot = MinoTimeEvolution.dϕ2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)

        ###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######
        # store multipole moments for waveform computation
        if e_t == 0.0 && θmin_t == π/2   # circular equatorial
            n_freqs=FourierFitGSL.compute_num_fitting_freqs_1(nHarm);
        elseif e_t != 0.0 && θmin_t != π/2   # generic case
            n_freqs=FourierFitGSL.compute_num_fitting_freqs_3(nHarm);
        else   # circular non-equatorial or non-circular equatorial — either way only two non-trivial frequencies
            n_freqs=FourierFitGSL.compute_num_fitting_freqs_2(nHarm);
        end
        
        # # MIGHT WANT TO USE VIEWS TO OPTIMIZE A BIT AND AVOID MAKING COPIES IN EACH CALL BELOW #
        chisq=[0.0];
        EstimateMultipoleDerivs.FourierFit.compute_waveform_moments_and_derivs_Mino!(a, E_t, L_t, C_t, m, M, xBL_wf, vBL_wf, aBL_wf, xH_wf, x_H_wf, rH_wf, vH_wf, v_H_wf, aH_wf, a_H_wf, v_wf, λλ[1:nPoints], rr[1:nPoints], r_dot[1:nPoints], r_ddot[1:nPoints], θθ[1:nPoints], θ_dot[1:nPoints], 
            θ_ddot[1:nPoints], ϕϕ[1:nPoints], ϕ_dot[1:nPoints], ϕ_ddot[1:nPoints], Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp,
            nHarm, ωr, ωθ, ωϕ, nPoints, n_freqs, chisq)

        # store trajectory, ignoring the overshot piece
        append!(λ, λλ[1:nPoints]); append!(t, tt[1:nPoints]); append!(dt_dτ, Γ[1:nPoints]); append!(r, rr[1:nPoints]); append!(dr_dt, r_dot[1:nPoints]); 
        append!(d2r_dt2, r_ddot[1:nPoints]); append!(θ, θθ[1:nPoints]); append!(dθ_dt, θ_dot[1:nPoints]); append!(d2θ_dt2, θ_ddot[1:nPoints]);
        append!(ϕ, ϕϕ[1:nPoints]); append!(dϕ_dt, ϕ_dot[1:nPoints]); append!(d2ϕ_dt2, ϕ_ddot[1:nPoints]);
        
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

        ###### COMPUTE SELF-FORCE ######
        fit_index_0 = nPoints - (nPoints÷2); fit_index_1 = nPoints + (nPoints÷2); compute_at=(nPoints÷2)+1; 

        if e_t == 0.0 && θmin_t == π/2   # circular equatorial
            n_freqs=FourierFitGSL.compute_num_fitting_freqs_1(nHarm);
        elseif e_t != 0.0 && θmin_t != π/2   # generic case
            n_freqs=FourierFitGSL.compute_num_fitting_freqs_3(nHarm);
        else   # circular non-equatorial or non-circular equatorial — either way only two non-trivial frequencies
            n_freqs=FourierFitGSL.compute_num_fitting_freqs_2(nHarm);
        end

        chisq=[0.0];
        SelfAcceleration.FourierFit.selfAcc_Mino!(aSF_H_temp, aSF_BL_temp, xBL, vBL, aBL, xH, x_H, rH, vH, v_H, aH, a_H, v, λλ[fit_index_0:fit_index_1], 
            rr[fit_index_0:fit_index_1], r_dot[fit_index_0:fit_index_1], r_ddot[fit_index_0:fit_index_1], θθ[fit_index_0:fit_index_1], 
            θ_dot[fit_index_0:fit_index_1], θ_ddot[fit_index_0:fit_index_1], ϕϕ[fit_index_0:fit_index_1], ϕ_dot[fit_index_0:fit_index_1], 
            ϕ_ddot[fit_index_0:fit_index_1], Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, 
            Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, E_t, L_t, M, m, compute_at, nHarm, ωr, ωθ, ωϕ, fit_array_length, n_freqs, chisq);
        
        EvolveConstants(tt[nPoints]-tt[1], a, tt[nPoints], rr[nPoints], θθ[nPoints], ϕϕ[nPoints], Γ[nPoints], r_dot[nPoints], θ_dot[nPoints], ϕ_dot[nPoints], aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPoints)

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)

        # update next ics for next piece
        t0 = tt[nPoints+1];
        λ0 = λλ[nPoints+1];
        ics = @SArray [t0, psi[nPoints+1], chi[nPoints+1], ϕϕ[nPoints+1]]
    end

    # delete final "extra" energies and fluxes
    delete_first = size(EE, 1) - (nPoints-1)
    deleteat!(EE, delete_first:(delete_first+nPoints-1))
    deleteat!(LL, delete_first:(delete_first+nPoints-1))
    deleteat!(QQ, delete_first:(delete_first+nPoints-1))
    deleteat!(CC, delete_first:(delete_first+nPoints-1))
    deleteat!(pArray, delete_first:(delete_first+nPoints-1))
    deleteat!(ecc, delete_first:(delete_first+nPoints-1))
    deleteat!(θmin, delete_first:(delete_first+nPoints-1))

    delete_first = size(Edot, 1) - (nPoints-2)
    deleteat!(Edot, delete_first:(delete_first+nPoints-2))
    deleteat!(Ldot, delete_first:(delete_first+nPoints-2))
    deleteat!(Qdot, delete_first:(delete_first+nPoints-2))
    deleteat!(Cdot, delete_first:(delete_first+nPoints-2))

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end

    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPoints).jld2"
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

end