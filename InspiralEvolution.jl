module InspiralEvolution

Z_1(a::Float64, M::Float64) = 1 + (1 - a^2 / M^2)^(1/3) * ((1 + a / M)^(1/3) + (1 - a / M)^(1/3))
Z_2(a::Float64, M::Float64) = sqrt(3 * (a / M)^2 + Z_1(a, M)^2)
LSO_r(a::Float64, M::Float64) = M * (3 + Z_2(a, M) - sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # retrograde LSO
LSO_p(a::Float64, M::Float64) = M * (3 + Z_2(a, M) + sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # prograde LSO

module FourierFit

module BLTime
using LinearAlgebra
using Combinatorics
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ....Kerr
using ....HJEvolution
using ....FourierFitGSL
using ....CircularNonEquatorial
using ....HarmonicCoords
using ....ConstructSymmetricArrays
using ....SelfForce
using ....EvolveConstants
using JLD2
using FileIO
using ...InspiralEvolution

#=
    This comment explains the methodology in the function below. At the end of each piecewise geodesic, we must compute the self-force in order to update the orbital 
    parameters and move to the next geodesic piece in the trajectory. The method we employ in computing the self-force is to fit the multipole moments
    to a fourier series expanded in terms of the fundamental frequencies, and then take high-order derivates from a simple formula. Empirically, this fit
    is ``best'' at the middle of the data set (e.g., it tends to be worse at the edges, which is common in interpolation methods, for example). As a result,
    we would like the point at which we wish to compute the high-order derivatives (and the self-force) to be at the midpoint of the data array. Suppose we
    want to compute the self force at t=T. Then, we evolve the geodesic past t=T into the future, using an odd number of points, and then perform the fit
    to data for a time range which lies an odd number of points in the future and past of t=T, so that t=T is exactly at the midpoint of the data arrays. Note
    we will discard of the data for t>T since it was only computed as an auxiliary for the fitting process.
=#

function compute_inspiral_HJE!(tOrbit::Float64, compute_SF::Float64, fit_time_range_factor::Float64, nPointsGeodesic::Int64, nPointsFit::Int64, M::Float64, m::Float64, a::Float64, p::Float64, 
    e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function,
    gRR::Function, gThTh::Function, gΦΦ::Function, nHarm::Int64, reltol::Float64=1e-10, abstol::Float64=1e-10; data_path::String="Data/")
    if iseven(nPointsFit)
        throw(DomainError(nPointsFit, "nPointsFit must be odd"))
    end

    # create arrays for trajectory
    t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[];
    # create arrays to store multipole moments necessary for waveform computation
    Mij2_wf = [Float64[] for i=1:3, j=1:3];
    Mijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf = [Float64[] for i=1:3, j=1:3];
    Sijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    
    # initialize data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()
    xBL_fit = [Float64[] for i in 1:nPointsFit]; xBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    vBL_fit = [Float64[] for i in 1:nPointsFit]; vBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    aBL_fit = [Float64[] for i in 1:nPointsFit]; aBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    xH_fit = [Float64[] for i in 1:nPointsFit];  xH_wf = [Float64[] for i in 1:nPointsGeodesic];
    x_H_fit = [Float64[] for i in 1:nPointsFit]; x_H_wf = [Float64[] for i in 1:nPointsGeodesic];
    vH_fit = [Float64[] for i in 1:nPointsFit];  vH_wf = [Float64[] for i in 1:nPointsGeodesic];
    v_H_fit = [Float64[] for i in 1:nPointsFit]; v_H_wf = [Float64[] for i in 1:nPointsGeodesic];
    v_fit = zeros(nPointsFit);   v_wf = zeros(nPointsGeodesic);
    rH_fit = zeros(nPointsFit);  rH_wf = zeros(nPointsGeodesic);
    aH_fit = [Float64[] for i in 1:nPointsFit];  aH_wf = [Float64[] for i in 1:nPointsGeodesic];
    a_H_fit = [Float64[] for i in 1:nPointsFit]; a_H_wf = [Float64[] for i in 1:nPointsGeodesic];

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

    # compute number of fitting frequencies used in fits to the fourier series expansion of the multipole moments
    if e == 0.0 && θi == π/2   # circular equatorial
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_1(nHarm);
    elseif e != 0.0 && θi != π/2   # generic case
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_3(nHarm);
    else   # circular non-equatorial or non-circular equatorial — either way only two non-trivial frequencies
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_2(nHarm);
    end

    # compute apastron
    ra = p * M / (1 - e);

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi, CCi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)   

    # store orbital params in arrays
    EE = ones(nPointsGeodesic) * EEi; 
    Edot = zeros(nPointsGeodesic-1);
    LL = ones(nPointsGeodesic) * LLi; 
    Ldot = zeros(nPointsGeodesic-1);
    CC = ones(nPointsGeodesic) * CCi;
    Cdot = zeros(nPointsGeodesic-1);
    QQ = ones(nPointsGeodesic) * QQi
    Qdot = zeros(nPointsGeodesic-1);
    pArray = ones(nPointsGeodesic) * p;
    ecc = ones(nPointsGeodesic) * e;
    θmin = ones(nPointsGeodesic) * θi;

    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    geodesic_ics = HJEvolution.HJ_ics(ra, p, e, M);

    rLSO = InspiralEvolution.LSO_p(a, M)

    use_custom_ics = true; use_specified_params = true;
    save_at_trajectory = compute_SF / (nPointsGeodesic - 1); Δti=save_at_trajectory;    # initial time step for geodesic integration

    # in the code, we will want to compute the geodesic with an additional time step at the end so that these coordinate values can be used as initial conditions for the
    # subsequent geodesic
    geodesic_time_length = compute_SF + save_at_trajectory;
    num_points_geodesic = nPointsGeodesic + 1;

    while tOrbit > t0
        print("Completion: $(100 * t0/tOrbit)%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # orbital parameters
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θmin); e_t = last(ecc);  

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t * M / (1.0 - e_t); rp=p_t * M / (1.0 + e_t);
        A = M / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t) / M; p4 = r4 * (1.0 + e_t) / M    # Above Eq. 96

        # geodesic
        tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi = HJEvolution.compute_kerr_geodesic(a, p_t, e_t, θmin_t, num_points_geodesic, use_custom_ics, use_specified_params, geodesic_time_length, Δti, reltol, abstol;
        ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false, inspiral = true)

        tt = tt .+ t0   # tt from the above function call starts from zero

        # check that geodesic output is as expected
        if (length(tt) != num_points_geodesic) || !isapprox(tt[nPointsGeodesic], t0 + compute_SF)
            println("Integration terminated at t = $(last(t))")
            println("total_num_points - len(sol) = $(num_points_geodesic-length(tt))")
            println("tt[nPointsGeodesic] = $(tt[nPointsGeodesic])")
            println("t0 + compute_SF = $(t0 + compute_SF)")
            break
        end

        # extract initial conditions for next geodesic, then remove these points from the data array
        t0 = last(tt); geodesic_ics = @SArray [last(psi), last(chi), last(ϕϕ)];
        pop!(tt); pop!(rr); pop!(θθ); pop!(ϕϕ); pop!(r_dot); pop!(θ_dot); pop!(ϕ_dot);
        pop!(r_ddot); pop!(θ_ddot); pop!(ϕ_ddot); pop!(Γ); pop!(psi); pop!(chi);

        # store physical trajectory
        append!(t, tt); append!(dt_dτ, Γ); append!(r, rr); append!(dr_dt, r_dot); append!(d2r_dt2, r_ddot); 
        append!(θ, θθ); append!(dθ_dt, θ_dot); append!(d2θ_dt2, θ_ddot); append!(ϕ, ϕϕ); 
        append!(dϕ_dt, ϕ_dot); append!(d2ϕ_dt2, ϕ_ddot);

        ###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######
        chisq=[0.0];
        # compute fundamental frequencies
        ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p_t, e_t, θmin_t, E_t, L_t, Q_t, C_t, rplus, rminus, M);    # Mino time frequencies
        Ω=ω[1:3]/ω[4]; 
        Ωr, Ωθ, Ωϕ = Ω;   # BL time frequencies

        # compute waveform
        SelfForce.compute_waveform_moments_and_derivs!(a, m, M, xBL_wf, vBL_wf, aBL_wf, xH_wf, x_H_wf, rH_wf, vH_wf, v_H_wf, aH_wf, a_H_wf, v_wf, 
            tt, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp,
            Sij2_wf_temp, Sijk3_wf_temp, nHarm, Ωr, Ωθ, Ωϕ, nPointsGeodesic, n_freqs, chisq)

        # store multipole data for waveforms — we only save the independent components
        @inbounds Threads.@threads for indices in ConstructSymmetricArrays.waveform_indices
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
        # we want to perform each fit over a set of points which span a physical time range T_fit. In some cases, the frequencies are infinite and we 
        # ignore them in our fitting procedure
        if e_t == 0.0 && θmin_t == π/2   # circular equatorial
            Ωr = 1e50; Ωθ =1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/Ωϕ)
        elseif e_t == 0.0   # circular non-equatorial
            Ωr = 1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/[Ωθ, Ωϕ])
        elseif θmin_t == π/2   # non-circular equatorial
            Ωθ = 1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/[Ωr, Ωϕ])
        else   # generic case
            T_Fit = fit_time_range_factor * minimum(@. 2π/Ω)
        end

        saveat_fit = T_Fit / (nPointsFit-1);    # the user specifies the number of points in each fit, i.e., the resolution, which determines at which points the interpolator should save data points
        Δti_fit = saveat_fit;
        # compute geodesic into future and past of the final point in the (physical) piecewise geodesic computed above
        midpoint_ics = @SArray [last(psi), last(chi), last(ϕϕ)];
        
        tt_fit, rr_fit, θθ_fit, ϕϕ_fit, r_dot_fit, θ_dot_fit, ϕ_dot_fit, r_ddot_fit, θ_ddot_fit, ϕ_ddot_fit, Γ_fit, psi_fit, chi_fit = 
        HJEvolution.compute_kerr_geodesic_past_and_future(midpoint_ics, a, p_t, e_t, θmin_t, use_specified_params, nPointsFit, T_Fit, Δti_fit, reltol, abstol;
        E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, inspiral=true)
        
        compute_at=(nPointsFit÷2)+1;   # by construction, the end point of the physical geoodesic is at the center of the geodesic computed for the fit
        # println(("T_Fit = $(T_Fit)"))
        # check that that the midpoint of the fit geodesic arrays are equal to the final point of the physical arrays
        if rr_fit[compute_at] != last(rr) || θθ_fit[compute_at] != last(θθ) || ϕϕ_fit[compute_at] != last(ϕϕ) ||
            r_dot_fit[compute_at] != last(r_dot) || θ_dot_fit[compute_at] != last(θ_dot) || ϕ_dot_fit[compute_at] != last(ϕ_dot) || 
            r_ddot_fit[compute_at] != last(r_ddot)|| θ_ddot_fit[compute_at] != last(θ_ddot)|| ϕ_ddot_fit[compute_at] != last(ϕ_ddot) ||
            Γ_fit[compute_at] != last(Γ) || psi_fit[compute_at] != last(psi) || chi_fit[compute_at] != last(chi)
            println("Integration terminated at t = $(last(t)). Reason: midpoint of fit geodesic does not align with final point of physical geodesic")
            break
        end

        # compute self-force at end of physical geodesic
        SelfForce.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL_fit, vBL_fit, aBL_fit, xH_fit, x_H_fit, rH_fit, vH_fit, v_H_fit, aH_fit, a_H_fit, v_fit, tt_fit, rr_fit, r_dot_fit,
            r_ddot_fit, θθ_fit, θ_dot_fit, θ_ddot_fit, ϕϕ_fit, ϕ_dot_fit, ϕ_ddot_fit, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data,
            Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m, compute_at, nHarm, Ωr, Ωθ, Ωϕ, nPointsFit, n_freqs, chisq);

        # evolve orbital parameters using self-force
        EvolveConstants.Evolve_BL(compute_SF, a, last(tt), last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPointsGeodesic)

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)
    end
    print("Completion: 100%   \r")
    flush(stdout) 

    # delete final "extra" energies and fluxes
    delete_first = size(EE, 1) - (nPointsGeodesic-1)
    deleteat!(EE, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(LL, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(QQ, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(CC, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(pArray, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(ecc, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(θmin, delete_first:(delete_first+nPointsGeodesic-1))

    delete_first = size(Edot, 1) - (nPointsGeodesic-2)
    deleteat!(Edot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Ldot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Qdot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Cdot, delete_first:(delete_first+nPointsGeodesic-2))

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end

    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end



    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.jld2"
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, nHarm::Int64, nPointsFit::Int64, reltol::Float64, fit_time_range_factor::Float64, data_path::String)
    # load ODE solution
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.txt"
    sol = readdlm(ODE_filename)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; dr_dt=sol[5,:]; dθ_dt=sol[6,:]; dϕ_dt=sol[7,:]; d2r_dt2=sol[8,:]; d2θ_dt2=sol[9,:]; d2ϕ_dt2=sol[10,:]; dt_dτ=sol[11,:]
    return t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, nHarm::Int64, nPointsFit::Int64, reltol::Float64, fit_time_range_factor::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.txt"
    constants=readdlm(constants_filename)
    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.txt"
    constants_derivs = readdlm(constants_derivs_filename)
    EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :]
    Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]
    return EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, t::Vector{Float64}, a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, 
    nHarm::Int64, nPointsFit::Int64, reltol::Float64, fit_time_range_factor::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_fit.jld2"
    waveform_data = load(waveform_filename)["data"]
    Mij2 = waveform_data["Mij2"]; ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij2);
    Mijk3 = waveform_data["Mijk3"]; ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk3);
    Mijkl4 = waveform_data["Mijkl4"]; ConstructSymmetricArrays.SymmetrizeFourIndexTensor!(Mijkl4);
    Sij2 = waveform_data["Sij2"]; ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij2);
    Sijk3 = waveform_data["Sijk3"]; ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Sijk3);

    # compute h_{ij} tensor
    hij = [zeros(length(t)) for i=1:3, j=1:3];
    SelfForce.hij!(hij, t, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    # project h_{ij} tensor
    h_plus = SelfForce.hplus(hij, Θ, Φ);
    h_cross = SelfForce.hcross(hij, Θ, Φ);
    return h_plus, h_cross
end

end

module MinoTime

using LinearAlgebra
using Combinatorics
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ....Kerr
using ....MinoEvolution
using ....FourierFitGSL
using ....CircularNonEquatorial
import ....HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ....HarmonicCoords
using ....SelfForce
using ....ConstructSymmetricArrays
using ....EvolveConstants
using JLD2
using FileIO
using ...InspiralEvolution


function compute_inspiral_HJE!(tOrbit::Float64, compute_SF::Float64, fit_time_range_factor::Float64, nPointsGeodesic::Int64, nPointsFit::Int64, M::Float64, m::Float64, a::Float64, p::Float64,
    e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function,
    gRR::Function, gThTh::Function, gΦΦ::Function, nHarm::Int64, reltol::Float64=1e-10, abstol::Float64=1e-10; data_path::String="Data/")
    if iseven(nPointsFit)
        throw(DomainError(nPointsFit, "nPointsFit must be odd"))
    end

    # create arrays for trajectory
    λ = Float64[]; t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[]; dt_dλ = Float64[];
    Mij2_wf = [Float64[] for i=1:3, j=1:3];
    Mijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf = [Float64[] for i=1:3, j=1:3];
    Sijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    
    # initialize data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()
    xBL_fit = [Float64[] for i in 1:nPointsFit]; xBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    vBL_fit = [Float64[] for i in 1:nPointsFit]; vBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    aBL_fit = [Float64[] for i in 1:nPointsFit]; aBL_wf = [Float64[] for i in 1:nPointsGeodesic];
    xH_fit = [Float64[] for i in 1:nPointsFit];  xH_wf = [Float64[] for i in 1:nPointsGeodesic];
    x_H_fit = [Float64[] for i in 1:nPointsFit]; x_H_wf = [Float64[] for i in 1:nPointsGeodesic];
    vH_fit = [Float64[] for i in 1:nPointsFit];  vH_wf = [Float64[] for i in 1:nPointsGeodesic];
    v_H_fit = [Float64[] for i in 1:nPointsFit]; v_H_wf = [Float64[] for i in 1:nPointsGeodesic];
    v_fit = zeros(nPointsFit);   v_wf = zeros(nPointsGeodesic);
    rH_fit = zeros(nPointsFit);  rH_wf = zeros(nPointsGeodesic);
    aH_fit = [Float64[] for i in 1:nPointsFit];  aH_wf = [Float64[] for i in 1:nPointsGeodesic];
    a_H_fit = [Float64[] for i in 1:nPointsFit]; a_H_wf = [Float64[] for i in 1:nPointsGeodesic];

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

    # compute number of fitting frequencies used in fits to the fourier series expansion of the multipole moments
    if e == 0.0 && θi == π/2   # circular equatorial
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_1(nHarm);
    elseif e != 0.0 && θi != π/2   # generic case
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_3(nHarm);
    else   # circular non-equatorial or non-circular equatorial — either way only two non-trivial frequencies
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_2(nHarm);
    end

    # compute apastron
    ra = p * M / (1 - e);

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi, CCi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)   

    # store orbital params in arrays
    EE = ones(nPointsGeodesic) * EEi; 
    Edot = zeros(nPointsGeodesic-1);
    LL = ones(nPointsGeodesic) * LLi; 
    Ldot = zeros(nPointsGeodesic-1);
    CC = ones(nPointsGeodesic) * CCi;
    Cdot = zeros(nPointsGeodesic-1);
    QQ = ones(nPointsGeodesic) * QQi
    Qdot = zeros(nPointsGeodesic-1);
    pArray = ones(nPointsGeodesic) * p;
    ecc = ones(nPointsGeodesic) * e;
    θmin = ones(nPointsGeodesic) * θi;

    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    λ0 = 0.0
    geodesic_ics = MinoEvolution.Mino_ics(t0, ra, p, e, M);

    rLSO = InspiralEvolution.LSO_p(a, M)

    use_custom_ics = true; use_specified_params = true;
    save_at_trajectory = compute_SF / (nPointsGeodesic - 1); Δλi=save_at_trajectory;    # initial time step for geodesic integration

    # in the code, we will want to compute the geodesic with an additional time step at the end so that these coordinate values can be used as initial conditions for the
    # subsequent geodesic
    geodesic_time_length = compute_SF + save_at_trajectory;
    num_points_geodesic = nPointsGeodesic + 1;

    while tOrbit > t0
        print("Completion: $(100 * t0/tOrbit)%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # orbital parameters
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θmin); e_t = last(ecc);

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t * M / (1.0 - e_t); rp=p_t * M / (1.0 + e_t);
        A = M / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t) / M; p4 = r4 * (1.0 + e_t) / M    # Above Eq. 96

        # geodesic
        λλ, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi, dt_dλλ = MinoEvolution.compute_kerr_geodesic(a, p_t, e_t, θmin_t, num_points_geodesic, use_custom_ics,
        use_specified_params, geodesic_time_length, Δλi, reltol, abstol; ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false, inspiral = true)
        
        λλ = λλ .+ λ0   # λλ from the above function call starts from zero 

        # check that geodesic output is as expected
        if (length(λλ) != num_points_geodesic) || !isapprox(λλ[nPointsGeodesic], λ0 + compute_SF)
            println("Integration terminated at t = $(last(t))")
            println("total_num_points - len(sol) = $(num_points_geodesic-length(λλ))")
            println("λλ[nPointsGeodesic] = $(λλ[nPointsGeodesic])")
            println("λ0 + compute_SF = $(λ0 + compute_SF)")
            break
        end

        # extract initial conditions for next geodesic, then remove these points from the data array
        λ0 = last(λλ); t0 = last(tt); geodesic_ics = @SArray [t0, last(psi), last(chi), last(ϕϕ)];

        pop!(λλ); pop!(tt); pop!(rr); pop!(θθ); pop!(ϕϕ); pop!(r_dot); pop!(θ_dot); pop!(ϕ_dot);
        pop!(r_ddot); pop!(θ_ddot); pop!(ϕ_ddot); pop!(Γ); pop!(psi); pop!(chi); pop!(dt_dλλ)

        # store physical trajectory
        append!(λ, λλ); append!(t, tt); append!(dt_dτ, Γ); append!(r, rr); append!(dr_dt, r_dot); append!(d2r_dt2, r_ddot); 
        append!(θ, θθ); append!(dθ_dt, θ_dot); append!(d2θ_dt2, θ_ddot); append!(ϕ, ϕϕ); append!(dϕ_dt, ϕ_dot);
        append!(d2ϕ_dt2, ϕ_ddot); append!(dt_dλ, dt_dλλ);


        ###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######
        chisq=[0.0];
        # compute fundamental frequencies
        ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p_t, e_t, θmin_t, E_t, L_t, Q_t, C_t, rplus, rminus, M);    # Mino time frequencies
        ωr=ω[1]; ωθ=ω[2]; ωϕ=ω[3];   # mino time frequencies

        # compute waveform
        SelfForce.compute_waveform_moments_and_derivs_Mino!(a, E_t, L_t, m, M, xBL_wf, vBL_wf, aBL_wf, xH_wf, x_H_wf, rH_wf, vH_wf, v_H_wf, aH_wf, a_H_wf, v_wf, 
        λλ, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp,
        Sij2_wf_temp, Sijk3_wf_temp, nHarm, ωr, ωθ, ωϕ, nPointsGeodesic, n_freqs, chisq)
        
        # store multipole data for waveforms — note that we only save the independent components
        @inbounds Threads.@threads for indices in ConstructSymmetricArrays.waveform_indices
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
        #  we want to perform each fit over a set of points which span a physical time range T_fit. In some cases, the frequencies are infinite, and we thus ignore them in our fitting procedure
        if e_t == 0.0 && θmin_t == π/2   # circular equatorial
            ωr = 1e50; ωθ =1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/ωϕ)
        elseif e_t == 0.0   # circular non-equatorial
            ωr = 1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/[ωθ, ωϕ])
        elseif θmin_t == π/2   # non-circular equatorial
            ωθ = 1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/[ωr, ωϕ])
        else   # generic case
            T_Fit = fit_time_range_factor * minimum(@. 2π/ω[1:3])
        end

        saveat_fit = T_Fit / (nPointsFit-1);    # the user specifies the number of points in each fit, i.e., the resolution, which determines at which points the interpolator should save data points
        Δλi_fit = saveat_fit;
        # compute geodesic into future and past of the final point in the (physical) piecewise geodesic computed above
        midpoint_ics = @SArray [last(tt), last(psi), last(chi), last(ϕϕ)];
        
        λλ_fit, tt_fit, rr_fit, θθ_fit, ϕϕ_fit, r_dot_fit, θ_dot_fit, ϕ_dot_fit, r_ddot_fit, θ_ddot_fit, ϕ_ddot_fit, Γ_fit, psi_fit, chi_fit, dt_dλ_fit = 
        MinoEvolution.compute_kerr_geodesic_past_and_future(midpoint_ics, a, p_t, e_t, θmin_t, use_specified_params, nPointsFit, T_Fit, Δλi_fit, reltol, abstol;
        E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, inspiral=true)

        compute_at=(nPointsFit÷2)+1;   # by construction, the end point of the physical geoodesic is at the center of the geodesic computed for the fit
        # check that that the midpoint of the fit geodesic arrays are equal to the final point of the physical arrays
        if tt_fit[compute_at] != last(tt) || rr_fit[compute_at] != last(rr) || θθ_fit[compute_at] != last(θθ) || ϕϕ_fit[compute_at] != last(ϕϕ) ||
            r_dot_fit[compute_at] != last(r_dot) || θ_dot_fit[compute_at] != last(θ_dot) || ϕ_dot_fit[compute_at] != last(ϕ_dot) || 
            r_ddot_fit[compute_at] != last(r_ddot)|| θ_ddot_fit[compute_at] != last(θ_ddot)|| ϕ_ddot_fit[compute_at] != last(ϕ_ddot) ||
            Γ_fit[compute_at] != last(Γ) || psi_fit[compute_at] != last(psi) || chi_fit[compute_at] != last(chi)
            println("Integration terminated at t = $(last(t)). Reason: midpoint of fit geodesic does not align with final point of physical geodesic")
            break
        end

        chisq=[0.0];
        SelfForce.selfAcc_Mino!(aSF_H_temp, aSF_BL_temp, xBL_fit, vBL_fit, aBL_fit, xH_fit, x_H_fit, rH_fit, vH_fit, v_H_fit, aH_fit, a_H_fit, v_fit, λλ_fit, 
            rr_fit, r_dot_fit, r_ddot_fit, θθ_fit, θ_dot_fit, θ_ddot_fit, ϕϕ_fit, ϕ_dot_fit, ϕ_ddot_fit, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6,
            Mij2_data, Mijk2_data, Sij1_data, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, E_t, L_t, M, m, compute_at, nHarm,
            ωr, ωθ, ωϕ, nPointsFit, n_freqs, chisq);
        
        Δt = last(tt) - tt[1]
        EvolveConstants.Evolve_BL(Δt, a, last(tt), last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPointsGeodesic)

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)
    end
    print("Completion: 100%   \r")
    flush(stdout) 

    # delete final "extra" energies and fluxes
    delete_first = size(EE, 1) - (nPointsGeodesic-1)
    deleteat!(EE, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(LL, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(QQ, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(CC, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(pArray, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(ecc, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(θmin, delete_first:(delete_first+nPointsGeodesic-1))

    delete_first = size(Edot, 1) - (nPointsGeodesic-2)
    deleteat!(Edot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Ldot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Qdot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Cdot, delete_first:(delete_first+nPointsGeodesic-2))

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end

    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.jld2"
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, nHarm::Int64, nPointsFit::Int64, reltol::Float64, fit_time_range_factor::Float64, data_path::String)
    # load ODE solution
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.txt"
    sol = readdlm(ODE_filename)
    λ=sol[1,:]; t=sol[2,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt2=sol[9,:]; d2θ_dt2=sol[10,:]; d2ϕ_dt2=sol[11,:]; dt_dτ=sol[12,:]; dt_dλ=sol[13,:]
    return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ
end


function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, nHarm::Int64, nPointsFit::Int64, reltol::Float64, fit_time_range_factor::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.txt"
    constants=readdlm(constants_filename)
    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.txt"
    constants_derivs = readdlm(constants_derivs_filename)
    EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :]
    Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]
    return EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, t::Vector{Float64}, a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, 
    nHarm::Int64, nPointsFit::Int64, reltol::Float64, fit_time_range_factor::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPointsFit)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_fit.jld2"
    waveform_data = load(waveform_filename)["data"]
    Mij2 = waveform_data["Mij2"]; ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij2);
    Mijk3 = waveform_data["Mijk3"]; ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk3);
    Mijkl4 = waveform_data["Mijkl4"]; ConstructSymmetricArrays.SymmetrizeFourIndexTensor!(Mijkl4);
    Sij2 = waveform_data["Sij2"]; ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij2);
    Sijk3 = waveform_data["Sijk3"]; ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Sijk3);

    # compute h_{ij} tensor
    hij = [zeros(length(t)) for i=1:3, j=1:3];
    SelfForce.hij!(hij, t, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    # project h_{ij} tensor
    h_plus = SelfForce.hplus(hij, Θ, Φ);
    h_cross = SelfForce.hcross(hij, Θ, Φ);
    return h_plus, h_cross
end

end

end

module FiniteDifferences

module BLTime

using LinearAlgebra
using Combinatorics
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ....Kerr
using ....HJEvolution
using ....FourierFitGSL
using ....CircularNonEquatorial
import ....HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ....HarmonicCoords
using ....SelfForce
using ....SelfForce_numerical
using ....ConstructSymmetricArrays
using ....EvolveConstants
using JLD2
using FileIO
using ...InspiralEvolution

### following function not maintained ###
function compute_inspiral_HJE!(tOrbit::Float64, nPointsGeodesic::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, h::Float64=0.15, reltol::Float64=1e-10, abstol::Float64=1e-10; data_path::String="Data/")
    # create arrays for trajectory
    t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[];
    
    # initialize data arrays
    aSF_BL = Vector{Vector{Float64}}()
    aSF_H = Vector{Vector{Float64}}()
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    stencil_array_length = 11
    xBL = [Float64[] for i in 1:stencil_array_length]
    vBL = [Float64[] for i in 1:stencil_array_length]
    aBL = [Float64[] for i in 1:stencil_array_length]
    xH = [Float64[] for i in 1:stencil_array_length]
    x_H = [Float64[] for i in 1:stencil_array_length]
    vH = [Float64[] for i in 1:stencil_array_length]
    v_H = [Float64[] for i in 1:stencil_array_length]
    v = zeros(stencil_array_length)
    rH = zeros(stencil_array_length)
    aH = [Float64[] for i in 1:stencil_array_length]
    a_H = [Float64[] for i in 1:stencil_array_length]
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
    EE = ones(nPointsGeodesic) * EEi; 
    Edot = zeros(nPointsGeodesic-1);
    LL = ones(nPointsGeodesic) * LLi; 
    Ldot = zeros(nPointsGeodesic-1);
    CC = ones(nPointsGeodesic) * CCi;
    Cdot = zeros(nPointsGeodesic-1);
    QQ = ones(nPointsGeodesic) * QQi
    Qdot = zeros(nPointsGeodesic-1);
    pArray = ones(nPointsGeodesic) * p;
    ecc = ones(nPointsGeodesic) * e;
    θmin = ones(nPointsGeodesic) * θi;
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    ics = HJEvolution.HJ_ics(ra, p, e, M);

    rLSO = LSO_p(a, M)
    while tOrbit > t0
        # orbital parameters during current piecewise geodesic
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θmin); e_t = last(ecc);
        print("Completion: $(100 * t0/tOrbit)%   \r")
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

        # to compute the self force at a point, we must overshoot the solution into the future
        tF = t0 + (nPointsGeodesic-1) * h + (stencil_array_length÷2) * h   # evolve geodesic up to tF
        total_num_points = nPointsGeodesic+(stencil_array_length÷2)   # total number of points in geodesic since we overshoot
        Δti=h;    # initial time step for geodesic integration

        saveat_t = t0:h:tF |> collect    # saveat time array for solver
        tspan=(t0, tF)

        # stop when it reaches LSO
        condition(u, t , integrator) = u[1] - rLSO # Is zero when r = rLSO (to 5 d.p)
        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(condition, affect!)

        # numerically solve for geodesic motion
        prob = ODEProblem(HJEvolution.geodesicEq, ics, tspan, params);
        
        if e==0.0
            sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t, callback = cb);
        else
            sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);
        end

        
        # deconstruct solution
        tt = sol.t;
        psi = sol[1, :];
        chi = mod.(sol[2, :], 2π);
        ϕϕ = sol[3, :];

        if (length(sol[1, :]) < total_num_points)
            println("Integration terminated at t = $(last(t))")
            println("(nPointsGeodesic+1) - len(sol) = $(nPointsGeodesic+1-length(sol[1,:]))")
            break
        elseif length(tt)>total_num_points
            tt = sol.t[:total_num_points];
            psi = sol[1, 1:total_num_points];
            chi = mod.(sol[2, 1:total_num_points], 2π);
            ϕϕ = sol[3, 1:total_num_points];
        end

        # compute time derivatives
        psi_dot = HJEvolution.psi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        chi_dot = HJEvolution.chi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        ϕ_dot = HJEvolution.phi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)

        # compute BL coordinates t, r, θ and their time derivatives
        rr = HJEvolution.r.(psi, p_t, e_t, M)
        θθ = [acos((π/2<chi[i]<1.5π) ? -sqrt(HJEvolution.z(chi[i], θmin_t)) : sqrt(HJEvolution.z(chi[i], θmin_t))) for i in eachindex(chi)]

        r_dot = HJEvolution.dr_dt.(psi_dot, psi, p_t, e_t, M);
        θ_dot = HJEvolution.dθ_dt.(chi_dot, chi, θθ, θmin_t);
        v_spatial = [[r_dot[i], θ_dot[i], ϕ_dot[i]] for i in eachindex(tt)];
        Γ = @. HJEvolution.Γ(tt, rr, θθ, ϕϕ, v_spatial, a, M)

        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        r_ddot = HJEvolution.dr2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)
        θ_ddot = HJEvolution.dθ2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)
        ϕ_ddot = HJEvolution.dϕ2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)

        ###### MIGHT WANT TO USE VIEWS TO OPTIMIZE A BIT AND AVOID MAKING COPIES IN EACH CALL BELOW ######

        # store trajectory, ignoring the overshot piece
        append!(t, tt[1:nPointsGeodesic]); append!(dt_dτ, Γ[1:nPointsGeodesic]); append!(r, rr[1:nPointsGeodesic]); append!(dr_dt, r_dot[1:nPointsGeodesic]); append!(d2r_dt2, r_ddot[1:nPointsGeodesic]); 
        append!(θ, θθ[1:nPointsGeodesic]); append!(dθ_dt, θ_dot[1:nPointsGeodesic]); append!(d2θ_dt2, θ_ddot[1:nPointsGeodesic]); append!(ϕ, ϕϕ[1:nPointsGeodesic]); 
        append!(dϕ_dt, ϕ_dot[1:nPointsGeodesic]); append!(d2ϕ_dt2, ϕ_ddot[1:nPointsGeodesic]);
        
        ###### COMPUTE SELF-FORCE ######
        fit_index_0 = nPointsGeodesic - (stencil_array_length÷2); fit_index_1 = nPointsGeodesic + (stencil_array_length÷2); compute_at=stencil_array_length÷2+1;
        
        SelfForce_numerical.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL, vBL, aBL, xH, x_H, rH, vH, v_H, aH, a_H, v, tt[fit_index_0:fit_index_1], 
        rr[fit_index_0:fit_index_1], r_dot[fit_index_0:fit_index_1], r_ddot[fit_index_0:fit_index_1], θθ[fit_index_0:fit_index_1], 
        θ_dot[fit_index_0:fit_index_1], θ_ddot[fit_index_0:fit_index_1], ϕϕ[fit_index_0:fit_index_1], ϕ_dot[fit_index_0:fit_index_1], 
        ϕ_ddot[fit_index_0:fit_index_1], Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, 
        Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m, compute_at, h);

        EvolveConstants(tt[nPointsGeodesic]-tt[1], a, tt[nPointsGeodesic], rr[nPointsGeodesic], θθ[nPointsGeodesic], ϕϕ[nPointsGeodesic], Γ[nPointsGeodesic], r_dot[nPointsGeodesic], θ_dot[nPointsGeodesic], ϕ_dot[nPointsGeodesic], aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPointsGeodesic)
        
        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)

        # update next ics for next piece
        t0 = tt[nPointsGeodesic+1];
        ics = @SArray [psi[nPointsGeodesic+1], chi[nPointsGeodesic+1], ϕϕ[nPointsGeodesic+1]]
    end

    # delete final "extra" energies and fluxes
    delete_first = size(EE, 1) - (nPointsGeodesic-1)
    deleteat!(EE, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(LL, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(QQ, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(CC, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(pArray, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(ecc, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(θmin, delete_first:(delete_first+nPointsGeodesic-1))

    delete_first = size(Edot, 1) - (nPointsGeodesic-2)
    deleteat!(Edot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Ldot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Qdot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Cdot, delete_first:(delete_first+nPointsGeodesic-2))

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end


    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # number of data points
    n_OrbPoints = size(r, 1)

    # save trajectory
    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_fdm.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

end

module MinoTime

using LinearAlgebra
using Combinatorics
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ....Kerr
using ....MinoEvolution
using ....FourierFitGSL
using ....CircularNonEquatorial
import ....HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ....HarmonicCoords
using ....SelfForce
using ....SelfForce_numerical
using ....ConstructSymmetricArrays
using ....EvolveConstants
using JLD2
using FileIO
using ...InspiralEvolution

function compute_inspiral_HJE!(tOrbit::Float64, nPointsGeodesic::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64, 
    Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function,
    gRR::Function, gThTh::Function, gΦΦ::Function, h::Float64=0.15, reltol::Float64=1e-10, abstol::Float64=1e-10; data_path::String="Data/")

    # create arrays for trajectory
    λ = Float64[]; t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[]; dt_dλ = Float64[];
    Mij2_wf = [Float64[] for i=1:3, j=1:3];
    Mijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf = [Float64[] for i=1:3, j=1:3];
    Sijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];

    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    stencil_array_length = 11;   # set by number of points in FDM stencil
    
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
    
    # arrays for multipole moments
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Mijkl2_data = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    Sijk1_data= [Float64[] for i=1:3, j=1:3, k=1:3]

    # "temporary" mulitpole arrays which contain the multipole data for a given piecewise geodesic
    Mij2_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3];
    Mijk3_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3, k=1:3];
    Mijkl4_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3];
    Sijk3_wf_temp = [zeros(nPointsGeodesic) for i=1:3, j=1:3, k=1:3];

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
    EE = ones(nPointsGeodesic) * EEi; 
    Edot = zeros(nPointsGeodesic-1);
    LL = ones(nPointsGeodesic) * LLi; 
    Ldot = zeros(nPointsGeodesic-1);
    CC = ones(nPointsGeodesic) * CCi;
    Cdot = zeros(nPointsGeodesic-1);
    QQ = ones(nPointsGeodesic) * QQi
    Qdot = zeros(nPointsGeodesic-1);
    pArray = ones(nPointsGeodesic) * p;
    ecc = ones(nPointsGeodesic) * e;
    θmin = ones(nPointsGeodesic) * θi;
    # initial condition for Kerr geodesic trajectory
    λ0 = 0.0;
    t0 = 0.0;
    geodesic_ics = MinoEvolution.Mino_ics(t0, ra, p, e, M);

    rLSO = InspiralEvolution.LSO_p(a, M)

    use_custom_ics = true; use_specified_params = true;
    Δλi=h/10;    # initial time step for geodesic integration

    # in the code, we will want to compute the geodesic with an additional time step at the end so that these coordinate values can be used as initial conditions for the
    # subsequent geodesic
    num_points_geodesic = nPointsGeodesic + 1;
    geodesic_time_length = h * (num_points_geodesic-1);

    while tOrbit > t0
        print("Completion: $(100 * t0/tOrbit)%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # orbital parameters
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θmin); e_t = last(ecc);

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t * M / (1.0 - e_t); rp=p_t * M / (1.0 + e_t);
        A = M / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t) / M; p4 = r4 * (1.0 + e_t) / M    # Above Eq. 96

        # geodesic
        λλ, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi, dt_dλλ = MinoEvolution.compute_kerr_geodesic(a, p_t, e_t, θmin_t, num_points_geodesic, use_custom_ics,
        use_specified_params, geodesic_time_length, Δλi, reltol, abstol; ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false, inspiral = true)
        
        λλ = λλ .+ λ0   # λλ from the above function call starts from zero 

        # check that geodesic output is as expected
        if (length(λλ) != num_points_geodesic) || !isapprox(λλ[nPointsGeodesic], λ0 + h * (nPointsGeodesic-1))
            println("Integration terminated at t = $(last(t))")
            println("total_num_points - len(sol) = $(num_points_geodesic-length(λλ))")
            println("λλ[nPointsGeodesic] = $(λλ[nPointsGeodesic])")
            println("λ0 + compute_SF = $(λ0 + compute_SF)")
            break
        end

        # extract initial conditions for next geodesic, then remove these points from the data array
        λ0 = last(λλ); t0 = last(tt); geodesic_ics = @SArray [t0, last(psi), last(chi), last(ϕϕ)];

        pop!(λλ); pop!(tt); pop!(rr); pop!(θθ); pop!(ϕϕ); pop!(r_dot); pop!(θ_dot); pop!(ϕ_dot);
        pop!(r_ddot); pop!(θ_ddot); pop!(ϕ_ddot); pop!(Γ); pop!(psi); pop!(chi); pop!(dt_dλλ)

        # store physical trajectory
        append!(λ, λλ); append!(t, tt); append!(dt_dτ, Γ); append!(r, rr); append!(dr_dt, r_dot); append!(d2r_dt2, r_ddot); 
        append!(θ, θθ); append!(dθ_dt, θ_dot); append!(d2θ_dt2, θ_ddot); append!(ϕ, ϕϕ); append!(dϕ_dt, ϕ_dot);
        append!(d2ϕ_dt2, ϕ_ddot); append!(dt_dλ, dt_dλλ);


        ###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######
        SelfForce_numerical.compute_waveform_moments_and_derivs_Mino!(a, E_t, L_t, m, M, xBL_wf, vBL_wf, aBL_wf, xH_wf, x_H_wf, rH_wf, vH_wf, v_H_wf, aH_wf, a_H_wf, v_wf, λλ, rr, r_dot, r_ddot, θθ, θ_dot, 
            θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp, nPointsGeodesic, h)
        
        # store multipole data for waveforms — note that we only save the independent components
        @inbounds Threads.@threads for indices in ConstructSymmetricArrays.waveform_indices
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
        # compute past and future geodesic at end point of physical geodesic
        midpoint_ics = @SArray [last(tt), last(psi), last(chi), last(ϕϕ)];
        T_Fit = (stencil_array_length - 1) * h;
        λλ_stencil, tt_stencil, rr_stencil, θθ_stencil, ϕϕ_stencil, r_dot_stencil, θ_dot_stencil, ϕ_dot_stencil, r_ddot_stencil, θ_ddot_stencil, ϕ_ddot_stencil, Γ_stencil, psi_stencil, chi_stencil, dt_dλ_stencil = 
        MinoEvolution.compute_kerr_geodesic_past_and_future(midpoint_ics, a, p_t, e_t, θmin_t, use_specified_params, stencil_array_length, T_Fit, Δλi, reltol, abstol;
        E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, inspiral=true)


        compute_at=stencil_array_length÷2+1;    # by construction, the end point of the physical geoodesic is at the center of the stencil array for the future and past geodesic
        # check that that the midpoint of the fit geodesic arrays are equal to the final point of the physical arrays
        if tt_stencil[compute_at] != last(tt) || rr_stencil[compute_at] != last(rr) || θθ_stencil[compute_at] != last(θθ) || ϕϕ_stencil[compute_at] != last(ϕϕ) ||
            r_dot_stencil[compute_at] != last(r_dot) || θ_dot_stencil[compute_at] != last(θ_dot) || ϕ_dot_stencil[compute_at] != last(ϕ_dot) || 
            r_ddot_stencil[compute_at] != last(r_ddot)|| θ_ddot_stencil[compute_at] != last(θ_ddot)|| ϕ_ddot_stencil[compute_at] != last(ϕ_ddot) ||
            Γ_stencil[compute_at] != last(Γ) || psi_stencil[compute_at] != last(psi) || chi_stencil[compute_at] != last(chi)
            println("Integration terminated at t = $(last(t)). Reason: midpoint of fit geodesic does not align with final point of physical geodesic")
            break
        end
        SelfForce_numerical.selfAcc_mino!(a, M, E_t, L_t, aSF_H_temp, aSF_BL_temp, xBL_stencil, vBL_stencil, aBL_stencil, xH_stencil, x_H_stencil,
        rH_stencil, vH_stencil, v_H_stencil, aH_stencil, a_H_stencil, v_stencil, tt_stencil, rr_stencil, r_dot_stencil, r_ddot_stencil, θθ_stencil, 
        θ_dot_stencil, θ_ddot_stencil, ϕϕ_stencil, ϕ_dot_stencil, ϕ_ddot_stencil, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, 
        Mijk2_data, Sij1_data, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, m, compute_at, h);

        Δt = last(tt) - tt[1];
        EvolveConstants.Evolve_BL(Δt, a, last(tt), last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPointsGeodesic)

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)
    end
    print("Completion: 100%   \r")
    flush(stdout) 

    # delete final "extra" energies and fluxes
    delete_first = size(EE, 1) - (nPointsGeodesic-1)
    deleteat!(EE, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(LL, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(QQ, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(CC, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(pArray, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(ecc, delete_first:(delete_first+nPointsGeodesic-1))
    deleteat!(θmin, delete_first:(delete_first+nPointsGeodesic-1))

    delete_first = size(Edot, 1) - (nPointsGeodesic-2)
    deleteat!(Edot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Ldot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Qdot, delete_first:(delete_first+nPointsGeodesic-2))
    deleteat!(Cdot, delete_first:(delete_first+nPointsGeodesic-2))

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end


    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end


    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_mino_fdm.jld2"
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(reltol)_Mino_fdm.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end


function load_trajectory(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, h::Float64, reltol::Float64, data_path::String)
    # load ODE solution
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm.txt"
    sol = readdlm(ODE_filename)
    λ=sol[1,:]; t=sol[2,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt2=sol[9,:]; d2θ_dt2=sol[10,:]; d2ϕ_dt2=sol[11,:]; dt_dτ=sol[12,:]; dt_dλ=sol[13,:]
    return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, h::Float64, reltol::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm.txt"
    constants=readdlm(constants_filename)
    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_Mino_fdm.txt"
    constants_derivs = readdlm(constants_derivs_filename)
    EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :]
    Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]
    return EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin
end


function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, t::Vector{Float64}, a::Float64, p::Float64, e::Float64, θi::Float64, q::Float64, 
    h::Float64, reltol::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=data_path *  "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(q)_h_$(h)_tol_$(reltol)_mino_fdm.jld2"
    waveform_data = load(waveform_filename)["data"]
    Mij2 = waveform_data["Mij2"]; ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij2);
    Mijk3 = waveform_data["Mijk3"]; ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk3);
    Mijkl4 = waveform_data["Mijkl4"]; ConstructSymmetricArrays.SymmetrizeFourIndexTensor!(Mijkl4);
    Sij2 = waveform_data["Sij2"]; ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij2);
    Sijk3 = waveform_data["Sijk3"]; ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Sijk3);

    # compute h_{ij} tensor
    hij = [zeros(length(t)) for i=1:3, j=1:3];
    SelfForce.hij!(hij, t, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    # project h_{ij} tensor
    h_plus = SelfForce.hplus(hij, Θ, Φ);
    h_cross = SelfForce.hcross(hij, Θ, Φ);
    return h_plus, h_cross
end


end
end

end