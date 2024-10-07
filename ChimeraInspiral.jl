#=
    In this module we write the master functions for computing EMRIs based on the Chimera kludge scheme presented in arXiv:1109.0572v2 (hereafter Ref. [1]), which introduced a local treatment of the self-force approximated using post-Newtonian, post-Minkowskian and
    black hole perturbation theoretic methods, hence the kludge nature of the scheme. In particular, Ref. [1] employs PN and PM expansions to obtain expressions for the regularized metric perturbations in terms of time-asymmetric radiation reaction potentials which are
    functions of the EMRI trajectory. These metric perturbations are then substituted into MiSaTaQuWa equation from black hole perturbation theory to obtain an approximation of the self-force, which is then used to compute radiative fluxes of the constants of motion. 
    
    Schematically, our implementation of the Chimera is as follows (noting that Eq. XX refers to Ref. [1]):

    (1) Numerically evolve the geodesic equation with the initial constants of motion and initial conditions for a time ΔT. (Actually, the geodesic is evolved for one additional time step, which is used as the initial
        condition for the evolution of the subsequent geodesic, and is not used in any computations done with the current piecewise geodesic. Note that we do not yet consider conservative effects, which would introduce a correction when "jumping" between
        geodesics in the method of osculating orbits.)
    
    (2) Compute the multipole moment derivatives required for waveform waveform generation (Eqs. 48-49, 85-86). These are computed at every point on each piecewise geodesic. They are not used to evolve the inspiral, but are saved for waveform computation outside
        these master functions.

    (3) Compute the multipole moment derivatives required for the self-force computation (Eqs. 48-49). These are computed at the end point of each piecewise geodesic.
    
    (4) Compute the self-acceleration and use this to update the constants of motion.
    
    (5) Evolve the next geodesic using the intial conditions as taken from the additional time step in (1) with the updated constants of motion. Repeat this process until the inspiral has been evolved for the desired amount of time.

    Steps (1-5) are schematic, and, in practice, we have several functions which carry out steps (2-3) in different ways:

    (i) Estimate all the multipole moment derivatves using a least-squares fitting algorithm. In particular, the multipole moments are orbital functionals and thus possess a Fourier series expansion in terms of the fundamental frequencies of motion (e.g., see Ref. [1] and
        arXiv:astro-ph/0308479). We can compute time series data (from analytic expressions) of the first and second derivatives of the mass and current multipole moments, fit these to their fourier series expansion for the coefficients, and take analytic time derivatives
        of the (truncated) Fourier series to approximate the high order derivatives (see Eqs. 98-99).
        
        For the waveform moments, this is done by taking the physical trajectory from a given piecewise geodesic and using the corresponding time series data in the fit. This physical trajectory is in contrast to a "fictitous" trajectory we use to compute the multipole
        moments for the self-force computation. We wish to compute the self force at the endpoint of each piecewise geodesic. We could do this by using the fit carried out for the waveform moments, but this would place the point of interest at the final
        point in the time series, which is where the fit performs the worst (i.e., at the edges). So, to get a better fit, we artifically place the point of interest at the center of a fictitous geodesic trajectory and then perform the fit using this time series data.
        This ficticious trajectory is constructed by taking the point of interest and evolving the geodesic equation into the past and future of this point to obtain a time series with the point of interest at its center (this approach was taken by the authors of Ref. [1] in their
        original implementation). This fictitious trajectory is only used to compute the self-force and is discarded thereafter. Once these steps have been carried out to approximate the relevant multipole derivatives at the point of interest, one can then compute the self-force
        there, update the constants of motion and continue in the inspiral evolution.

        We provide two options for carrying out this fitting procedure. The first uses GSL's multilinear fitting algorithm to fit the data to its fourier series expansion. This method is slow, and gets increasingly slow as one includes more harmonic
        frequencies in the fit (which doesn't even necessarily improve the fit). For this option, we recommend using N=2 harmonics, for which we have found the sixth time derivative of test orbital functionals (e.g., r cos(θ) sin (θ)) to be accurate to 1 part in 
        10^{5} (which is consistent with that found in [1]). The second method for implementing these fits useσ Julia's base least squares algorithm, which we have generally found to be faster, and which can be tuned to give more accurate fits. For test orbital
        functionals, we found an accuracy in the sixth time derivative of 1 part in 10^{7} for N=3 harmonics and N=200 points in the time series.

        In practice, both of these approaches are slow (in relative terms). We outline two faster methods below, with option (iii) being the one we recommend using.
    
    (ii) Estimate all the multipole moment derivatives using finite difference formulae. Finite differences as implemented in coordinate time leads to catastrophic cancellations in the high order derivative estimation (this was also reported in Ref. [1]).
         However, when applying finite difference formulae in Mino time, we have not observed the same numerical instabilities in the limited testing we have done so far. We are still to carry out convergence tests, but we do expect that this approach would
         still require too much fine tuning to be reliable for inspirals across the entire parameter space.
    
    (iii) Estimate only the multipole moment derivatives required for the self-force computation using a least-squares fitting algorithm, and estimate those required for the waveform using finite differences. To compute the radiation reaction fluxes, we must
          numerically take up to six more time derivatives of time series data of the multipole moments. However, to generate the waveform, we need only take up to two more time derivatives of the multipole moments. Thus, we can reliably use finite differences
          to estimate these derivatives, while using the more robust fourier fitting to compute the fluxes. This leads to a significant speed up because, in total, there are 22 independent components of the multipole moments which must be computed for the self-force,
          while there are 41 components which must be computed for the waveform. Thus, by using finite differences for the waveform moments, we can reduce the number of fits we must perform by a factor of ~2/3 (since we compute the fits for the waveform moments
          and the self-force moments separately). This approach is close in efficiency to (ii), but is more reliable because we still use the robust fourier fitting algorithm to compute the multipole derivatives for the fluxes, while the finite difference formulae
          for the waveform moments are accurate within O(h^{5}) and are less prone to numerical instabilities than (ii) is because (a) we no longer use formaulae with h^{3} up to h^{6} in the denominator (where, here, h is typically taken to be ~1e-3), and (b) we are
          are estimating lower-order derivatives, which are of a higher order of magnitude, allowing us to use larger step sizes (for a fixed relative error) and thus avoid the effects of catastrophic cancellation. For example, estimating a derivative of the order 10^{-10} requires a smaller
          step size for a fixed relative error. For the same relative error, a larger absolute error is acceptable for estimating a derivative of the order 10^{-5}, thus allowing a large step size.

    Using approach (iii), we expect a year-long EMRI to take ~10 hours to compute. This will be ~2x faster using approach (ii), but may be accompanied by significant accumulation of numerical errors (or quantum fluctuations, depending on one's perspective). Approach (i) will take infinte time.
=#

module ChimeraInspiral

Z_1(a::Float64) = 1 + (1 - a^2 / 1.0)^(1/3) * ((1 + a)^(1/3) + (1 - a)^(1/3))
Z_2(a::Float64) = sqrt(3 * a^2 + Z_1(a)^2)
LSO_r(a::Float64) = (3 + Z_2(a) - sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # retrograde LSO
LSO_p(a::Float64) = (3 + Z_2(a) + sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # prograde LSO

module FourierFit

module BLTime
using LinearAlgebra
using Combinatorics
using StaticArrays
using HDF5
using DifferentialEquations
using ....Kerr
using ....ConstantsOfMotion
using ....BLTimeGeodesics
using ....FourierFitGSL
using ....CircularNonEquatorial
using ....HarmonicCoords
using ....SymmetricTensors
using ....SelfAcceleration
using ....EstimateMultipoleDerivs
using ....EvolveConstants
using ....Waveform
using JLD2
using Printf
using ...ChimeraInspiral
using .....MultipoleFitting

"""
    compute_inspiral(args...)

Evolve inspiral with coordinate time parameterization and estimating the high order multipole derivatives using Fourier fits with respect to t.

- `tInspiral::Float64`: total coordinate time to evolve the inspiral.
- `compute_SF::Float64`: time interval between self-force computations.
- `fit_time_range_factor::Float64`: time range over which to perform the fourier fits as a fraction of the minimum time period associated with the fundamental frequencies.
- `nPointsGeodesic::Int64`: number of points in each piecewise geodesic.
- `nPointsFit::Int64`: number of points in each fit.
- `q::Float64`: mass ratio.
- `a::Float64`: black hole spin 0 < a < 1.
- `p::Float64`: initial semi-latus rectum.
- `e::Float64`: initial eccentricity.
- `θmin::Float64`: initial inclination angle.
- `sign_Lz::Int64`: sign of the z-component of the angular momentum (+1 for prograde, -1 for retrograde).
- `psi_0::Float64`: initial radial angle variable.
- `chi_0::Float64`: initial polar angle variable.
- `phi_0::Float64`: initial azimuthal angle.
- `fit::String`: type of fit to perform. Options are "GSL" or "Julia" to use Julia's GSL wrapper for a multilinear fit, or to use Julia's base least squares solver.
- `nHarm::Int64`: number of radial harmonics to include in the fit (see `FourierFitGSL.jl` or `FourierFitJuliaBase.jl`).
- `reltol`: relative tolerance for ODE solver.
- `abstol`: absolute tolerance for ODE solver.
- `data_path::String`: path to save data.
"""

function compute_inspiral(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nPointsGeodesic::Int64, nPointsFit::Int64, nHarm::Int64, fit_time_range_factor::Float64,
    compute_SF::Float64, tInspiral::Float64, fit::String, reltol::Float64=1e-14, abstol::Float64=1e-14; data_path::String="Data/")
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
    xBL_fit = [zeros(3) for i in 1:nPointsFit]; xBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    vBL_fit = [zeros(3) for i in 1:nPointsFit]; vBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    aBL_fit = [zeros(3) for i in 1:nPointsFit]; aBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    xH_fit = [zeros(3) for i in 1:nPointsFit]; xH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    vH_fit = [zeros(3) for i in 1:nPointsFit]; vH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    v_fit = zeros(nPointsFit);   v_wf = zeros(nPointsGeodesic);
    rH_fit = zeros(nPointsFit);  rH_wf = zeros(nPointsGeodesic);
    aH_fit = [zeros(3) for i in 1:nPointsFit];  aH_wf = [zeros(3) for i in 1:nPointsGeodesic];

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

    # compute number of fitting frequencies used in fits to the fourier series expansion of the multipole moments
    if e == 0.0 && θmin == π/2   # circular equatorial
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_1(nHarm);
    elseif e != 0.0 && θmin != π/2   # generic case
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_3(nHarm);
    else   # circular non-equatorial or non-circular equatorial — either way only two non-trivial frequencies
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_2(nHarm);
    end

    # compute apastron
    ra = p / (1 - e);

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

    rplus = Kerr.KerrMetric.rplus(a); rminus = Kerr.KerrMetric.rminus(a);
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    t_Fluxes = ones(1) * t0
    geodesic_ics = @SArray [psi_0, chi_0, phi_0];

    rLSO = ChimeraInspiral.LSO_p(a)

    use_custom_ics = true; use_specified_params = true;
    save_at_trajectory = compute_SF / (nPointsGeodesic - 1); Δti=save_at_trajectory;    # initial time step for geodesic integration

    # in the code, we will want to compute the geodesic with an additional time step at the end so that these coordinate values can be used as initial conditions for the
    # subsequent geodesic
    geodesic_time_length = compute_SF + save_at_trajectory;
    num_points_geodesic = nPointsGeodesic + 1;

    while tInspiral > t0
        print("Completion: $(round(100 * t0/tInspiral; digits=5))%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # orbital parameters
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θminArray); e_t = last(ecc);  

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t / (1.0 - e_t); rp=p_t / (1.0 + e_t);
        A = 1.0 / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t); p4 = r4 * (1.0 + e_t)    # Above Eq. 96

        # geodesic
        tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi = BLTimeGeodesics.compute_kerr_geodesic(a, p_t, e_t, θmin_t, sign_Lz, num_points_geodesic, use_specified_params, geodesic_time_length, Δti, reltol, abstol;
        ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)

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
        ω = ConstantsOfMotion.KerrFreqs(a, p_t, e_t, θmin_t, E_t, L_t, Q_t, C_t, rplus, rminus);    # Mino time frequencies
        Ω=ω[1:3]/ω[4]; 
        Ωr, Ωθ, Ωϕ = Ω;   # BL time frequencies

        # # compute waveform
        EstimateMultipoleDerivs.FourierFit.compute_waveform_moments_and_derivs!(a, q, xBL_wf, vBL_wf, aBL_wf, xH_wf, rH_wf, vH_wf, aH_wf, v_wf, 
            tt, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp,
            Sij2_wf_temp, Sijk3_wf_temp, nHarm, Ωr, Ωθ, Ωϕ, nPointsGeodesic, n_freqs, chisq, fit)

        # store multipole data for waveforms — we only save the independent components
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
        BLTimeGeodesics.compute_kerr_geodesic_past_and_future(a, p_t, e_t, θmin_t, sign_Lz, use_specified_params, nPointsFit, T_Fit, Δti_fit, reltol, abstol; ics = midpoint_ics,
        E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)
        
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

        # compute multipole moments and self-acceleration
        SelfAcceleration.FourierFit.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL_fit, vBL_fit, aBL_fit, xH_fit, rH_fit, vH_fit, aH_fit, v_fit, tt_fit, rr_fit, r_dot_fit, 
        r_ddot_fit, θθ_fit, θ_dot_fit, θ_ddot_fit, ϕϕ_fit, ϕ_dot_fit, ϕ_ddot_fit, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6,
            Mij2_data, Mijk2_data, Sij1_data, a, q, compute_at, nHarm, Ωr, Ωθ, Ωϕ, nPointsFit, n_freqs, chisq, fit)

        # evolve orbital parameters using self-force
        EvolveConstants.Evolve_BL(compute_SF, a, last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θminArray)
        
        push!(t_Fluxes, last(tt))

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)
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
    waveform_filename=waveform_moments_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)
    

    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    h5open(sol_filename, "w") do file
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

function solution_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    return data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_nHarm_$(nHarm)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_"*fit*"_fit.h5"
end

function waveform_moments_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    return data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_nHarm_$(nHarm)_fit_range_factor_$(fit_time_range_factor)_BL_fourier_"*fit*"_fit.jld2"
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    h5f = h5open(sol_filename, "r")
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
    close(h5f)
    return t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    h5f = h5open(sol_filename, "r")
    t_Fluxes = h5f["t_Fluxes"][:]
    EE = h5f["Energy"][:]
    Edot = h5f["Edot"][:]
    LL = h5f["AngularMomentum"][:]
    Ldot = h5f["Ldot"][:]
    CC = h5f["CarterConstant"][:]
    Cdot = h5f["Cdot"][:]
    QQ = h5f["AltCarterConstant"][:]
    Qdot = h5f["Qdot"][:]
    pArray = h5f["p"][:]
    ecc = h5f["eccentricity"][:]
    θminArray = h5f["theta_min"][:]
    close(h5f)
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θminArray
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    # load waveform multipole moments
    waveform_filename=waveform_moments_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
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
function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    rm(sol_filename)

    waveform_filename=waveform_moments_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    rm(waveform_filename)
end


end

module MinoTime

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
using ....SymmetricTensors
using ....EvolveConstants
using ....Waveform
using JLD2
using Printf

using ...ChimeraInspiral
using ....MultipoleFitting

"""
    compute_inspiral(args...)

Evolve inspiral with with Mino time parameterization and estimating the high order multipole derivatives using Fourier fits with respect to λ.

- `tInspiral::Float64`: total coordinate time to evolve the inspiral.
- `compute_SF::Float64`: Mino time interval between self-force computations.
- `fit_time_range_factor::Float64`: time range over which to perform the fourier fits as a fraction of the minimum time period associated with the Mino time fundamental frequencies.
- `nPointsFit::Int64`: number of points in each fit.
- `q::Float64`: mass ratio.
- `a::Float64`: black hole spin 0 < a < 1.
- `p::Float64`: initial semi-latus rectum.
- `e::Float64`: initial eccentricity.
- `θmin::Float64`: initial inclination angle.
- `sign_Lz::Int64`: sign of the z-component of the angular momentum (+1 for prograde, -1 for retrograde).
- `psi_0::Float64`: initial radial angle variable.
- `chi_0::Float64`: initial polar angle variable.
- `phi_0::Float64`: initial azimuthal angle.
- `fit::String`: type of fit to perform. Options are "GSL" or "Julia" to use Julia's GSL wrapper for a multilinear fit, or to use Julia's base least squares solver.
- `nHarm::Int64`: number of radial harmonics to include in the fit (see `FourierFitGSL.jl` or `FourierFitJuliaBase.jl`).
- `use_FDM::Bool`: whether to use finite difference formulae to compute the derivatives of the multipole moments required for the waveform.
- `reltol`: relative tolerance for ODE solver.
- `abstol`: absolute tolerance for ODE solver.
- `h::Float64`: if use_FDM=true, then h is the step size for the ODE solver (and one does not specify nPointsGeodesic).
- `nPointsGeodesic::Int64`: if use_FDM=false, this sets the number of points in the geodesic.
- `data_path::String`: path to save data.

# Notes
- We provide the option to use finite differences in Mino time to compute the derivatives of the multipole moments required for the waveform (while the derivatives of the moments for computing the fluxes is done using the more robust method of Fourier
fitting). In constructing waveforms, one needs lower order derivatives of the respective multipole moments, meaning that the waveform can be more accurately constructed from FDM than the self-acceleration can (since this requires taking up to six derivatives
compared to only up to two for the waveforms). Using this hybrid approach allows one to significantly speed up the inspiral evolution.
"""
function compute_inspiral(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nPointsFit::Int64, nHarm::Int64, fit_time_range_factor::Float64,
    compute_SF::Float64, tInspiral::Float64, use_FDM::Bool, fit::String, reltol::Float64=1e-14, abstol::Float64=1e-14; h::Float64 = 0.001, nPointsGeodesic::Int64 = 500, data_path::String="Data/")

    if iseven(nPointsFit)
        throw(DomainError(nPointsFit, "nPointsFit must be odd"))
    end

    if use_FDM
        nPointsGeodesic = floor(Int, compute_SF / h)
        save_at_trajectory = h; Δλi=h/10;    # initial time step for geodesic integration
    else
        save_at_trajectory = compute_SF / (nPointsGeodesic - 1); Δλi=save_at_trajectory/10;    # initial time step for geodesic integration
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
    xBL_fit = [zeros(3) for i in 1:nPointsFit]; xBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    vBL_fit = [zeros(3) for i in 1:nPointsFit]; vBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    aBL_fit = [zeros(3) for i in 1:nPointsFit]; aBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    xH_fit = [zeros(3) for i in 1:nPointsFit];  xH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    vH_fit = [zeros(3) for i in 1:nPointsFit];  vH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    v_fit = zeros(nPointsFit);   v_wf = zeros(nPointsGeodesic);
    rH_fit = zeros(nPointsFit);  rH_wf = zeros(nPointsGeodesic);
    aH_fit = [zeros(3) for i in 1:nPointsFit];  aH_wf = [zeros(3) for i in 1:nPointsGeodesic];

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

    # compute number of fitting frequencies used in fits to the fourier series expansion of the multipole moments
    if e == 0.0 && θmin == π/2   # circular equatorial
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_1(nHarm);
    elseif e != 0.0 && θmin != π/2   # generic case
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_3(nHarm);
    else   # circular non-equatorial or non-circular equatorial — either way only two non-trivial frequencies
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_2(nHarm);
    end

    # compute apastron
    ra = p / (1 - e);

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

    rplus = Kerr.KerrMetric.rplus(a); rminus = Kerr.KerrMetric.rminus(a);
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    t_Fluxes = ones(1) * t0
    λ0 = 0.0
    geodesic_ics = @SArray [t0, psi_0, chi_0, phi_0];

    rLSO = ChimeraInspiral.LSO_p(a)

    use_custom_ics = true; use_specified_params = true;

    # in the code, we will want to compute the geodesic with an additional time step at the end so that these coordinate values can be used as initial conditions for the
    # subsequent geodesic
    geodesic_time_length = compute_SF + save_at_trajectory;
    num_points_geodesic = nPointsGeodesic + 1;

    while tInspiral > t0
        print("Completion: $(round(100 * t0/tInspiral; digits=5))%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # orbital parameters
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θminArray); e_t = last(ecc);

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t / (1.0 - e_t); rp=p_t / (1.0 + e_t);
        A = 1.0 / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t); p4 = r4 * (1.0 + e_t)    # Above Eq. 96

        # geodesic
        λλ, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi, dt_dλλ = MinoTimeGeodesics.compute_kerr_geodesic(a, p_t, e_t, θmin_t, sign_Lz, num_points_geodesic,
        use_specified_params, geodesic_time_length, Δλi, reltol, abstol; ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)
        
        λλ = λλ .+ λ0   # λλ from the above function call starts from zero 

        # check that geodesic output is as expected
        if (length(λλ) != num_points_geodesic) || !isapprox(λλ[nPointsGeodesic], λ0 + compute_SF, rtol=1e-3)
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
        ω = ConstantsOfMotion.KerrFreqs(a, p_t, e_t, θmin_t, E_t, L_t, Q_t, C_t, rplus, rminus);    # Mino time frequencies
        ωr=ω[1]; ωθ=ω[2]; ωϕ=ω[3];   # mino time frequencies

        if use_FDM
            EstimateMultipoleDerivs.FiniteDifferences.compute_waveform_moments_and_derivs_Mino!(a, E_t, L_t, C_t, q, xBL_wf, vBL_wf, aBL_wf, xH_wf, rH_wf, vH_wf, aH_wf, v_wf, tt, rr,
            r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp, nPointsGeodesic, h)
        else
            EstimateMultipoleDerivs.FourierFit.compute_waveform_moments_and_derivs_Mino!(a, E_t, L_t, C_t, q, xBL_wf, vBL_wf, aBL_wf, xH_wf, rH_wf, vH_wf, aH_wf, v_wf, 
            λλ, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp,
            Sij2_wf_temp, Sijk3_wf_temp, nHarm, ωr, ωθ, ωϕ, nPointsGeodesic, n_freqs, chisq, fit)
        end


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
        MinoTimeGeodesics.compute_kerr_geodesic_past_and_future(a, p_t, e_t, θmin_t, sign_Lz, use_specified_params, nPointsFit, T_Fit, Δλi_fit, reltol, abstol; ics=midpoint_ics,
        E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)

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
        # SelfAcceleration.FourierFit.selfAcc_Mino!(aSF_H_temp, aSF_BL_temp, xBL_fit, vBL_fit, aBL_fit, xH_fit, rH_fit, vH_fit, aH_fit, v_fit, λλ_fit, 
        #     rr_fit, r_dot_fit, r_ddot_fit, θθ_fit, θ_dot_fit, θ_ddot_fit, ϕϕ_fit, ϕ_dot_fit, ϕ_ddot_fit, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6,
        #     Mij2_data, Mijk2_data, Sij1_data, a, E_t, L_t, C_t, q, compute_at, nHarm,
        #     ωr, ωθ, ωϕ, nPointsFit, n_freqs, chisq, fit);

        SelfAcceleration.FourierFit.selfAcc_Mino!(aSF_H_temp, aSF_BL_temp, xBL_fit, vBL_fit, aBL_fit, xH_fit, rH_fit, vH_fit, aH_fit, v_fit, λλ_fit,
            rr_fit, r_dot_fit, r_ddot_fit, θθ_fit, θ_dot_fit, θ_ddot_fit, ϕϕ_fit, ϕ_dot_fit, ϕ_ddot_fit, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6,
            Mij2_data, Mijk2_data, Sij1_data, a, q, E_t, L_t, C_t,
            compute_at, nHarm, ωr, ωθ, ωϕ, nPointsFit, n_freqs, chisq, fit)
        
        Δt = last(tt) - tt[1]
        EvolveConstants.Evolve_BL(Δt, a, last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θminArray)
        push!(t_Fluxes, last(tt))

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)
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
    waveform_filename=waveform_moments_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)
    

    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
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

function solution_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    return data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_nHarm_$(nHarm)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_"*fit*"_fit.h5"
end

function waveform_moments_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    return data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_nHarm_$(nHarm)_fit_range_factor_$(fit_time_range_factor)_Mino_fourier_"*fit*"_fit.jld2"
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
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


function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    h5f = h5open(sol_filename, "r")
    t_Fluxes = h5f["t_Fluxes"][:]
    EE = h5f["Energy"][:]
    LL = h5f["AngularMomentum"][:]
    CC = h5f["CarterConstant"][:]
    QQ = h5f["AltCarterConstant"][:]
    pArray = h5f["p"][:]
    ecc = h5f["eccentricity"][:]
    θminArray = h5f["theta_min"][:]
    Edot = h5f["Edot"][:]
    Ldot = h5f["Ldot"][:]
    Qdot = h5f["Qdot"][:]
    Cdot = h5f["Cdot"][:]
    close(h5f)
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θminArray
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64,a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    # load waveform multipole moments
    waveform_filename=waveform_moments_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
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

function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    rm(sol_filename)

    waveform_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, data_path)
    rm(waveform_filename)
end

end
end

module FiniteDifferences

module MinoTime

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
using JLD2
using Printf
using ...ChimeraInspiral


"""
    compute_inspiral(args...)

Evolve inspiral with Mino time parameterization and estimating the high order multipole derivatives via finite difference formulae.

- `tInspiral::Float64`: total coordinate time to evolve the inspiral.
- `compute_SF::Float64`: Mino time interval between self-force computations.
- `q::Float64`: mass ratio.
- `a::Float64`: black hole spin 0 < a < 1.
- `p::Float64`: initial semi-latus rectum.
- `e::Float64`: initial eccentricity.
- `θmin::Float64`: initial inclination angle.
- `sign_Lz::Int64`: sign of the z-component of the angular momentum (+1 for prograde, -1 for retrograde).
- `psi_0::Float64`: initial radial angle variable.
- `chi_0::Float64`: initial polar angle variable.
- `phi_0::Float64`: initial azimuthal angle.
- `h::Float64`: step size for the ODE solver.
- `reltol`: relative tolerance for ODE solver.
- `abstol`: absolute tolerance for ODE solver.
- `data_path::String`: path to save data.
"""
function compute_inspiral(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, compute_SF::Float64, tInspiral::Float64,
    reltol::Float64=1e-14, abstol::Float64=1e-14; data_path::String="Data/")

    nPointsGeodesic = floor(Int, compute_SF / h)

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

    xBL_stencil = [zeros(3) for i in 1:stencil_array_length]; xBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    vBL_stencil = [zeros(3) for i in 1:stencil_array_length]; vBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    aBL_stencil = [zeros(3) for i in 1:stencil_array_length]; aBL_wf = [zeros(3) for i in 1:nPointsGeodesic];
    xH_stencil = [zeros(3) for i in 1:stencil_array_length];  xH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    vH_stencil = [zeros(3) for i in 1:stencil_array_length];  vH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    v_stencil = zeros(stencil_array_length);   v_wf = zeros(nPointsGeodesic);
    rH_stencil = zeros(stencil_array_length);  rH_wf = zeros(nPointsGeodesic);
    aH_stencil = [zeros(3) for i in 1:stencil_array_length];  aH_wf = [zeros(3) for i in 1:nPointsGeodesic];
    
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
    ra = p / (1 - e);

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
    λ0 = 0.0;
    t0 = 0.0
    t_Fluxes = ones(1) * t0;
    geodesic_ics = @SArray [t0, psi_0, chi_0, phi_0];

    rLSO = ChimeraInspiral.LSO_p(a)

    use_custom_ics = true; use_specified_params = true;
    Δλi=h/10;    # initial time step for geodesic integration

    # in the code, we will want to compute the geodesic with an additional time step at the end so that these coordinate values can be used as initial conditions for the subsequent geodesic
    save_at_trajectory = h; Δλi=h/10;    # initial time step for geodesic integration
    num_points_geodesic = nPointsGeodesic + 1;
    geodesic_time_length = h * (num_points_geodesic-1);

    while tInspiral > t0
        print("Completion: $(round(100 * t0/tInspiral; digits=5))%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # orbital parameters
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θminArray); e_t = last(ecc);

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t / (1.0 - e_t); rp=p_t / (1.0 + e_t);
        A = 1.0 / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t); p4 = r4 * (1.0 + e_t)    # Above Eq. 96

        # geodesic
        λλ, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi, dt_dλλ = MinoTimeGeodesics.compute_kerr_geodesic(a, p_t, e_t, θmin_t, sign_Lz, num_points_geodesic,
        use_specified_params, geodesic_time_length, Δλi, reltol, abstol; ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)
        
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
        EstimateMultipoleDerivs.FiniteDifferences.compute_waveform_moments_and_derivs_Mino!(a, E_t, L_t, C_t, q, xBL_wf, vBL_wf, aBL_wf, xH_wf, rH_wf, vH_wf, aH_wf, v_wf, tt, rr, r_dot, r_ddot, θθ, θ_dot, 
            θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp, nPointsGeodesic, h)
        
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
        # compute past and future geodesic at end point of physical geodesic
        midpoint_ics = @SArray [last(tt), last(psi), last(chi), last(ϕϕ)];
        T_Fit = (stencil_array_length - 1) * h;
        λλ_stencil, tt_stencil, rr_stencil, θθ_stencil, ϕϕ_stencil, r_dot_stencil, θ_dot_stencil, ϕ_dot_stencil, r_ddot_stencil, θ_ddot_stencil, ϕ_ddot_stencil, Γ_stencil, psi_stencil, chi_stencil, dt_dλ_stencil = 
        MinoTimeGeodesics.compute_kerr_geodesic_past_and_future(a, p_t, e_t, θmin_t, sign_Lz, use_specified_params, stencil_array_length, T_Fit, Δλi, reltol, abstol; ics=midpoint_ics,
        E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)
        
        # println(length(λλ_stencil))
        # println(λλ_stencil)
        # println(diff(λλ_stencil))
        # throw(DomainError())
        # break

        compute_at=stencil_array_length÷2+1;    # by construction, the end point of the physical geoodesic is at the center of the stencil array for the future and past geodesic
        # check that that the midpoint of the fit geodesic arrays are equal to the final point of the physical arrays
        if tt_stencil[compute_at] != last(tt) || rr_stencil[compute_at] != last(rr) || θθ_stencil[compute_at] != last(θθ) || ϕϕ_stencil[compute_at] != last(ϕϕ) ||
            r_dot_stencil[compute_at] != last(r_dot) || θ_dot_stencil[compute_at] != last(θ_dot) || ϕ_dot_stencil[compute_at] != last(ϕ_dot) || 
            r_ddot_stencil[compute_at] != last(r_ddot)|| θ_ddot_stencil[compute_at] != last(θ_ddot)|| ϕ_ddot_stencil[compute_at] != last(ϕ_ddot) ||
            Γ_stencil[compute_at] != last(Γ) || psi_stencil[compute_at] != last(psi) || chi_stencil[compute_at] != last(chi)
            println("Integration terminated at t = $(last(t)). Reason: midpoint of fit geodesic does not align with final point of physical geodesic")
            break
        end
        SelfAcceleration.FiniteDifferences.selfAcc_mino!(a, E_t, L_t, C_t, aSF_H_temp, aSF_BL_temp, xBL_stencil, vBL_stencil, aBL_stencil, xH_stencil,
        rH_stencil, vH_stencil, aH_stencil, v_stencil, tt_stencil, rr_stencil, r_dot_stencil, r_ddot_stencil, θθ_stencil, 
        θ_dot_stencil, θ_ddot_stencil, ϕϕ_stencil, ϕ_dot_stencil, ϕ_ddot_stencil, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, 
        Mijk2_data, Sij1_data, q, compute_at, h);

        Δt = last(tt) - tt[1];
        EvolveConstants.Evolve_BL(Δt, a, last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θminArray)
        push!(t_Fluxes, last(tt))

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)
        # break
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
    waveform_filename=waveform_moments_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, h, data_path)
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, h, data_path)
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

function solution_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    return data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm.h5"
end

function waveform_moments_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    return data_path *  "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm.jld2"
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, h, data_path)
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
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, h, data_path)
    h5f = h5open(sol_filename, "r")
    t_Fluxes = h5f["t_Fluxes"][:]
    EE = h5f["Energy"][:]
    LL = h5f["AngularMomentum"][:]
    CC = h5f["CarterConstant"][:]
    QQ = h5f["AltCarterConstant"][:]
    pArray = h5f["p"][:]
    ecc = h5f["eccentricity"][:]
    θminArray = h5f["theta_min"][:]
    Edot = h5f["Edot"][:]
    Ldot = h5f["Ldot"][:]
    Qdot = h5f["Qdot"][:]
    Cdot = h5f["Cdot"][:]
    close(h5f)
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θminArray
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=waveform_moments_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, h, data_path)
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

function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, h::Float64, data_path::String)
    sol_filename=data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm.h5"
    rm(sol_filename)

    waveform_filename=data_path *  "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_h_$(h)_Mino_fdm.jld2"
    rm(waveform_filename)
end

end
end

end