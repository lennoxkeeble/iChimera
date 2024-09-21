module RyanInspiral
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ..Kerr
using ..BLTimeEvolution
using ..FourierFitGSL
using ..SelfAcceleration
using ..RyanFluxes
using ..Kerr
using StaticArrays

Z_1(a::Float64, M::Float64) = 1 + (1 - a^2 / M^2)^(1/3) * ((1 + a / M)^(1/3) + (1 - a / M)^(1/3))
Z_2(a::Float64, M::Float64) = sqrt(3 * (a / M)^2 + Z_1(a, M)^2)
LSO_r(a::Float64, M::Float64) = M * (3 + Z_2(a, M) - sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # retrograde LSO
LSO_p(a::Float64, M::Float64) = M * (3 + Z_2(a, M) + sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # prograde LSO

function compute_inspiral!(t_range_factor::Float64, tOrbit::Float64, nPoints::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function, gRR::Function, gThTh::Function, gΦΦ::Function, nHarm::Int64, reltol::Float64=1e-12, abstol::Float64=1e-10; data_path::String="Data/")
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

    fit_fname_params="fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm).txt";

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
    ι = ones(nPoints) * acos(LLi / sqrt(LLi^2 + CCi));

    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    ics = BLTimeEvolution.HJ_ics(ra, p, e, M);

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
        Ω=ω[1:3]/ω[4]; Ωr=Ω[1]; Ωθ=Ω[2]; Ωϕ=Ω[3];   # BL time frequencies

        #  we want to perform each fit over a set of points which span a physical time range T_fit. In some cases, the frequencies are infinite, and we thus ignore them in our fitting procedure
        if e_t == 0.0 && θmin_t == π/2   # circular equatorial
            Ωr = 1e50; Ωθ =1e50;
            T_Fit = t_range_factor * minimum(@. 2π/Ωϕ)
        elseif e_t == 0.0   # circular non-equatorial
            Ωr = 1e50;
            T_Fit = t_range_factor * minimum(@. 2π/[Ωθ, Ωϕ])
        elseif θmin_t == π/2   # non-circular equatorial
            Ωθ = 1e50;
            T_Fit = t_range_factor * minimum(@. 2π/[Ωr, Ωϕ])
        else   # generic case
            T_Fit = t_range_factor * minimum(@. 2π/Ω)
        end

        saveat = T_Fit / (nPoints-1);    # the user specifies the Float64 of points in each fit, i.e., the resolution, which determines at which points the interpolator should save data points

        # to compute the self force at a point, we must overshoot the solution into the future
        tF = t0 + (nPoints-1) * saveat + (nPoints÷2) * saveat   # evolve geodesic up to tF
        total_num_points = nPoints+(nPoints÷2)   # total Float64 of points in geodesic since we overshoot
        Δti=saveat;    # initial time step for geodesic integration

        saveat_t = range(t0, tF, total_num_points) |> collect
        tspan=(t0, tF)
        # println(tspan)

        # stop when it reaches LSO
        condition(u, t , integrator) = u[1] - rLSO # Is zero when r = rLSO (to 5 d.p)
        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(condition, affect!)

        # numerically solve for geodesic motion
        prob = e == 0.0 ? ODEProblem(BLTimeEvolution.HJ_Eqns_circular, ics, tspan, params) : ODEProblem(BLTimeEvolution.HJ_Eqns, ics, tspan, params);
        
        # if e==0.0
        #     sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t, callback = cb);
        # else
        #     sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);
        # end
      
        sol = solve(prob, AutoTsit5(Rodas4P()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_t);

        # deconstruct solution
        tt = sol.t;
        psi = sol[1, :];
        chi = mod.(sol[2, :], 2π);
        ϕϕ = sol[3, :];

        if (length(sol[1, :]) < total_num_points)
            println("Integration terminated at t = $(last(t))")
            println("total_num_points - len(sol) = $(total_num_points-length(sol[1,:]))")
            println("t0 = $(t0), tF = $(tF), total_num_points = $(total_num_points)\n")
            println("saveat_t:")
            println(saveat_t)
            println("\nsol.t:")
            println(sol.t)
            break
        elseif length(tt)>total_num_points
            tt = sol.t[:total_num_points];
            psi = sol[1, 1:total_num_points];
            chi = mod.(sol[2, 1:total_num_points], 2π);
            ϕϕ = sol[3, 1:total_num_points];
        end

        # compute time derivatives
        psi_dot = BLTimeEvolution.psi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        chi_dot = BLTimeEvolution.chi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        ϕ_dot = BLTimeEvolution.phi_dot.(psi, chi, ϕϕ, a, M, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)

        # compute BL coordinates t, r, θ and their time derivatives
        rr = BLTimeEvolution.r.(psi, p_t, e_t, M)
        θθ = [acos((π/2<chi[i]<1.5π) ? -sqrt(BLTimeEvolution.z(chi[i], θmin_t)) : sqrt(BLTimeEvolution.z(chi[i], θmin_t))) for i in eachindex(chi)]

        r_dot = BLTimeEvolution.dr_dt.(psi_dot, psi, p_t, e_t, M);
        θ_dot = BLTimeEvolution.dθ_dt.(chi_dot, chi, θθ, θmin_t);
        v_spatial = [[r_dot[i], θ_dot[i], ϕ_dot[i]] for i in eachindex(tt)];
        Γ = @. BLTimeEvolution.Γ(tt, rr, θθ, ϕϕ, v_spatial, a, M)

        # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
        r_ddot = BLTimeEvolution.dr2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)
        θ_ddot = BLTimeEvolution.dθ2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)
        ϕ_ddot = BLTimeEvolution.dϕ2_dt2.(tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, a, M)

        ###### MIGHT WANT TO USE VIEWS TO OPTIMIZE A BIT AND AVOID MAKING COPIES IN EACH CALL BELOW ######

        # store trajectory, ignoring the overshot piece
        append!(t, tt[1:nPoints]); append!(dt_dτ, Γ[1:nPoints]); append!(r, rr[1:nPoints]); append!(dr_dt, r_dot[1:nPoints]); append!(d2r_dt2, r_ddot[1:nPoints]); 
        append!(θ, θθ[1:nPoints]); append!(dθ_dt, θ_dot[1:nPoints]); append!(d2θ_dt2, θ_ddot[1:nPoints]); append!(ϕ, ϕϕ[1:nPoints]); 
        append!(dϕ_dt, ϕ_dot[1:nPoints]); append!(d2ϕ_dt2, ϕ_ddot[1:nPoints]);
        
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
        SelfAcceleration.FourierFit.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL, vBL, aBL, xH, x_H, rH, vH, v_H, aH, a_H, v, tt[fit_index_0:fit_index_1], 
            rr[fit_index_0:fit_index_1], r_dot[fit_index_0:fit_index_1], r_ddot[fit_index_0:fit_index_1], θθ[fit_index_0:fit_index_1], 
            θ_dot[fit_index_0:fit_index_1], θ_ddot[fit_index_0:fit_index_1], ϕϕ[fit_index_0:fit_index_1], ϕ_dot[fit_index_0:fit_index_1], 
            ϕ_ddot[fit_index_0:fit_index_1], Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data, 
            Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m, compute_at, nHarm, Ωr, Ωθ, Ωϕ, fit_array_length, n_freqs, chisq, fit_fname_params);


        RyanFluxes.EvolveConstants(tt[nPoints]-tt[1], a, rr[nPoints], θθ[nPoints], EE, Edot, LL, Ldot, CC, Cdot, pArray, ecc, ι, m, M, nPoints)
        # println("Edot = $(Edot[length(Edot)-(nPoints-1)])")
        # println("Edot * Δt = $(Edot[length(Edot)-(nPoints-1)] * (tF-t0))")
        # println("ΔE = $(last(EE)-EE[length(EE)-nPoints])")

        # compute Q
        append!(QQ, ones(nPoints) * (last(CC) + (last(LL) - a * last(EE))^2))
        push!(Qdot, (last(QQ) - Q_t) / (tt[nPoints]-tt[1]))
        append!(Qdot, zeros(nPoints-1))

        # compute new θmin
        c0 = last(CC) / (a^2 * (1.0 - last(EE)^2))
        c1 = -1.0 - (last(LL)^2 + last(CC)) / (a^2 * (1.0 - last(EE)^2))
        append!(θmin, ones(nPoints) * (acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))))

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)

        # update next ics for next piece
        t0 = tt[nPoints+1];
        ics = @SArray [psi[nPoints+1], chi[nPoints+1], ϕϕ[nPoints+1]]
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
    deleteat!(ι, delete_first:(delete_first+nPoints-1))

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


    # save trajectory
    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(reltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin, ι)
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

    println("Self-force file saved to: " * SF_filename)
    println("ODE saved to: " * ODE_filename)
end

end
