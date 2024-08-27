### NEW ###
function compute_inspiral_HJE!(tOrbit::Float64, compute_SF::Float64, fit_time_range_factor::Float64, nPointsGeodesic::Int64, nPointsFit::Int64, M::Float64, m::Float64, a::Float64, p::Float64, 
    e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function,
    gRR::Function, gThTh::Function, gΦΦ::Function, nHarm::Int64, reltol::Float64=1e-10, abstol::Float64=1e-10; data_path::String="Data/")

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
    t_Fluxes = ones(1) * t0

    while tOrbit > t0
        print("Completion: $(round(100 * t0/tOrbit; digits=5))%   \r")
        flush(stdout) 


        # compute waveform
        EstimateMultipoleDerivs.FourierFit.compute_waveform_moments_and_derivs!(a, m, M, xBL_wf, vBL_wf, aBL_wf, xH_wf, x_H_wf, rH_wf, vH_wf, v_H_wf, aH_wf, a_H_wf, v_wf, 
            tt, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp,
            Sij2_wf_temp, Sijk3_wf_temp, nHarm, Ωr, Ωθ, Ωϕ, nPointsGeodesic, n_freqs, chisq)



        # compute self-force at end of physical geodesic
        SelfAcceleration.FourierFit.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL_fit, vBL_fit, aBL_fit, xH_fit, x_H_fit, rH_fit, vH_fit, v_H_fit, aH_fit, a_H_fit, v_fit, tt_fit, rr_fit, r_dot_fit,
            r_ddot_fit, θθ_fit, θ_dot_fit, θ_ddot_fit, ϕϕ_fit, ϕ_dot_fit, ϕ_ddot_fit, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data,
            Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m, compute_at, nHarm, Ωr, Ωθ, Ωϕ, nPointsFit, n_freqs, chisq);

        # evolve orbital parameters using self-force
        EvolveConstants.Evolve_BL(compute_SF, a, last(tt), last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPointsGeodesic)
        
        push!(t_Fluxes, last(tt))


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

    

    

    # save params
    constants = (t_Fluxes, EE, LL, QQ, CC, pArray, ecc, θmin)

end

function compute_inspiral_HJE!(tOrbit::Float64, compute_SF::Float64, fit_time_range_factor::Float64, nPointsGeodesic::Int64, nPointsFit::Int64, M::Float64, m::Float64, a::Float64, p::Float64, 
    e::Float64, θi::Float64,  Γαμν::Function, g_μν::Function, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, gTT::Function, gTΦ::Function,
    gRR::Function, gThTh::Function, gΦΦ::Function, nHarm::Int64, reltol::Float64=1e-10, abstol::Float64=1e-10; data_path::String="Data/")

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

    while tOrbit > t0
        print("Completion: $(100 * t0/tOrbit)%   \r")
        flush(stdout) 


        # compute waveform
        SelfForce.compute_waveform_moments_and_derivs!(a, m, M, xBL_wf, vBL_wf, aBL_wf, xH_wf, x_H_wf, rH_wf, vH_wf, v_H_wf, aH_wf, a_H_wf, v_wf, 
            tt, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3_wf_temp, Mijkl4_wf_temp,
            Sij2_wf_temp, Sijk3_wf_temp, nHarm, Ωr, Ωθ, Ωϕ, nPointsGeodesic, n_freqs, chisq)



        # compute self-force at end of physical geodesic
        SelfForce.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL_fit, vBL_fit, aBL_fit, xH_fit, x_H_fit, rH_fit, vH_fit, v_H_fit, aH_fit, a_H_fit, v_fit, tt_fit, rr_fit, r_dot_fit,
            r_ddot_fit, θθ_fit, θ_dot_fit, θ_ddot_fit, ϕϕ_fit, ϕ_dot_fit, ϕ_ddot_fit, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Mij2_data, Mijk2_data, Sij1_data,
            Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, a, M, m, compute_at, nHarm, Ωr, Ωθ, Ωϕ, nPointsFit, n_freqs, chisq);

        # evolve orbital parameters using self-force
        EvolveConstants.Evolve_BL(compute_SF, a, last(tt), last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, M, nPointsGeodesic)


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

    



    

    # save params
    constants = (EE, LL, QQ, CC, pArray, ecc, θmin)

end