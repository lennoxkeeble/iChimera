module Resonances

module Location
using Elliptic, GSL, Roots
# coefficients of polynomial in E, L (Eq. E3)
αI(a::Float64, M::Float64, rI::Float64, zm::Float64) = (rI^2 + a^2) * (rI^2 + a^2 * zm) + 2.0M * rI * a^2 * (1.0 - zm)   # Eq. E4
βI(a::Float64, M::Float64, rI::Float64, zm::Float64) = - 2.0M * rI * a    # Eq. E5
γI(a::Float64, M::Float64, rI::Float64, zm::Float64) = -(1.0 / (1.0 - zm)) * (rI^2 + a^2 * zm - 2.0 * M * rI)    # Eq. E6
λI(a::Float64, M::Float64, rI::Float64, zm::Float64) = -(rI^2 + a^2 * zm) * (rI^2 - 2.0M * rI + a^2)    # Eq. E7

# for circular orbits
α_2(a::Float64, M::Float64, r0::Float64, zm::Float64) = 2.0r0 * (r0^2 + a^2) - a^2 * (r0 - M) * (1.0 - zm)    # Eq. E8
β_2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -a * M    # Eq. E9
γ_2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -(r0 - M) / (1.0 - zm)    # Eq. E10
λ_2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -r0 * (r0^2 - 2.0M * r0 + a^2) - (r0 - M) * (r0^2 + a^2 * zm)    # Eq. E11

# define [*, *] operation in Eq. E3
commute(Πa::Float64, Πp::Float64, Ωa::Float64, Ωp::Float64) = Πa * Ωp - Πp * Ωa

# test prograde / retrograde orbits
function compute_ELQ(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64)
    M=1.0;

    ### COMPUTE ELQ ###
    zm = cos(θmin)^2
    if e==0.0
        r0 = p * M
        α1 = αI(a, M, r0, zm)
        α2 = α_2(a, M, r0, zm)
        β1 = βI(a, M, r0, zm)
        β2 = β_2(a, M, r0, zm)
        γ1 = γI(a, M, r0, zm)
        γ2 = γ_2(a, M, r0, zm)
        λ1 = λI(a, M, r0, zm)
        λ2 = λ_2(a, M, r0, zm)
    else
        rp = p * M / (1 + e)
        ra = p * M / (1 - e)
        α1 = αI(a, M, ra, zm)
        α2 = αI(a, M, rp, zm)
        β1 = βI(a, M, ra, zm)
        β2 = βI(a, M, rp, zm)
        γ1 = γI(a, M, ra, zm)
        γ2 = γI(a, M, rp, zm)
        λ1 = λI(a, M, ra, zm)
        λ2 = λI(a, M, rp, zm)
    end
    
    # write out coefficients of Eq. E12 in the form ax^2 + bx + c
    aa = (commute(α1, α2, γ1, γ2)^2 + 4.0 * commute(α1, α2, β1, β2) * commute(γ1, γ2, β1, β2))
    b = 2.0 * (commute(α1, α2, γ1, γ2) * commute(λ1, λ2, γ1, γ2) + 2.0 * commute(γ1, γ2, β1, β2) * commute(λ1, λ2, β1, β2))
    c = commute(λ1, λ2, γ1, γ2)^2

    # prograde
    if sign_Lz>0
        # prograge energy (Eq. E12) - retrograde is other root
        E = sqrt((-b - sqrt(b^2 - 4.0aa * c))/ 2.0aa)
        # prograde z-component of angular momentum (Eq. E14) - retrograde is negative root
        L = sqrt((commute(α1, α2, β1, β2) * E^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))
    else
        # retrograde
        E = sqrt((-b + sqrt(b^2 - 4.0aa * c))/ 2.0aa)
        L = -sqrt((commute(α1, α2, β1, β2) * E^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))
    end


    if θmin==0.0
        C = 0.0
    else
        C = zm * (L^2 / (1.0 - zm) + a^2 * (1.0 - E^2))    # Eq. E2
    end
    
    return E, L, C
end

function compute_iota(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64)
    E, L, C = compute_ELQ(a, p, e, θmin, sign_Lz)
    return acos(L / sqrt(L^2 + C))
end

# compute iota corresponding to θmin
function iota_to_theta_min(a::Float64, p::Float64, e::Float64, ι::Float64)
    if ι < 0. || ι > π
        throw(DomainError("ι must be in the range [0, π]"))
    else
        sign_Lz = ι < π/2 ? +1 : -1;
        iota(θmin::Float64) = compute_iota(a, p, e, θmin, sign_Lz) - ι
        θmin = find_zeros(iota, 0.001, π/2-0.001)
    end
    return length(θmin) == 1 ? θmin[1] : throw(DomainError())
end

function compute_ωθ_over_ωr(a::Float64, p::Float64, e::Float64, ι::Float64)
    M=1.0;

    sign_Lz = ι < π/2 ? +1 : -1;

    θmin = iota_to_theta_min(a, p, e, ι)

    ### COMPUTE ELQ ###
    zm = cos(θmin)^2
    if e==0.0
        r0 = p * M
        α1 = αI(a, M, r0, zm)
        α2 = α_2(a, M, r0, zm)
        β1 = βI(a, M, r0, zm)
        β2 = β_2(a, M, r0, zm)
        γ1 = γI(a, M, r0, zm)
        γ2 = γ_2(a, M, r0, zm)
        λ1 = λI(a, M, r0, zm)
        λ2 = λ_2(a, M, r0, zm)
    else
        rp = p * M / (1 + e)
        ra = p * M / (1 - e)
        α1 = αI(a, M, ra, zm)
        α2 = αI(a, M, rp, zm)
        β1 = βI(a, M, ra, zm)
        β2 = βI(a, M, rp, zm)
        γ1 = γI(a, M, ra, zm)
        γ2 = γI(a, M, rp, zm)
        λ1 = λI(a, M, ra, zm)
        λ2 = λI(a, M, rp, zm)
    end
    
    # write out coefficients of Eq. E12 in the form ax^2 + bx + c
    aa = (commute(α1, α2, γ1, γ2)^2 + 4.0 * commute(α1, α2, β1, β2) * commute(γ1, γ2, β1, β2))
    b = 2.0 * (commute(α1, α2, γ1, γ2) * commute(λ1, λ2, γ1, γ2) + 2.0 * commute(γ1, γ2, β1, β2) * commute(λ1, λ2, β1, β2))
    c = commute(λ1, λ2, γ1, γ2)^2

    # prograde
    if sign_Lz>0
        # prograge energy (Eq. E12) - retrograde is other root
        E = sqrt((-b - sqrt(b^2 - 4.0aa * c))/ 2.0aa)
        # prograde z-component of angular momentum (Eq. E14) - retrograde is negative root
        L = sqrt((commute(α1, α2, β1, β2) * E^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))
    else
        # retrograde
        E = sqrt((-b + sqrt(b^2 - 4.0aa * c))/ 2.0aa)
        L = -sqrt((commute(α1, α2, β1, β2) * E^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))
    end

    if θmin==0.0
        C = 0.0
    else
        C = zm * (L^2 / (1.0 - zm) + a^2 * (1.0 - E^2))    # Eq. E2
    end
    
    ### COMPUTE FREQUENCIES ###
    zp = C / (a^2 * (1.0-E^2) * zm)    # Eq. E23
    ra=p * M / (1.0 - e); rp=p * M / (1.0 + e);
    A = M / (1.0 - E^2) - (ra + rp) / 2.0    # Eq. E20
    B = a^2 * C / ((1.0 - E^2) * ra * rp)    # Eq. E21
    r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19

    kr = sqrt((ra-rp) * (r3-r4) / ((ra-r3) * (rp-r4)))    # Eq. F5
    kθ = sqrt(zm/zp)    # Eq. F5
    
    K_kr = Elliptic.K(kr^2)
    K_kθ = Elliptic.K(kθ^2)

    γr = π * sqrt((1.0-E^2) * (ra-r3) * (rp-r4)) / (2.0K_kr)    # Eq. F3
    γθ = π * a * sqrt((1.0-E^2)*zp)/(2.0K_kθ)    # Eq. F4

    return γθ / γr
end

function compute_semi_latus(a::Float64, e::Float64, ι::Float64, resonance_value::Float64)
    #### COMPUTING RESONANCES ####
    T = gsl_root_fsolver_bisection
    solver = root_fsolver_alloc(T)
    # define lower p_value from the periastron value at the LSO
    p_min = 5.0; 
    p_max = 16.0;

    @eval compute_p(p) = compute_ωθ_over_ωr($a, p, $e, $ι) - $resonance_value

    # Wrapping structs
    f = @gsl_function(compute_p)

    root_solver_set = false
    while !root_solver_set
        try
            root_fsolver_set(solver, f, p_min, p_max)
            root_solver_set = true
        catch error
            if isa(error, DomainError)
                p_min = p_min + 0.1
                continue  # Skip to the next parameter set
            else
                rethrow(error)
            end
        end
        if p_min > 0.95 * p_max
            root_solver_set = true
            throw(DomainError("Increase p_max"))
        end
    end

    status = GSL_CONTINUE
    maxiter = 3000
    iter = 0
    while status == GSL_CONTINUE
        root_fsolver_iterate(solver)
        x = root_fsolver_root(solver)
        status = root_test_residual(Base.invokelatest(compute_p, x), 1e-10)
        iter += 1
        if iter==maxiter
            error("No convergence")
        end
    end

    p = root_fsolver_root(solver)

    root_fsolver_free(solver)

    # final check #
    if !isapprox(compute_ωθ_over_ωr(a, p, e, ι), resonance_value)
        throw(DomainError("Value of semi-latus rectum doesn't satisfy resonance condition"))
    end

    return p
end

end

module Evolution
using ...Kerr, ...FiniteDiff_5, ..Location, ...SemiRelativisticKludge, ...InspiralEvolution
### COMPUTE RESONANCE DURATION ###
t_res(l::Int64, m::Int64, ωr0_dot::Float64, ωθ0_dot::Float64) = sqrt(4π / abs(l * ωr0_dot + m * ωθ0_dot))

### THRESHOLD FUNCTION - EQ. 12 in arXiv:2103.06306 ###
ξstar(t_res::Float64, ωr0::Float64) = π / (t_res * ωr0)

function compute_resonance_time(a::Float64, e::Float64, ι::Float64, l::Int64, m::Int64, q::Float64, nPointsGeodesic::Int64, kerrReltol::Float64, kerrAbstol::Float64,
    inspiral_type::String; t_range_factor::Float64=NaN, nPointsFit::Float64=NaN, nHarm::Float64=NaN, h::Float64=NaN, data_path="Results/")
    MBH_geometric_mass = 1.; SCO_geometric_mass = q;
    rplus = Kerr.KerrMetric.rplus(a, MBH_geometric_mass); rminus = Kerr.KerrMetric.rminus(a, MBH_geometric_mass);

    # compute parameter space location of the EXACT resonance
    exact_resonance_value = l / m;
    p_0 = Location.compute_semi_latus(a, e, ι, exact_resonance_value)   # semi-latus rectum
    θmin_0 = Location.iota_to_theta_min(a, p_0, e, ι)

    # calculate integrals of motion at resonance
    EE_0, LL_0, QQ_0, CC_0, iota_0 = Kerr.ConstantsOfMotion.ELQ(a, p_0, e, θmin_0, MBH_geometric_mass)

    # frequencies at resonance
    ω0 = Kerr.ConstantsOfMotion.KerrFreqs(a, p_0, e, θmin_0, EE_0, LL_0, QQ_0, CC_0, rplus, rminus, MBH_geometric_mass);
    Ωr0 = ω0[1] / ω0[4]; Ωθ0 = ω0[2] / ω0[4];
    t_max_M = sqrt(1/q) * 2π / (maximum([Ωr0, Ωθ0]))

    # compute resonance ratio at start of inspiral run
    resonance_value = exact_resonance_value - ξstar(t_max_M, Ωr0)
    println("exact_resonance_value = $(exact_resonance_value)")
    println("resonance_value = $(resonance_value)")

    # compute parameter space location at inspiral start
    p = Location.compute_semi_latus(a, e, ι, resonance_value)   # semi-latus rectum
    θmin = Location.iota_to_theta_min(a, p, e, ι)

    # Kerr covariant metric components
    g_tt(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_tt(t, r, θ, ϕ, a, M);
    g_tϕ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_tϕ(t, r, θ, ϕ, a, M);
    g_rr(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_rr(t, r, θ, ϕ, a, M);
    g_θθ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_θθ(t, r, θ, ϕ, a, M);
    g_ϕϕ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_ϕϕ(t, r, θ, ϕ, a, M);
    g_μν(t, r, θ, ϕ, a, M, μ, ν) = Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, μ, ν); 

    # contravariant metric components
    gTT(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gTT(t, r, θ, ϕ, a, M);
    gTΦ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gTΦ(t, r, θ, ϕ, a, M);
    gRR(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gRR(t, r, θ, ϕ, a, M);
    gThTh(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gThTh(t, r, θ, ϕ, a, M);
    gΦΦ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gΦΦ(t, r, θ, ϕ, a, M);
    ginv(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.ginv(t, r, θ, ϕ, a, M);

    ## compute frequency of flux computation to match Chimera ##
    EEi, LLi, QQi, CCi, ιi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θmin, MBH_geometric_mass);
    ωi = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θmin, EEi, LLi, QQi, CCi, rplus, rminus, MBH_geometric_mass);    # Mino time frequencies
    Ωi = ωi[1:3]/ωi[4];  # BL time frequencies
    compute_SF_frac = 0.05;
    compute_fluxes_BL = compute_SF_frac * minimum(@. 2π /Ωi);   # in units of M (not seconds)
    compute_fluxes_Mino = compute_SF_frac * minimum(@. 2π /ωi[1:3]);   # in units of M (not seconds)

    ## Float64 of points in Finite Difference geodesic ##
    if isequal(h, NaN)
        nothing
    else
        nPointsFDM = Int(floor(compute_fluxes_Mino / h));
    end

    mkpath(data_path)

    if isequal(inspiral_type, "NK")
        # run inspiral
        SemiRelativisticKludge.Inspiral.compute_inspiral_HJE!(a, p, e, ι,  q, t_max_M, compute_fluxes_BL, nPointsGeodesic, kerrReltol, kerrAbstol; data_path=data_path)
        # load constants of motion and fluxes
        t_Fluxes, EE, Edot, LL, Ldot, QQ, CC, Cdot, pArray, ecc, ι, θmin = SemiRelativisticKludge.Inspiral.load_constants_of_motion(a, p, e, ι, q, nPointsGeodesic, kerrReltol, data_path)
        compute_fluxes = compute_fluxes_BL
    elseif isequal(inspiral_type, "Chimera_BL_FF")
        # run inspiral
        InspiralEvolution.FourierFit.BLTime.compute_inspiral_HJE!(t_max_M, compute_fluxes_BL, t_range_factor, nPointsGeodesic, nPointsFit, MBH_geometric_mass, SCO_geometric_mass,
        a, p, e, θmin, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, nHarm, kerrReltol, kerrAbstol; data_path=data_path)
        # load constants of motion and fluxes
        t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin = InspiralEvolution.FourierFit.BLTime.load_constants_of_motion(a, p, e, θmin, q, nHarm, nPointsFit, kerrReltol, t_range_factor_BL, data_path);
        ι = @. acos(LL / sqrt(LL^2 + CC));
        # delete saved files
        InspiralEvolution.FourierFit.BLTime.delete_EMRI_data(a, p, e, θmin, q, nHarm, nPointsFit, kerrReltol, t_range_factor, data_path)

    elseif isequal(inspiral_type, "Chimera_Mino_FF")
        # run inspiral
        InspiralEvolution.FourierFit.MinoTime.compute_inspiral_HJE!(t_max_M, compute_fluxes_Mino, t_range_factor, nPointsGeodesic, nPointsFit, MBH_geometric_mass, SCO_geometric_mass,
        a, p, e, θmin, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, nHarm, kerrReltol, kerrAbstol; data_path=data_path)
        # load constants of motion and fluxes
        t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin = InspiralEvolution.FourierFit.MinoTime.load_constants_of_motion(a, p, e, θmin, q, nHarm, nPointsFit, kerrReltol, t_range_factor_Mino_FF, data_path);
        ι = @. acos(LL / sqrt(LL^2 + CC));
        # delete saved files
        InspiralEvolution.FourierFit.MinoTime.delete_EMRI_data(a, p, e, θmin, q, nHarm, nPointsFit, kerrReltol, t_range_factor, data_path)
    
    elseif isequal(inspiral_type, "Chimera_Mino_FDM")
        # run insprial
        InspiralEvolution.FiniteDifferences.MinoTime.compute_inspiral_HJE!(t_max_M, nPointsFDM, MBH_geometric_mass, SCO_geometric_mass,
        a, p, e, θmin, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, h, kerrReltol, kerrAbstol; data_path=data_path)
        # load constants of motion and fluxes
        t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin = 
        InspiralEvolution.FiniteDifferences.MinoTime.load_constants_of_motion(a, p, e, θmin, q, h, kerrReltol, data_path);
        ι = @. acos(LL / sqrt(LL^2 + CC));
        # delete saved files
        InspiralEvolution.FiniteDifferences.MinoTime.delete_EMRI_data(a, p, e, θmin, q, h, kerrReltol, data_path)
    else
        throw(DomainError("The options for the argument 'inspiral_type' are 'NK', 'Chimera_BL_FF', 'Chimera_Mino_FF', and 'Chimera_Mino_FDM'"))
    end

    ### INITIALIZE ARRAYS FOR FUNDAMENTAL FREQUENCIES, THEIR TIME DERIVATIVE, AND THEIR RATIO ###
    ωr = Float64[]; ωθ = Float64[];
    ωr_dot = Float64[]; ωθ_dot = Float64[];
    ωθ_over_ωr = Float64[];


    ### COMPUTE FUNDAMENTAL FREQUENCIES AS A FUNCTION OF TIME OVER THE EVOLUTION ###
    @inbounds for i in eachindex(pArray)
        ω = Kerr.ConstantsOfMotion.KerrFreqs(a, pArray[i], ecc[i], θmin[i], EE[i], LL[i], QQ[i], CC[i], rplus, rminus, MBH_geometric_mass);    # Mino time frequencies
        append!(ωr, ω[1]/ω[4]); append!(ωθ, ω[2]/ω[4]); append!(ωθ_over_ωr, ωθ[i]/ωr[i]);
    end

    ### COMPUTE TIME DERIVATIVES ###
    @inbounds for i in eachindex(ωr)
        append!(ωr_dot, FiniteDiff_5.compute_first_derivative(i,  ωr, compute_fluxes, length(ωr))); 
        append!(ωθ_dot, FiniteDiff_5.compute_first_derivative(i,  ωθ, compute_fluxes, length(ωθ))); 
    end

    ### EXTRACT FREQUENCIES AT RESONANCE ###
    resonance_index = argmin(abs.(ωθ_over_ωr.-l/m))
    println("min(ωθ/ωr - 1.5) = $(ωθ_over_ωr[resonance_index])")
    ωr0_dot = ωr_dot[resonance_index]; ωθ0_dot = ωθ_dot[resonance_index]; 

    ### COMPUTE RESONANCE DURATION ###
    t_res = Evolution.t_res(l, m, ωr0_dot, ωθ0_dot)
    return t_res
end

end 
end