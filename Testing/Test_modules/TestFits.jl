#=

In this module we write several functions which we repeatedly use in analysing the performance of various fitting techniques

=#

module TestFitsBLTime
using DelimitedFiles, Peaks, ..BLTimeEvolution, ..FourierFitGSL, ..FourierFitGSL_Derivs, JLD2, FileIO
using ..Deriv2, ..Deriv3, ..Deriv4, ..Deriv5, ..Deriv6, ..FourierFunctions, ..Kerr

# returns elements-wise percentage deviation between two arrays
function compute_deviation(y_true::Vector{Float64}, y_approx::Vector{Float64})
    return @. abs(100 * (y_true-y_approx)/y_true)
end

# compute fundamental frequencies
function compute_frequencies(a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64)
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M)
    return ω
end

# compute fit to Boyer-Lindquist r
function compute_fit(f::Function, df_dt::Function, d2f_dt::Function, d3f_dt::Function, d4f_dt::Function, d5f_dt::Function, d6f_dt::Function,
    t_range_factor::Float64, nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, kerrReltol::Float64, kerrAbstol::Float64, 
    r_frequency::Bool, θ_frequency::Bool, ϕ_frequency::Bool, data_path::String)
    println("nHarm = $(nHarm), nPoints = $(nPoints)")

    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);
    Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];
    Ωr = r_frequency ? Ωr : 1e10; Ωθ = θ_frequency ? Ωθ : 1e10; Ωϕ = ϕ_frequency ? Ωϕ : 1e10; 
    Ω = [Ωr, Ωθ, Ωϕ]

    tmax =  t_range_factor * minimum(@. 2π/Ω[Ω .< 1e9]); saveat = tmax / (nPoints-1); Δti=saveat;

    ##### compute geodesic for reltol=1e-12 #####
    BLTimeEvolution.compute_kerr_geodesic(a, p, e, θi, nPoints, tmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; 
    r_dot=sol[5,:]; θ_dot=sol[6,:]; ϕ_dot=sol[7,:]; r_ddot=sol[8,:]; θ_ddot=sol[9,:]; ϕ_ddot=sol[10,:]; dt_dτ=sol[11,:];

    ##### compute test function values and its derivatives #####
    test_func_data_0 = zeros(size(t, 1))
    test_func_data_1 = zeros(size(t, 1))
    test_func_data_2 = zeros(size(t, 1))
    test_func_data_3 = zeros(size(t, 1))
    test_func_data_4 = zeros(size(t, 1))
    test_func_data_5 = zeros(size(t, 1))
    test_func_data_6 = zeros(size(t, 1))
    x = [Float64[] for i in 1:size(t, 1)]
    dx = [Float64[] for i in 1:size(t, 1)]
    d2x = [Float64[] for i in 1:size(t, 1)]
    d3x = [Float64[] for i in 1:size(t, 1)]
    d4x = [Float64[] for i in 1:size(t, 1)]
    d5x = [Float64[] for i in 1:size(t, 1)]
    d6x = [Float64[] for i in 1:size(t, 1)]
    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        dx[i] = Vector{Float64}([r_dot[i], θ_dot[i], ϕ_dot[i]]);
        d2x[i] = Vector{Float64}([r_ddot[i], θ_ddot[i], ϕ_ddot[i]]);
    end

    @inbounds for i in eachindex(t)
        d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
        d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
        d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        test_func_data_0[i] = f(x[i])
        test_func_data_1[i] = df_dt(dx[i], x[i]) 
        test_func_data_2[i] = d2f_dt(d2x[i], dx[i], x[i]) 
        test_func_data_3[i] = d3f_dt(d3x[i], d2x[i], dx[i], x[i])
        test_func_data_4[i] = d4f_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i]) 
        test_func_data_5[i] = d5f_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
        test_func_data_6[i] = d6f_dt(d6x[i], d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
    end

    # test_func_data = [test_func_data_0, test_func_data_1, test_func_data_2, test_func_data_3,
    # test_func_data_4, test_func_data_5, test_func_data_6];

    test_func_data = [test_func_data_0, test_func_data_2,
    test_func_data_4, test_func_data_6];

    ##### perform fit #####
    xdata = t; ydata=test_func_data_0; 
    fit_fname_save=data_path * "Test_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"

    n_freqs = FourierFitGSL.compute_num_fitting_freqs_master(nHarm, Ω); chisq=[0.0];
    fit_params = zeros(2 * n_freqs + 1);

    if length(t) == nPoints
        Ω_fit = FourierFitGSL.GSL_fit_master!(xdata, ydata, nPoints, nHarm, chisq, Ω, fit_params)
    else
        println("nPoints = $(nPoints), length(t) = $(length(t))")
        throw(BoundsError)
    end

    # compute power spectrum
    power = zeros(nFreqs)
    for i=1:nFreqs
        power[i] = sqrt(fit_params[i+1]^2 + fit_params[i+1+nFreqs]^2)
    end

    # Creating a Typed Dictionary 
    fit_dictionary = Dict{String, AbstractArray}("xdata" => xdata, "test_func_data" => test_func_data, "fit_params" => fit_params, "fit_freqs" => Ω_fit, "power" => power) 
    # save fit #
    println(fit_fname_save)
    save(fit_fname_save, "data", fit_dictionary)
end


##### define function to carry out fit and store fit parameters #####
function compute_fit_FFT(f::Function, df_dt::Function, d2f_dt::Function, d3f_dt::Function, d4f_dt::Function, d5f_dt::Function, d6f_dt::Function,
    t_range_factor::Float64, nHarm::Int64, nFFTFreqs::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, kerrReltol::Float64, kerrAbstol::Float64, 
    r_frequency::Bool, θ_frequency::Bool, ϕ_frequency::Bool, data_path::String)
    println("nHarm = $(nHarm), nPoints = $(nPoints)")

    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);
    Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];

    # include desired frequencies
    Ωr = r_frequency ? Ωr : 1e10; Ωθ = θ_frequency ? Ωθ : 1e10; Ωϕ = ϕ_frequency ? Ωϕ : 1e10; 
    Ω = [Ωr, Ωθ, Ωϕ]

    tmax =  t_range_factor * minimum(@. 2π/Ω[Ω .< 1e9]); saveat = tmax / (nPoints-1); Δti=saveat;

    # extract FFT frequencies
    t_range_factor_FFT = 100.0; nPointsFFT=50000; t_max_FFT = t_range_factor_FFT * minimum(@. 2π/Ω[Ω .< 1e9]);
    saveat_FFT = t_max_FFT / (nPointsFFT-1); Δti_FFT=saveat_FFT;
    F, freqs = TestFitsBLTime.extract_fourier_transform(f, a, p, e, θi, t_max_FFT, Δti_FFT, saveat_FFT, nPointsFFT, kerrReltol, kerrAbstol, data_path)

    nPeaks = -1    # value -1 is a flag to include all peak frequencies
    exclude_zero=true    # exclude zero frequency
    peak_freqs, peak_F_vals = FourierFunctions.extract_frequencies!(F, freqs, nPeaks, exclude_zero)

    # compute geodesic
    tmax =  t_range_factor * minimum(@. 2π/Ω[Ω .< 1e9]); saveat = tmax / (nPoints-1); Δti=saveat;

    ##### compute geodesic for reltol=1e-12 #####
    BLTimeEvolution.compute_kerr_geodesic(a, p, e, θi, nPoints, tmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; 
    r_dot=sol[5,:]; θ_dot=sol[6,:]; ϕ_dot=sol[7,:]; r_ddot=sol[8,:]; θ_ddot=sol[9,:]; ϕ_ddot=sol[10,:]; dt_dτ=sol[11,:];

    ##### compute test function values and its derivatives #####
    test_func_data_0 = zeros(size(t, 1))
    test_func_data_1 = zeros(size(t, 1))
    test_func_data_2 = zeros(size(t, 1))
    test_func_data_3 = zeros(size(t, 1))
    test_func_data_4 = zeros(size(t, 1))
    test_func_data_5 = zeros(size(t, 1))
    test_func_data_6 = zeros(size(t, 1))
    x = [Float64[] for i in 1:size(t, 1)]
    dx = [Float64[] for i in 1:size(t, 1)]
    d2x = [Float64[] for i in 1:size(t, 1)]
    d3x = [Float64[] for i in 1:size(t, 1)]
    d4x = [Float64[] for i in 1:size(t, 1)]
    d5x = [Float64[] for i in 1:size(t, 1)]
    d6x = [Float64[] for i in 1:size(t, 1)]
    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        dx[i] = Vector{Float64}([r_dot[i], θ_dot[i], ϕ_dot[i]]);
        d2x[i] = Vector{Float64}([r_ddot[i], θ_ddot[i], ϕ_ddot[i]]);
    end

    @inbounds for i in eachindex(t)
        d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
        d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
        d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        test_func_data_0[i] = f(x[i])
        test_func_data_1[i] = df_dt(dx[i], x[i]) 
        test_func_data_2[i] = d2f_dt(d2x[i], dx[i], x[i]) 
        test_func_data_3[i] = d3f_dt(d3x[i], d2x[i], dx[i], x[i])
        test_func_data_4[i] = d4f_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i]) 
        test_func_data_5[i] = d5f_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
        test_func_data_6[i] = d6f_dt(d6x[i], d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
    end


    # test_func_data = [test_func_data_0, test_func_data_1, test_func_data_2, test_func_data_3,
    # test_func_data_4, test_func_data_5, test_func_data_6];
    test_func_data = [test_func_data_0, test_func_data_2,
    test_func_data_4, test_func_data_6];

    ##### perform fit #####
    # println("Length(t) = $(length(t))")
    fit_fname_save=data_path * "Test_FFT_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    xdata = t; ydata=test_func_data_0;  chisq=[0.0];
    
    # frequencies for r are just Ωr, Ωθ
    orbital_freqs = FourierFitGSL.compute_fitting_frequencies_master(nHarm, Ω)
    nFreqs = FourierFitGSL.compute_num_fitting_freqs_master(nHarm, Ω)

    fit_params = zeros(2 * nFreqs + 1);
    Ω_fit, FFT_freqs = FourierFunctions.GSL_fit!(xdata, ydata, peak_freqs, orbital_freqs, nPoints, nFreqs, nFFTFreqs, chisq, fit_params) 

    # compute power spectrum
    power = zeros(nFreqs)
    for i=1:nFreqs
        power[i] = sqrt(fit_params[i+1]^2 + fit_params[i+1+nFreqs]^2)
    end

    # Creating a Typed Dictionary 
    fit_dictionary = Dict{String, AbstractArray}("xdata" => xdata, "test_func_data" => test_func_data, "fit_params" => fit_params, "fit_freqs" => Ω_fit, "power" => power) 
    # save fit #
    println(fit_fname_save)
    save(fit_fname_save, "data", fit_dictionary)
    return Ω_fit, F, freqs, peak_freqs, peak_F_vals, FFT_freqs, power
end

# load fourier fit and power spectrum
function load_fit(a::Float64, p::Float64, e::Float64, θi::Float64, nHarm::Int64, nPoints::Int64, data_path::String, FFT_flag::Bool)
    if FFT_flag == false
        fit_fname_save=data_path * "Test_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    else
        fit_fname_save=data_path * "Test_FFT_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    end
    return load(fit_fname_save)["data"]
end


# function to compute the derivative from the fit
function compute_derivatives(nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, data_path::String, FFT_flag::Bool)
    saved_data = load_fit(a, p, e, θi, nHarm, nPoints, data_path, FFT_flag)
    xdata = saved_data["xdata"]
    test_func_data = saved_data["test_func_data"]
    fit_params = saved_data["fit_params"]
    Ω_fit = saved_data["fit_freqs"]
    n_freqs = length(Ω_fit)
    return xdata, [FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, N) for N in [0, 2, 4, 6]], test_func_data
end


# function to extract fourier frequencies from geodesic data
function extract_fourier_transform(f::Function, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, t_range_factor_FFT::Float64, nPointsFFT::Int64, kerrReltol::Float64, kerrAbstol::Float64, 
    r_frequency::Bool, θ_frequency::Bool, ϕ_frequency::Bool, data_path::String)
    
    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);
    Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];

    # include desired frequencies
    Ωr = r_frequency ? Ωr : 1e10; Ωθ = θ_frequency ? Ωθ : 1e10; Ωϕ = ϕ_frequency ? Ωϕ : 1e10; 
    Ω = [Ωr, Ωθ, Ωϕ]

    t_max_FFT = t_range_factor_FFT * minimum(@. 2π/Ω[Ω .< 1e9]); saveat_FFT = t_max_FFT / (nPointsFFT-1); Δti=saveat_FFT;

    ##### compute geodesic for reltol=1e-12 #####
    BLTimeEvolution.compute_kerr_geodesic(a, p, e, θi, nPointsFFT, t_max_FFT, Δti, kerrReltol, kerrAbstol, saveat_FFT, data_path=data_path)

    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat_FFT)_T_$(t_max_FFT)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:];

    x = [Float64[] for i in 1:size(t, 1)]    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
    end

    # fourier transform of r #
    fs = nPointsFFT / (last(t) - t[1])

    signal = f.(x)
    F, freqs = FourierFunctions.compute_real_FFT(signal, fs, nPointsFFT)
    return F, freqs
end

# function to extract fourier frequencies from geodesic data
function extract_fourier_transform(f::Function, a::Float64, p::Float64, e::Float64, θi::Float64, tmax::Float64, Δti::Float64, saveat::Float64, nPointsFFT::Int64, kerrReltol::Float64, kerrAbstol::Float64, data_path::String)
    ##### compute geodesic for reltol=1e-12 #####
    BLTimeEvolution.compute_kerr_geodesic(a, p, e, θi, nPointsFFT, tmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:];

    x = [Float64[] for i in 1:size(t, 1)]    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
    end

    # fourier transform of r #
    fs = nPointsFFT / (last(t) - t[1])

    signal = f.(x)
    F, freqs = FourierFunctions.compute_real_FFT(signal, fs, nPointsFFT)
    return F, freqs
end

end

module TestFitsMinoTime
using DelimitedFiles, Peaks, ..MinoTimeEvolution, ..FourierFitGSL, JLD2, FileIO, ..MinoTimeDerivs, ..ParameterizedDerivs
using ..Deriv2, ..Deriv3, ..Deriv4, ..Deriv5, ..Deriv6, ..FourierFunctions, ..Kerr, ..TestFitsBLTime

##### define function to carry out fit and store fit parameters #####
function compute_fit(f::Function, df_dt::Function, d2f_dt::Function, d3f_dt::Function, d4f_dt::Function, d5f_dt::Function, d6f_dt::Function, t_range_factor::Float64,
    nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, kerrReltol::Float64, kerrAbstol::Float64,
    r_frequency::Bool, θ_frequency::Bool, ϕ_frequency::Bool, data_path::String)
    println("nHarm = $(nHarm), nPoints = $(nPoints)")

    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);
    ωr, ωθ, ωϕ = ω[1:3];
    ωr = r_frequency ? ωr : 1e10; ωθ = θ_frequency ? ωθ : 1e10; ωϕ = ϕ_frequency ? ωϕ : 1e10; 
    ω = [ωr, ωθ, ωϕ]

    λmax =  t_range_factor * minimum(@. 2π/ω[ω .< 1e9]); saveat = λmax / (nPoints-1); Δλi=saveat;

    ##### compute geodesic for reltol=1e-12 #####
    @time MinoTimeEvolution.compute_kerr_geodesic(a, p, e, θi, λmax, Δλi, kerrReltol, kerrAbstol, saveat, data_path=data_path)
    
    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "Mino_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(λmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    λ=sol[1,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; 
    dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt=sol[9,:]; d2θ_dt=sol[10,:]; d2ϕ_dt=sol[11,:]; dt_dλ=sol[12,:];

    ##### compute test function values and its derivatives #####
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    test_func_data_0 = zeros(size(λ, 1))
    test_func_data_1 = zeros(size(λ, 1))
    test_func_data_2 = zeros(size(λ, 1))
    test_func_data_3 = zeros(size(λ, 1))
    test_func_data_4 = zeros(size(λ, 1))
    test_func_data_5 = zeros(size(λ, 1))
    test_func_data_6 = zeros(size(λ, 1))
    x = [Float64[] for i in 1:size(λ, 1)]
    dx = [Float64[] for i in 1:size(λ, 1)]
    d2x = [Float64[] for i in 1:size(λ, 1)]
    d3x = [Float64[] for i in 1:size(λ, 1)]
    d4x = [Float64[] for i in 1:size(λ, 1)]
    d5x = [Float64[] for i in 1:size(λ, 1)]
    d6x = [Float64[] for i in 1:size(λ, 1)]
    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(λ)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        dx[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
        d2x[i] = Vector{Float64}([d2r_dt[i], d2θ_dt[i], d2ϕ_dt[i]]);
    end

    @inbounds for i in eachindex(λ)
        d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
        d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
        d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        test_func_data_0[i] = f(x[i])
        test_func_data_1[i] = df_dt(dx[i], x[i]) 
        test_func_data_2[i] = d2f_dt(d2x[i], dx[i], x[i]) 
        test_func_data_3[i] = d3f_dt(d3x[i], d2x[i], dx[i], x[i])
        test_func_data_4[i] = d4f_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i]) 
        test_func_data_5[i] = d5f_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
        test_func_data_6[i] = d6f_dt(d6x[i], d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
    end

    # test_func_data = [test_func_data_0, test_func_data_1, test_func_data_2, test_func_data_3,
    # test_func_data_4, test_func_data_5, test_func_data_6];

    test_func_data = [test_func_data_0, test_func_data_2, test_func_data_4, test_func_data_6];

    ##### perform fit #####
    # println("Length(λ) = $(length(λ))")
    xdata = λ; ydata=test_func_data_0; 
    fit_fname_save=data_path * "Test_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"

    n_freqs = FourierFitGSL.compute_num_fitting_freqs_master(nHarm, ω); chisq=[0.0];
    fit_params = zeros(2 * n_freqs + 1);

    if length(λ) == nPoints
        Ω_fit = FourierFitGSL.GSL_fit_master!(xdata, ydata, nPoints, nHarm, chisq,  ω, fit_params)
    else
        println("nPoints = $(nPoints), length(λ) = $(length(λ))")
        throw(BoundsError)
    end

    # compute power spectrum
    power = zeros(n_freqs)
    for i=1:n_freqs
        power[i] = sqrt(fit_params[i+1]^2 + fit_params[i+1+n_freqs]^2)
    end

    # Creating a Typed Dictionary 
    fit_dictionary = Dict("ODE_fname" => kerr_ode_sol_fname, "xdata" => xdata, "test_func_data" => test_func_data, "fit_params" => fit_params, "fit_freqs" => Ω_fit, "power" => power) 
    # save fit #
    println(fit_fname_save)
    save(fit_fname_save, "data", fit_dictionary)
end


##### define function to carry out fit and store fit parameters #####
function compute_fit_FFT(f::Function, df_dt::Function, d2f_dt::Function, d3f_dt::Function, d4f_dt::Function, d5f_dt::Function, d6f_dt::Function,
    t_range_factor::Float64, nHarm::Int64, nFFTFreqs::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, kerrReltol::Float64, kerrAbstol::Float64, 
    r_frequency::Bool, θ_frequency::Bool, ϕ_frequency::Bool, data_path::String)
    println("nHarm = $(nHarm), nPoints = $(nPoints)")

    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);
    ωr, ωθ, ωϕ = ω[1:3];
    ωr = r_frequency ? ωr : 1e10; ωθ = θ_frequency ? ωθ : 1e10; ωϕ = ϕ_frequency ? ωϕ : 1e10; 
    ω = [ωr, ωθ, ωϕ]

    # extract FFT frequencies
    t_range_factor_FFT = 100.0; nPointsFFT=50000; λ_max_FFT = t_range_factor_FFT * minimum(@. 2π/ω[ω .< 1e9]);
    saveat_FFT = λ_max_FFT / (nPointsFFT-1); Δλi_FFT=saveat_FFT;
    F, freqs = TestFitsMinoTime.extract_fourier_transform(f, a, p, e, θi, λ_max_FFT, Δλi_FFT, saveat_FFT, nPointsFFT, kerrReltol, kerrAbstol, data_path)

    nPeaks = -1    # value -1 is a flag to include all peak frequencies
    exclude_zero=true    # exclude zero frequency
    peak_freqs, peak_F_vals = FourierFunctions.extract_frequencies!(F, freqs, nPeaks, exclude_zero)

    # compute geodesic
    λmax =  t_range_factor * minimum(@. 2π/ω[ω .< 1e9]); saveat = λmax / (nPoints-1); Δλi=saveat;

    ##### compute geodesic for reltol=1e-12 #####
    @time MinoTimeEvolution.compute_kerr_geodesic(a, p, e, θi, λmax, Δλi, kerrReltol, kerrAbstol, saveat, data_path=data_path)
    
    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "Mino_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(λmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    λ=sol[1,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; 
    dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt=sol[9,:]; d2θ_dt=sol[10,:]; d2ϕ_dt=sol[11,:]; dt_dλ=sol[12,:];

    ##### compute test function values and its derivatives #####
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    test_func_data_0 = zeros(size(λ, 1))
    test_func_data_1 = zeros(size(λ, 1))
    test_func_data_2 = zeros(size(λ, 1))
    test_func_data_3 = zeros(size(λ, 1))
    test_func_data_4 = zeros(size(λ, 1))
    test_func_data_5 = zeros(size(λ, 1))
    test_func_data_6 = zeros(size(λ, 1))
    x = [Float64[] for i in 1:size(λ, 1)]
    dx = [Float64[] for i in 1:size(λ, 1)]
    d2x = [Float64[] for i in 1:size(λ, 1)]
    d3x = [Float64[] for i in 1:size(λ, 1)]
    d4x = [Float64[] for i in 1:size(λ, 1)]
    d5x = [Float64[] for i in 1:size(λ, 1)]
    d6x = [Float64[] for i in 1:size(λ, 1)]
    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(λ)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        dx[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
        d2x[i] = Vector{Float64}([d2r_dt[i], d2θ_dt[i], d2ϕ_dt[i]]);
    end

    @inbounds for i in eachindex(λ)
        d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
        d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
        d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        test_func_data_0[i] = f(x[i])
        test_func_data_1[i] = df_dt(dx[i], x[i]) 
        test_func_data_2[i] = d2f_dt(d2x[i], dx[i], x[i]) 
        test_func_data_3[i] = d3f_dt(d3x[i], d2x[i], dx[i], x[i])
        test_func_data_4[i] = d4f_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i]) 
        test_func_data_5[i] = d5f_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
        test_func_data_6[i] = d6f_dt(d6x[i], d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
    end

    # test_func_data = [test_func_data_0, test_func_data_1, test_func_data_2, test_func_data_3,
    # test_func_data_4, test_func_data_5, test_func_data_6];

    test_func_data = [test_func_data_0, test_func_data_2, test_func_data_4, test_func_data_6];

    ##### perform fit #####
    # println("Length(t) = $(length(t))")
    fit_fname_save=data_path * "Test_FFT_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    xdata = λ; ydata=test_func_data_0;  chisq=[0.0];
    
    # frequencies for r are just Ωr, Ωθ
    orbital_freqs = FourierFitGSL.compute_fitting_frequencies_master(nHarm, ω)
    nFreqs = FourierFitGSL.compute_num_fitting_freqs_master(nHarm, ω)

    fit_params = zeros(2 * nFreqs + 1);
    Ω_fit, FFT_freqs = FourierFunctions.GSL_fit!(xdata, ydata, peak_freqs, orbital_freqs, nPoints, nFreqs, nFFTFreqs, chisq, fit_params) 

    # compute power spectrum
    power = zeros(nFreqs)
    for i=1:nFreqs
        power[i] = sqrt(fit_params[i+1]^2 + fit_params[i+1+nFreqs]^2)
    end

    # Creating a Typed Dictionary 
    fit_dictionary = Dict("ODE_fname" => kerr_ode_sol_fname, "xdata" => xdata, "test_func_data" => test_func_data, "fit_params" => fit_params, "fit_freqs" => Ω_fit, "power" => power) 
    # save fit #
    println(fit_fname_save)
    save(fit_fname_save, "data", fit_dictionary)
    return Ω_fit, F, freqs, peak_freqs, peak_F_vals, FFT_freqs, power
end

# function to compute the derivative from the fit
function compute_derivatives(nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, data_path::String, FFT_flag::Bool)
    saved_data = TestFitsBLTime.load_fit(a, p, e, θi, nHarm, nPoints, data_path, FFT_flag)
    test_func_data = saved_data["test_func_data"]
    kerr_ode_sol_fname = saved_data["ODE_fname"]
    xdata = saved_data["xdata"]
    fit_params = saved_data["fit_params"]
    Ω_fit = saved_data["fit_freqs"]
    n_freqs = length(Ω_fit)

    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)

    # compute f^{(n)}(λ)
    fλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 0)
    df_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 1)
    d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 2)
    d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 3)
    d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 4)
    d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 5)
    d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 6)


    # compute λ^{(n)}(t)
    sol = readdlm(kerr_ode_sol_fname)
    r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; 
    dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt=sol[9,:]; d2θ_dt=sol[10,:]; d2ϕ_dt=sol[11,:];

    ## initialize arrays
    x = [Float64[] for i in 1:size(r, 1)]; dx = [Float64[] for i in 1:size(r, 1)];
    d2x = [Float64[] for i in 1:size(r, 1)]; d3x = [Float64[] for i in 1:size(r, 1)];
    d4x = [Float64[] for i in 1:size(r, 1)]; d5x = [Float64[] for i in 1:size(r, 1)];
    d6x = [Float64[] for i in 1:size(r, 1)];

    dλ_dt = zeros(length(r)); d2λ_dt = zeros(length(r)); d3λ_dt = zeros(length(r));
    d4λ_dt = zeros(length(r)); d5λ_dt = zeros(length(r)); d6λ_dt = zeros(length(r));
    df_dt = zeros(length(r)); d2f_dt = zeros(length(r)); d3f_dt = zeros(length(r));
    d4f_dt = zeros(length(r)); d5f_dt = zeros(length(r)); d6f_dt = zeros(length(r));

    @inbounds Threads.@threads for i in eachindex(r)
       x[i] = [r[i], θ[i], ϕ[i]]
       dx[i] = [dr_dt[i], dθ_dt[i], dϕ_dt[i]]
       d2x[i] = [d2r_dt[i], d2θ_dt[i], d2ϕ_dt[i]]
       d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
       d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
       d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
       d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]

       dλ_dt[i] = MinoTimeDerivs.dλ_dt(x[i], a, M, E, L)
       d2λ_dt[i] = MinoTimeDerivs.d2λ_dt(dx[i], x[i], a, M, E, L)
       d3λ_dt[i] = MinoTimeDerivs.d3λ_dt(d2x[i], dx[i], x[i], a, M, E, L)
       d4λ_dt[i] = MinoTimeDerivs.d4λ_dt(d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
       d5λ_dt[i] = MinoTimeDerivs.d5λ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
       d6λ_dt[i] = MinoTimeDerivs.d6λ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a, M, E, L)

       df_dt[i] = ParameterizedDerivs.df_dt(df_dλ[i], dλ_dt[i])
       d2f_dt[i] = ParameterizedDerivs.d2f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i])
       d3f_dt[i] = ParameterizedDerivs.d3f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i])
       d4f_dt[i] = ParameterizedDerivs.d4f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i])
       d5f_dt[i] = ParameterizedDerivs.d5f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i], d5f_dλ[i], d5λ_dt[i])
       d6f_dt[i] = ParameterizedDerivs.d6f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i], d5f_dλ[i], d5λ_dt[i], d6f_dλ[i], d6λ_dt[i])
    end

    return xdata, [fλ, d2f_dt, d4f_dt, d6f_dt], test_func_data
end


# function to extract fourier frequencies from geodesic data
function extract_fourier_transform(f::Function, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, t_range_factor_FFT::Float64, nPointsFFT::Int64, kerrReltol::Float64, kerrAbstol::Float64, 
    r_frequency::Bool, θ_frequency::Bool, ϕ_frequency::Bool, data_path::String)
    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);
    ωr, ωθ, ωϕ = ω[1:3];
    ωr = r_frequency ? ωr : 1e10; ωθ = θ_frequency ? ωθ : 1e10; ωϕ = ϕ_frequency ? ωϕ : 1e10; 
    ω = [ωr, ωθ, ωϕ]

    λmax =  t_range_factor_FFT * minimum(@. 2π/ω[ω .< 1e9]); saveat = λmax / (nPointsFFT-1); Δλi=saveat;

    ##### compute geodesic for reltol=1e-12 #####
    @time MinoTimeEvolution.compute_kerr_geodesic(a, p, e, θi, λmax, Δλi, kerrReltol, kerrAbstol, saveat, data_path=data_path)
    
    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "Mino_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(λmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    λ=sol[1,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:];

    x = [Float64[] for i in 1:size(λ, 1)]    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(λ)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
    end

    # fourier transform of r #
    fs = nPointsFFT / (last(λ) - λ[1])

    signal = f.(x)
    F, freqs = FourierFunctions.compute_real_FFT(signal, fs, nPointsFFT)
    return F, freqs
end

# function to extract fourier frequencies from geodesic data
function extract_fourier_transform(f::Function, a::Float64, p::Float64, e::Float64, θi::Float64, λmax::Float64, Δλi::Float64, saveat::Float64, nPointsFFT::Int64, kerrReltol::Float64, kerrAbstol::Float64, data_path::String)
    @time MinoTimeEvolution.compute_kerr_geodesic(a, p, e, θi, λmax, Δλi, kerrReltol, kerrAbstol, saveat, data_path=data_path)
    
    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "Mino_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(λmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    λ=sol[1,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:];

    x = [Float64[] for i in 1:size(λ, 1)]    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(λ)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
    end

    # fourier transform of r #
    fs = nPointsFFT / (last(λ) - λ[1])

    signal = f.(x)
    F, freqs = FourierFunctions.compute_real_FFT(signal, fs, nPointsFFT)
    return F, freqs
end

end

module TestFitsDerivs
using DelimitedFiles, Peaks, ..BLTimeEvolution, ..FourierFitGSL_Derivs, JLD2, FileIO
using ..Deriv2, ..Deriv3, ..Deriv4, ..Deriv5, ..Deriv6, ..FourierFunctions, ..Kerr

# compute fit to Boyer-Lindquist r
function compute_fit(f::Function, df_dt::Function, d2f_dt::Function, d3f_dt::Function, d4f_dt::Function, d5f_dt::Function, d6f_dt::Function,
    t_range_factor::Float64, nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, kerrReltol::Float64, kerrAbstol::Float64, 
    r_frequency::Bool, θ_frequency::Bool, ϕ_frequency::Bool, data_path::String)
    println("nHarm = $(nHarm), nPoints = $(nPoints)")

    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);
    Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];
    Ωr = r_frequency ? Ωr : 1e10; Ωθ = θ_frequency ? Ωθ : 1e10; Ωϕ = ϕ_frequency ? Ωϕ : 1e10; 
    Ω = [Ωr, Ωθ, Ωϕ]

    tmax =  t_range_factor * minimum(@. 2π/Ω[Ω .< 1e9]); saveat = tmax / (nPoints-1); Δti=saveat;

    ##### compute geodesic for reltol=1e-12 #####
    BLTimeEvolution.compute_kerr_geodesic(a, p, e, θi, nPoints, tmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; 
    r_dot=sol[5,:]; θ_dot=sol[6,:]; ϕ_dot=sol[7,:]; r_ddot=sol[8,:]; θ_ddot=sol[9,:]; ϕ_ddot=sol[10,:]; dt_dτ=sol[11,:];

    ##### compute test function values and its derivatives #####
    test_func_data_0 = zeros(size(t, 1))
    test_func_data_1 = zeros(size(t, 1))
    test_func_data_2 = zeros(size(t, 1))
    test_func_data_3 = zeros(size(t, 1))
    test_func_data_4 = zeros(size(t, 1))
    test_func_data_5 = zeros(size(t, 1))
    test_func_data_6 = zeros(size(t, 1))
    x = [Float64[] for i in 1:size(t, 1)]
    dx = [Float64[] for i in 1:size(t, 1)]
    d2x = [Float64[] for i in 1:size(t, 1)]
    d3x = [Float64[] for i in 1:size(t, 1)]
    d4x = [Float64[] for i in 1:size(t, 1)]
    d5x = [Float64[] for i in 1:size(t, 1)]
    d6x = [Float64[] for i in 1:size(t, 1)]
    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        dx[i] = Vector{Float64}([r_dot[i], θ_dot[i], ϕ_dot[i]]);
        d2x[i] = Vector{Float64}([r_ddot[i], θ_ddot[i], ϕ_ddot[i]]);
    end

    @inbounds for i in eachindex(t)
        d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
        d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
        d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        test_func_data_0[i] = f(x[i])
        test_func_data_1[i] = df_dt(dx[i], x[i]) 
        test_func_data_2[i] = d2f_dt(d2x[i], dx[i], x[i]) 
        test_func_data_3[i] = d3f_dt(d3x[i], d2x[i], dx[i], x[i])
        test_func_data_4[i] = d4f_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i]) 
        test_func_data_5[i] = d5f_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
        test_func_data_6[i] = d6f_dt(d6x[i], d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
    end

    # test_func_data = [test_func_data_0, test_func_data_1, test_func_data_2, test_func_data_3,
    # test_func_data_4, test_func_data_5, test_func_data_6];

    test_func_data = [test_func_data_0, test_func_data_2,
    test_func_data_4, test_func_data_6];

    ##### perform fit #####
    xdata = t; ydata=test_func_data_0; ydata_1 = test_func_data_1; ydata_2 = test_func_data_2;

    fit_fname_save=data_path * "Test_deriv_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"

    n_freqs = FourierFitGSL_Derivs.compute_num_fitting_freqs_master(nHarm, Ω); chisq=[0.0];
    fit_params = zeros(2 * n_freqs + 1);

    if length(t) == nPoints
        Ω_fit = FourierFitGSL_Derivs.GSL_fit_master!(xdata, ydata, ydata_1, ydata_2, nPoints, nHarm, chisq, Ω, fit_params)
    else
        println("nPoints = $(nPoints), length(t) = $(length(t))")
        throw(BoundsError)
    end

    # compute power spectrum
    power = zeros(n_freqs)
    for i=1:n_freqs
        power[i] = sqrt(fit_params[i+1]^2 + fit_params[i+1+n_freqs]^2)
    end

    # Creating a Typed Dictionary 
    fit_dictionary = Dict{String, AbstractArray}("xdata" => xdata, "test_func_data" => test_func_data, "fit_params" => fit_params, "fit_freqs" => Ω_fit, "power" => power) 
    # save fit #
    println(fit_fname_save)
    save(fit_fname_save, "data", fit_dictionary)
end

# load fourier fit and power spectrum
function load_fit(a::Float64, p::Float64, e::Float64, θi::Float64, nHarm::Int64, nPoints::Int64, data_path::String, FFT_flag::Bool)
    if FFT_flag == false
        fit_fname_save=data_path * "Test_deriv_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    else
        fit_fname_save=data_path * "Test_FFT_deriv_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    end
    return load(fit_fname_save)["data"]
end


# function to compute the derivative from the fit
function compute_derivatives(nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, data_path::String, FFT_flag::Bool)
    saved_data = load_fit(a, p, e, θi, nHarm, nPoints, data_path, FFT_flag)
    xdata = saved_data["xdata"]
    test_func_data = saved_data["test_func_data"]
    fit_params = saved_data["fit_params"]
    Ω_fit = saved_data["fit_freqs"]
    n_freqs = length(Ω_fit)
    return xdata, [FourierFitGSL_Derivs.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, N) for N in [0, 2, 4, 6]], test_func_data
end

end

module TestFitsFDM
using DelimitedFiles, Peaks, ..MinoTimeEvolution, JLD2, FileIO, ..MinoTimeDerivs, ..ParameterizedDerivs
using ..Deriv2, ..Deriv3, ..Deriv4, ..Deriv5, ..Deriv6, ..FourierFunctions, ..Kerr, ..TestFitsBLTime, ..FiniteDiff_5

##### define function to carry out fit and store fit parameters #####
function compute_orbital_functional_derivs(f::Function, df_dt::Function, d2f_dt::Function, d3f_dt::Function, d4f_dt::Function, d5f_dt::Function, d6f_dt::Function,
    h::Float64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, kerrReltol::Float64, kerrAbstol::Float64)
    println("h = $(h), nPoints = $(nPoints)")

    λ0 = 0.0; t0 = 0.0; ra = p * M / (1 - e);

    # compute fundamental frequencies 
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)

    saveat_λ = range(start=λ0, step=h, length=nPoints)|>collect
    λspan=(saveat_λ[1], last(saveat_λ))
    Δλi = h/10
    ics = MinoTimeEvolution.Mino_ics(t0, ra, p, e, M);
    sol = MinoTimeEvolution.compute_kerr_geodesic_inspiral(a, p, e, θi, E, L, Q, C, M, ics, saveat_λ, λspan, Δλi, kerrReltol, kerrAbstol)

    λ=sol[1,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; 
    dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt=sol[9,:]; d2θ_dt=sol[10,:]; d2ϕ_dt=sol[11,:]; dt_dλ=sol[12,:];

    ##### INTIALIZE DATA ARRAYS #####
    # orbital functional analytic derivatives w.r.t BL time
    test_func_data_0 = zeros(nPoints)
    test_func_data_1 = zeros(nPoints)
    test_func_data_2 = zeros(nPoints)
    test_func_data_3 = zeros(nPoints)
    test_func_data_4 = zeros(nPoints)
    test_func_data_5 = zeros(nPoints)
    test_func_data_6 = zeros(nPoints)

    # BL coordinate derivatives
    x = [Float64[] for i in 1:nPoints]
    dx = [Float64[] for i in 1:nPoints]
    d2x = [Float64[] for i in 1:nPoints]
    d3x = [Float64[] for i in 1:nPoints]
    d4x = [Float64[] for i in 1:nPoints]
    d5x = [Float64[] for i in 1:nPoints]
    d6x = [Float64[] for i in 1:nPoints]

    # derivatives of Mino λ w.r.t BL t
    dλ_dt = zeros(nPoints)
    d2λ_dt = zeros(nPoints)
    d3λ_dt = zeros(nPoints)
    d4λ_dt = zeros(nPoints)
    d5λ_dt = zeros(nPoints)
    d6λ_dt = zeros(nPoints)

    # numerical derivatives of the orbital functional w.r.t λ
    df_dλ = zeros(nPoints)
    d2f_dλ = zeros(nPoints)
    d3f_dλ = zeros(nPoints)
    d4f_dλ = zeros(nPoints)
    d5f_dλ = zeros(nPoints)
    d6f_dλ = zeros(nPoints)

    # numerical derivatives of the orbital functional w.r.t t
    df__dt = zeros(nPoints)
    d2f__dt = zeros(nPoints)
    d3f__dt = zeros(nPoints)
    d4f__dt = zeros(nPoints)
    d5f__dt = zeros(nPoints)
    d6f__dt = zeros(nPoints)
    
    @inbounds for i in eachindex(λ)
        # compute time derivatives of the BL coordinates
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        dx[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
        d2x[i] = Vector{Float64}([d2r_dt[i], d2θ_dt[i], d2ϕ_dt[i]]);
        d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
        d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
        d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]

        # compute time derivatives of the orbital functional
        test_func_data_0[i] = f(x[i])
        test_func_data_1[i] = df_dt(dx[i], x[i]) 
        test_func_data_2[i] = d2f_dt(d2x[i], dx[i], x[i]) 
        test_func_data_3[i] = d3f_dt(d3x[i], d2x[i], dx[i], x[i])
        test_func_data_4[i] = d4f_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i]) 
        test_func_data_5[i] = d5f_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
        test_func_data_6[i] = d6f_dt(d6x[i], d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])

        # compute auxiliary derivaties of Mino time, λ, w.r.t BL time t
        dλ_dt[i] = MinoTimeDerivs.dλ_dt(x[i], a, M, E, L)
        d2λ_dt[i] = MinoTimeDerivs.d2λ_dt(dx[i], x[i], a, M, E, L)
        d3λ_dt[i] = MinoTimeDerivs.d3λ_dt(d2x[i], dx[i], x[i], a, M, E, L)
        d4λ_dt[i] = MinoTimeDerivs.d4λ_dt(d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
        d5λ_dt[i] = MinoTimeDerivs.d5λ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
        d6λ_dt[i] = MinoTimeDerivs.d6λ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
    end

    # cannot parallelize with above loop since need various points in a stencil
    @inbounds for i in eachindex(λ)
        ##### COMPUTE NUMERICAL DERIVATIVES W.R.T MINO TIME #####
        df_dλ[i] = FiniteDiff_5.compute_first_derivative(i, test_func_data_0, h, nPoints)
        d2f_dλ[i] = FiniteDiff_5.compute_second_derivative(i, test_func_data_0, h, nPoints)
        d3f_dλ[i] = FiniteDiff_5.compute_third_derivative(i, test_func_data_0, h, nPoints)
        d4f_dλ[i] = FiniteDiff_5.compute_fourth_derivative(i, test_func_data_0, h, nPoints)
        d5f_dλ[i] = FiniteDiff_5.compute_fifth_derivative(i, test_func_data_0, h, nPoints)
        d6f_dλ[i] = FiniteDiff_5.compute_sixth_derivative(i, test_func_data_0, h, nPoints)

        ##### CONVERT TO DERIVATIVES W.R.T BL TIME #####
        df__dt[i] =  ParameterizedDerivs.df_dt(df_dλ[i], dλ_dt[i])
        d2f__dt[i] =  ParameterizedDerivs.d2f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i])
        d3f__dt[i] =  ParameterizedDerivs.d3f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i])
        d4f__dt[i] =  ParameterizedDerivs.d4f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i])
        d5f__dt[i] =  ParameterizedDerivs.d5f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i], d5f_dλ[i], d5λ_dt[i])
        d6f__dt[i] =  ParameterizedDerivs.d6f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i], d5f_dλ[i], d5λ_dt[i], d6f_dλ[i], d6λ_dt[i])
    end

    test_func_data = [test_func_data_2, test_func_data_4, test_func_data_6];
    numerical_data = [d2f__dt, d4f__dt, d6f__dt];
    
    return λ, test_func_data, numerical_data
end

end