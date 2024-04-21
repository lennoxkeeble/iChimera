include("/home/lkeeble/GRSuite/main.jl")
using DelimitedFiles, Statistics, BenchmarkTools, Plots, LaTeXStrings, LsqFit, Distributions
using .FourierFit

# returns percentaage deviation
function deviation(y_true::Float64, y_approx::Float64)
    return 100 * (y_true-y_approx)/y_true
end

# prints maximum, minimum and average error of derivative N
function print_errors(data::Vector{Float64}, fitted_data::Vector{Float64}, N::Int)
    # compute percentage difference in real data and best-fit data 
    deviations = @. deviation(data, fitted_data)

    println("Error in fit to function f, derivative order $(N)")
    # println("Minimum deviation =$(minimum(abs.(deviations)))%")
    println("Maxmium deviation =$(maximum(abs.(deviations)))%")
    println("Average deviation =$(mean(abs.(deviations)))%")
end


# path for saving data and plots
data_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_data/";
plot_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_plots/";
fourier_fit_test_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_data/fourier_fit_p0";
# mkpath(data_path)
# mkpath(plot_path)
mkpath(fourier_fit_test_path)

##### specify orbital params to produce example frequencies ####
p=7.0; q=1e-5; e=0.6; a=0.98; θi=0.570798
nPoints = 200;

# calculate orbital frequencies (wrt τ, NOT t)
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi); ωr, ωθ, ωϕ = ω[1:3]; Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];

##### generate dummy data for some number of harmonics #####
# we evolve the trajectory for a time τ = max(2π/ωi)
τmax = maximum(@. 2π/[Ωr, Ωθ, Ωϕ]);
saveat = τmax / (nPoints-1)
saveat=0.5

t = 0.0:saveat:saveat * (nPoints-1)|>collect

# implement method used in Chimera code (e.g., the way of constructing the fitting frequencies)
nHarm_true = 10
Ω = Float64[]
@inbounds for i_r in 0:nHarm_true
    @inbounds for i_θ in -i_r:nHarm_true
        @inbounds for i_ϕ in -(i_r+i_θ):nHarm_true
            append!(Ω, i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ)
        end
    end
end

n_freqs = size(Ω, 1)
# specify coefficients arbitrarily - we require 2 * n_freqs of them since we have one cosine and sine term per frequency
coefs = rand(Uniform(-1, 1), 2 * n_freqs)

# array of data points containing 0th - 10th derivatives
data = [zeros(nPoints) for i=1:11]
data[1] = FourierFit.curve_fit_functional(zeros(nPoints), t, Ω, coefs, n_freqs)
@inbounds for i in 1:10
    data[i+1] = FourierFit.curve_fit_functional_derivs(t, Ω, coefs, i)
end

#### Given artificial data, now attempt to fit to low harmonic number ####
nHarm_fit = 3

# load fit (if it exists)
fit_fname_params="fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm_fit).txt";
fit_fname_save=fourier_fit_test_path * "test_func_" * fit_fname_params
isfile(fit_fname_save) ? p0 = readdlm(fit_fname_save)[:] : p0 = Float64[];

# carry out fit
@time Ωfit, ffit, fitted_data = FourierFit.fourier_fit(t, data[1], Ωr, Ωθ, Ωϕ, nHarm_fit, p0=p0)
fit_params=coef(ffit)

# save fit 
open(fit_fname_save, "w") do io
    writedlm(io, fit_params)
end

# array of arrays for the 0th - 10th derivatives
fitted_data = [FourierFit.curve_fit_functional_derivs(t, Ωfit, fit_params, N) for N=0:10]

### plot ###
print_errors(data[1], fitted_data[1], 0)
print_errors(data[2], fitted_data[2], 1)
print_errors(data[3], fitted_data[3], 2)

# plot attribtutes #
ms = 2

# plotting the 0th derivative #
scatter(t, data[1], label="Exact", xlabel=L"t",
framsetyle=:box,
legend=:topright,
foreground_color_legend = nothing,
background_color_legend = nothing,
size=(500, 300),
markersize=ms)

scatter!(t, fitted_data[1], label="Fit",
markersize=ms)
annotate!(maximum(t)*0.9, minimum(data[1])*0.9, Plots.text(L"N=%$(nHarm_fit)", :left, 12))


# plotting the 1st derivative #
N=1
fitted_1 = FourierFit.curve_fit_functional_derivs(t, Ωfit, fit_params, N)

scatter(t, data[N+1], label="Exact", xlabel=L"t",
framsetyle=:box,
legend=:topright,
foreground_color_legend = nothing,
background_color_legend = nothing,
size=(500, 300),
markersize=ms)

scatter!(t, fitted_1, label="Fit",
markersize=ms)
annotate!(maximum(t)*0.9, minimum(data[N+1])*0.9, Plots.text(L"N=%$(nHarm_fit)", :left, 12))

# plotting the 1st derivative #
N=1
fitted_1 = FourierFit.curve_fit_functional_derivs(t, Ωfit, fit_params, N)

scatter(t, data[N+1], label="Exact", xlabel=L"t",
framsetyle=:box,
legend=:topright,
foreground_color_legend = nothing,
background_color_legend = nothing,
size=(500, 300),
markersize=ms)

scatter!(t, fitted_1, label="Fit",
markersize=ms)
annotate!(maximum(t)*0.9, minimum(data[N+1])*0.9, Plots.text(L"N=%$(nHarm_fit)", :left, 12))