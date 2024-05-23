include("/home/lkeeble/GRSuite/main.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/fourier_fit_test_funcs.jl")
using DelimitedFiles, Statistics, BenchmarkTools, Plots, LaTeXStrings, LsqFit, Distributions
using .TestFunctions
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
data_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_data";
plot_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_plots/";

##### specify orbital params to produce example frequencies ####
nPoints = 500;

# calculate orbital frequencies (wrt t, NOT t)
# ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi); ωr, ωθ, ωϕ = ω[1:3];
ωr = 0.0132324317837;
ωθ = 0.11421324314134;
ωϕ = 0.087541442292912;

##### generate dummy data for some number of harmonics #####
# we evolve the trajectory for a time t = max(2π/ωi)
τmax = minimum(@. 2π/[ωr, ωθ, ωϕ]);
saveat = τmax / (nPoints-1)
t = 0.0:saveat:saveat * (nPoints-1)|>collect

# implement method used in Chimera code (e.g., the way of constructing the fitting frequencies)
data = [zeros(nPoints) for i=0:10]
a2=0.5; b2=0.6; p2=0.7;
data[1] = TestFunctions.func1_0.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[2] = TestFunctions.func1_1.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[3] = TestFunctions.func1_2.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[4] = TestFunctions.func1_3.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[5] = TestFunctions.func1_4.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[6] = TestFunctions.func1_5.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[7] = TestFunctions.func1_6.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[8] = TestFunctions.func1_7.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[9] = TestFunctions.func1_8.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[10] = TestFunctions.func1_9.(t, ωr, ωθ, ωϕ, a2, b2, p2)
data[11] = TestFunctions.func1_10.(t, ωr, ωθ, ωϕ, a2, b2, p2)

#### Given artificial data, now attempt to fit to low harmonic number ####
nHarm_fit = 2    # first plot with 500 points and 5 harmonics takes: 7376.201090 seconds (595.20 M allocations: 133.247 GiB, 0.17% gc time), 500 points with 4 harmonics takes: 

# load fit (if it exists)
# fit_fname_params="fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm_fit)_nPoints_$(nPoints)_tmax_$(τmax).txt";
# fit_fname_save=data_path * "test_func_" * fit_fname_params
# isfile(fit_fname_save) ? p0 = readdlm(fit_fname_save)[:] : p0 = Float64[];

p0 = Float64[];

# carry out fit - 146.989705 seconds (73.57 M allocations: 12.494 GiB, 0.64% gc time, 1.03% compilation time) for N=2
@time Ωfit, ffit, fitted_data = FourierFit.fourier_fit(t, data[1], ωr, ωθ, ωϕ, nHarm_fit, p0=p0)  
fit_params=coef(ffit)

# save fit 
# open(fit_fname_save, "w") do io
#     writedlm(io, fit_params)
# end

# array of arrays for the 0th - 10th derivatives
fitted_data = [FourierFit.curve_fit_functional_derivs(t, Ωfit, fit_params, N) for N=0:10]

### plot ###
print_errors(data[1][2:nPoints], fitted_data[1][2:nPoints], 0)
print_errors(data[2][2:nPoints], fitted_data[2][2:nPoints], 1)
print_errors(data[3][2:nPoints], fitted_data[3][2:nPoints], 2)
print_errors(data[6][2:nPoints], fitted_data[6][2:nPoints], 2)

# devitations #
diff = [@. 100 * (data[i][2:nPoints]-fitted_data[i][2:nPoints])/data[i][2:nPoints]  for i=1:11]    # avoid t=0 to not divide by zero

# plot attributes
shape_1 = :star
color_1 = :yellow
shape_4 = :circle
color_4 = :blue
shape_5 = :rect
color_5 = :red
shape_6 = :diamond
color_6 = :green
markerstrokewidth=0;
# plotting deviations of the 4th, 5th, 6th derivative #
ms=2
N=0
deviation_plot_LsqFit = scatter(t, abs.(diff[N+1]), xlabel=L"t", ylabel=L"$\mathrm{Error}\,(\%)$",
framsetyle=:box,
legend=:bottomright,
foreground_color_legend = nothing,
background_color_legend = nothing,
size=(500, 300),
markersize=ms,
ylims=(1e-16,1e16), dpi=1000,
label=L"d^{(%$(N))}", markerstrokewidth=markerstrokewidth, yscale=:log10, markershape=shape_1, color=color_1)
vline!([t[nPoints ÷ 2]], label=false, linestyle=:dash)
yticks!(deviation_plot_LsqFit, [1e-16, 1e-12, 1e-8, 1e-4, 1e0, 1e4, 1e8, 1e12, 1e16])
annotate!(maximum(t)*0.3, 1.5e14, Plots.text(L"N=%$(nHarm_fit)", :center, 12))
annotate!(maximum(t)*0.3, 1.5e11, Plots.text(L"n_{\mathrm{p}}=%$(nPoints)", :center, 12))
annotate!(maximum(t)*0.7, 1.5e14, Plots.text("LsqFit", :center, 12))

N=3
scatter!(deviation_plot_LsqFit, t, abs.(diff[N+1]),
markersize=ms, markershape=shape_4, color=color_4, label=L"d^{(%$(N))}", markerstrokewidth=markerstrokewidth)

N=6
scatter!(deviation_plot_LsqFit, t, abs.(diff[N+1]),
markersize=ms, markershape=shape_5, color=color_5, label=L"d^{(%$(N))}", markerstrokewidth=markerstrokewidth)

N=10
scatter!(deviation_plot_LsqFit, t, abs.(diff[N+1]),
markersize=ms, markershape=shape_6, color=color_6, label=L"d^{(%$(N))}", markerstrokewidth=markerstrokewidth)
display(deviation_plot_LsqFit)

plot_name="test_func_1_deltas_plot_LsqFit_nHarm_$(nHarm_fit)_nPoints_$(nPoints).png"
savefig(deviation_plot_LsqFit, plot_path * plot_name)