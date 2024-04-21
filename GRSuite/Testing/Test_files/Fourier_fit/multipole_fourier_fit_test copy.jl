include("/home/lkeeble/GRSuite/main.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/curve_fit.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/TestMultipoleFit.jl")
using DelimitedFiles, Statistics, BenchmarkTools, Plots, Plots.PlotMeasures, LaTeXStrings, LsqFit, .TestMultipoleFit

# path for saving data and plots
data_path="/home/lkeeble/GRSuite/Testing/Test_results/GSL/Test_data/";
plot_path="/home/lkeeble/GRSuite/Testing/Test_results/GSL/Test_plots/";
fourier_fit_test_path="/home/lkeeble/GRSuite/Testing/Test_results/GSL/Test_data/fourier_fit_p0";
mkpath(data_path)
mkpath(plot_path)
mkpath(fourier_fit_test_path)



###### COMPUTE MULTOPOLE MOMENTS #####

Γαμν(t, r, θ, ϕ, a, M, α, μ, ν) = Kerr.KerrMetric.Γαμν(t, r, θ, ϕ, a, M, α, μ, ν);   # Christoffel symbols

# covariant metric components
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

# constants
c = 2.99792458 * 1e8; Grav_Newton = 6.67430 * 1e-11; Msol = (1.988) * 1e30; yr = 365 * 24 * 60 * 60;

points = [1000]; index1 = 2; index2 =3; nHarm = [2];

nPointsMultipoleFit=points[1]
println("\nnPointsMultipoleFit = $(nPointsMultipoleFit)\n")

### COMPUTE INSPIRAL ###

# # orbit parameters - Alejo Example
# M = 1e6 * Msol; p=6.0; q=1e-5; e=0.7; a=0.9; theta_inc_o=0.349065850398866; θi=π/2 - theta_inc_o; Δti=1.0; kerrReltol=1e-10; kerrAbstol=1e-10;
# nPoints = 100;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points

# orbit parameters - Paper Fig. 5 Example
M = 1e6 * Msol; p=7.0; q=1e-5; e=0.6; a=0.98; theta_inc_o=57.39 * π/180.0; θi=π/2 - theta_inc_o; Δti=1.0; kerrReltol=1e-10; kerrAbstol=1e-10;
nPoints = 1000;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points

θi=0.570798

# conversion from t(M) -> t(s)
MtoSecs = M * Grav_Newton / c^3

# time parameters for orbit
saveat_secs = 0.5    # sample trajectory every X secs
saveat_geometric = saveat_secs / MtoSecs    # convert t= Xs to units of M
τMax_secs = 10^(-2) * yr   # evolve inspiral for a maximum time of X
τMax_geometric = τMax_secs / MtoSecs;   # convert inspiral evolution time to units of M

# now convert back to geometric units
M=1.0; m=q;

Mij_data, Mijk_data, Sij_data, Mij2_data, Mijk2_data, Sij1_data, Ωr, Ωθ, Ωϕ, t, τ, tdot = TestMultipoleFit.compute_multipoles(nPoints, nPointsMultipoleFit, M, m, a, p, e, θi,  g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, saveat_geometric, Δti, kerrReltol, kerrAbstol);

###### PERFORM FOURIER FIT ######

fitted_data = [[] for i in eachindex(nHarm)]    # will contain 5 array, each of which will contain three vectors corresponding to the fitted 0-, 1-, and 2- derivative of the mulitpole moment

for i in eachindex(nHarm)
    N=nHarm[i]
    println("Fitting for N = $(N) harmonics")
    @time fitted_data[i] = TestMultipoleFit.compute_fourier_fit_Mij(Mij_data, t, index1, index2, N, nPointsMultipoleFit, Ωr, Ωθ, Ωϕ, a, p, e, θi, data_path)
end


# print some error statistics
for i in eachindex(nHarm)
    N = nHarm[i]
    println("\n$(N) harmonics:")
    TestMultipoleFit.print_errors(Mij_data[index1, index2], fitted_data[i][1], 0)
    TestMultipoleFit.print_errors(Mij2_data[index1, index2], fitted_data[i][3], 2)
end


# # plot 0th derivative
# scatter(t, Mij_data[index1, index2], color=:red, label="Geodesic", xlabel=L"t")
# scatter!(t, fitted_data[5][1], color=:blue, label="Fit")
# vline!([t[(nPointsMultipoleFit ÷ 2) + 1]], linestyle=:dash, label=false)

# plot 2nd derivative
Mij2_plot=scatter(t, Mij2_data[index1, index2], color=:red, label="Geodesic", xlabel=L"t")
scatter!(Mij2_plot, t, fitted_data[2][3], color=:blue, label="Fit")
vline!(Mij2_plot, [t[(nPointsMultipoleFit ÷ 2) + 1]], linestyle=:dash, label=false)
display(Mij2_plot)
# compute errors harmonic-by-harmonic 

errors_1 = [@. 100 * (Mij_data[index1, index2]-fitted_data[i][1])/Mij_data[index1, index2]  for i in eachindex(nHarm)]    # avoid t=0 to not divide by zero
errors_2 = [@. 100 * (Mij2_data[index1, index2]-fitted_data[i][3])/Mij2_data[index1, index2]  for i in eachindex(nHarm)]    # avoid t=0 to not divide by zero


# plot attributes
shapes = [:star4, :xcross, :diamond, :rect, :circle]
colors = [:yellow, :magenta, :green, :red, :blue]
# alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
alphas = [0.2, 0.2, 0.2, 0.2, 0.2]

# plot errors

# plotting deviations of the 0th derivative #
ms=2
error_plot_1 = scatter(t, abs.(errors_1[1]), xlabel=L"t", ylabel=L"M_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
        framsetyle=:box,
        legend=:topright,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        size=(500, 300),
        markersize=ms,
        ylims=(1e-8,1e8), dpi=1000,
        label=L"N=%$(nHarm[1])", yscale=:log10, markershape=shapes[1], color=colors[1], fillalpha=alphas[1])
        vline!([t[nPointsMultipoleFit ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(t) + (maximum(t)-minimum(t))*0.1, 5e6, Plots.text(L"n_{\mathrm{p}}=%$(nPointsMultipoleFit)", :center, 12))

for i in eachindex(nHarm)
    N=nHarm[i]
    if i!=1
        scatter!(error_plot_1, t, abs.(errors_1[i]),
        markersize=ms, markershape=shapes[i], color=colors[i], fillalpha=alphas[i], label=L"N=%$(N)")
    end
end
yticks!(error_plot_1, [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_1)

plot_name="Mij_$(index1)_$(index2)_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nPoints_$(nPointsMultipoleFit)_tmin_$(first(t))_tmax_$(last(t)).png"
savefig(error_plot_1, plot_path * plot_name)

# plotting deviations of the 2nd derivative #


error_plot_2 = scatter(t, abs.(errors_2[1]), xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
framsetyle=:box,
legend=:bottomright,
foreground_color_legend = nothing,
background_color_legend = nothing,
size=(500, 300),
markersize=ms,
ylims=(1e-8,1e8), dpi=1000,
label=L"N=%$(nHarm[1])", yscale=:log10, markershape=shapes[1], color=colors[1], fillalpha=alphas[1])
vline!([t[nPointsMultipoleFit ÷ 2]], label=false, linestyle=:dash)
annotate!(minimum(t) + (maximum(t)-minimum(t))*0.1, 5e-7, Plots.text(L"n_{\mathrm{p}}=%$(nPointsMultipoleFit)", :center, 12))

for i in eachindex(nHarm)
    N=nHarm[i]

    if i!=1
        scatter!(error_plot_2, t, abs.(errors_2[i]),
        markersize=ms, markershape=shapes[i], color=colors[i], fillalpha=alphas[i], label=L"N=%$(N)")
    end
    
end
yticks!(error_plot_2, [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_2)

plot_name="Mij2_$(index1)_$(index2)_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nPoints_$(nPointsMultipoleFit)_tmin_$(first(t))_tmax_$(last(t)).png"
savefig(error_plot_2, plot_path * plot_name)



# now compute the numerical derivatives wrt τ
nPoints=size(τ, 1); h=τ[2]-τ[1];
derivs_fdm_4=[zeros(nPoints) for i=1:2]; FiniteDiff_4.compute_derivs(derivs_fdm_4, Mij_data[index1,index2], h, nPoints)
derivs_fdm_5=[zeros(nPoints) for i=1:2]; FiniteDiff_5.compute_derivs(derivs_fdm_5, Mij_data[index1,index2], h, nPoints)
# convert to proper time
for i in eachindex(derivs_fdm_4)
    for j in eachindex(derivs_fdm_5[i])
        derivs_fdm_4[i][j] = derivs_fdm_4[i][j] / (tdot[j]^i)
        derivs_fdm_5[i][j] = derivs_fdm_5[i][j] / (tdot[j]^i)
    end
end

numerical_error_analytic = @. 100 * (Mij2_data[index1,index2]-derivs_fdm_5[2]) / Mij2_data[index1,index2]
numerical_error_fitted = @. 100 * (fitted_data[2][3]-derivs_fdm_5[2]) / fitted_data[2][3]

shapes = [:star4, :circle]
colors = [:red, :blue]
alphas = [0.2, 0.2]

# plot errors

# plotting deviations of the 0th derivative #
ms=2

error_plot_numerical = scatter(t, abs.(numerical_error_analytic), xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
    framsetyle=:box,
    legend=:false,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    size=(600, 400),
    markersize=ms,
    ylims=(1e-8,1e8),dpi=1000,
    yscale=:log10, markershape=shapes[1], color=colors[1], fillalpha=alphas[1],
    framestyle=:box)
    vline!([t[nPointsMultipoleFit ÷ 2]], label=false, linestyle=:dash)
    annotate!(minimum(t) + (maximum(t)-minimum(t))*0.25, 5e5, Plots.text(L"\textrm{Numerical\;derivative}"*"\n"* L"n_{\mathrm{p}}=%$(nPointsMultipoleFit)", :center, 12))
savefig(plot_path*"numerical_deriv_error_Mij2_$(index1)_$(index2).png")
# scatter!(error_plot_numerical, t, abs.(numerical_error_fitted),
#         markersize=ms, markershape=shapes[2], color=colors[2], fillalpha=alphas[2], label=L"\textrm{Fit, Num}")

#=
Original Fourier Fit

Fitting statistics for nPointsMultipoleFit=50: 
    Fitting for N = 2 harmonics
        0.101019 seconds (95.92 k allocations: 7.390 MiB, 87.44% compilation time)
    Fitting for N = 3 harmonics
        6.590201 seconds (4.17 M allocations: 570.572 MiB, 2.26% gc time, 1.02% compilation time)
    Fitting for N = 4 harmonics
        4.925577 seconds (2.13 M allocations: 332.436 MiB, 0.99% gc time)
    Fitting for N = 5 harmonics
        29.022246 seconds (9.68 M allocations: 1.924 GiB, 1.16% gc time)
    Fitting for N = 6 harmonics
        143.734090 seconds (43.51 M allocations: 11.600 GiB, 0.52% gc time)

Fitting statistics for nPointsMultipoleFit=100:
    Fitting for N = 2 harmonics
        0.612105 seconds (732.04 k allocations: 92.812 MiB, 5.65% gc time)
    Fitting for N = 3 harmonics
        2.877693 seconds (2.06 M allocations: 290.562 MiB, 2.64% gc time)
    Fitting for N = 4 harmonics
        23.567727 seconds (8.86 M allocations: 1.515 GiB, 0.67% gc time)
    Fitting for N = 5 harmonics
        135.201470 seconds (40.97 M allocations: 9.042 GiB, 0.53% gc time)
    Fitting for N = 6 harmonics
        229.302681 seconds (54.69 M allocations: 15.178 GiB, 0.48% gc time)

Fitting statistics for nPointsMultipoleFit=200:

    Fitting for N = 2 harmonics
        1.369426 seconds (1.23 M allocations: 173.956 MiB, 5.35% gc time)
    Fitting for N = 3 harmonics
        4.481821 seconds (2.34 M allocations: 369.757 MiB, 1.29% gc time)
    Fitting for N = 4 harmonics
        43.216203 seconds (10.06 M allocations: 1.849 GiB, 0.53% gc time)
    Fitting for N = 5 harmonics
        229.652172 seconds (45.75 M allocations: 10.573 GiB, 0.40% gc time)
    Fitting for N = 6 harmonics
        146.745532 seconds (14.23 M allocations: 3.943 GiB, 0.27% gc time)

Fitting statistics for nPointsMultipoleFit=400:

    Fitting for N = 2 harmonics
        2.297094 seconds (1.42 M allocations: 235.691 MiB, 1.26% gc time)
    Fitting for N = 3 harmonics
        22.304434 seconds (5.38 M allocations: 993.856 MiB, 0.28% gc time)
    Fitting for N = 4 harmonics
        104.626495 seconds (16.37 M allocations: 3.400 GiB, 0.21% gc time)
    Fitting for N = 5 harmonics
        397.902863 seconds (39.36 M allocations: 10.077 GiB, 0.13% gc time)
    Fitting for N = 6 harmonics
        1208.104373 seconds (83.15 M allocations: 27.090 GiB, 0.10% gc time)

Fitting statistics for nPointsMultipoleFit=500:

    Fitting for N = 2 harmonics
        3.188422 seconds (1.73 M allocations: 307.730 MiB, 1.11% gc time)
    Fitting for N = 3 harmonics
        26.808618 seconds (5.30 M allocations: 1.011 GiB, 0.29% gc time)
    Fitting for N = 4 harmonics
        115.232147 seconds (16.16 M allocations: 3.555 GiB, 0.19% gc time)
    Fitting for N = 5 harmonics
        1001.183606 seconds (84.45 M allocations: 22.622 GiB, 0.11% gc time)
    Fitting for N = 6 harmonics
        3231.829708 seconds (183.99 M allocations: 61.433 GiB, 0.09% gc time)

Fitting statistics for nPointsMultipoleFit=750:

    Fitting for N = 2 harmonics
        6.375215 seconds (2.50 M allocations: 522.730 MiB, 0.66% gc time)
    Fitting for N = 3 harmonics
        31.195339 seconds (4.54 M allocations: 1.000 GiB, 0.22% gc time)
    Fitting for N = 4 harmonics
        243.559132 seconds (24.31 M allocations: 6.091 GiB, 0.14% gc time)
    Fitting for N = 5 harmonics
        890.572737 seconds (55.05 M allocations: 16.647 GiB, 0.09% gc time)
    Fitting for N = 6 harmonics
        2518.560960 seconds (95.35 M allocations: 34.508 GiB, 0.07% gc time)

Fitting statistics for nPointsMultipoleFit=1000:
    Fitting for N = 2 harmonics
        9.030767 seconds (2.82 M allocations: 680.413 MiB, 0.48% gc time)
    Fitting for N = 3 harmonics
        47.309655 seconds (6.07 M allocations: 1.538 GiB, 0.19% gc time)
    Fitting for N = 4 harmonics
        230.692314 seconds (17.79 M allocations: 5.020 GiB, 0.12% gc time)
    Fitting for N = 5 harmonics
        3522.755484 seconds (166.90 M allocations: 54.944 GiB, 0.07% gc time)

Fitting statistics for nPointsMultipoleFit=1500:

Fitting statistics for nPointsMultipoleFit=2000:

Fitting statistics for nPointsMultipoleFit=3000:



=#


#=
Owren Fourier Fit

Fitting statistics for nPointsMultipoleFit=200:

    Fitting for N = 2 harmonics
        2.318673 seconds (1.68 M allocations: 204.947 MiB, 2.99% gc time, 33.04% compilation time)
    Fitting for N = 5 harmonics
        521.221738 seconds (45.84 M allocations: 10.578 GiB, 0.31% gc time, 0.02% compilation time)

Fitting statistics for nPointsMultipoleFit=500:

    Fitting for N = 2 harmonics
        4.901897 seconds (1.73 M allocations: 307.730 MiB, 1.75% gc time)
    Fitting for N = 5 harmonics
        2516.071141 seconds (84.47 M allocations: 22.623 GiB, 0.08% gc time)

=#

#=

DP8() Fourier Fit

Fitting statistics for nPointsMultipoleFit=200:

    Fitting for N = 2 harmonics
        2.286420 seconds (1.68 M allocations: 205.010 MiB, 2.68% gc time, 31.32% compilation time)
    Fitting for N = 5 harmonics
        474.849439 seconds (45.79 M allocations: 10.576 GiB, 0.31% gc time, 0.02% compilation time)

Fitting statistics for nPointsMultipoleFit=500:

    Fitting for N = 2 harmonics
        4.908216 seconds (1.73 M allocations: 307.730 MiB, 1.85% gc time)
    Fitting for N = 5 harmonics
        2364.086663 seconds (84.46 M allocations: 22.622 GiB, 0.09% gc time)

=#