include("/home/lkeeble/GRSuite/main.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/curve_fit.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/TestMultipoleFit.jl")
using DelimitedFiles, Statistics, BenchmarkTools, Plots, Plots.PlotMeasures, LaTeXStrings, LsqFit, .TestMultipoleFit

# path for saving data and plots
# data_path="TestFiles/Multiple_fit_test_results_v2/";
# plot_path="TestFiles/Multiple_fit_test_plots_v2/";
# data_path="TestFiles/Multiple_fit_test_results_Owren/";
# plot_path="TestFiles/Multiple_fit_test_plots_Owren/";
# path for saving data and plots
data_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_data/";
plot_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_plots/";
fourier_fit_test_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_data/fourier_fit_p0";
# mkpath(data_path)
# mkpath(plot_path)
mkpath(fourier_fit_test_path)
# mkdir(data_path)
# mkdir(plot_path)


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

points = [1000]; index1 = 2; index2 =3; nHarm = [2, 2];

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

Mij_data, Mijk_data, Sij_data, Sij_data, Mijk2_data, Sij1_data, Ωr, Ωθ, Ωϕ, t, τ, tdot = TestMultipoleFit.compute_multipoles(nPoints, nPointsMultipoleFit, M, m, a, p, e, θi,  g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, saveat_geometric, Δti, kerrReltol, kerrAbstol);

###### PERFORM FOURIER FIT ######

fitted_data = [[] for i in eachindex(nHarm)]    # will contain 5 array, each of which will contain three vectors corresponding to the fitted 0-, 1-, and 2- derivative of the mulitpole moment

for i in eachindex(nHarm)
    N=nHarm[i]
    println("Fitting for N = $(N) harmonics")
    @time fitted_data[i] = TestMultipoleFit.compute_fourier_fit_Mij(Sij_data, t, index1, index2, N, nPointsMultipoleFit, Ωr, Ωθ, Ωϕ, a, p, e, θi, data_path)
end


# print some error statistics
for i in eachindex(nHarm)
    N = nHarm[i]
    println("\n$(N) harmonics:")
    TestMultipoleFit.print_errors(Sij_data[index1, index2], fitted_data[i][1], 0)
    TestMultipoleFit.print_errors(Sij1_data[index1, index2], fitted_data[i][2], 1)
end


# # plot 0th derivative
# scatter(t, Sij_data[index1, index2], color=:red, label="Geodesic", xlabel=L"t")
# scatter!(t, fitted_data[5][1], color=:blue, label="Fit")
# vline!([t[(nPointsMultipoleFit ÷ 2) + 1]], linestyle=:dash, label=false)

# plot 2nd derivative
Sij_plot=scatter(t, Sij1_data[index1, index2], color=:red, label="Geodesic", xlabel=L"t")
scatter!(Sij_plot, t, fitted_data[2][2], color=:blue, label="Fit")
vline!(Sij_plot, [t[(nPointsMultipoleFit ÷ 2) + 1]], linestyle=:dash, label=false)
display(Sij_plot)
# compute errors harmonic-by-harmonic 

errors_1 = [@. 100 * (Sij_data[index1, index2]-fitted_data[i][1])/Sij_data[index1, index2]  for i in eachindex(nHarm)]    # avoid t=0 to not divide by zero
errors_2 = [@. 100 * (Sij1_data[index1, index2]-fitted_data[i][2])/Sij1_data[index1, index2]  for i in eachindex(nHarm)]    # avoid t=0 to not divide by zero


# plot attributes
shapes = [:star4, :xcross, :diamond, :rect, :circle]
colors = [:yellow, :magenta, :green, :red, :blue]
# alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
alphas = [0.2, 0.2, 0.2, 0.2, 0.2]

# plot errors
# plotting deviations of the 0th derivative #
ms=2
error_plot_1 = scatter(t, abs.(errors_1[1]), xlabel=L"t", ylabel=L"S_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
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


error_plot_2 = scatter(t, abs.(errors_2[1]), xlabel=L"t", ylabel=L"\dot{S}_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
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

plot_name="Sij_$(index1)_$(index2)_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nPoints_$(nPointsMultipoleFit)_tmin_$(first(t))_tmax_$(last(t)).png"
savefig(error_plot_2, plot_path * plot_name)



# now compute the numerical derivatives wrt τ
nPoints=size(τ, 1); h=τ[2]-τ[1];
derivs_fdm_4=[zeros(nPoints) for i=1:2]; FiniteDiff_4.compute_derivs(derivs_fdm_4, Sij_data[index1,index2], h, nPoints)
derivs_fdm_5=[zeros(nPoints) for i=1:2]; FiniteDiff_5.compute_derivs(derivs_fdm_5, Sij_data[index1,index2], h, nPoints)
# convert to proper time
for i in eachindex(derivs_fdm_4)
    for j in eachindex(derivs_fdm_5[i])
        derivs_fdm_4[i][j] = derivs_fdm_4[i][j] / (tdot[j]^i)
        derivs_fdm_5[i][j] = derivs_fdm_5[i][j] / (tdot[j]^i)
    end
end

numerical_error_analytic = @. 100 * (Sij1_data[index1,index2]-derivs_fdm_5[2]) / Sij1_data[index1,index2]
numerical_error_fitted = @. 100 * (fitted_data[2][2]-derivs_fdm_5[2]) / fitted_data[2][2]

shapes = [:star4, :circle]
colors = [:red, :blue]
alphas = [0.2, 0.2]

# plot errors

# plotting deviations of the 0th derivative #
ms=2

error_plot_numerical = scatter(t, abs.(numerical_error_analytic), xlabel=L"t", ylabel=L"\dot{S}_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
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
savefig(plot_path*"numerical_deriv_error_Sij_$(index1)_$(index2).png")