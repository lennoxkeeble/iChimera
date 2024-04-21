# include files
include("main.jl")
using DelimitedFiles, LaTeXStrings, Plots.PlotMeasures, Plots

# path for saving data and plots
data_path="Test_results_fdm/HJE";
plot_path="Test_plots_fdm/HJE";
mkpath(data_path)
mkpath(plot_path)

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

### COMPUTE INSPIRAL ###

# # orbit parameters - Alejo Example
# M = 1e6 * Msol; p=6.0; q=1e-5; e=0.7; a=0.9; theta_inc_o=0.349065850398866; θi=π/2 - theta_inc_o; Δti=1.0; kerrReltol=1e-10; kerrAbstol=1e-10;
# nPoints = 100;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points

# orbit parameters - Paper Fig. 5 Example
M = 1e6 * Msol; p=7.0; q=1e-5; e=0.6; a=0.98; theta_inc_o=57.39 * π/180.0; θi=π/2 - theta_inc_o; Δti=1.0; kerrReltol=1e-12; kerrAbstol=1e-10; MtoSecs = M * Grav_Newton / c^3'
nPoints = 500;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points

θi=0.570798

tOrbit = 10^(-2) * yr   # evolve inspiral for a maximum time of X


# now convert back to geometric units
M=1.0; m=q;
h=0.2;
# compute inspiral
@time SelfForce_numerical.compute_inspiral_HJE!(tOrbit, nPoints, M, m, a, p, e, θi,  Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, h, kerrReltol, kerrAbstol, data_path=data_path)

# # plot orbit solution
# EMRI_ode_sol_fname = data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(kerrReltol)_fdm.txt"
# EMRI_orbit_fname = "EMRI_3d_orbit_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(kerrReltol)_fdm.txt"
# zlims = (-30, 30);
# GRPlotLib.plot_orbit(EMRI_ode_sol_fname, EMRI_orbit_fname, zlims, plot_path=plot_path)

# # plot orbit projected onto x-y plane
# xlims = (-11, 11)
# ylims =  (-11, 11)
# EMRI_xy_fname = "EMRI_xy_orbit__a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tstep_$(saveat_geometric)_tol_$(kerrReltol).png"
# GRPlotLib.plot_xy_orbit(EMRI_ode_sol_fname, EMRI_xy_fname, xlims, ylims, plot_path=plot_path,
# xlims=(-20, 20), xticks=(-20:10:20, ["-20" "-10" "0" "10" "20"]),
# ylims=(-20, 20), yticks=(-20:10:20, ["-20" "-10" "0" "10" "20"]),
# size=(500, 500), color=:black, linewidth=0.5,
# legend=false, grid=false, framestyle=:box)

#### parameters ###

# load ODE solution
ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(kerrReltol)_fdm.txt"
sol = readdlm(ODE_filename)
t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; 
dr_dt=sol[5,:]; dθ_dt=sol[6,:]; dϕ_dt=sol[7,:]; 
d2r_dt2=sol[8,:]; d2θ_dt2=sol[9,:]; d2ϕ_dt2=sol[10,:]; dt_dτ=sol[11,:];

constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(kerrReltol)_fdm.txt"
constants=readdlm(constants_filename)
constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_h_$(h)_tol_$(kerrReltol)_fdm.txt"
constants_derivs = readdlm(constants_derivs_filename)

EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :]
Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]

### PLOT FLUXES AND CONSTANTS OF MOTION ###

# margins
left_margin = 8mm; right_margin=8mm; top_margin=8mm; bottom_margin=8mm;

gr()
plotΔE = Plots.plot(t, Edot, ylabel=L"\dot{E}", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotΔL = Plots.plot(t, Ldot, ylabel=L"\dot{L}", xlabel=L"t\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotΔQ = Plots.plot(t, Qdot, ylabel=L"\dot{Q}", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
# title = plot(title ="p0=$(p)M, e0=$(e), a=$(a)M, ι0=$(round(theta_inc_o*180/π; digits=4)), q=$(q)", grid = false, showaxis = false, bottom_margin = -50Plots.px)
title = plot(title =L"p_{0}=%$(p)M,\,e_{0}=%$(e),\,a=%$(a)M,\,ι_{0}=%$(θi),\,q=%$(q),\,n_{\mathrm{geo}}=%$(nPoints),\,h=%$(h)", grid = false, showaxis = false, bottom_margin = -50Plots.px, dpi=1000)
fluxplot = plot(title, plotΔE, plotΔL, plotΔQ, layout = @layout([A{0.01h}; [B C D]]), size=(1500, 300))
display(fluxplot)

gr()
plotE = Plots.plot(t, EE, ylabel=L"E", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotL = Plots.plot(t, LL, ylabel=L"L", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotC = Plots.plot(t, CC, ylabel=L"C", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

orbitalParamsPlot=plot(plotE, plotL, plotC, layout=(1, 3), size=(1500, 300))
plotP = Plots.plot(t, pArray, ylabel=L"p", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotEcc = Plots.plot(t, ecc, ylabel=L"e", xlabel=L"t\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotθ = Plots.plot(t, θmin, ylabel=L"θ_{\mathrm{min}}", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

orbitalParamsPlot=plot(title, plotE, plotL, plotC, plotP, plotEcc, plotθ, layout = @layout([A{0.01h}; [B C D]; [E F G]]), size=(1500, 600), dpi=1000)
display(orbitalParamsPlot)

savefig(fluxplot, plot_path * "test_inspiral_flux_h_$(h)_nGeo_$(nPoints).png")
savefig(orbitalParamsPlot, plot_path * "test_inspiral_orbital_constants_h_$(h)_nGeo_$(nPoints).png")


# #### COMPUTE WAVEFORM #### 
# # this takes a long time on my computer, but maybe it won't take as long on your high-tech machinery.
# Θ=0.0; Φ=0.0; obs_distance=1.0;

# waveform_fname = data_path * "kerr_RR_GW_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat_geometric)_T_$(τMax_geometric)_tol_$(kerrReltol).txt"

# # compute kerr waveform in the new kludge scheme
# @time KludgeWaveforms.NewKludge.Kerr_waveform(a, M, m, EMRI_ode_sol_fname, waveform_fname, Θ, Φ, obs_distance)
# waveform_plot_fname="kerr_RR_WF_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat_geometric)_T_$(τMax_geometric)_tol_$(kerrReltol).png"
# ylims=(-0.003, 0.003)
# GRPlotLib.plot_waveform(waveform_fname, waveform_plot_fname; plot_path=plot_path, color=:black, legend=:false, ylims=ylims) 