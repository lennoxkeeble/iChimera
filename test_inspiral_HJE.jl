# include files
include("main.jl")
using DelimitedFiles, LaTeXStrings, Plots.PlotMeasures, Plots

##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results/HJE"
# data_path=test_results_path * "/Test_data/diff_vec_potential";
# plot_path=test_results_path * "/Test_plots/diff_vec_potential/";
# data_path=test_results_path * "/Test_data/";
# plot_path=test_results_path * "/Test_plots/";
data_path=test_results_path * "/Test_data/diff_delta_m/";
plot_path=test_results_path * "/Test_plots/diff_delta_m/";
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
M = 1e6 * Msol; p=7.0; q=1e-5; e=0.6; a=0.98; theta_inc_o=57.39 * π/180.0; θi=π/2 - theta_inc_o; Δti=1.0; kerrReltol=1e-10; kerrAbstol=1e-10;
nPoints = 1500;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points

θi=0.570798


# # calculate integrals of motion from orbital parameters
# EEi, LLi, QQi, CCi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)   
# rplus = Kerr.KerrMetric.rplus(a, M); rminus=Kerr.KerrMetric.rminus(a, M);
# # ωωω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, EEi, LLi, QQi, CCi, rplus, rminus, M)

# # ΩΩΩ = ωωω[1:3]/ωωω[4];
# # tMin = 0.5 * minimum(@. 2π/ΩΩΩ)

# Tmin = zeros(length(EE))
# @inbounds Threads.@threads for i in eachindex(EE)
#     ωωω = Kerr.ConstantsOfMotion.KerrFreqs(a, pArray[i], ecc[i], θmin[i], EE[i], LL[i], QQ[i], CC[i], rplus, rminus, M)
#     ΩΩΩ = ωωω[1:3]/ωωω[4];
#     Tmin[i] = 0.5 * minimum(@. 2π / ΩΩΩ)
# # end
# Edot[5500]
# positive_E_flux_index = Int64[]
# positive_L_flux_index = Int64[]
# positive_Q_flux_index = Int64[]
# positive_C_flux_index = Int64[]
# total_SF=0
# total_E_increase = 0
# total_L_increase = 0
# total_Q_increase=0
# total_C_increase=0
# total_p_increase = 0
# total_ecc_increase = 0
# total_ι_decrease=0
# @inbounds for i in eachindex(EE)
#     if Edot[i] != 0
#         total_SF = total_SF+1
#     end
#     if Edot[i]>0
#         append!(positive_E_flux_index, i)
#         total_E_increase = total_E_increase + 1
#     end
#     if Ldot[i]>0
#         append!(positive_L_flux_index, i)
#         total_L_increase = total_L_increase + 1
#     end
#     if Qdot[i]>0
#         append!(positive_Q_flux_index, i)
#         total_Q_increase = total_Q_increase + 1
#     end
#     if Cdot[i]>0
#         append!(positive_C_flux_index, i)
#         total_C_increase = total_C_increase + 1
#     end
#     if i!=1
#         if pArray[i] - pArray[i-1] > 0
#             total_p_increase = total_p_increase +1
#         end
#         if ecc[i] - ecc[i-1] > 0
#             total_ecc_increase = total_ecc_increase +1
#         end
#         if ι[i] - ι[i-1] < 0
#             total_ι_decrease = total_ι_decrease +1
#         end
#     end
# end
# η(q::Float64) = q/((1+q)^2)   # q = mass ratio
# μ(q::Float64) = 
# η_to_μ = 

# maximum(@. Edot[positive_E_flux_index])
# 100 * length(positive_E_flux_index) / total_SF 

# total_E_increase
# total_L_increase
# total_Q_increase
# total_C_increase
# total_p_increase
# total_ecc_increase
# total_ι_decrease
# # length(Edot)
# positive_E_flux_index
# positive_L_flux_index
# positive_Q_flux_index
# positive_C_flux_index
# test = Kerr.ConstantsOfMotion.KerrFreqs(a, pArray[1], ecc[1], θmin[1], EE[1], LL[1], QQ[1], CC[1], rplus, rminus, M)
# # test

# minimum(diff(Tmin))
# # Tmin[1]

# # last(Tmin)

# conversion from t(M) -> t(s)
MtoSecs = M * Grav_Newton / c^3

t_max = (10^-2) * yr


# now convert back to geometric units
M=1.0; m=q;

nHarm=4    # number of harmonics

# # compute inspiral
@time SelfForce.compute_inspiral_HJE!(t_max, nPoints, M, m, a, p, e, θi,  Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, nHarm, kerrReltol, kerrAbstol, data_path=data_path)

# 24 mins, to run 3.7 days, or yr/100 w 2 harmonics

# # plot orbit solution
# EMRI_ode_sol_fname = data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(kerrReltol)_nHarm_$(nHarm)_Mij_Sij.txt"
# EMRI_orbit_fname = "EMRI_3d_orbit_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(kerrReltol)_nHarm_$(nHarm)_Mij_Sij.txt"
# zlims = (-30, 30);
# GRPlotLib.plot_orbit(EMRI_ode_sol_fname, EMRI_orbit_fname, zlims, plot_path=plot_path)

# # plot orbit projected onto x-y plane
# xlims = (-11, 11)
# ylims =  (-11, 11)
# EMRI_xy_fname = "EMRI_xy_orbit__a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(kerrReltol).png"
# GRPlotLib.plot_xy_orbit(EMRI_ode_sol_fname, EMRI_xy_fname, xlims, ylims, plot_path=plot_path,
# xlims=(-20, 20), xticks=(-20:10:20, ["-20" "-10" "0" "10" "20"]),
# ylims=(-20, 20), yticks=(-20:10:20, ["-20" "-10" "0" "10" "20"]),
# size=(500, 500), color=:black, linewidth=0.5,
# legend=false, grid=false, framestyle=:box)

#### parameters ###

# load ODE solution
ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(kerrReltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
sol = readdlm(ODE_filename)
t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; dr_dt=sol[5,:]; dθ_dt=sol[6,:]; dϕ_dt=sol[7,:]; d2r_dt2=sol[8,:]; d2θ_dt2=sol[9,:]; d2ϕ_dt2=sol[10,:]; dt_dτ=sol[11,:]

constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(kerrReltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
constants=readdlm(constants_filename)
constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tol_$(kerrReltol)_nHarm_$(nHarm)_n_fit_$(nPoints).txt"
constants_derivs = readdlm(constants_derivs_filename)

EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :]
Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]

ι = @. acos(LL / sqrt(LL^2 + CC))

### PLOT FLUXES AND CONSTANTS OF MOTION ###

Edot_nonzero=[]

for i in eachindex(Edot)
    if Edot[i] ≠ 0
        append!(Edot_nonzero, Edot[i])
    end
end

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
title = plot(title =L"p_{0}=%$(p)M,\,e_{0}=%$(e),\,a=%$(a)M,\,ι_{0}=%$(θi),\,q=%$(q),\,n_{\mathrm{geo}}=%$(nPoints),\,n_{\mathrm{fit}}=%$(nPoints),\,N=%$(nHarm)", grid = false, showaxis = false, bottom_margin = -50Plots.px)
fluxplot = plot(title, plotΔE, plotΔL, plotΔQ, layout = @layout([A{0.01h}; [B C D]]), size=(1500, 300), dpi=1000)
display(fluxplot)

gr()
plotE = Plots.plot(t, 1e4 * (EE .- EE[1]), ylabel=L"E", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
# ylims=(-3, 0.5)
# yticks!(plotE, [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5])

plotL = Plots.plot(t, LL, ylabel=L"L", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
# ylims=(1.733, 1.735)
# yticks!(plotL, [1.733, 1.7335, 1.734, 1.7345, 1.735])

plotC = Plots.plot(t, CC, ylabel=L"C", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
# ylims=(7.345, 7.354)
# yticks!(plotC, [7.346, 7.348, 7.35, 7.352, 7.354])

# orbitalParamsPlot=plot(plotE, plotL, plotC, layout=(1, 3), size=(1500, 300))
plotP = Plots.plot(t, pArray, ylabel=L"p", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
# ylims=(6.985, 7.0)
# yticks!(plotP, [6.985, 6.99, 6.995, 7.0])

plotEcc = Plots.plot(t, ecc, ylabel=L"e", xlabel=L"t\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
# ylims=(0.5985, 0.6)
# yticks!(plotEcc, [0.5985, 0.599, 0.5995, 0.6])

plotι = Plots.plot(t, 1e4 * (ι .- ι[1]), ylabel=L"\iota", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
#  ylims=(0, 2)
# yticks!(plotι,[0, 0.5, 1.0, 1.5, 2] )

orbitalParamsPlot=plot(title, plotE, plotL, plotC, plotP, plotEcc, plotι, layout = @layout([A{0.01h}; [B C D]; [E F G]]), size=(1500, 600), dpi=1000)
display(orbitalParamsPlot)

savefig(fluxplot, plot_path * "test_inspiral_flux_nHarm_$(nHarm)_nGeo_$(nPoints)_nFit_$(nPoints).png")
savefig(orbitalParamsPlot, plot_path * "test_inspiral_orbital_constants_nHarm_$(nHarm)_nGeo_$(nPoints)_nFit_$(nPoints)_natural_ranges.png")


gr()
plotE = Plots.plot(t, 1e4 * (EE .- EE[1]), ylabel=L"E", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin, 
    ylims=(-3, 0.5))
yticks!(plotE, [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5])

plotL = Plots.plot(t, LL, ylabel=L"L", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin,
    ylims=(1.733, 1.735))

yticks!(plotL, [1.733, 1.7335, 1.734, 1.7345, 1.735])

plotC = Plots.plot(t, CC, ylabel=L"C", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin, 
    ylims=(7.345, 7.354))
yticks!(plotC, [7.346, 7.348, 7.35, 7.352, 7.354])

# orbitalParamsPlot=plot(plotE, plotL, plotC, layout=(1, 3), size=(1500, 300))
plotP = Plots.plot(t, pArray, ylabel=L"p", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin, 
    ylims=(6.985, 7.0))
yticks!(plotP, [6.985, 6.99, 6.995, 7.0])

plotEcc = Plots.plot(t, ecc, ylabel=L"e", xlabel=L"t\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin,
    ylims=(0.5985, 0.6))
yticks!(plotEcc, [0.5985, 0.599, 0.5995, 0.6])

plotι = Plots.plot(t, 1e4 * (ι .- ι[1]), ylabel=L"\iota", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin,
    ylims=(0, 2))
yticks!(plotι,[0, 0.5, 1.0, 1.5, 2] )

orbitalParamsPlot=plot(title, plotE, plotL, plotC, plotP, plotEcc, plotι, layout = @layout([A{0.01h}; [B C D]; [E F G]]), size=(1500, 600), dpi=1000)
display(orbitalParamsPlot)

savefig(orbitalParamsPlot, plot_path * "test_inspiral_orbital_constants_nHarm_$(nHarm)_nGeo_$(nPoints)_nFit_$(nPoints)_NK_ranges.png")

max_time_index = 53000

gr()
plotE = Plots.plot(t[1:max_time_index], EE[1:max_time_index], ylabel=L"E", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin, 
    xlims=(0, 1e4),)

plotL = Plots.plot(t[1:max_time_index], LL[1:max_time_index], ylabel=L"L", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin,
    xlims=(0, 1e4))



plotC = Plots.plot(t[1:max_time_index], CC[1:max_time_index], ylabel=L"C", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin, 
    xlims=(0, 1e4))


# orbitalParamsPlot=plot(plotE, plotL, plotC, layout=(1, 3), size=(1500, 300))
plotP = Plots.plot(t[1:max_time_index], pArray[1:max_time_index], ylabel=L"p", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin, 
    xlims=(0, 1e4))


    
plotEcc = Plots.plot(t[1:max_time_index], ecc[1:max_time_index], ylabel=L"e", xlabel=L"t\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin,
    xlims=(0, 1e4))


plotι = Plots.plot(t[1:max_time_index], ι[1:max_time_index], ylabel=L"\iota", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin,
    xlims=(0, 1e4))

orbitalParamsPlot=plot(title, plotE, plotL, plotC, plotP, plotEcc, plotι, layout = @layout([A{0.01h}; [B C D]; [E F G]]), size=(1500, 600), dpi=1000)
display(orbitalParamsPlot)

savefig(orbitalParamsPlot, plot_path * "test_inspiral_orbital_constants_nHarm_$(nHarm)_nGeo_$(nPoints)_nFit_$(nPoints)_short_t_range.png")