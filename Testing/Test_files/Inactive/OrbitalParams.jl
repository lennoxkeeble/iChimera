# include files
# include("main.jl")
include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/GRSuite/main.jl")
using DelimitedFiles, LaTeXStrings, Plots.PlotMeasures, Plots

# path for saving data and plots
data_path="Test_results/";
plot_path="Test_plots/";
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
# nPiecewise = 100;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points

# orbit parameters - Paper Fig. 5 Example
M = 1e6 * Msol; p=7.0; q=1e-5; e=0.6; a=0.98; theta_inc_o=57.39 * π/180.0; θi=π/2 - theta_inc_o; Δti=1.0; kerrReltol=1e-10; kerrAbstol=1e-10;
nPiecewise = 100;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points

# θi=0.570798

# # orbit parameters
# M = 4.5e6 * Msol; p=8.0; q=1e-5; e=0.0; a=0.98; theta_inc_o=0.0; θi=π/2 - theta_inc_o; Δti=1.0; kerrReltol=1e-10; kerrAbstol=1e-10;
# nPiecewise = 20;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points


# conversion from t(M) -> t(s)
MtoSecs = M * Grav_Newton / c^3

# time parameters for orbit
saveat_secs = 5    # sample trajectory every X secs
saveat_geometric = saveat_secs / MtoSecs    # convert t= Xs to units of M
τMax_secs = 10^(-2) * yr    # evolve inspiral for a maximum time of X
τMax_geometric = τMax_secs / MtoSecs;   # convert inspiral evolution time to units of M

a=0.4; p=10.0; e=0.0; θi=π/2; M=1.0; m=10^-1;
τMax_geometric=10000.0
saveat_geometric=0.5

# # now convert back to geometric units
# M=1.0; m=q;

# compute inspiral
@time SelfForce.compute_inspiral!(τMax_geometric, nPiecewise, M, m, a, p, e, θi,  Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, saveat_geometric, Δti, kerrReltol, kerrAbstol, data_path=data_path)


### PLOT INSPIRAL ###

EMRI_ode_sol_fname = data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tstep_$(saveat_geometric)_tol_$(kerrReltol).txt"
EMRI_orbit_fname = "EMRI_3d_orbit_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tstep_$(saveat_geometric)_tol_$(kerrReltol).txt"
zlims = (-8, 8);    # MAY NEED TO ADAPT Z-LIMITS DEPENDING ON CHOICE OF ORBITAL PARAMS
GRPlotLib.plot_orbit(EMRI_ode_sol_fname, EMRI_orbit_fname, zlims; plot_path=plot_path)


### COMPUTE ORBITAL PARAMETERS ###

# load ODE solution
ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tstep_$(saveat_geometric)_tol_$(kerrReltol).txt"
sol = readdlm(ODE_filename)
τ = sol[1, :]; t = sol[2, :]; r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :]; tdot = sol[6, :]; rdot = sol[7, :]; θdot = sol[8, :]; ϕdot = sol[9, :]; tddot = sol[10, :]; rddot = sol[11, :]; θddot = sol[12, :]; ϕddot = sol[13, :];

# load self force data
SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tstep_$(saveat_geometric)_tol_$(kerrReltol).txt"
aSF_BL = readdlm(SF_filename)
n_OrbPoints = size(aSF_BL, 2)    # number of data points

# initial params
EEi, LLi, QQi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi)   # dimensionless constants

# save "constants" of motion
E_BL = zeros(n_OrbPoints); @views E_BL[1] = EEi; 
Edot_BL = zeros(n_OrbPoints-1);
L_BL = zeros(n_OrbPoints); @views L_BL[1] = LLi; 
Ldot_BL = zeros(n_OrbPoints-1);
C_BL = zeros(n_OrbPoints); @views C_BL[1] = QQi;    # note that C in the new kludge is equal to Schmidt's Q
Cdot_BL = zeros(n_OrbPoints-1);
Q_BL = zeros(n_OrbPoints); @views Q_BL[1] = C_BL[1] + (L_BL[1] - a * E_BL[1])^2;    # Eq. 17
Qdot_BL = zeros(n_OrbPoints-1);

# compute constants of motion using the self-force (Eqs. 30-33)
@inbounds for i=2:n_OrbPoints
    dE_dτ = - Kerr.KerrMetric.g_μν(t[i-1], r[i-1], θ[i-1], ϕ[i-1], a, M, 1, 1) * aSF_BL[1, i-1] - Kerr.KerrMetric.g_μν(t[i-1], r[i-1], θ[i-1], ϕ[i-1], a, M, 4, 1) * aSF_BL[4, i-1]    # Eq. 30
    dL_dτ = Kerr.KerrMetric.g_μν(t[i-1], r[i-1], θ[i-1], ϕ[i-1], a, M, 1, 4) * aSF_BL[1, i-1] + Kerr.KerrMetric.g_μν(t[i-1], r[i-1], θ[i-1], ϕ[i-1], a, M, 4, 4) * aSF_BL[4, i-1]    # Eq. 31
    
    dQ_dτ = 0 
    @inbounds for α=1:4, β=1:4
        dQ_dτ += 2 * Kerr.KerrMetric.ξ_μν(t[i-1], r[i-1], θ[i-1], ϕ[i-1], a, M, α, β) * (α==1 ? tdot[i-1] : α==2 ? rdot[i-1] : α==3 ? θdot[i-1] : ϕdot[i-1]) * aSF_BL[β, i-1]    # Eq. 32
    end
    
    dC_dτ = dQ_dτ + 2 * (a * E_BL[i-1] - L_BL[i-1]) * (dL_dτ - a * dE_dτ)

    # constants of motion
    @views E_BL[i] = E_BL[i-1] + dE_dτ * saveat_geometric
    @views L_BL[i] = L_BL[i-1] + dL_dτ * saveat_geometric
    @views Q_BL[i] = Q_BL[i-1] + dQ_dτ * saveat_geometric
    @views C_BL[i] = C_BL[i-1] + dC_dτ * saveat_geometric

    # fluxes
    @views Edot_BL[i-1] = dE_dτ;
    @views Ldot_BL[i-1] = dL_dτ;
    @views Qdot_BL[i-1] = dQ_dτ;
    @views Cdot_BL[i-1] = dC_dτ;
end


# computing p, e, θmin_BL. Note that this may fail, e.g., for the case where e=0 and Q goes negative (in our computation), the function will run into
# square roots of negative numbers
pArray_BL = zeros(n_OrbPoints); @views pArray_BL[1]=p;                      # semilatus rectum p(t)
ecc_BL = zeros(n_OrbPoints); @views ecc_BL[1]=e;                            # ecc_BLentricity e(t)
θmin_BL = zeros(n_OrbPoints); @views θmin_BL[1]=θi;                         # θmin_BL(t)
ι_BL = zeros(n_OrbPoints); @views ι_BL[1] = sign(L_BL[1]) * (π/2 - θmin_BL[1]);   # inclination ι_BL(t)

println("Ei = $(E_BL[1])")
println("Li = $(L_BL[1])")
println("Ci = $(C_BL[1])")
println("Qi = $(Q_BL[1])")

@inbounds for i=2:n_OrbPoints
    pArray_BL[i], ecc_BL[i], θmin_BL[i] = Kerr.ConstantsOfMotion.peθ(a, E_BL[i], L_BL[i], Q_BL[i], C_BL[i], M)
    @views ι_BL[i] = sign(L_BL[i]) * (π/2 - θmin_BL[i])
end

### PLOT FLUXES AND CONSTANTS OF MOTION ###

nPlotTime = Int(1e4 ÷ saveat_secs)
plotτFlux = τ[2:nPlotTime+1] * MtoSecs
plotτOrbParams = τ[1:nPlotTime] * MtoSecs

# margins
left_margin = 8mm; right_margin=8mm; top_margin=8mm; bottom_margin=8mm;

gr()
plotΔE = Plots.plot(plotτFlux, Edot_BL[1:nPlotTime], ylabel=L"\dot{E}", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotΔL = Plots.plot(plotτFlux, Ldot_BL[1:nPlotTime], ylabel=L"\dot{L}", xlabel=L"τ\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotΔQ = Plots.plot(plotτFlux, Qdot_BL[1:nPlotTime], ylabel=L"\dot{Q}", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
# title = plot(title ="p0=$(p)M, e0=$(e), a=$(a)M, ι_BL0=$(round(theta_inc_o*180/π; digits=4)), q=$(q)", grid = false, showaxis = false, bottom_margin = -50Plots.px)
title = plot(title ="p0=$(p)M, e0=$(e), a=$(a)M, ι_BL0=$(0), q=$(q)", grid = false, showaxis = false, bottom_margin = -50Plots.px)
fluxplot = plot(title, plotΔE, plotΔL, plotΔQ, layout = @layout([A{0.01h}; [B C D]]), size=(1500, 300))
display(fluxplot)

gr()
plotE = Plots.plot(plotτOrbParams, E_BL[1:nPlotTime], ylabel=L"E", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotL = Plots.plot(plotτOrbParams, L_BL[1:nPlotTime], ylabel=L"L", xlabel=L"τ\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotC = Plots.plot(plotτOrbParams, C_BL[1:nPlotTime], ylabel=L"C", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

orbitalParamsPlot=plot(plotE, plotL, plotC, layout=(1, 3), size=(1500, 300))
plotP = Plots.plot(plotτOrbParams, pArray_BL[1:nPlotTime], ylabel=L"p", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotEcc_BL = Plots.plot(plotτOrbParams, ecc_BL[1:nPlotTime], ylabel=L"e", xlabel=L"τ\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotι_BL = Plots.plot(plotτOrbParams, ι_BL[1:nPlotTime], ylabel=L"ι_BL", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

orbitalParamsPlot=plot(plotE, plotL, plotC, plotP, plotEcc_BL, plotι_BL, layout=(2, 3), size=(1500, 600))
display(orbitalParamsPlot)

# #### COMPUTE WAVEFORM #### 
# # this takes a long time on my computer, but maybe it won't take as long on your high-tech machinery.
# Θ=0.0; Φ=0.0; obs_distance=1.0;

# waveform_fname = data_path * "kerr_RR_GW_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat_geometric)_T_$(τMax_geometric)_tol_$(kerrReltol).txt"

# # compute kerr waveform in the new kludge scheme
# @time KludgeWaveforms.NewKludge.Kerr_waveform(a, M, m, EMRI_ode_sol_fname, waveform_fname, Θ, Φ, obs_distance)
# waveform_plot_fname="kerr_RR_WF_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat_geometric)_T_$(τMax_geometric)_tol_$(kerrReltol).png"
# ylims=(-0.003, 0.003)
# GRPlotLib.plot_waveform(waveform_fname, waveform_plot_fname; plot_path=plot_path, color=:black, legend=:false, ylims=ylims)