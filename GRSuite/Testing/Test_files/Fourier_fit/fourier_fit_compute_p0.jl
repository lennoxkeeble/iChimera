include("/home/lkeeble/GRSuite/main.jl")
@time include("/home/lkeeble/GRSuite/TestFiles/CoordinateDerivs/derivs.jl")   # ~ 3 minutes
using DelimitedFiles, Statistics, BenchmarkTools, Plots, LaTeXStrings

#### begin by computing a geodesic trajectory ####

# path for saving data and plots
data_path="/home/lkeeble/GRSuite/TestFiles/Test_results/";
plot_path="/home/lkeeble/GRSuite/TestFiles/Test_plots/";
fourier_fit_p0="/home/lkeeble/GRSuite/fourier_fit_p0/";
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

# specify params
# a = 0.8; p = 6.5; e = 0.5; θi = π/6; M=1.0; Δti=1.0; kerrReltol=1e-10; kerrAbstol=1e-10;
p=7.0; q=1e-5; e=0.6; a=0.98; θi=0.570798; kerrReltol=1e-10; kerrAbstol=1e-10; Δti=1.0

M=1.0

# calculate orbital frequencies (wrt τ, NOT t)
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi); ωr, ωθ, ωϕ = ω[1:3]; Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];

# we evolve the trajectory for a time τ = max(2π/ωi)
τmax = maximum(@. 2π/ω);

# determine sampling rate by the number of points we wish to have in our "grid"
nPoints=3000
n=1:nPoints|>collect

saveat = τmax / (nPoints-1)

# solve geodesic equation, specifically built for Kerr
@time Kerr.KerrGeodesics.compute_kerr_geodesic(a, p, e, θi, τmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

# ODE sol filename copied from output in cell above
kerr_ode_sol_fname = data_path * "ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(τmax)_tol_$(kerrReltol).txt"
sol = readdlm(kerr_ode_sol_fname)
τ = sol[1, :]; t = sol[2, :]; r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :]; tdot = sol[6, :]; rdot = sol[7, :]; θdot = sol[8, :]; ϕdot = sol[9, :]; tddot = sol[10, :]; rddot = sol[11, :]; θddot = sol[12, :]; ϕddot = sol[13, :];

# initialize data arrays
Mijk_data = [Float64[] for i=1:3, j=1:3, k=1:3]
Mij_data = [Float64[] for i=1:3, j=1:3]
Sij_data = [Float64[] for i=1:3, j=1:3]
Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
Mij2_data = [Float64[] for i=1:3, j=1:3]
Sij1_data = [Float64[] for i=1:3, j=1:3]
xBL = [Float64[] for i in 1:nPoints]
vBL = [Float64[] for i in 1:nPoints]
aBL = [Float64[] for i in 1:nPoints]
xH = [Float64[] for i in 1:nPoints]
x_H = [Float64[] for i in 1:nPoints]
vH = [Float64[] for i in 1:nPoints]
v_H = [Float64[] for i in 1:nPoints]
v = zeros(nPoints)
rH = zeros(nPoints)
aH = [Float64[] for i in 1:nPoints]
a_H = [Float64[] for i in 1:nPoints]
Mij5 = zeros(3, 3)
Mij6 = zeros(3, 3)
Mij7 = zeros(3, 3)
Mij8 = zeros(3, 3)
Mijk7 = zeros(3, 3, 3)
Mijk8 = zeros(3, 3, 3)
Sij5 = zeros(3, 3)
Sij6 = zeros(3, 3)

# convert trajectories to BL coords
@inbounds Threads.@threads for i in n
    xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
    vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]) / tdot[i];             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
    aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]) / (tdot[i]^2);      # divide by (dt/dτ)² to get accelerations wrt BL time
end
@inbounds Threads.@threads for i in n
    xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
    x_H[i] = xH[i]
    rH[i] = SelfForce.norm_3d(xH[i]);
end
@inbounds Threads.@threads for i in n
    vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
    v_H[i] = vH[i]; 
    v[i] = SelfForce.norm_3d(vH[i]);
end
@inbounds Threads.@threads for i in n
    aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
    a_H[i] = aH[i]
end

#### compute multipole moments and their derivatives ####
q=10^-5   # mass ratio

SelfForce.multipole_moments_tr!(vH, xH, x_H, q, M, Mij_data, Mijk_data, Sij_data)
SelfForce.moments_tr!(aH, a_H, vH, v_H, xH, x_H, q, M, Mij2_data, Mijk2_data, Sij1_data)

#### compute best fit parameters ####
multipoles = ["mass_q_2nd", "mass_o_2nd", "current_1st"]; 

index_pairs = [(i, j) for i=1:3, j=1:3]
n_harmonics=5
tdata = t

# compute-at index
compute_at = nPoints ÷ 2
fit_fname_params="fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(n_harmonics).txt";
@time SelfForce.moment_derivs_tr_p0!(tdata, Mij2_data, Mijk2_data, Sij1_data, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, compute_at, n_harmonics, Ωr, Ωθ, Ωϕ, fit_fname_params)

index1=1; index2=3;
multipole=multipoles[1]
n_harmonics=2    # n=6 took 61 seconds the first time
data=Mij2_data[index1, index2]
tdata = t
fit_fname_save=fourier_fit_p0 * multipole * (isequal(multipole, "mass_o") ? "_i_$(index1)_j_$(index2)_k_$(index3)_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(n_harmonics).txt" : "_i_$(index1)_j_$(index2)_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(n_harmonics).txt")
fit_fname_p0=data_path * "mass" * "_i_$(1)_j_$(1)_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(n_harmonics).txt"
coef_saved = readdlm(fit_fname_p0)
p0=coef_saved[:]

n_freqs=size(p0, 1)
# bounds 
bound_fact = 100.0
minVal = minimum(data); lbVal = minVal < 0 ? bound_fact * minVal : minVal / bound_fact
maxVal = maximum(data); ubVal = maxVal < 0 ? maxVal / bound_fact : bound_fact * maxVal
lb=lbVal * ones(2 * n_freqs)
ub=ubVal * ones(2 * n_freqs)

# initial guess
p0 = 0.5 * (lb .+ ub)

@time Ω, fit, fitted_data = FourierFit.fourier_fit(tdata, data, n_harmonics, p0)

# save fit #
open(fit_fname_save, "w") do io
    writedlm(io, coef(fit))
end

# compute percentage difference in real data and best-fit data 
deviation = @. 100 * (data-fitted_data) / data

println("Error in fit to function f")
println("Minimum deviation =$(minimum(abs.(deviation))) %")
println("Maxmium deviation =$(maximum(abs.(deviation))) %")
println("Average deviation =$(mean(abs.(deviation))) %")

## 2nd derivative ### 

isequal(multipole, "current") ? N=1 : N=2

d2f = curve_fit_functional_derivs(tdata, Ω, coef(fit), N)
deviation_deriv = @. (d2f-d2data) / d2f    ### can't compute percentage since rdot goes to zero so will get 

println("$(N)th derivative")
println("Minimum deviation =$(minimum(abs.(deviation_deriv))) %")
println("Maxmium deviation =$(maximum(abs.(deviation_deriv))) %")
println("Average deviation =$(mean(abs.(deviation_deriv))) %")

# construct subplots for different numbers of harmonics
n_1=1; n_2=6; harmonic_array=n_1:n_2|>collect
d2f=[Float64[] for i=n_1:n_2];

for i in harmonic_array
    fit_fname_p0=data_path * "mass" * "_i_$(1)_j_$(1)_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(harmonic_array[i]).txt"
    coef_saved = readdlm(fit_fname_p0)
    p0=coef_saved[:]
    tdata = t
    @time Ω, fit, fitted_data = FourierFit.fourier_fit(tdata, data, harmonic_array[i], p0)
    d2f[i] = curve_fit_functional_derivs(tdata, Ω, coef(fit), N)
end

plot_array = Any[]
ms=3
for i in harmonic_array
    # plot
    if i==5
        push!(plot_array, scatter(t, d2data, label="Analytic", xlabel=L"t", ylabel=L"\ddot{M}_{11}",
        markersize=ms,
        framestyle=:box,
        legend=:topleft,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        ylims=(1.1 * minimum(d2data), 1.1 * maximum(d2data)),
        xlims=(minimum(tdata), 1.1 * maximum(tdata))))
        scatter!(plot_array[i], t, d2f[i], markersize=ms, label="Fit")
        vline!([tdata[nPoints ÷ 2]], linestyle=:dash)  # midpoint of data points
        annotate!(plot_array[i], maximum(t)*0.9, maximum(d2data)*0.9, Plots.text(L"N=%$(harmonic_array[i])", :left, 12))
    else
        push!(plot_array, scatter(t, d2data,
        markersize=ms,
        legend=:false,
        framestyle=:box,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        ylims=(1.1 * minimum(d2data), 1.1 * maximum(d2data)),
        xlims=(minimum(tdata), 1.1 * maximum(tdata))))
        scatter!(plot_array[i], t, d2f[i], markersize=ms)
        vline!([tdata[nPoints ÷ 2]], linestyle=:dash)
        annotate!(plot_array[i], maximum(t)*0.9, maximum(d2data)*0.9, Plots.text(L"N=%$(harmonic_array[i])", :left, 12))
    end
end

plot_name=multipole * "_i_$(index1)_j_$(index2)_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(last(harmonic_array))_2.png"

fit_subplot = plot(plot_array..., layout = (3, 2), size=(1000, 1000), dpi=1000)
savefig(fit_subplot, plot_path * plot_name)

# ######## extra code #######
# if isequal(multipole, "mass")
#     data = Mij_data[index1, index2];
# elseif isequal(multipole, "current")
#     data = Sij_data[index1, index2];
# end
 
# if isfile(fit_fname)
#     coef_saved = readdlm(fit_fname)
#     p0=coef_saved[:]
# else
#     p0=Float64[]
# end