include("/home/lkeeble/GRSuite/main.jl")
include("/home/lkeeble/GRSuite/TestFiles/curve_fit.jl")
@time include("/home/lkeeble/GRSuite/TestFiles/CoordinateDerivs/derivs.jl")   # ~ 3 minutes
using DelimitedFiles, Statistics, BenchmarkTools, Plots, LaTeXStrings

#### begin by computing a geodesic trajectory ####

# path for saving data and plots
data_path="/home/lkeeble/GRSuite/TestFiles/Test_results/";
plot_path="/home/lkeeble/GRSuite/TestFiles/Test_plots/";
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
a = 0.8; p = 6.5; e = 0.5; θi = π/6; M=1.0; Δti=1.0; kerrReltol=1e-10; kerrAbstol=1e-10;

# calculate orbital frequencies (wrt τ, NOT t)
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi); ωr, ωθ, ωϕ = ω[1:3]; Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];

# we evolve the trajectory for a time τ = max(2π/ωi)
τmax = maximum(@. 2π/ω);

# determine sampling rate by the number of points we wish to have in our "grid"
nPoints=200
n=1:nPoints|>collect

saveat = τmax / (nPoints-1)

# solve geodesic equation, specifically built for Kerr
@time Kerr.KerrGeodesics.compute_kerr_geodesic(a, p, e, θi, τmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

# ODE sol filename copied from output in cell above
kerr_ode_sol_fname = data_path * "ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(τmax)_tol_$(kerrReltol).txt"
sol = readdlm(kerr_ode_sol_fname)
τ = sol[1, :]; t = sol[2, :]; r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :]; tdot = sol[6, :]; rdot = sol[7, :]; θdot = sol[8, :]; ϕdot = sol[9, :]; tddot = sol[10, :]; rddot = sol[11, :]; θddot = sol[12, :]; ϕddot = sol[13, :];

# initialize data arrays
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
Mij5 = zeros(3, 3, nPoints)
Mij6 = zeros(3, 3, nPoints)
Mij7 = zeros(3, 3, nPoints)
Mij8 = zeros(3, 3, nPoints)
Mijk7 = zeros(3, 3, 3, nPoints)
Mijk8 = zeros(3, 3, 3, nPoints)
Sij5 = zeros(3, 3, nPoints)
Sij6 = zeros(3, 3, nPoints)

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

#### compute multipole moments ####

q=10^-5
# calculate ddotMijk, ddotMijk, dotSij "analytically"
SelfForce.moments_tr!(aH, a_H, vH, v_H, xH, x_H, q, M, Mij2_data, Mijk2_data, Sij1_data)


data = Mij2_data[1, 1]
# data = Sij1_data[1, 1]
# data=r
# tdata = t .- t[nPoints ÷ 2]
tdata = t
f_container=zeros(size(t, 1))    # container for curve_fit_functional to store ydata

# implement method used in Chimera code (any more than 3 harmonics takes an unfeasibly long time)
n_harmonics = 4
Ω = Float64[]
@inbounds for i_r in 0:n_harmonics
    @inbounds for i_θ in -i_r:n_harmonics
        @inbounds for i_ϕ in -(i_r+i_θ):n_harmonics
            append!(Ω, i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ)
        end
    end
end
n_freqs = size(Ω, 1)

#### carry out fit ####

# bounds 
bound_fact = 100.0
minVal = minimum(data); lbVal = minVal < 0 ? bound_fact * minVal : minVal / bound_fact
maxVal = maximum(data); ubVal = maxVal < 0 ? maxVal / bound_fact : bound_fact * maxVal
lb=lbVal * ones(2 * n_freqs)
ub=ubVal * ones(2 * n_freqs)

# initial guess
p0 = 0.5 * (lb .+ ub)
fit_fname=data_path * "fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(n_harmonics).txt"
# coef_saved = readdlm(fit_fname)
# p0=coef_saved[:]
# compute best fit parameters
# data_fit = ls_fit(tdata, data, Ω, lb, ub, p0)
@time data_fit = ls_fit(f_container, tdata, data, Ω, p0, n_freqs)

# save fit
open(fit_fname, "w") do io
    writedlm(io, coef(data_fit))
end

# compute ydata with best fit parameters
data_fitted = curve_fit_functional(f_container, tdata, Ω, coef(data_fit), n_freqs)

# compute percentage difference in real data and best-fit data 
deviation = @. 100 * (data-data_fitted) / data

println("Error in fit to function f")
println("Minimum deviation =$(minimum(abs.(deviation))) %")
println("Maxmium deviation =$(maximum(abs.(deviation))) %")
println("Average deviation =$(mean(abs.(deviation))) %")

#### compute derivatives ####
SinTheta=sin.(θ); Sin2Theta=sin.(2.0*θ); Sin3Theta=sin.(3.0*θ); Sin4Theta=sin.(4.0 * θ); CosTheta=cos.(θ); Cos2Theta=cos.(2.0*θ); Cos3Theta=cos.(3.0*θ); Cos4Theta=cos.(4.0*θ); TanTheta=tan.(θ); CscTheta=csc.(θ); SecTheta=sec.(θ); CotTheta=cot.(θ);
x = [[t[i], r[i], θ[i], ϕ[i]] for i=1:nPoints]
dx = [[tdot[i], rdot[i], θdot[i], ϕdot[i]] for i=1:nPoints]
dx2_geodesic=[[tddot[i], rddot[i], θddot[i], ϕddot[i]] for i=1:nPoints]
dx2_analytic = d2x.(dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
dx3_analytic = d3x.(dx2_analytic, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
dx4_analytic = d4x.(dx3_analytic, dx2_analytic, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)

# compile d5x #
@time d5x(dx4_analytic[1], dx3_analytic[1], dx2_analytic[1], dx[1], x[1], SinTheta[1], Sin2Theta[1], Sin3Theta[1], Sin4Theta[1], CosTheta[1], Cos2Theta[1], Cos3Theta[1], Cos4Theta[1], TanTheta[1], CscTheta[1], SecTheta[1], CotTheta[1], a)

@benchmark d5x(dx4_analytic[1], dx3_analytic[1], dx2_analytic[1], dx[1], x[1], SinTheta[1], Sin2Theta[1], Sin3Theta[1], Sin4Theta[1], CosTheta[1], Cos2Theta[1], Cos3Theta[1], Cos4Theta[1], TanTheta[1], CscTheta[1], SecTheta[1], CotTheta[1], a)

# compute 5th derivatives 
@time dx5_analytic = d5x.(dx4_analytic, dx3_analytic, dx2_analytic, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
## 1st derivative ###

N=1

df = curve_fit_functional_derivs(tdata, Ω, coef(data_fit), N)
deviation_deriv = @. (df-rdot) / df    ### can't compute percentage since rdot goes to zero so will get 

println("$(N)th derivative")
println("Minimum deviation =$(minimum(abs.(deviation_deriv))) %")
println("Maxmium deviation =$(maximum(abs.(deviation_deriv))) %")
println("Average deviation =$(mean(abs.(deviation_deriv))) %")


N_1_plot=scatter(t, rdot, label="Geodesic", xlabel=L"t", ylabel=L"\dot{r}",
framestyle=:box,
legend=:true,
foreground_color_legend = nothing,
background_color_legend = nothing)
scatter!(t, df, label="Fourier Fit")
annotate!(maximum(t)*0.9, minimum(rdot)*0.9, Plots.text(L"N=%$(n_harmonics)", :left, 12))

## 2nd derivative ### 

N=2

d2f = curve_fit_functional_derivs(tdata, Ω, coef(data_fit), N)
deviation_deriv = @. (d2f-rddot) / d2f    ### can't compute percentage since rdot goes to zero so will get 

println("$(N)th derivative")
println("Minimum deviation =$(minimum(abs.(deviation_deriv))) %")
println("Maxmium deviation =$(maximum(abs.(deviation_deriv))) %")
println("Average deviation =$(mean(abs.(deviation_deriv))) %")


N_2_plot=scatter(t, rddot, label="Geodesic", xlabel=L"t", ylabel=L"\ddot{r}",
framestyle=:box,
legend=:true,
foreground_color_legend = nothing,
background_color_legend = nothing)
scatter!(t, d2f, label="Fit")
annotate!(maximum(t)*0.9, minimum(rddot)*0.9, Plots.text(L"N=%$(n_harmonics)", :left, 12))



## 3nd derivative ### 

N=3
dr3 = [du3[2] for du3 in dx3_analytic]

d3f = curve_fit_functional_derivs(tdata, Ω, coef(data_fit), N)
deviation_deriv = @. (d3f-dr3) / d3f    ### can't compute percentage since rdot goes to zero so will get 

println("$(N)th derivative")
println("Minimum deviation =$(minimum(abs.(deviation_deriv))) %")
println("Maxmium deviation =$(maximum(abs.(deviation_deriv))) %")
println("Average deviation =$(mean(abs.(deviation_deriv))) %")


N_3_plot=scatter(t, dr3, label="Analytic Order-$(N) Derivative", xlabel=L"t", ylabel=L"\ddot{r}",
framestyle=:box,
legend=:topright,
foreground_color_legend = nothing,
background_color_legend = nothing)
scatter!(t, d3f, label="Fit")
annotate!(maximum(t)*0.9, minimum(dr3)*0.9, Plots.text(L"N=%$(n_harmonics)", :left, 12))


## 4nd derivative ### 

N=4
dr4 = [du4[2] for du4 in dx4_analytic]

d4f = curve_fit_functional_derivs(tdata, Ω, coef(data_fit), N)
deviation_deriv = @. (d4f-dr4) / d4f    ### can't compute percentage since rdot goes to zero so will get 

println("$(N)th derivative")
println("Minimum deviation =$(minimum(abs.(deviation_deriv))) %")
println("Maxmium deviation =$(maximum(abs.(deviation_deriv))) %")
println("Average deviation =$(mean(abs.(deviation_deriv))) %")


N_4_plot=scatter(t, dr4, label="Analytic Order-$(N) Derivative", xlabel=L"t", ylabel=L"\ddot{r}",
framestyle=:box,
legend=:topright,
foreground_color_legend = nothing,
background_color_legend = nothing)
scatter!(t, d4f, label="Fit")
annotate!(maximum(t)*0.9, minimum(dr4)*0.9, Plots.text(L"N=%$(n_harmonics)", :left, 12))


## 5nd derivative ### 

N=5
dr5 = [du5[2] for du5 in dx5_analytic]

d5f = curve_fit_functional_derivs(tdata, Ω, coef(data_fit), N)
deviation_deriv = @. (d5f-dr5) / d5f    ### can't compute percentage since rdot goes to zero so will get 

println("$(N)th derivative")
println("Minimum deviation =$(minimum(abs.(deviation_deriv))) %")
println("Maxmium deviation =$(maximum(abs.(deviation_deriv))) %")
println("Average deviation =$(mean(abs.(deviation_deriv))) %")


N_5_plot=scatter(t, dr5, label="Analytic Order-$(N) Derivative", xlabel=L"t", ylabel=L"\ddot{r}",
framestyle=:box,
legend=:topright,
foreground_color_legend = nothing,
background_color_legend = nothing)
scatter!(t, d5f, label="Fit")
annotate!(maximum(t)*0.9, minimum(dr5)*0.9, Plots.text(L"N=%$(n_harmonics)", :left, 12))