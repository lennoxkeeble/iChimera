include("/home/lkeeble/GRSuite/main.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings, Peaks, .MinoEvolution

##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results/Mino"
data_path=test_results_path * "/Test_data/";
plot_path=test_results_path * "/Test_plots/";


##### define function to compute percentage difference #####
# returns elements-wise percentage deviation between two arrays
function compute_deviation(y_true::Vector{Float64}, y_approx::Vector{Float64})
    return @. abs(100 * (y_true-y_approx)/y_true)
end

#=

    We would like to check how the solutions change with decreasing tolerance. We shall fix the absolute tolerance to be 1e-10,
    and vary the relative tolerance. We will copmute a geodesic for reltol=1e-10, and take this to be the "true" one. That is,
    we will compute the errors in the geodesics for reltol<1e-10, with respect to the numerical solution at reltol=1e-10.

=#

##### specify geodesic parameters #####
a = 0.8; p = 10.5; e = 0.5; θi = π/6; M=1.0; m=1.0; λmax=3000.0; Δλi=0.1; kerrReltol=1e-12; kerrAbstol=1e-10; saveat=0.005;

##### compute geodesic for reltol=1e-12 #####
@time MinoEvolution.compute_kerr_geodesic(a, p, e, θi, λmax, Δλi, kerrReltol, kerrAbstol, saveat, data_path=data_path)

# load geodesic and store in array #
kerr_ode_sol_fname=data_path * "Mino_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(λmax)_tol_$(kerrReltol).txt"
sol = readdlm(kerr_ode_sol_fname)
λ=sol[1,:]; t=sol[2,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dt_dλ=sol[6,:]; dt_dτ=sol[7,:]; dr_dτ=sol[8,:];
dθ_dτ=sol[9,:]; dϕ_dτ=sol[10,:]; d2t_dτ=sol[11,:]; d2r_dτ=sol[12,:]; d2θ_dτ=sol[13,:]; d2ϕ_dτ=sol[14,:]

# plot(λ[1:5000], mod(chi[1:5000], 2π))
##### compute fundamental frequencies #####
rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);

#### plots ####
# construct x-values for vertical lines to denote period
x_lims = (0., 10.); λmax=x_lims[2];
r_vlines = Float64[]
θ_vlines = Float64[]
ϕ_vlines = Float64[]

λ_r = 2π / ω[1]
while λ_r<λmax
    append!(r_vlines, λ_r)
    λ_r += 2π / ω[1]
end

λ_θ = 2π / ω[2]
while λ_θ<λmax
    append!(θ_vlines, λ_θ)
    λ_θ += 2π / ω[2]
end

λ_ϕ = 2π / ω[3]
while λ_ϕ<λmax
    append!(ϕ_vlines, λ_ϕ)
    λ_ϕ += 2π / ω[3]
end

# plot
plot(λ, r, label=:false, ylabel=L"r", xlabel=L"λ", color=:red, dpi=1000, xlims = x_lims)
vline!(r_vlines, linestyle=:dash, label=L"k\cdot \frac{2\pi}{\omega_{r}}", color=:blue, legend=:bottomright)
savefig(plot_path*"r_plot_mino_time.png")

plot(λ, θ, label=:false, ylabel=L"θ", xlabel=L"λ", color=:red, dpi=1000, xlims = x_lims)
vline!(θ_vlines, linestyle=:dash, label=L"k\cdot \frac{2\pi}{\omega_{\theta}}", color=:blue, legend=:bottomright)
savefig(plot_path*"theta_plot_mino_time.png")

plot(λ, ϕ, label=:false, ylabel=L"\phi", xlabel=L"λ", color=:red, dpi=1000, xlims = x_lims)
vline!(ϕ_vlines, linestyle=:dash, label=L"k\cdot \frac{2\pi}{\omega_{\phi}}", color=:blue, legend=:bottomright)
savefig(plot_path*"phi_plot_mino_time.png")

periods_r = diff(tt[argmaxima(rr)])
periods_θ = diff(tt[argmaxima(θθ)])
periods_ϕ = diff(tt[argmaxima(ϕϕ)])

freqs_r = @. (2π / periods_r)
freqs_θ = @. (2π / periods_θ)
freqs_ϕ = @. (2π / periods_ϕ)

println("r")
freqs_r .- Ω[1]
println("θ")
freqs_θ .- Ω[2]
println("ϕ")
freqs_ϕ .- Ω[3]

println(Ω)
freqs_θ

plot(τ, θθ)

println(sol[4,:])