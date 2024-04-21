include("/home/lkeeble/GRSuite/main.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings, Peaks, .HJEvolution

##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results/HJE"
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
a = 0.8; p = 10.5; e = 0.5; θi = π/6; M=1.0; m=1.0; tmax=3000.0; Δti=0.01; kerrReltol=1e-12; kerrAbstol=1e-10; saveat=0.05;

##### compute geodesic for reltol=1e-12 #####
@time HJEvolution.compute_kerr_geodesic(a, p, e, θi, tmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

# load geodesic and store in array #
kerr_ode_sol_fname=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(kerrReltol).txt"
sol = readdlm(kerr_ode_sol_fname)
t = sol[1, :]; r = sol[2, :]; θ = sol[3, :]; ϕ = mod.(sol[4, :], 2π); 
t_dot = sol[5, :]; r_dot = sol[6, :]; θ_dot = sol[7, :]; ϕ_dot = sol[8, :]; 
t_ddot = sol[9, :]; r_ddot = sol[10, :]; θ_ddot = sol[11, :]; ϕ_ddot = sol[12, :];

# compute frequencies of motion
rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M); Ω=ω[1:3]/ω[4]

#### plots ####
# construct x-values for vertical lines to denote period
r_vlines = Float64[]
θ_vlines = Float64[]
ϕ_vlines = Float64[]

t_r = 2π / Ω[1]
while t_r<tmax
    append!(r_vlines, t_r)
    t_r += 2π / Ω[1]
end

t_θ = 2π / Ω[2]
while t_θ<tmax
    append!(θ_vlines, t_θ)
    t_θ += 2π / Ω[2]
end

t_ϕ = 2π / Ω[3]
while t_ϕ<tmax
    append!(ϕ_vlines, t_ϕ)
    t_ϕ += 2π / Ω[3]
end

# plot
plot(t, r, label=:false, ylabel=L"r", xlabel=L"t", color=:red, dpi=1000)
vline!(r_vlines, linestyle=:dash, label=L"k\cdot \frac{2\pi}{\Omega_{r}}", color=:blue, legend=:bottomright)
# savefig(plot_path*"r_plot_BL_time.png")

plot(t, θ, label=:false, ylabel=L"θ", xlabel=L"t", color=:red, dpi=1000)
vline!(θ_vlines, linestyle=:dash, label=L"k\cdot \frac{2\pi}{\Omega_{\theta}}", color=:blue, legend=:bottomright)
# savefig(plot_path*"theta_plot_BL_time.png")

plot(t, ϕ, label=:false, ylabel=L"\phi", xlabel=L"t", color=:red, dpi=1000)
vline!(ϕ_vlines, linestyle=:dash, label=L"k\cdot \frac{2\pi}{\Omega_{\phi}}", color=:blue, legend=:bottomright)
# savefig(plot_path*"phi_plot_BL_time.png")

# numerically compute periods

periods_r = diff(t[argmaxima(r)])
periods_θ = diff(t[argmaxima(θ)])
periods_ϕ = diff(t[argmaxima(ϕ)])

freqs_r = @. (2π / periods_r)
freqs_θ = @. (2π / periods_θ)
freqs_ϕ = @. (2π / periods_ϕ)

println("r")
freqs_r .- Ω[1]
println("θ")
freqs_θ .- Ω[2]
println("ϕ")
freqs_ϕ .- Ω[3]