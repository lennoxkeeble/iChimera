include("/home/lkeeble/GRSuite/main.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings, Peaks

##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results"
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
a = 0.8; p = 10.5; e = 0.5; θi = π/6; M=1.0; m=1.0; τmax=3000.0; Δti=0.01; kerrReltol=1e-12; kerrAbstol=1e-10; saveat=0.05;


##### compute geodesic for reltol=1e-12 #####
@time Kerr.KerrGeodesics.compute_kerr_geodesic(a, p, e, θi, τmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

# load geodesic and store in array #
kerr_ode_sol_fname=data_path * "ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(τmax)_tol_$(kerrReltol).txt"
sol = readdlm(kerr_ode_sol_fname)
τ = sol[1,:]; tt=sol[2,:]; rr=sol[3,:]; θθ=sol[4,:]; ϕϕ=mod.(sol[5,:], 2π); ttdot=sol[6,:]; rrdot=sol[7,:]; θθdot=sol[8,:]; ϕϕdot=sol[9,:]
plot(tt, θθ)

cosψ = @. (p * M / rr - 1.0) / e; cos2χ = @. (cos(θθ) / cos(θi))^2
plot(τ, cosψ)
plot(τ, cos2χ)


rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
EE, LL, QQ = Kerr.ConstantsOfMotion.SchmidtELQ(a, p, e, θi)
p2, e2, θi2 = Kerr.ConstantsOfMotion.peθ(a, E, L, Q, C, M)
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M); Ω=ω[1:3]/ω[4]

E, L, C

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