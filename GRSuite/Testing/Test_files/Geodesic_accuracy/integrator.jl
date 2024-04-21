include("/home/lkeeble/GRSuite/main.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings, DifferentialEquations, StaticArrays, .Kerr

##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results"
data_path=test_results_path * "/Test_data/";
plot_path=test_results_path * "/Test_plots/";

##### define function to compute percentage difference #####
# returns elements-wise percentage deviation between two arrays
function compute_deviation(y_true::Vector, y_approx::Vector)
    return @. abs(100 * (y_true-y_approx)/y_true)
end

function compute_geodesic(a::Float64, p::Float64, e::Float64, θi::Float64, integrator, τmax::Float64=3000.0, Δti::Float64=0.1, reltol::Float64=1e-16, abstol::Float64=1e-16, saveat::Float64=0.5; data_path::String="Results/")
    # define periastron and apastron
    rp = p * M / (1 + e);   # Eq. 6.1
    ra = p * M / (1 - e);   # Eq. 6.1

    # calculate integrals of motion from orbital parameters
    E, L, Q = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi)   # dimensionless constants

    # initial conditions for Kerr geodesic trajectory
    ri = ra; τspan = (0.0, τmax); params = @SArray [a, M];
    τ = 0:saveat:τmax |> collect

    ics = Kerr.KerrGeodesics.boundKerr_ics(a, M, E, L, ri, θi, Kerr.KerrMetric.g_tt,  Kerr.KerrMetric.g_tϕ,  Kerr.KerrMetric.g_rr, Kerr.KerrMetric.g_θθ, Kerr.KerrMetric.g_ϕϕ);
    prob = SecondOrderODEProblem(Kerr.KerrGeodesics.geodesicEq, ics..., τspan, params);
    sol = solve(prob, integrator, adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=τ);
    return sol
end

#=

    We would like to check how the solutions change with decreasing tolerance. We shall fix the absolute tolerance to be 1e-10,
    and vary the relative tolerance. We will copmute a geodesic for reltol=1e-10, and take this to be the "true" one. That is,
    we will compute the errors in the geodesics for reltol<1e-10, with respect to the numerical solution at reltol=1e-10.

=#


##### compute geodesics for lower tolerances #####

##### specify geodesic parameters #####
a = 0.8; p = 6.5; e = 0.5; θi = π/6; M=1.0; m=1.0; τmax=3000.0; Δti=0.1; kerrReltol=1e-12; kerrAbstol=1e-10; saveat=0.5;

tolerances = [1e-4, 1e-8, 1e-12]
integrators = [AutoTsit5(RK4()), AutoTsit5(DP8()), AutoTsit5(OwrenZen5())]; integrator_names=["RK4" "DP8" "OwrenZen5"];

# create data arrays 
t = [[[] for i in eachindex(tolerances)] for j in integrators]; r = [[[] for i in eachindex(tolerances)] for j in integrators];
θ = [[[] for i in eachindex(tolerances)] for j in integrators]; ϕ = [[[] for i in eachindex(tolerances)] for j in integrators];
tdot = [[[] for i in eachindex(tolerances)] for j in integrators]; rdot = [[[] for i in eachindex(tolerances)] for j in integrators];
θdot = [[[] for i in eachindex(tolerances)] for j in integrators]; ϕdot = [[[] for i in eachindex(tolerances)] for j in integrators];

# compute geodesics
for i in eachindex(integrators)
    for j in eachindex(tolerances)
        integrator = integrators[i]
        reltol=tolerances[j]
        sol = compute_geodesic(a, p, e, θi, integrator, τmax, Δti, reltol, kerrAbstol, saveat, data_path=data_path)
        t[i][j]=sol[5,:]; r[i][j]=sol[6,:]; θ[i][j]=sol[7,:]; ϕ[i][j]=sol[8,:]; tdot[i][j]=sol[1,:]; rdot[i][j]=sol[2,:]; θdot[i][j]=sol[3,:]; ϕdot[i][j]=sol[4,:]
    end
end

##### compute errors at reltol=1e-12 relative to DP8 #####
tError = [compute_deviation(t[2][1], t[i][1]) for i in [1, 3]];
tdotError = [compute_deviation(tdot[2][1], tdot[i][1]) for i in [1, 3]]; 
rError = [compute_deviation(r[2][1], r[i][1]) for i in [1, 3]]; 
rdotError = [compute_deviation(rdot[2][1], rdot[i][1]) for i in [1, 3]]; 
θError = [compute_deviation(θ[2][1], θ[i][1]) for i in [1, 3]]; 
θdotError = [compute_deviation(θdot[2][1], θdot[i][1]) for i in [1, 3]]; 
ϕError = [compute_deviation(ϕ[2][1], ϕ[i][1]) for i in [1, 3]]; 
ϕdotError = [compute_deviation(ϕdot[2][1], ϕdot[i][1]) for i in [1, 3]];


##### plot errors #####
gr()
# plot attributes #
shapes = [:rect :x :circle :rect]
colors = [:green :red :blue :magenta]
alpha=1.0
ms=[1.5 1.5]
left_margin=8mm; bottom_margin=8mm; right_margin=8mm; top_margin=8mm;
xtickfontsize=18;ytickfontsize=18;xguidefontsize=18;yguidefontsize=18;legendfontsize=18
ylim=(1e-12, 1e4)
xtick_vals = range(start=τ[1], stop=last(τ), length=11)|>collect;
xlim=(τ[1]-saveat, last(τ)+saveat)

# plots #
tPlot = scatter([t[2][3], t[2][3], t[2][3]], tError, markershape=shapes, color=colors, label=[integrator_names[1], integrator_names[3]], markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)",
xlabel=L"τ", alpha=alpha, markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(minimum(τ)+0.2 * (maximum(τ)-minimum(τ)), 5e2, text(L"t", 18, :center))

# plots #
rPlot = scatter([t[2][3], t[2][3], t[2][3]], rError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)", 
xlabel=L"τ", alpha=alpha,markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(minimum(τ)+0.2 * (maximum(τ)-minimum(τ)), 5e2, text(L"r", 18, :center))

# plots #
θPlot = scatter([t[2][3], t[2][3], t[2][3]], θError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)",
xlabel=L"τ", alpha=alpha, markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(minimum(τ)+0.2 * (maximum(τ)-minimum(τ)), 5e2, text(L"θ", 18, :center))

# plots #
ϕPlot = scatter([t[2][3], t[2][3], t[2][3]], ϕError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)", 
xlabel=L"τ", alpha=alpha,markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(minimum(τ)+0.2 * (maximum(τ)-minimum(τ)), 5e2, text(L"ϕ", 18, :center))

plot(tPlot, rPlot, θPlot, ϕPlot, layout=(2,2), size=(1600, 1200))
savefig(plot_path*"x_integrator_errors.png")



##### plots for derivatives #####

# plots #
tdotPlot = scatter([t[2][3], t[2][3], t[2][3]], tdotError, markershape=shapes, color=colors, label=[integrator_names[1], integrator_names[3]], markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)",
xlabel=L"τ", alpha=alpha, markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(maximum(τ)-0.2 * (maximum(τ)-minimum(τ)), 8e-12, text(L"\dot{t}", 18, :center))

# plots #
rdotPlot = scatter([t[2][3], t[2][3], t[2][3]], rdotError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)", 
xlabel=L"τ", alpha=alpha,markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(maximum(τ)-0.2 * (maximum(τ)-minimum(τ)), 8e-12, text(L"\dot{r}", 18, :center))

# plots #
θdotPlot = scatter([t[2][3], t[2][3], t[2][3]], θdotError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)",
xlabel=L"τ", alpha=alpha, markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(maximum(τ)-0.2 * (maximum(τ)-minimum(τ)), 8e-12, text(L"\dot{θ}", 18, :center))

# plots #
ϕdotPlot = scatter([t[2][3], t[2][3], t[2][3]], ϕdotError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)", 
xlabel=L"τ", alpha=alpha,markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(maximum(τ)-0.2 * (maximum(τ)-minimum(τ)), 8e-12, text(L"\dot{ϕ}", 18, :center))

plot(tdotPlot, rdotPlot, θdotPlot, ϕdotPlot, layout=(2,2), size=(1600, 1200))
savefig(plot_path*"xdot_integrator_errors.png")



a = 0.95; p = 7.0; e = 0.0; θmin = 60.43 * π/180;
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θmin); Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];
println("Ωr = $(Ωr)")
println("Ωθ = $(Ωθ)")
println("Ωφ = $(Ωϕ)")