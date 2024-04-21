include("/home/lkeeble/GRSuite/main.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings

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
a = 0.8; p = 6.5; e = 0.5; θi = π/6; M=1.0; m=1.0; τmax=3000.0; Δti=0.1; kerrReltol=1e-12; kerrAbstol=1e-10; saveat=0.5;


##### compute geodesic for reltol=1e-12 #####
@time Kerr.KerrGeodesics.compute_kerr_geodesic(a, p, e, θi, τmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

# load geodesic and store in array #
kerr_ode_sol_fname = data_path * "ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=1))_tstep_$(saveat)_T_$(τmax)_tol_$(kerrReltol).txt"
sol = readdlm(kerr_ode_sol_fname)
τ = sol[1,:]; tt=sol[2,:]; rr=sol[3,:]; θθ=sol[4,:]; ϕϕ=sol[5,:]; ttdot=sol[6,:]; rrdot=sol[7,:]; θθdot=sol[8,:]; ϕϕdot=sol[9,:]


##### compute geodesics for lower tolerances #####
tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
# create data arrays 
t = [Float64[] for i in eachindex(tolerances)]; r = [Float64[] for i in eachindex(tolerances)];
θ = [Float64[] for i in eachindex(tolerances)]; ϕ = [Float64[] for i in eachindex(tolerances)];
tdot = [Float64[] for i in eachindex(tolerances)]; rdot = [Float64[] for i in eachindex(tolerances)];
θdot = [Float64[] for i in eachindex(tolerances)]; ϕdot = [Float64[] for i in eachindex(tolerances)];

# compute geodesics
for i in eachindex(tolerances)
    reltol=tolerances[i];
    Kerr.KerrGeodesics.compute_kerr_geodesic(a, p, e, θi, τmax, Δti, reltol, kerrAbstol, saveat, data_path=data_path)
    # load geodesic and store in array #
    kerr_ode_sol_fname = data_path * "/ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=1))_tstep_$(saveat)_T_$(τmax)_tol_$(reltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    t[i]=sol[2,:]; r[i]=sol[3,:]; θ[i]=sol[4,:]; ϕ[i]=sol[5,:]; tdot[i]=sol[6,:]; rdot[i]=sol[7,:]; θdot[i]=sol[8,:]; ϕdot[i]=sol[9,:]
end

##### compute errors #####
tError = [compute_deviation(tt, t[i]) for i in eachindex(tolerances)]; 
tdotError = [compute_deviation(ttdot, tdot[i]) for i in eachindex(tolerances)]; 
rError = [compute_deviation(rr, r[i]) for i in eachindex(tolerances)]; 
rdotError = [compute_deviation(rrdot, rdot[i]) for i in eachindex(tolerances)]; 
θError = [compute_deviation(θθ, θ[i]) for i in eachindex(tolerances)]; 
θdotError = [compute_deviation(θθdot, θdot[i]) for i in eachindex(tolerances)]; 
ϕError = [compute_deviation(ϕϕ, ϕ[i]) for i in eachindex(tolerances)]; 
ϕdotError = [compute_deviation(ϕϕdot, ϕdot[i]) for i in eachindex(tolerances)];

##### plot errors #####
gr()
# plot attributes #
shapes = [:rect :x :circle :rect]
colors = [:green :red :blue :magenta]
alpha=1.0
ms=[1.5 1.5 1.5 1.5]
left_margin=8mm; bottom_margin=8mm; right_margin=8mm; top_margin=8mm;
xtickfontsize=18;ytickfontsize=18;xguidefontsize=18;yguidefontsize=18;legendfontsize=18
ylim=(1e-12, 1e4)
xtick_vals = range(start=τ[1], stop=last(τ), length=11)|>collect;
xlim=(τ[1]-saveat, last(τ)+saveat)

# plots #
tPlot = scatter([τ for i in eachindex(tolerances)], tError, markershape=shapes, color=colors, label=[tolerances[1] tolerances[2] tolerances[3] tolerances[4]], markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)",
xlabel=L"τ", alpha=alpha, markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(minimum(τ)+0.2 * (maximum(τ)-minimum(τ)), 5e2, text(L"t", 18, :center))

# plots #
rPlot = scatter([τ for i in eachindex(tolerances)], rError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)", 
xlabel=L"τ", alpha=alpha,markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(minimum(τ)+0.2 * (maximum(τ)-minimum(τ)), 5e2, text(L"r", 18, :center))

# plots #
θPlot = scatter([τ for i in eachindex(tolerances)], θError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)",
xlabel=L"τ", alpha=alpha, markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(minimum(τ)+0.2 * (maximum(τ)-minimum(τ)), 5e2, text(L"θ", 18, :center))

# plots #
ϕPlot = scatter([τ for i in eachindex(tolerances)], ϕError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)", 
xlabel=L"τ", alpha=alpha,markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(minimum(τ)+0.2 * (maximum(τ)-minimum(τ)), 5e2, text(L"ϕ", 18, :center))

plot(tPlot, rPlot, θPlot, ϕPlot, layout=(2,2), size=(1600, 1200))
savefig(plot_path*"x_reltol_errors.png")

##### plots for derivatives #####

# plots #
tdotPlot = scatter([τ for i in eachindex(tolerances)], tdotError, markershape=shapes, color=colors, label=[tolerances[1] tolerances[2] tolerances[3] tolerances[4]], markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)",
xlabel=L"τ", alpha=alpha, markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(maximum(τ)-0.2 * (maximum(τ)-minimum(τ)), 8e-12, text(L"\dot{t}", 18, :center))

# plots #
rdotPlot = scatter([τ for i in eachindex(tolerances)], rdotError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)", 
xlabel=L"τ", alpha=alpha,markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(maximum(τ)-0.2 * (maximum(τ)-minimum(τ)), 8e-12, text(L"\dot{r}", 18, :center))

# plots #
θdotPlot = scatter([τ for i in eachindex(tolerances)], θdotError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)",
xlabel=L"τ", alpha=alpha, markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(maximum(τ)-0.2 * (maximum(τ)-minimum(τ)), 8e-12, text(L"\dot{θ}", 18, :center))

# plots #
ϕdotPlot = scatter([τ for i in eachindex(tolerances)], ϕdotError, markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, ylims=ylim, yticks=([10.0^(i) for i=(-12):4], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-12):4]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=1))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, top_margin=top_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\;(\%)", 
xlabel=L"τ", alpha=alpha,markerstrokewidth=0, xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize,
framestyle=:box)
annotate!(maximum(τ)-0.2 * (maximum(τ)-minimum(τ)), 8e-12, text(L"\dot{ϕ}", 18, :center))

plot(tdotPlot, rdotPlot, θdotPlot, ϕdotPlot, layout=(2,2), size=(1600, 1200))
savefig(plot_path*"xdot_reltol_errors.png")