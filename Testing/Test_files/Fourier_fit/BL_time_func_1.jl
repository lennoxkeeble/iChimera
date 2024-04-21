include("/home/lkeeble/GRSuite/main.jl");
include("/home/lkeeble/GRSuite/Testing/BL_time_derivs/Test_functions.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings, Peaks, .HJEvolution, .FourierFitGSL, JLD2, FileIO
using .Deriv2, .Deriv3, .Deriv4, .Deriv5, .Deriv6, .HJEvolution, .Function_1, .Function_2


##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results/BLtime/func1"
data_path=test_results_path * "/Test_data/trange_0.5_min/";
plot_path=test_results_path * "/Test_plots/trange_0.5_min/";
mkpath(plot_path)
mkpath(data_path)


##### define function to compute percentage difference #####
# returns elements-wise percentage deviation between two arrays
function compute_deviation(y_true::Vector{Float64}, y_approx::Vector{Float64})
    return @. abs(100 * (y_true-y_approx)/y_true)
end

##### define function to carry out fit and store fit parameters #####
function compute_fit(nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, kerrReltol::Float64, kerrAbstol::Float64)
    println("nHarm = $(nHarm), nPoints = $(nPoints)")

    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M); Ω=ω[1:3]/ω[4];
    tmax =  0.5 * minimum(@. 2π/Ω); saveat = tmax / (nPoints-1); Δti=saveat;
    println(length(0.0:saveat:tmax|>collect))

    ##### compute geodesic for reltol=1e-12 #####
    HJEvolution.compute_kerr_geodesic(a, p, e, θi, nPoints, tmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; 
    r_dot=sol[5,:]; θ_dot=sol[6,:]; ϕ_dot=sol[7,:]; r_ddot=sol[8,:]; θ_ddot=sol[9,:]; ϕ_ddot=sol[10,:]; dt_dτ=sol[11,:];

    ##### compute test function values and its derivatives #####
    # print(t)
    # println("t0 = $(t[1]), tF = $(last(t))")
    # println("tmax = $(tmax)")
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    test_func_data_0 = zeros(size(t, 1))
    test_func_data_1 = zeros(size(t, 1))
    test_func_data_2 = zeros(size(t, 1))
    test_func_data_3 = zeros(size(t, 1))
    test_func_data_4 = zeros(size(t, 1))
    test_func_data_5 = zeros(size(t, 1))
    test_func_data_6 = zeros(size(t, 1))
    x = [Float64[] for i in 1:size(t, 1)]
    dx = [Float64[] for i in 1:size(t, 1)]
    d2x = [Float64[] for i in 1:size(t, 1)]
    d3x = [Float64[] for i in 1:size(t, 1)]
    d4x = [Float64[] for i in 1:size(t, 1)]
    d5x = [Float64[] for i in 1:size(t, 1)]
    d6x = [Float64[] for i in 1:size(t, 1)]
    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        dx[i] = Vector{Float64}([r_dot[i], θ_dot[i], ϕ_dot[i]]);
        d2x[i] = Vector{Float64}([r_ddot[i], θ_ddot[i], ϕ_ddot[i]]);
    end

    @inbounds for i in eachindex(t)
        d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
        d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
        d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
        test_func_data_0[i] = Function_1.f(x[i])
        test_func_data_1[i] = Function_1.df_dt(dx[i], x[i])
        test_func_data_2[i] = Function_1.d2f_dt(d2x[i], dx[i], x[i])
        test_func_data_3[i] = Function_1.d3f_dt(d3x[i], d2x[i], dx[i], x[i])
        test_func_data_4[i] = Function_1.d4f_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i])
        test_func_data_5[i] = Function_1.d5f_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
        test_func_data_6[i] = Function_1.d6f_dt(d6x[i], d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i])
    end

    test_func_data = [test_func_data_0, test_func_data_1, test_func_data_2, test_func_data_3,
    test_func_data_4, test_func_data_5, test_func_data_6];
    # deriv_dictionary = Dict{String, Vector{Float64}}("xdata" => xdata, "test_func_data" => test_func_data, "fit_params" => fit_params, "fit_freqs" => Ω_fit) 

    ##### perform fit #####
    # println("Length(t) = $(length(t))")
    xdata = t; ydata=test_func_data_0; 
    fit_fname_save=data_path * "Test_func_1_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"

    n_freqs = FourierFitGSL.compute_num_freqs(nHarm); chisq=[0.0];
    fit_params = zeros(2 * n_freqs + 1);

    if length(t) == nPoints
        Ω_fit = FourierFitGSL.GSL_fit!(xdata, ydata, nPoints, nHarm, chisq, Ω[1], Ω[2], Ω[3], fit_params)
    else
        println("nPoints = $(nPoints), length(t) = $(length(t))")
        throw(BoundsError)
    end

    # Creating a Typed Dictionary 
    fit_dictionary = Dict{String, AbstractArray}("xdata" => xdata, "test_func_data" => test_func_data, "fit_params" => fit_params, "fit_freqs" => Ω_fit) 
    # save fit #
    println(fit_fname_save)
    save(fit_fname_save, "data", fit_dictionary)
end

# function to compute the derivative from the fit
function compute_derivatives(nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64)
    fit_fname_save=data_path * "Test_func_1_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    saved_data = load(fit_fname_save)["data"]
    xdata = saved_data["xdata"]
    test_func_data = saved_data["test_func_data"]
    fit_params = saved_data["fit_params"]
    Ω_fit = saved_data["fit_freqs"]
    n_freqs = length(Ω_fit)
    return xdata, [FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, N) for N in [0, 1, 2, 3, 4, 5, 6]], test_func_data
end

##### specify geodesic parameters #####
a = 0.8; p = 10.5; e = 0.5; θi = π/6; M=1.0; m=1e-5; kerrReltol=1e-12; kerrAbstol=1e-10; index1 = 2; index2 = 3

# harmonics = [2, 3, 4, 5]
# points = [100, 200, 500, 1000]
harmonics = [2, 5]
points = [1500, 3000]
# harmonics = [2]
# points = [500]
for nHarm in harmonics
    for nPoints in points
        compute_fit(nHarm, nPoints, a, p, e, θi, M, kerrReltol, kerrAbstol)
    end
end

# load fits and compute derivatives
xdata2 = []; fit_data2 = []; test_func_data2 = [];
xdata5 = []; fit_data5 = []; test_func_data5 = [];

for nPoints in points
    xdata_2, fit_data_2, ydata_2 = compute_derivatives(2, nPoints, a, p, e, θi)
    push!(xdata2, xdata_2); push!(fit_data2, fit_data_2); push!(test_func_data2, ydata_2);

    xdata_5, fit_data_5, ydata_5 = compute_derivatives(5, nPoints, a, p, e, θi)
    push!(xdata5, xdata_5); push!(fit_data5, fit_data_5); push!(test_func_data5, ydata_5);
end


##### plotting routine #####
# plot attributes
left_margin=10mm; right_margin=10mm; bottom_margin=10mm; top_margin=10mm;
color=[:blue :red]; linewidth = [2.0 1.5]; alpha=[0.6 0.8]; linestyle = [:solid :dash];     
xtickfontsize=10; ytickfontsize=10; guidefontsize=13; legendfontsize=13;


### N=2 harmonics ####

index1_points = 1; index2_points=2;

# plotting errors (i.e., percentage difference) #
errors_N_2_points_1 = [@.(100 * abs((test_func_data2[index1_points][i]-fit_data2[index1_points][i])/test_func_data2[index1_points][i])) for i=[1, 3, 5, 7]]; # get derivatives: 0, 2, 4, 6
errors_N_2_points_2 = [@.(100 * abs((test_func_data2[index2_points][i]-fit_data2[index2_points][i])/test_func_data2[index2_points][i])) for i=[1, 3, 5, 7]]; # get derivatives: 0, 2, 4, 6
errors_N_5_points_1 = [@.(100 * abs((test_func_data5[index1_points][i]-fit_data5[index1_points][i])/test_func_data5[index1_points][i])) for i=[1, 3, 5, 7]];
errors_N_5_points_2 = [@.(100 * abs((test_func_data5[index2_points][i]-fit_data5[index2_points][i])/test_func_data5[index2_points][i])) for i=[1, 3, 5, 7]];

# plot attributes
# shapes = [:rect, :circle, :diamond, :star, :cross]
# colors = [:red, :blue, :yellow, :magenta, :green]
# alphas = [1.0, 0.5, 0.2, 0.2, 0.2]

shapes = [:circle :circle :circle :circle]
colors = [:red :blue :magenta :green]
alphas = [1.0 0.8 0.6 0.4] 

# plot errors

# plotting deviations for first number of points #
ms=1.5; markerstrokewidth=0;
error_plot_fit_N_2_points_1 = scatter([xdata2[index1_points] for i=[1, 3, 5, 7]], 
        errors_N_2_points_1, xlabel=L"t", ylabel=L"f_{1}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-12,1e8), dpi=600,
        label=[L"f^{(0)}" L"f^{(2)}" L"f^{(4)}" L"f^{(6)}"], yscale=:log10, markershape=shapes, color=colors, alpha=alphas)
        vline!([xdata2[index1_points][points[index1_points] ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(xdata2[index1_points]) + (maximum(xdata2[index1_points])-minimum(xdata2[index1_points]))*0.2, 5e6, Plots.text(L"N=2, n_{p}=%$(points[index1_points])", :center, 12))

yticks!(error_plot_fit_N_2_points_1, [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_fit_N_2_points_1)

# plot_name="Test_func_1_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_2_nPoints_$(points[index1_points])_0.5_min.png"
# savefig(error_plot_fit_N_2_points_1, plot_path * plot_name)

# plotting deviations for second number of points #
ms=1.5; markerstrokewidth=0;
error_plot_fit_N_2_points_2 = scatter([xdata2[index2_points] for i=[1, 3, 5, 7]], 
        errors_N_2_points_2, xlabel=L"t", ylabel=L"f_{1}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-12,1e8), dpi=600,
        label=[L"f^{(0)}" L"f^{(2)}" L"f^{(4)}" L"f^{(6)}"], yscale=:log10, markershape=shapes, color=colors, alpha=alphas)
        vline!([xdata2[index2_points][points[index2_points] ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(xdata2[index2_points]) + (maximum(xdata2[index2_points])-minimum(xdata2[index2_points]))*0.2, 5e6, Plots.text(L"N=2, n_{p}=%$(points[index2_points])", :center, 12))

yticks!(error_plot_fit_N_2_points_2, [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_fit_N_2_points_2)

error_plot_fit_N_2 = plot(error_plot_fit_N_2_points_1, error_plot_fit_N_2_points_2, layout=(1, 2), size=(1000, 400))

plot_name="Test_func_1_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_2_nPoints_$(points[index1_points])_$(points[index2_points])_0.5_min.png"
savefig(error_plot_fit_N_2, plot_path * plot_name)


### N=5 harmonics ####

# plotting deviations for first number of points #
ms=1.5; markerstrokewidth=0;
error_plot_fit_N_5_points_1 = scatter([xdata5[index1_points] for i=[1, 3, 5, 7]], 
        errors_N_5_points_1, xlabel=L"t", ylabel=L"f_{1}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-12,1e8), dpi=600,
        label=[L"f^{(0)}" L"f^{(2)}" L"f^{(4)}" L"f^{(6)}"], yscale=:log10, markershape=shapes, color=colors, alpha=alphas)
        vline!([xdata5[index1_points][points[index1_points] ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(xdata5[index1_points]) + (maximum(xdata5[index1_points])-minimum(xdata5[index1_points]))*0.2, 5e6, Plots.text(L"N=5, n_{p}=%$(points[index1_points])", :center, 12))

yticks!(error_plot_fit_N_5_points_1, [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_fit_N_5_points_1)


# plotting deviations for second number of points #
ms=1.5; markerstrokewidth=0;
error_plot_fit_N_5_points_2 = scatter([xdata5[index2_points] for i=[1, 3, 5, 7]], 
        errors_N_5_points_2, xlabel=L"t", ylabel=L"f_{1}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-12,1e8), dpi=600,
        label=[L"f^{(0)}" L"f^{(2)}" L"f^{(4)}" L"f^{(6)}"], yscale=:log10, markershape=shapes, color=colors, alpha=alphas)
        vline!([xdata5[index2_points][points[index2_points] ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(xdata5[index2_points]) + (maximum(xdata5[index2_points])-minimum(xdata5[index2_points]))*0.2, 5e6, Plots.text(L"N=5, n_{p}=%$(points[index2_points])", :center, 12))

yticks!(error_plot_fit_N_5_points_2, [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_fit_N_5_points_2)

error_plot_fit_N_5 = plot(error_plot_fit_N_5_points_1, error_plot_fit_N_5_points_2, layout=(1, 2), size=(1000, 400))

plot_name="Test_func_1_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_5_nPoints_$(points[index1_points])_$(points[index2_points])_0.5_min.png"
savefig(error_plot_fit_N_5, plot_path * plot_name)