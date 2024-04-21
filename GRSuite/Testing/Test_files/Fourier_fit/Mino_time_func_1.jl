include("/home/lkeeble/GRSuite/main.jl");
include("/home/lkeeble/GRSuite/Testing/BL_time_derivs/Test_functions.jl");
include("/home/lkeeble/GRSuite/Testing/Test_modules/ParameterizedDerivs.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings, Peaks, .FourierFitGSL, JLD2, FileIO, .MinoEvolution
using .ParameterizedDerivs, .MinoTimeDerivs, .Function_1, .Function_2
using .Deriv2, .Deriv3, .Deriv4, .Deriv5, .Deriv6

##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results/Mino/GSL"
data_path=test_results_path * "/Test_data/trange_5_min/";
plot_path=test_results_path * "/Test_plots/trange_5_min/";
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
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);
    λmax =  5.0 * minimum(@. 2π/ω); saveat = λmax / (nPoints-1); Δλi=saveat;


    ##### compute geodesic for reltol=1e-12 #####
    @time MinoEvolution.compute_kerr_geodesic(a, p, e, θi, λmax, Δλi, kerrReltol, kerrAbstol, saveat, data_path=data_path)
    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "Mino_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(λmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    λ=sol[1,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; 
    dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt=sol[9,:]; d2θ_dt=sol[10,:]; d2ϕ_dt=sol[11,:]; dt_dλ=sol[12,:];

    ##### compute test function values and its derivatives #####
    # print(λ)
    # println("t0 = $(λ[1]), tF = $(last(λ))")
    # println("λmax = $(λmax)")
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    test_func_data_0 = zeros(size(λ, 1))
    test_func_data_1 = zeros(size(λ, 1))
    test_func_data_2 = zeros(size(λ, 1))
    test_func_data_3 = zeros(size(λ, 1))
    test_func_data_4 = zeros(size(λ, 1))
    test_func_data_5 = zeros(size(λ, 1))
    test_func_data_6 = zeros(size(λ, 1))
    x = [Float64[] for i in 1:size(λ, 1)]
    dx = [Float64[] for i in 1:size(λ, 1)]
    d2x = [Float64[] for i in 1:size(λ, 1)]
    d3x = [Float64[] for i in 1:size(λ, 1)]
    d4x = [Float64[] for i in 1:size(λ, 1)]
    d5x = [Float64[] for i in 1:size(λ, 1)]
    d6x = [Float64[] for i in 1:size(λ, 1)]
    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(λ)
        x[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        dx[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
        d2x[i] = Vector{Float64}([d2r_dt[i], d2θ_dt[i], d2ϕ_dt[i]]);
    end

    @inbounds for i in eachindex(λ)
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
    # println("Length(λ) = $(length(λ))")
    xdata = λ; ydata=test_func_data_0; 
    fit_fname_save=data_path * "Test_func_1_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"

    n_freqs = FourierFitGSL.compute_num_freqs(nHarm); chisq=[0.0];
    fit_params = zeros(2 * n_freqs + 1);

    if length(λ) == nPoints
        Ω_fit = FourierFitGSL.GSL_fit!(xdata, ydata, nPoints, nHarm, chisq, ω[1], ω[2], ω[3], fit_params)
    else
        println("nPoints = $(nPoints), length(λ) = $(length(λ))")
        throw(BoundsError)
    end

    # Creating a Typed Dictionary 
    fit_dictionary = Dict("ODE_fname" => kerr_ode_sol_fname, "xdata" => xdata, "test_func_data" => test_func_data, "fit_params" => fit_params, "fit_freqs" => Ω_fit) 
    # save fit #
    println(fit_fname_save)
    save(fit_fname_save, "data", fit_dictionary)
end


# function to compute the derivative from the fit
function compute_derivatives(nHarm::Int64, nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64)
    fit_fname_save=data_path * "Test_func_1_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    saved_data = load(fit_fname_save)["data"]
    test_func_data = saved_data["test_func_data"]
    kerr_ode_sol_fname = saved_data["ODE_fname"]
    xdata = saved_data["xdata"]
    fit_params = saved_data["fit_params"]
    Ω_fit = saved_data["fit_freqs"]
    n_freqs = length(Ω_fit)

    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)

    # compute f^{(n)}(λ)
    fλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 0)
    df_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 1)
    d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 2)
    d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 3)
    d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 4)
    d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 5)
    d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, 6)


    # compute λ^{(n)}(t)
    sol = readdlm(kerr_ode_sol_fname)
    r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; 
    dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt=sol[9,:]; d2θ_dt=sol[10,:]; d2ϕ_dt=sol[11,:];

    ## initialize arrays
    x = [Float64[] for i in 1:size(r, 1)]; dx = [Float64[] for i in 1:size(r, 1)];
    d2x = [Float64[] for i in 1:size(r, 1)]; d3x = [Float64[] for i in 1:size(r, 1)];
    d4x = [Float64[] for i in 1:size(r, 1)]; d5x = [Float64[] for i in 1:size(r, 1)];
    d6x = [Float64[] for i in 1:size(r, 1)];

    dλ_dt = zeros(length(r)); d2λ_dt = zeros(length(r)); d3λ_dt = zeros(length(r));
    d4λ_dt = zeros(length(r)); d5λ_dt = zeros(length(r)); d6λ_dt = zeros(length(r));
    df_dt = zeros(length(r)); d2f_dt = zeros(length(r)); d3f_dt = zeros(length(r));
    d4f_dt = zeros(length(r)); d5f_dt = zeros(length(r)); d6f_dt = zeros(length(r));

    @inbounds Threads.@threads for i in eachindex(r)
       x[i] = [r[i], θ[i], ϕ[i]]
       dx[i] = [dr_dt[i], dθ_dt[i], dϕ_dt[i]]
       d2x[i] = [d2r_dt[i], d2θ_dt[i], d2ϕ_dt[i]]
       d3x[i] = [Deriv3.d3r_dt(d2x[i], dx[i], x[i], a), Deriv3.d3θ_dt(d2x[i], dx[i], x[i], a), Deriv3.d3ϕ_dt(d2x[i], dx[i], x[i], a)]
       d4x[i] = [Deriv4.d4r_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4θ_dt(d3x[i], d2x[i], dx[i], x[i], a), Deriv4.d4ϕ_dt(d3x[i], d2x[i], dx[i], x[i], a)]
       d5x[i] = [Deriv5.d5r_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5θ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv5.d5ϕ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]
       d6x[i] = [Deriv6.d6r_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6θ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a), Deriv6.d6ϕ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a)]

       dλ_dt[i] = MinoTimeDerivs.dλ_dt(x[i], a, M, E, L)
       d2λ_dt[i] = MinoTimeDerivs.d2λ_dt(dx[i], x[i], a, M, E, L)
       d3λ_dt[i] = MinoTimeDerivs.d3λ_dt(d2x[i], dx[i], x[i], a, M, E, L)
       d4λ_dt[i] = MinoTimeDerivs.d4λ_dt(d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
       d5λ_dt[i] = MinoTimeDerivs.d5λ_dt(d4x[i], d3x[i], d2x[i], dx[i], x[i], a, M, E, L)
       d6λ_dt[i] = MinoTimeDerivs.d6λ_dt(d5x[i], d4x[i], d3x[i], d2x[i], dx[i], x[i], a, M, E, L)

       df_dt[i] = ParameterizedDerivs.df_dt(df_dλ[i], dλ_dt[i])
       d2f_dt[i] = ParameterizedDerivs.d2f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i])
       d3f_dt[i] = ParameterizedDerivs.d3f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i])
       d4f_dt[i] = ParameterizedDerivs.d4f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i])
       d5f_dt[i] = ParameterizedDerivs.d5f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i], d5f_dλ[i], d5λ_dt[i])
       d6f_dt[i] = ParameterizedDerivs.d6f_dt(df_dλ[i], dλ_dt[i], d2f_dλ[i], d2λ_dt[i], d3f_dλ[i], d3λ_dt[i], d4f_dλ[i], d4λ_dt[i], d5f_dλ[i], d5λ_dt[i], d6f_dλ[i], d6λ_dt[i])
    end

    return xdata, [fλ, df_dt, d2f_dt, d3f_dt, d4f_dt, d5f_dt, d6f_dt], test_func_data
end

##### specify geodesic parameters #####
a = 0.8; p = 10.5; e = 0.5; θi = π/6; M=1.0; m=1e-5; kerrReltol=1e-12; kerrAbstol=1e-10; index1 = 2; index2 = 3

# harmonics = [2, 3, 4, 5]
# points = [100, 200, 500, 1000]
# harmonics = [2, 5]
# points = [1500, 3000]
harmonics = [2]
points = [200]
for nHarm in harmonics
    for nPoints in points
        compute_fit(nHarm, nPoints, a, p, e, θi, M, kerrReltol, kerrAbstol)
    end
end

# load fits and compute derivatives
xdata2 = []; fit_data2 = []; test_func_data2 = [];
xdata5 = []; fit_data5 = []; test_func_data5 = [];

for nPoints in points
    xdata_2, fit_data_2, ydata_2 = compute_derivatives(harmonics[1], nPoints, a, p, e, θi)
    push!(xdata2, xdata_2); push!(fit_data2, fit_data_2); push!(test_func_data2, ydata_2);

    # xdata_5, fit_data_5, ydata_5 = compute_derivatives(5, nPoints, a, p, e, θi)
    # push!(xdata5, xdata_5); push!(fit_data5, fit_data_5); push!(test_func_data5, ydata_5);
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
# errors_N_2_points_2 = [@.(100 * abs((test_func_data2[index2_points][i]-fit_data2[index2_points][i])/test_func_data2[index2_points][i])) for i=[1, 3, 5, 7]]; # get derivatives: 0, 2, 4, 6
# errors_N_5_points_1 = [@.(100 * abs((test_func_data5[index1_points][i]-fit_data5[index1_points][i])/test_func_data5[index1_points][i])) for i=[1, 3, 5, 7]];
# errors_N_5_points_2 = [@.(100 * abs((test_func_data5[index2_points][i]-fit_data5[index2_points][i])/test_func_data5[index2_points][i])) for i=[1, 3, 5, 7]];

# errors_N_5_points_2[3]

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
        errors_N_2_points_1, xlabel=L"λ", ylabel=L"f_{1}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-12,1e8), dpi=600,
        label=[L"f^{(0)}" L"f^{(2)}" L"f^{(4)}" L"f^{(6)}"], yscale=:log10, markershape=shapes, color=colors, alpha=alphas)
        vline!([xdata2[index1_points][points[index1_points] ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(xdata2[index1_points]) + (maximum(xdata2[index1_points])-minimum(xdata2[index1_points]))*0.2, 5e6, Plots.text(L"N=%$(harmonics[1]), n_{p}=%$(points[index1_points])", :center, 12))

yticks!(error_plot_fit_N_2_points_1, [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_fit_N_2_points_1)

plot_name="Test_func_1_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(harmonics[1])_nPoints_$(points[index1_points])_5_min.png"
savefig(error_plot_fit_N_2_points_1, plot_path * plot_name)

# plotting deviations for second number of points #
ms=1.5; markerstrokewidth=0;
error_plot_fit_N_2_points_2 = scatter([xdata2[index2_points] for i=[1, 3, 5, 7]], 
        errors_N_2_points_2, xlabel=L"λ", ylabel=L"f_{1}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-12,1e8), dpi=600,
        label=[L"f^{(0)}" L"f^{(2)}" L"f^{(4)}" L"f^{(6)}"], yscale=:log10, markershape=shapes, color=colors, alpha=alphas)
        vline!([xdata2[index2_points][points[index2_points] ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(xdata2[index2_points]) + (maximum(xdata2[index2_points])-minimum(xdata2[index2_points]))*0.2, 5e6, Plots.text(L"N=%$(harmonics[1]), n_{p}=%$(points[index2_points])", :center, 12))

yticks!(error_plot_fit_N_2_points_2, [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_fit_N_2_points_2)

error_plot_fit_N_2 = plot(error_plot_fit_N_2_points_1, error_plot_fit_N_2_points_2, layout=(1, 2), size=(1000, 400))

plot_name="Test_func_1_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_2_nPoints_$(points[index1_points])_$(points[index2_points])_5_min.png"
savefig(error_plot_fit_N_2, plot_path * plot_name)


### N=5 harmonics ####

# plotting deviations for first number of points #
ms=1.5; markerstrokewidth=0;
error_plot_fit_N_5_points_1 = scatter([xdata5[index1_points] for i=[1, 3, 5, 7]], 
        errors_N_5_points_1, xlabel=L"λ", ylabel=L"f_{1}\,\mathrm{error}\,(\%)",
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
        errors_N_5_points_2, xlabel=L"λ", ylabel=L"f_{1}\,\mathrm{error}\,(\%)",
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

plot_name="Test_func_1_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_5_nPoints_$(points[index1_points])_$(points[index2_points])_5_min.png"
savefig(error_plot_fit_N_5, plot_path * plot_name)