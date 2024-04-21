include("/home/lkeeble/GRSuite/main.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings, Peaks, .HJEvolution, .FourierFitGSL, JLD2, FileIO

##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results/HJE/GSL"
data_path=test_results_path * "/Test_data/trange_0.1_min/";
plot_path=test_results_path * "/Test_plots/trange_0.1_min/";
mkpath(plot_path)
mkpath(data_path)


##### define function to compute percentage difference #####
# returns elements-wise percentage deviation between two arrays
function compute_deviation(y_true::Vector{Float64}, y_approx::Vector{Float64})
    return @. abs(100 * (y_true-y_approx)/y_true)
end

##### define function to carry out fit and store fit parameters #####
function compute_fit(nHarm::Int64, nPoints::Int64, index1::Int64, index2::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, m::Float64, kerrReltol::Float64, kerrAbstol::Float64)
    println("nHarm = $(nHarm), nPoints = $(nPoints)")

    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M); Ω=ω[1:3]/ω[4];
    tmax=  0.1 * minimum(@. 2π/Ω); saveat = tmax / (nPoints-1); Δti=saveat;

    ##### compute geodesic for reltol=1e-12 #####
    HJEvolution.compute_kerr_geodesic(a, p, e, θi, tmax, Δti, kerrReltol, kerrAbstol, saveat, data_path=data_path)

    # load geodesic and store in array #
    kerr_ode_sol_fname=data_path * "HJ_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(tmax)_tol_$(kerrReltol).txt"
    sol = readdlm(kerr_ode_sol_fname)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; 
    r_dot=sol[5,:]; θ_dot=sol[6,:]; ϕ_dot=sol[7,:]; r_ddot=sol[8,:]; θ_ddot=sol[9,:]; ϕ_ddot=sol[10,:]; dt_dτ=sol[11,:];

    ##### compute multipole moments #####
    # initialize ydata arrays
    Mijk_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij_data = [Float64[] for i=1:3, j=1:3]
    Sij_data = [Float64[] for i=1:3, j=1:3]

    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]

    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    xBL = [Float64[] for i in 1:size(t, 1)]
    vBL = [Float64[] for i in 1:size(t, 1)]
    aBL = [Float64[] for i in 1:size(t, 1)]
    xH = [Float64[] for i in 1:size(t, 1)]
    x_H = [Float64[] for i in 1:size(t, 1)]
    vH = [Float64[] for i in 1:size(t, 1)]
    v_H = [Float64[] for i in 1:size(t, 1)]
    v = zeros(size(t, 1))
    rH = zeros(size(t, 1))
    aH = [Float64[] for i in 1:size(t, 1)]
    a_H = [Float64[] for i in 1:size(t, 1)]


    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([r_dot[i], θ_dot[i], ϕ_dot[i]]);
        aBL[i] = Vector{Float64}([r_ddot[i], θ_ddot[i], ϕ_ddot[i]]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
        x_H[i] = xH[i]
        rH[i] = SelfForce.norm_3d(xH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        v_H[i] = vH[i]; 
        v[i] = SelfForce.norm_3d(vH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
        a_H[i] = aH[i]
    end

    SelfForce.multipole_moments_tr!(vH, xH, x_H, m/M, M, Mij_data, Mijk_data, Sij_data)
    SelfForce.moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)

    ##### perform fit #####
    xdata = t; ydata=Mij_data[index1, index2]; y2data = Mij2_data[index1, index2];
    fit_fname_save=data_path * "Mij_$(index1)_$(index2)_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"

    n_freqs = FourierFitGSL.compute_num_freqs(nHarm); chisq=[0.0];
    fit_params = zeros(2 * n_freqs + 1);

    Ω_fit = FourierFitGSL.GSL_fit!(xdata, ydata, nPoints, nHarm, chisq, Ω[1], Ω[2], Ω[3], fit_params)

    # Creating a Typed Dictionary 
    fit_dictionary = Dict{String, Vector{Float64}}("xdata" => xdata, "ydata" => ydata, "y2data" => y2data, "fit_params" => fit_params, "fit_freqs" => Ω_fit) 
    # save fit #
    println(fit_fname_save)
    save(fit_fname_save, "data", fit_dictionary)
end

# function to compute the derivative from the fit
function compute_derivatives(nHarm::Int64, nPoints::Int64, index1::Int64, index2::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, m::Float64, kerrReltol::Float64, kerrAbstol::Float64)
    fit_fname_save=data_path * "Mij_$(index1)_$(index2)_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints).jld2"
    saved_data = load(fit_fname_save)["data"]
    xdata = saved_data["xdata"]
    ydata = saved_data["ydata"]
    y2data = saved_data["y2data"]
    fit_params = saved_data["fit_params"]
    Ω_fit = saved_data["fit_freqs"]
    n_freqs = length(Ω_fit)
    return xdata, [FourierFitGSL.curve_fit_functional_derivs(xdata, Ω_fit, fit_params, n_freqs, nPoints, N) for N in [0, 2]], [ydata, y2data]
end

##### specify geodesic parameters #####
a = 0.8; p = 10.5; e = 0.5; θi = π/6; M=1.0; m=1e-5; kerrReltol=1e-12; kerrAbstol=1e-10; index1 = 2; index2 = 3

# harmonics = [2, 3, 4, 5]
# points = [100, 200, 500, 1000]
harmonics = [2, 5]
points = [1500, 3000]
for nHarm in harmonics
    for nPoints in points
        compute_fit(nHarm, nPoints, index1, index2, a, p, e, θi, M, m, kerrReltol, kerrAbstol)
    end
end

# load fits and compute derivatives
xdata_2 = []; fit_data_2 = []; ydata_2 = []; xdata_3 = []; fit_data_3 = []; ydata_3 = [];
xdata_4 = []; fit_data_4 = []; ydata_4 = []; xdata_5 = []; fit_data_5 = []; ydata_5 = [];
xdata_6 = []; fit_data_6 = []; ydata_6 = [];

for nPoints in points
    xdata2, fit_data2, ydata2 = compute_derivatives(2, nPoints, index1, index2, a, p, e, θi, M, m, kerrReltol, kerrAbstol)
    push!(xdata_2, xdata2); push!(fit_data_2, fit_data2); push!(ydata_2, ydata2);

    # xdata3, fit_data3, ydata3 = compute_derivatives(3, nPoints, index1, index2, a, p, e, θi, M, m, kerrReltol, kerrAbstol)
    # push!(xdata_3, xdata3); push!(fit_data_3, fit_data3); push!(ydata_3, ydata3);

    # xdata4, fit_data4, ydata4 = compute_derivatives(4, nPoints, index1, index2, a, p, e, θi, M, m, kerrReltol, kerrAbstol)
    # push!(xdata_4, xdata4); push!(fit_data_4, fit_data4); push!(ydata_4, ydata4);

    xdata5, fit_data5, ydata5 = compute_derivatives(5, nPoints, index1, index2, a, p, e, θi, M, m, kerrReltol, kerrAbstol)
    push!(xdata_5, xdata5); push!(fit_data_5, fit_data5); push!(ydata_5, ydata5);

    # xdata6, fit_data6, ydata6 = compute_derivatives(6, nPoints, index1, index2, a, p, e, θi, M, m, kerrReltol, kerrAbstol)
    # push!(xdata_6, xdata6); push!(fit_data_6, fit_data6); push!(ydata_6, ydata6);
end



##### plotting routine #####
# plot attributes
index1_points=1; index2_points=2;
# index1_points=3; index2_points=4;
left_margin=10mm; right_margin=10mm; bottom_margin=10mm; top_margin=10mm;
color=[:blue :red]; linewidth = [2.0 1.5]; alpha=[0.6 0.8]; linestyle = [:solid :dash];     
xtickfontsize=10; ytickfontsize=10; guidefontsize=13; legendfontsize=13;


### N=2 harmonics ####
# plotting fits for "by-eye" comparison #
plot_N_2_Mij_fit1 = plot([xdata_2[index1_points], xdata_2[index1_points]], [ydata_2[index1_points][1], fit_data_2[index1_points][1]], label=["Analytic" "Fit" "Fourier Fit"],
xlabel=L"t", ylabel=L"M_{%$(index1)%$(index2)}", framestyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=
bottom_margin, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, foreground_color_legend = nothing, background_color_legend = nothing, dpi=600, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize)
annotate!(plot_N_2_Mij_fit1, minimum(xdata_2[index1_points]) + (maximum(xdata_2[index1_points])-minimum(xdata_2[index1_points]))*0.1, 
minimum(ydata_2[index1_points][1]) + (maximum(ydata_2[index1_points][1])-minimum(ydata_2[index1_points][1]))*0.9,
Plots.text(L"N=2,\;n_{\mathrm{p}}=%$(points[index1_points])", :left, 12))

plot_N_2_Mij_fit2 = plot([xdata_2[index2_points], xdata_2[index2_points]], [ydata_2[index2_points][1], fit_data_2[index2_points][1]],
xlabel=L"t", ylabel=L"M_{%$(index1)%$(index2)}", framestyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=
bottom_margin, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, foreground_color_legend = nothing, background_color_legend = nothing, dpi=600, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize, legend=false)
annotate!(plot_N_2_Mij_fit2, minimum(xdata_2[index2_points]) + (maximum(xdata_2[index2_points])-minimum(xdata_2[index2_points]))*0.1, 
minimum(ydata_2[index2_points][1]) + (maximum(ydata_2[index2_points][1])-minimum(ydata_2[index2_points][1]))*0.9,
Plots.text(L"n_{\mathrm{p}}=%$(points[index2_points])", :left, 12))

plot_N_2_Mij2_fit1 = plot([xdata_2[index1_points], xdata_2[index1_points]], [ydata_2[index1_points][2], fit_data_2[index1_points][2]], label=["Analytic" "Fit" "Fourier Fit"],
xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}", framestyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=
bottom_margin, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, foreground_color_legend = nothing, background_color_legend = nothing, dpi=600, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize)
annotate!(plot_N_2_Mij2_fit1, minimum(xdata_2[index1_points]) + (maximum(xdata_2[index1_points])-minimum(xdata_2[index1_points]))*0.1, 
minimum(ydata_2[index1_points][2]) + (maximum(ydata_2[index1_points][2])-minimum(ydata_2[index1_points][2]))*0.9,
Plots.text(L"n_{\mathrm{p}}=%$(points[index1_points])", :left, 12), legend=false)

plot_N_2_Mij2_fit2 = plot([xdata_2[index2_points], xdata_2[index2_points]], [ydata_2[index2_points][2], fit_data_2[index2_points][2]],
xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}", framestyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=
bottom_margin, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, foreground_color_legend = nothing, background_color_legend = nothing, dpi=600, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize, legend=false)
annotate!(plot_N_2_Mij2_fit2, minimum(xdata_2[index2_points]) + (maximum(xdata_2[index2_points])-minimum(xdata_2[index2_points]))*0.1, 
minimum(ydata_2[index2_points][2]) + (maximum(ydata_2[index2_points][2])-minimum(ydata_2[index2_points][2]))*0.9,
Plots.text(L"n_{\mathrm{p}}=%$(points[index2_points])", :left, 12))

Mij_N_2_fit_plot=plot(plot_N_2_Mij_fit1, plot_N_2_Mij_fit2, plot_N_2_Mij2_fit1, plot_N_2_Mij2_fit2, layout=(2, 2), size=(1600, 800))
plot_N_2_name="Mij_$(index1)_$(index2)_fit_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_2_nPoints_$(points[index1_points])_$(points[index2_points])_0.1_min.png"
savefig(Mij_N_2_fit_plot, plot_path * plot_N_2_name)

# plotting errors (i.e., percentage difference) #
errors_fit_N_2 = [@.(100 * abs((ydata_2[index1_points][1]-fit_data_2[index1_points][1])/ydata_2[index1_points][1])), @.(100 * abs((ydata_2[index2_points][1]-fit_data_2[index2_points][1])/ydata_2[index2_points][1]))]; # error in fit itself
errors_deriv_N_2 = [@.(100 * abs((ydata_2[index1_points][2]-fit_data_2[index1_points][2])/ydata_2[index1_points][2])), @.(100 * abs((ydata_2[index2_points][2]-fit_data_2[index2_points][2])/ydata_2[index2_points][2]))]; # error in derivative

# plot attributes
shapes = [:rect, :circle, :diamond, :star, :cross]
colors = [:red, :blue, :yellow, :magenta, :green]
# alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
alphas = [1.0, 0.5, 0.2, 0.2, 0.2]

# plot errors

# plotting deviations of the 0th derivative #
ms=2; markerstrokewidth=0;
error_plot_fit_N_2 = scatter([xdata_2[index1_points], xdata_2[index2_points]], errors_fit_N_2, xlabel=L"t", ylabel=L"M_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-16,1e4), dpi=600,
        label=[L"n_{\mathrm{p}}=%$(points[index1_points])" L"n_{\mathrm{p}}=%$(points[index2_points])"], yscale=:log10, markershape=[shapes[1] shapes[2]], color=[colors[1] colors[2]], alpha=[alpha[1] alphas[2]])
        vline!([xdata_2[index2_points][points[index2_points] ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(xdata_2[index1_points]) + (maximum(xdata_2[index1_points])-minimum(xdata_2[index1_points]))*0.1, 5e2, Plots.text(L"N=2", :center, 12))

yticks!(error_plot_fit_N_2, [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
display(error_plot_fit_N_2)

# plotting deviations of the 2nd derivative #


error_plot_derivs_N_2 = scatter([xdata_2[index1_points], xdata_2[index2_points]], errors_deriv_N_2, xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=false, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-16,1e4), dpi=600, yscale=:log10, markershape=[shapes[1] shapes[2]], color=[colors[1] colors[2]], alpha=[alpha[1] alphas[2]])
        vline!([xdata_2[index2_points][points[index2_points] ÷ 2]], label=false, linestyle=:dash)
        # annotate!(minimum(xdata_2[index1_points]) + (maximum(xdata_2[index1_points])-minimum(xdata_2[index1_points]))*0.1, 5e2, Plots.text(L"N=2", :center, 12))

yticks!(error_plot_derivs_N_2, [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
display(error_plot_derivs_N_2)

Mij_error_plot_N_2=plot(error_plot_fit_N_2, error_plot_derivs_N_2, layout=(2, 1), size=(600, 800))
plot_name="Mij_$(index1)_$(index2)_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_2_nPoints_$(points[index1_points])_$(points[index2_points])_0.1_min.png"
savefig(Mij_error_plot_N_2, plot_path * plot_name)


### N=5 harmonics ####
# plotting fits for "by-eye" comparison #
plot_N_5_Mij_fit1 = plot([xdata_5[index1_points], xdata_5[index1_points]], [ydata_5[index1_points][1], fit_data_5[index1_points][1]], label=["Analytic" "Fit" "Fourier Fit"],
xlabel=L"t", ylabel=L"M_{%$(index1)%$(index2)}", framestyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=
bottom_margin, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, foreground_color_legend = nothing, background_color_legend = nothing, dpi=600, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize)
annotate!(plot_N_5_Mij_fit1, minimum(xdata_5[index1_points]) + (maximum(xdata_5[index1_points])-minimum(xdata_5[index1_points]))*0.1, 
minimum(ydata_5[index1_points][1]) + (maximum(ydata_5[index1_points][1])-minimum(ydata_5[index1_points][1]))*0.9,
Plots.text(L"N=5,\;n_{\mathrm{p}}=%$(points[index1_points])", :left, 12))

plot_N_5_Mij_fit2 = plot([xdata_5[index2_points], xdata_5[index2_points]], [ydata_5[index2_points][1], fit_data_5[index2_points][1]],
xlabel=L"t", ylabel=L"M_{%$(index1)%$(index2)}", framestyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=
bottom_margin, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, foreground_color_legend = nothing, background_color_legend = nothing, dpi=600, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize, legend=false)
annotate!(plot_N_5_Mij_fit2, minimum(xdata_5[index2_points]) + (maximum(xdata_5[index2_points])-minimum(xdata_5[index2_points]))*0.1, 
minimum(ydata_5[index2_points][1]) + (maximum(ydata_5[index2_points][1])-minimum(ydata_5[index2_points][1]))*0.9,
Plots.text(L"n_{\mathrm{p}}=%$(points[index2_points])", :left, 12))

plot_N_5_Mij2_fit1 = plot([xdata_5[index1_points], xdata_5[index1_points]], [ydata_5[index1_points][2], fit_data_5[index1_points][2]], label=["Analytic" "Fit" "Fourier Fit"],
xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}", framestyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=
bottom_margin, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, foreground_color_legend = nothing, background_color_legend = nothing, dpi=600, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize)
annotate!(plot_N_5_Mij2_fit1, minimum(xdata_5[index1_points]) + (maximum(xdata_5[index1_points])-minimum(xdata_5[index1_points]))*0.1, 
minimum(ydata_5[index1_points][2]) + (maximum(ydata_5[index1_points][2])-minimum(ydata_5[index1_points][2]))*0.9,
Plots.text(L"n_{\mathrm{p}}=%$(points[index1_points])", :left, 12), legend=false)

plot_N_5_Mij2_fit2 = plot([xdata_5[index2_points], xdata_5[index2_points]], [ydata_5[index2_points][2], fit_data_5[index2_points][2]],
xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}", framestyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=
bottom_margin, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, foreground_color_legend = nothing, background_color_legend = nothing, dpi=600, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize, legend=false)
annotate!(plot_N_5_Mij2_fit2, minimum(xdata_5[index2_points]) + (maximum(xdata_5[index2_points])-minimum(xdata_5[index2_points]))*0.1, 
minimum(ydata_5[index2_points][2]) + (maximum(ydata_5[index2_points][2])-minimum(ydata_5[index2_points][2]))*0.9,
Plots.text(L"n_{\mathrm{p}}=%$(points[index2_points])", :left, 12))

Mij_N_5_fit_plot=plot(plot_N_5_Mij_fit1, plot_N_5_Mij_fit2, plot_N_5_Mij2_fit1, plot_N_5_Mij2_fit2, layout=(2, 2), size=(1600, 800))
plot_N_5_name="Mij_$(index1)_$(index2)_fit_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_5_nPoints_$(points[index1_points])_$(points[index2_points])_0.1_min.png"
savefig(Mij_N_5_fit_plot, plot_path * plot_N_5_name)

# plotting errors (i.e., percentage difference) #
errors_fit_N_5 = [@.(100 * abs((ydata_5[index1_points][1]-fit_data_5[index1_points][1])/ydata_5[index1_points][1])), @.(100 * abs((ydata_5[index2_points][1]-fit_data_5[index2_points][1])/ydata_5[index2_points][1]))]; # error in fit itself
errors_deriv_N_5 = [@.(100 * abs((ydata_5[index1_points][2]-fit_data_5[index1_points][2])/ydata_5[index1_points][2])), @.(100 * abs((ydata_5[index2_points][2]-fit_data_5[index2_points][2])/ydata_5[index2_points][2]))]; # error in derivative

# plotting deviations of the 0th derivative #
error_plot_fit_N_5 = scatter([xdata_5[index1_points], xdata_5[index2_points]], errors_fit_N_5, xlabel=L"t", ylabel=L"M_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-16,1e4), dpi=600,
        label=[L"n_{\mathrm{p}}=%$(points[index1_points])" L"n_{\mathrm{p}}=%$(points[index2_points])"], yscale=:log10, markershape=[shapes[1] shapes[2]], color=[colors[1] colors[2]], alpha=[alpha[1] alphas[2]])
        vline!([xdata_5[index2_points][points[index2_points] ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(xdata_5[index1_points]) + (maximum(xdata_5[index1_points])-minimum(xdata_5[index1_points]))*0.1, 5e2, Plots.text(L"N=5", :center, 12))

yticks!(error_plot_fit_N_5, [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
display(error_plot_fit_N_5)

# plotting deviations of the 2nd derivative #

error_plot_derivs_N_5 = scatter([xdata_5[index1_points], xdata_5[index2_points]], errors_deriv_N_5, xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=false, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-16,1e4), dpi=600, yscale=:log10, markershape=[shapes[1] shapes[2]], color=[colors[1] colors[2]], alpha=[alpha[1] alphas[2]])
        vline!([xdata_5[index2_points][points[index2_points] ÷ 2]], label=false, linestyle=:dash)
        # annotate!(minimum(xdata_5[index1_points]) + (maximum(xdata_5[index1_points])-minimum(xdata_5[index1_points]))*0.1, 5e2, Plots.text(L"N=5", :center, 12))

yticks!(error_plot_derivs_N_5, [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
display(error_plot_derivs_N_5)

Mij_error_plot_N_5=plot(error_plot_fit_N_5, error_plot_derivs_N_5, layout=(2, 1), size=(600, 800))
plot_name="Mij_$(index1)_$(index2)_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_5_nPoints_$(points[index1_points])_$(points[index2_points])_0.1_min.png"
savefig(Mij_error_plot_N_5, plot_path * plot_name)