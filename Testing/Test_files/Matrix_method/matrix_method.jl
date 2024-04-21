include("/home/lkeeble/GRSuite/main.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/curve_fit.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/MatrixFitting.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/TestMultipoleFit.jl")
using DelimitedFiles, Statistics, BenchmarkTools, Plots, Plots.PlotMeasures, LaTeXStrings, LsqFit, .TestMultipoleFit
using LinearAlgebra

# path for saving data and plots
data_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_matrix_data/";
plot_path="/home/lkeeble/GRSuite/Testing/Test_results/Test_matrix_plots/";

##### define function to carry out fit and store fit parameters #####
function compute_multipoles(nPoints::Int64, a::Float64, p::Float64, e::Float64, θi::Float64, M::Float64, m::Float64, kerrReltol::Float64, kerrAbstol::Float64)
    println("nHarm = $(nHarm), nPoints = $(nPoints)")

    # compute fundamental frequencies 
    rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
    E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M); Ω=ω[1:3]/ω[4];
    tmax= 1.0 * minimum(@. 2π/Ω); saveat = tmax / (nPoints-1); Δti=saveat;

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

    return Mij_data, Mijk_data, Sij_data, Mij2_data, Mijk2_data, Sij1_data, Ω..., t
end

##### specify geodesic parameters #####
a = 0.8; p = 10.5; e = 0.5; θi = π/6; M=1.0; m=1e-5; kerrReltol=1e-12; kerrAbstol=1e-10; index1 = 2; index2 = 3;

M=1.0; nHarm=1;
n_freqs=MatrixFitting.compute_n_frequencies(nHarm)
harmonics = [2, 3, 4, 5]
points = [100, 200, 500, 1000]; nPoints=n_freqs


Mij_data, Mijk_data, Sij_data, Mij2_data, Mijk2_data, Sij1_data, Ωr, Ωθ, Ωϕ, t = compute_multipoles(nPoints, a, p, e, θi, M, m, kerrReltol, kerrAbstol)

Ω = Float64[];
MatrixFitting.compute_fitting_frequencies(Ω, Ωr, Ωθ, Ωϕ, nHarm)
fourier_coeff=im * zeros(n_freqs)
fitted_data=im * zeros(n_freqs)

auxiliary_matrix = im * zeros(n_freqs, n_freqs)
MatrixFitting.fill_matrix!(t, auxiliary_matrix, Ω, n_freqs)

MatrixFitting.compute_fourier_coeffs(auxiliary_matrix, fitted_data, Mij_data[index1, index2], fourier_coeff)
fitted_data_real=@. real(fitted_data)
fitted_deriv = real(MatrixFitting.compute_derivative(t, fourier_coeff, Ω, n_freqs, 2)[2:nPoints-1])


error = @. MatrixFitting.deviation(Mij_data[index1, index2], fitted_data_real)
error = @. MatrixFitting.deviation(Mij2_data[index1, index2][2:nPoints-1], fitted_deriv)

left_margin=10mm; right_margin=10mm; bottom_margin=10mm; top_margin=10mm;
color=[:blue :red]; linewidth = [2.0 1.5]; alpha=[0.6 0.8]; linestyle = [:solid :dash];     
xtickfontsize=10; ytickfontsize=10; guidefontsize=13; legendfontsize=13;

scatter(t, error, xlabel=L"t", ylabel=L"M_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
        framsetyle=:box, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin,
        legend=:topright, foreground_color_legend = nothing, background_color_legend = nothing, xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize, guidefontsize = guidefontsize, legendfontsize=legendfontsize,
        markersize=ms, markerstrokewidth=markerstrokewidth,
        ylims=(1e-8,1e8), dpi=600,
        label=[L"n_{\mathrm{p}}=%$(nPoints)" L"n_{\mathrm{p}}=%$(nPoints)"], yscale=:log10, markershape=[shapes[1] shapes[2]], color=[colors[1] colors[2]], alpha=[alpha[1] alphas[2]])
        vline!([t[nPoints ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(t) + (maximum(t)-minimum(t))*0.1, 5e6, Plots.text(L"N=1", :center, 12))


rank(auxiliary_matrix)
size(auxiliary_matrix, 1)