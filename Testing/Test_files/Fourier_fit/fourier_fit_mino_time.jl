include("/home/lkeeble/GRSuite/main.jl");
using DelimitedFiles, Plots, Plots.Measures, LaTeXStrings, Peaks, .MinoEvolution, LsqFit

##### define paths #####
test_results_path = "/home/lkeeble/GRSuite/Testing/Test_results/Mino"
data_path=test_results_path * "/Test_data/";
plot_path=test_results_path * "/Test_plots/";


##### define function to compute percentage difference #####
# returns elements-wise percentage deviation between two arrays
function compute_deviation(y_true::Vector{Float64}, y_approx::Vector{Float64})
    return @. abs(100 * (y_true-y_approx)/y_true)
end

##### specify geodesic parameters #####
a = 0.8; p = 10.5; e = 0.5; θi = π/6; M=1.0; m=1e-5; kerrReltol=1e-12; kerrAbstol=1e-10;

# compute fundamental frequencies 
rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);
E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi, M)
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi, E, L, Q, C, rplus, rminus, M);

λmax= 5.0 * minimum(@. 2π/ω[1:3]); nPoints=1000; saveat = λmax / (nPoints-1); Δλi=saveat; nPointsMultipoleFit=nPoints;

##### compute geodesic for reltol=1e-12 #####
@time MinoEvolution.compute_kerr_geodesic(a, p, e, θi, λmax, Δλi, kerrReltol, kerrAbstol, saveat, data_path=data_path)

# load geodesic and store in array #
kerr_ode_sol_fname=data_path * "Mino_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_tstep_$(saveat)_T_$(λmax)_tol_$(kerrReltol).txt"
sol = readdlm(kerr_ode_sol_fname)
λ=sol[1,:]; t=sol[2,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt2=sol[9,:]; d2θ_dt2=sol[10,:]; d2ϕ_dt2=sol[11,:]; dt_dλ=sol[12,:];

##### compute multipole moments #####
# initialize ydata arrays
Mijk_data = [Float64[] for i=1:3, j=1:3, k=1:3]
Mij_data = [Float64[] for i=1:3, j=1:3]
Sij_data = [Float64[] for i=1:3, j=1:3]

Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
Mij2_data = [Float64[] for i=1:3, j=1:3]
Sij1_data = [Float64[] for i=1:3, j=1:3]

# length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
xBL = [Float64[] for i in 1:size(λ, 1)]
vBL = [Float64[] for i in 1:size(λ, 1)]
aBL = [Float64[] for i in 1:size(λ, 1)]
xH = [Float64[] for i in 1:size(λ, 1)]
x_H = [Float64[] for i in 1:size(λ, 1)]
vH = [Float64[] for i in 1:size(λ, 1)]
v_H = [Float64[] for i in 1:size(λ, 1)]
v = zeros(size(λ, 1))
rH = zeros(size(λ, 1))
aH = [Float64[] for i in 1:size(λ, 1)]
a_H = [Float64[] for i in 1:size(λ, 1)]


# convert trajectories to BL coords
@inbounds Threads.@threads for i in eachindex(λ)
    xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
    vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
    aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);
end
@inbounds Threads.@threads for i in eachindex(λ)
    xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
    x_H[i] = xH[i]
    rH[i] = SelfForce.norm_3d(xH[i]);
end
@inbounds Threads.@threads for i in eachindex(λ)
    vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
    v_H[i] = vH[i]; 
    v[i] = SelfForce.norm_3d(vH[i]);
end
@inbounds Threads.@threads for i in eachindex(λ)
    aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
    a_H[i] = aH[i]
end

SelfForce.multipole_moments_tr!(vH, xH, x_H, m/M, M, Mij_data, Mijk_data, Sij_data)
SelfForce.moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)
##### perform fit #####
index1 = 2; index2 =3; nHarm = 2; plot(λ, Mij_data[index1, index2])
xdata = λ; ydata=Mij_data[index1, index2];
fit_fname_save=data_path * "Mij_$(index1)_$(index2)_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints)_tmin_$(round(first(xdata); digits=3))_tmax_$(round(last(xdata))).txt"

# load fit (if it exists)
isfile(fit_fname_save) ? p0 = readdlm(fit_fname_save)[:] : p0 = Float64[];
p0 = Float64[];

# carry out fit
@time Ω_mino, ffit, fitted_data = FourierFit.fourier_fit(xdata, ydata, ω[1], ω[2], ω[3], nHarm, p0=p0)

fit_params=coef(ffit)

# save fit #
open(fit_fname_save, "w") do io
    writedlm(io, fit_params)
end

fit_params_BL = zeros(size(fit_params, 1)); n_fit_freqs=size(Ω_mino, 1);

for i in eachindex(Ω_mino)
    fit_params_BL[i] = fit_params[i]/ω[4]
    fit_params_BL[i+n_fit_freqs] = fit_params[i+n_fit_freqs]/ω[4]
end

Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4]
# now must construct fourier series frequencies wrt BL time
Ω_BL=Float64[]
@inbounds for i_r in 0:nHarm
    @inbounds for i_θ in 0:(nHarm+i_r)
        @inbounds for i_ϕ in 0:(nHarm+i_r+i_θ)
            append!(Ω_BL, i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ)
        end
    end
end


fitted_data = [FourierFit.curve_fit_functional_derivs(xdata, Ω_mino, fit_params, N) for N=0:2]

plot([λ, λ], [Mij_data[index1, index2], fitted_data[1]], label=["Analytic" "Fit"])

#### need to convert fourier series
plot([λ, λ], [Mij2_data[index1, index2], @. fitted_data[3]/(dt_dλ^2)], label=["Analytic" "Fit"])
##### plot errors #####
errors_1 = @. 100 * (Mij_data[index1, index2]-fitted_data[1])/Mij_data[index1, index2];    # avoid t=0 to not divide by zero
errors_2 = @. 100 * (Mij2_data[index1, index2]-(fitted_data[3]/(dt_dλ^2)))/Mij2_data[index1, index2];    # avoid t=0 to not divide by zero

# plot attributes
shapes = [:star4, :xcross, :diamond, :rect, :circle]
colors = [:yellow, :magenta, :green, :red, :blue]
# alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
alphas = [0.2, 0.2, 0.2, 0.2, 0.2]

# plot errors

# plotting deviations of the 0th derivative #
ms=2
error_plot_1 = scatter(λ, abs.(errors_1), xlabel=L"λ", ylabel=L"M_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
        framsetyle=:box,
        legend=:topright,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        size=(500, 300),
        markersize=ms,
        ylims=(1e-8,1e8), dpi=1000,
        label=L"N=%$(nHarm)", yscale=:log10, markershape=shapes[1], color=colors[1], fillalpha=alphas[1])
        vline!([λ[nPointsMultipoleFit ÷ 2]], label=false, linestyle=:dash)
        annotate!(minimum(λ) + (maximum(λ)-minimum(λ))*0.1, 5e6, Plots.text(L"n_{\mathrm{p}}=%$(nPointsMultipoleFit)", :center, 12))

yticks!(error_plot_1, [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_1)

# plot_name="Mij_$(index1)_$(index2)_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nPoints_$(nPointsMultipoleFit)_tmin_$(first(λ))_tmax_$(last(λ)).png"
# savefig(error_plot_1, plot_path * plot_name)

# plotting deviations of the 2nd derivative #


error_plot_2 = scatter(λ, abs.(errors_2), xlabel=L"λ", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
framsetyle=:box,
legend=:bottomright,
foreground_color_legend = nothing,
background_color_legend = nothing,
size=(500, 300),
markersize=ms,
ylims=(1e-8,1e8), dpi=1000,
label=L"N=%$(nHarm)", yscale=:log10, markershape=shapes[1], color=colors[1], fillalpha=alphas[1])
vline!([λ[nPointsMultipoleFit ÷ 2]], label=false, linestyle=:dash)
annotate!(minimum(λ) + (maximum(λ)-minimum(λ))*0.1, 5e-7, Plots.text(L"n_{\mathrm{p}}=%$(nPointsMultipoleFit)", :center, 12))

yticks!(error_plot_2, [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
display(error_plot_2)

# plot_name="Mij2_$(index1)_$(index2)_error_plot_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nPoints_$(nPointsMultipoleFit)_tmin_$(first(λ))_tmax_$(last(λ)).png"
# savefig(error_plot_2, plot_path * plot_name)



# now compute the numerical derivatives wrt τ
nPoints=size(τ, 1); h=τ[2]-τ[1];
derivs_fdm_4=[zeros(nPoints) for i=1:2]; FiniteDiff_4.compute_derivs(derivs_fdm_4, Mij_data[index1,index2], h, nPoints)
derivs_fdm_5=[zeros(nPoints) for i=1:2]; FiniteDiff_5.compute_derivs(derivs_fdm_5, Mij_data[index1,index2], h, nPoints)
# convert to proper time
for i in eachindex(derivs_fdm_4)
    for j in eachindex(derivs_fdm_5[i])
        derivs_fdm_4[i][j] = derivs_fdm_4[i][j] / (tdot[j]^i)
        derivs_fdm_5[i][j] = derivs_fdm_5[i][j] / (tdot[j]^i)
    end
end

numerical_error_analytic = @. 100 * (Mij2_data[index1,index2]-derivs_fdm_5[2]) / Mij2_data[index1,index2]
numerical_error_fitted = @. 100 * (fitted_data[2][3]-derivs_fdm_5[2]) / fitted_data[2][3]

shapes = [:star4, :circle]
colors = [:red, :blue]
alphas = [0.2, 0.2]

# plot errors

# plotting deviations of the 0th derivative #
ms=2

error_plot_numerical = scatter(t, abs.(numerical_error_analytic), xlabel=L"t", ylabel=L"\ddot{M}_{%$(index1)%$(index2)}\,\mathrm{error}\,(\%)",
    framsetyle=:box,
    legend=:false,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    size=(600, 400),
    markersize=ms,
    ylims=(1e-8,1e8),dpi=1000,
    yscale=:log10, markershape=shapes[1], color=colors[1], fillalpha=alphas[1],
    framestyle=:box)
    vline!([t[nPointsMultipoleFit ÷ 2]], label=false, linestyle=:dash)
    annotate!(minimum(t) + (maximum(t)-minimum(t))*0.25, 5e5, Plots.text(L"\textrm{Numerical\;derivative}"*"\n"* L"n_{\mathrm{p}}=%$(nPointsMultipoleFit)", :center, 12))
savefig(plot_path*"numerical_deriv_error_Mij2_$(index1)_$(index2).png")