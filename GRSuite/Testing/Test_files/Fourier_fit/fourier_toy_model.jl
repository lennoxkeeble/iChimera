include("/home/lkeeble/GRSuite/main.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/curve_fit.jl")
include("/home/lkeeble/GRSuite/Testing/Test_modules/TestMultipoleFit.jl")
include("/home/lkeeble/GRSuite/Testing/Test_files/Fourier_fit/toy_model_functions.jl")
using .ToyFourierFunctions, .FourierFit, LsqFit, DelimitedFiles, Plots, Plots.Measures, LaTeXStrings

function compute_deviation(y_true::Vector{Float64}, y_approx::Vector{Float64})
    return @. abs(100 * (y_true-y_approx)/y_true)
end

function compute_fit(ydata::Vector{Float64}, xdata::Vector{Float64}, nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, derivatives::Vector{Int64}, data_path::String)
    fit_save_fname=data_path*"toy_model_nHarm_$(nHarm).txt"
    isfile(fit_save_fname) ? p0=readdlm(fit_save_fname)[:] : p0 = Float64[];

    # carry out fit
    Ω, ffit, fitted_data = FourierFit.fourier_fit(xdata, ydata, Ωr, Ωθ, Ωϕ, nHarm, p0=p0)

    fit_params=coef(ffit)

    # save fit #
    open(fit_save_fname, "w") do io
        writedlm(io, fit_params)
    end

    fitted_data = [FourierFit.curve_fit_functional_derivs(xdata, Ω, fit_params, N) for N in derivatives]

    return fitted_data
end

data_path="/home/lkeeble/GRSuite/Testing/Test_results/Toy_model/Data/";
plot_path="/home/lkeeble/GRSuite/Testing/Test_results/Toy_model/Plots/";
mkpath(data_path)
mkpath(plot_path)

###### first, let us check that the functions from mathematica match the julia code #####
nHarmMMA=3; Ωr=0.073209432; Ωθ=0.047683593; Ωϕ=0.0027498324; derivatives=[0, 2, 4, 6];

Ωtrue=Float64[]
@inbounds for i_r in 0:nHarmMMA
    @inbounds for i_θ in 0:(nHarmMMA+i_r)
        @inbounds for i_ϕ in 0:(nHarmMMA+i_r+i_θ)
            append!(Ωtrue, i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ)
        end
    end
end

# choose a time range and resolution #
tMax = minimum(@. 2π/[Ωr, Ωθ, Ωϕ]); nPoints=500; Δt = tMax/(nPoints-1); t=0:Δt:tMax|>collect;

mma_deriv_data = [ToyFourierFunctions.f.(t, Ωr, Ωθ, Ωϕ), ToyFourierFunctions.f2.(t, Ωr, Ωθ, Ωϕ), ToyFourierFunctions.f4.(t, Ωr, Ωθ, Ωϕ), ToyFourierFunctions.f6.(t, Ωr, Ωθ, Ωϕ)]

# construct coefficients in the same way as in the mathematica nb #
coeffs = [(-1.0)^i * (i/100.) for i=1:(2*size(Ωtrue, 1))]

# now test the two functions we using for fitting #
functional_form=zeros(size(t, 1)); FourierFit.curve_fit_functional(functional_form, t, Ωtrue, coeffs, size(Ωtrue, 1));
println("Max % difference between julia functional form and MMA functional form = $(100 * maximum(functional_form - mma_deriv_data[1]))")

julia_deriv_data = [FourierFit.curve_fit_functional_derivs(t, Ωtrue, coeffs, N) for N in derivatives]

max_analytical_error = maximum([maximum(@. (mma_deriv_data[i]-julia_deriv_data[i])/mma_deriv_data[i]) for i in eachindex(derivatives)])
println("Max % difference between julia derivative formula and MMA explicit formulas = $(100 * max_analytical_error)")

###### now let us investigate the fitting routine #####

# compute fitting frequencies #
Ωr=0.1    # introduce error to see how it affects fits
Ω=Float64[]
@inbounds for i_r in 0:nHarmMMA
    @inbounds for i_θ in 0:(nHarmMMA+i_r)
        @inbounds for i_ϕ in 0:(nHarmMMA+i_r+i_θ)
            append!(Ω, i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ)
        end
    end
end

# generate mock data from mathematica function #
ydata = mma_deriv_data[1]; nHarmFits=[1, 2, 3, 4, 5]; nHarmFits=[1, 2, 3, 4];

@time fitted_data_nH_1 = compute_fit(ydata, t, nHarmFits[1], Ωr, Ωθ, Ωϕ, derivatives, data_path);
@time fitted_data_nH_2 = compute_fit(ydata, t, nHarmFits[2], Ωr, Ωθ, Ωϕ, derivatives, data_path);
@time fitted_data_nH_3 = compute_fit(ydata, t, nHarmFits[3], Ωr, Ωθ, Ωϕ, derivatives, data_path);
@time fitted_data_nH_4 = compute_fit(ydata, t, nHarmFits[4], Ωr, Ωθ, Ωϕ, derivatives, data_path);
# @time fitted_data_nH_5 = compute_fit(ydata, t, 4, Ωr, Ωθ, Ωϕ, derivatives, data_path);

# print errors for 1 harmonic
N=nHarmFits[1]
println("\n$(N) harmonics:")
TestMultipoleFit.print_errors(ydata, fitted_data_nH_1[1], 0)
TestMultipoleFit.print_errors(mma_deriv_data[2], fitted_data_nH_1[2], 2)
TestMultipoleFit.print_errors(mma_deriv_data[3], fitted_data_nH_1[3], 4)
TestMultipoleFit.print_errors(mma_deriv_data[4], fitted_data_nH_1[4], 6)

# print errors for 2 harmonics
N=nHarmFits[2]
println("\n$(N) harmonics:")
TestMultipoleFit.print_errors(ydata, fitted_data_nH_2[1], 0)
TestMultipoleFit.print_errors(mma_deriv_data[2], fitted_data_nH_2[2], 2)
TestMultipoleFit.print_errors(mma_deriv_data[3], fitted_data_nH_2[3], 4)
TestMultipoleFit.print_errors(mma_deriv_data[4], fitted_data_nH_2[4], 6)

# print errors for 3 harmonics
N=nHarmFits[3]
println("\n$(N) harmonics:")
TestMultipoleFit.print_errors(ydata, fitted_data_nH_3[1], 0)
TestMultipoleFit.print_errors(mma_deriv_data[2], fitted_data_nH_3[2], 2)
TestMultipoleFit.print_errors(mma_deriv_data[3], fitted_data_nH_3[3], 4)
TestMultipoleFit.print_errors(mma_deriv_data[4], fitted_data_nH_3[4], 6)

# print errors for 4 harmonics
N=nHarmFits[4]
println("\n$(N) harmonics:")
TestMultipoleFit.print_errors(ydata, fitted_data_nH_3[1], 0)
TestMultipoleFit.print_errors(mma_deriv_data[2], fitted_data_nH_4[2], 2)
TestMultipoleFit.print_errors(mma_deriv_data[3], fitted_data_nH_4[3], 4)
TestMultipoleFit.print_errors(mma_deriv_data[4], fitted_data_nH_4[4], 6)

# # print errors for 4 harmonics
# N=4
# println("\n$(N) harmonics:")
# TestMultipoleFit.print_errors(ydata, fitted_data_nH_5[1], 0)
# TestMultipoleFit.print_errors(mma_deriv_data[2], fitted_data_nH_5[2], 2)
# TestMultipoleFit.print_errors(mma_deriv_data[3], fitted_data_nH_5[3], 4)
# TestMultipoleFit.print_errors(mma_deriv_data[4], fitted_data_nH_5[4], 6)

#=

Carrying out fit
    13645.850804 seconds (793.93 M allocations: 192.525 GiB, 0.09% gc time, 0.01% compilation time)
Carrying out fit
    3585.417696 seconds (438.59 M allocations: 91.350 GiB, 0.16% gc time)
Carrying out fit
    726.751219 seconds (136.99 M allocations: 27.077 GiB, 0.26% gc time)
Carrying out fit
    0.451898 seconds (94.16 k allocations: 9.604 MiB, 87.99% compilation time)
Carrying out fit
    0.005755 seconds (4.67 k allocations: 1.131 MiB)

=#

##### compute errors #####
nH1error = [compute_deviation(mma_deriv_data[i], fitted_data_nH_1[i]) for i in eachindex(derivatives)]; 
nH2error = [compute_deviation(mma_deriv_data[i], fitted_data_nH_2[i]) for i in eachindex(derivatives)]; 
nH3error = [compute_deviation(mma_deriv_data[i], fitted_data_nH_3[i]) for i in eachindex(derivatives)]; 
nH4error = [compute_deviation(mma_deriv_data[i], fitted_data_nH_4[i]) for i in eachindex(derivatives)]; 
# nH5error = [compute_deviation(mma_deriv_data[i], fitted_data_nH_5[i]) for i in eachindex(derivatives)];

##### plot errors for 2, 3, and 5 harmonics #####
gr()
# plot attributes #
shapes = [:rect :circle :rect]
colors = [:green :red :blue]
alpha=1.0
ms=[1.5 1.5 1.5]
left_margin=8mm; bottom_margin=8mm; right_margin=8mm
xtickfontsize=18;ytickfontsize=18;xguidefontsize=18;yguidefontsize=18;legendfontsize=18
ylim=(1e-9, 1e5)
xtick_vals = range(start=t[1], stop=last(t), length=11)|>collect;
xlim=(t[1], last(t))

# derivative order 0 #
i=1
plot_0 = scatter([t for i in 1:3], [nH2error[i], nH3error[i], nH4error[i]], markershape=shapes, color=colors, label=[L"N=%$(nHarmFits[2])" L"N=%$(nHarmFits[3])" L"N=%$(nHarmFits[4])"], markersize=ms,
yscale=:log10, framestyle=:box, ylims=ylim, yticks=([10.0^(i) for i=(-9):5], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-9):5]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=2))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\,(\%)", xlabel=L"t", alpha=alpha, markerstrokewidth=0,
xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize)
annotate!(minimum(t)+0.2 * (maximum(t)-minimum(t)), 1e4, text(L"f^{(%$(derivatives[i]))}"))


# derivative order 2 #
i=2
plot_2 = scatter([t for i in 1:3], [nH2error[i], nH3error[i], nH4error[i]], markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, framestyle=:box, ylims=ylim, yticks=([10.0^(i) for i=(-9):5], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-9):5]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=2))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\,(\%)", xlabel=L"t", alpha=alpha, markerstrokewidth=0,
xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize)
annotate!(minimum(t)+0.2 * (maximum(t)-minimum(t)), 1e4, text(L"f^{(%$(derivatives[i]))}"))


# derivative order 4 #
i=3
plot_4 = scatter([t for i in 1:3], [nH2error[i], nH3error[i], nH4error[i]], markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, framestyle=:box, ylims=ylim, yticks=([10.0^(i) for i=(-9):5], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-9):5]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=2))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\,(\%)", xlabel=L"t", alpha=alpha, markerstrokewidth=0,
xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize)
annotate!(minimum(t)+0.2 * (maximum(t)-minimum(t)), 1e4, text(L"f^{(%$(derivatives[i]))}"))


# derivative order 6 #
i=4
plot_6 = scatter([t for i in 1:3], [nH2error[i], nH3error[i], nH4error[i]], markershape=shapes, color=colors, legend=:false, markersize=ms,
yscale=:log10, framestyle=:box, ylims=ylim, yticks=([10.0^(i) for i=(-9):5], [iseven(i) ? L"10^{%$(i)}" : "" for i=(-9):5]), xlims=xlim, 
xticks = (xtick_vals, [isodd(i) ? L"%$(round(xtick_vals[i]; digits=2))" : "" for i=1:size(xtick_vals, 1)]),
left_margin=left_margin, bottom_margin=bottom_margin, right_margin=right_margin, ylabel=L"\textrm{Percentage}\;\textrm{error}\,(\%)", xlabel=L"t", alpha=alpha, markerstrokewidth=0,
xtickfontsize=xtickfontsize,ytickfontsize=ytickfontsize,xguidefontsize=xguidefontsize,yguidefontsize=yguidefontsize,legendfontsize=legendfontsize)
annotate!(minimum(t)+0.2 * (maximum(t)-minimum(t)), 1e4, text(L"f^{(%$(derivatives[i]))}"))

plot(plot_0, plot_2, plot_4, plot_6, layout=(2,2), size=(1600, 1200))
savefig(plot_path*"tweaked_toy_model_errors.png")