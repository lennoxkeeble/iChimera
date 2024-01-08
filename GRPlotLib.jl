module GRPlotLib
using LaTeXStrings
using DelimitedFiles
using Plots.PlotMeasures
using Plots
using ..EffPotential
using Roots

# plot effective potential for an arbitrary metric, specifying Ei, Li_{z}
function plot_vEff(θi::Float64, ϕi::Float64, a::Float64, M::Float64, E::Float64, L::Float64, m::Float64, g_tt::Function, g_tϕ::Function, g_ϕϕ::Function, ylims::Tuple=(-1, 1), x_width::Int=800; y_width::Int =600)    
    x_res = 0.01
    # margins
    left_margin = 8mm; right_margin=8mm; top_margin=8mm; bottom_margin=8mm;
    # font sizes
    xtickfontsize=10; ytickfontsize=10; guidefontsize=20; # guidefontsize is label font size

    # range of r values
    r = 2M:x_res:30M |> collect

    V = EffPotential.vEff.(0., r, θi, ϕi, a, M, E, L, m, g_tt, g_tϕ, g_ϕϕ)
    gr()
    vPlot=Plots.plot(r, V,
    ylims=ylims,
    xlabel=L"r\mathrm{\ (M)}",
    ylabel=L"V_{\mathrm{eff}}",
    legend=:true,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    label=:false,
    size=(x_width, y_width),
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin,
    xtickfontsize = xtickfontsize,
    ytickfontsize = ytickfontsize,
    guidefontsize = guidefontsize,
    framestyle=:box
    )

    # x-axis
    hline!([0], linecolor=:black, linestyle=:solid, label=false)

    # # plot vertically dashed lines where vEff crosses x-axis 
    # for i=2:size(V, 1)
    #     if (V[i-1] < 0 && V[i] > 0) || (V[i-1] > 0 && V[i] < 0)
    #         vline!([r[i]], linecolor=:black, linestyle=:dash, label=false)
    #         annotate!(r[i] + 50 * x_res, ylims[1] + 0.1 * (ylims[2] - ylims[1]), Plots.text(L"r\approx %$(r[i])", :left, 6))
    #     end
    # end

    f(r) = EffPotential.vEff(0., r, θi, ϕi, a, M, E, L, m, g_tt, g_tϕ, g_ϕϕ)
    roots = find_zeros(f, first(r), last(r));
    if !isempty(roots)
        for r in roots
            vline!([r], linecolor=:black, linestyle=:dash, label=false)
            annotate!(r + 50 * x_res, ylims[1] + 0.1 * (ylims[2] - ylims[1]), Plots.text(L"r= %$(round(r; digits=4))", :left, 6))
        end
    end
    display(vPlot)
end


# saves a html file containing a 3D plot of the trajectory
function plot_orbit(ode_sol_fname::String, plot_fname::String, zlims::Tuple=(-5, 5); plot_path::String="Plots/", kwargs...)
    sol = readdlm(ode_sol_fname)
    t = sol[2, :]; r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :]; tdot = sol[6, :]; rdot = sol[7, :]; θdot = sol[8, :]; ϕdot = sol[9, :];

    # project onto cartesian coordinates in flat space
    x = @. r * sin(θ) * cos(ϕ);   # Eq. 6.3
    y = @. r * sin(θ) * sin(ϕ);   # Eq. 6.4
    z = @. r * cos(θ);   # Eq. 6.5

    # 3d plot
    plotlyjs()
    orbit_plot = Plots.plot(x, y, z, markersize=0.5, background_color = :black, color = :white, axis=([], false), legend=false, zlims=zlims)
    mkpath(plot_path)
    Plots.html(orbit_plot, plot_path * plot_fname)
    println("File saved: " *  plot_path * plot_fname)
    display(orbit_plot)
end

# saves a plot of the trajectory projected onto the (Cartesian) xy-plane
function plot_xy_orbit(ode_sol_fname::String, plot_fname::String, xlims::Tuple=(-11, 11), ylims::Tuple=(-11, 11); plot_path::String="Plots/", kwargs...)
    sol = readdlm(ode_sol_fname)
    t = sol[2, :]; r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :];
    # project onto cartesian coordinates in flat space
    x = @. r * sin(θ) * cos(ϕ);   # Eq. 6.3
    y = @. r * sin(θ) * sin(ϕ);   # Eq. 6.4
    z = @. r * cos(θ);   # Eq. 6.5

    # plot attributes
    gr()
    equatorial_plot = Plots.plot(x, y, xlabel=L"\textrm{x\ (M)}", ylabel=L"\textrm{y\ (M)}"; kwargs...)
    mkpath(plot_path)
    savefig(equatorial_plot, plot_path * plot_fname)
    println("File saved: " *  plot_path * plot_fname)
    display(equatorial_plot)
end

# saves a plot of the waveform h_{+}
function plot_waveform(waveform_fname::String, plot_fname::String; plot_path::String="Plots/", kwargs...)
    wf = readdlm(waveform_fname)
    t = wf[1, :]; hplus = wf[2, :]
    gr()
    p = plot(t, hplus, xlabel = L"\textrm{t\ (M)}", ylabel = L"h_{+}"; kwargs...)
    mkpath(plot_path)
    savefig(p, plot_path * plot_fname)
    display(p)
end

# saves a gif containing a plot of the trajectory concurrently with its xy-plane-projected trajectory. Make sure plot_fname ends in ".gif"
function waveform_xy_orbit_gif(ode_sol_fname::String, waveform_fname::String, plot_fname::String; plot_path::String="Plots/", res::Int=25, xTicks, ylims::Tuple, yTicks, fps::Int=15)
    sol = readdlm(ode_sol_fname)
    r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :];
    wf = readdlm(waveform_fname)
    t = wf[1, :]; h_plus = wf[2, :]
    xlims=(trunc(Int, minimum(t)), trunc(Int, maximum(t)))

    # project onto cartesian coordinates in flat space
    x = @. r * sin(θ) * cos(ϕ);   # Eq. 6.3
    y = @. r * sin(θ) * sin(ϕ);   # Eq. 6.4
    z = @. r * cos(θ);   # Eq. 6.5

        
    # plot dimensions
    ## waveform plot
    wave_im_ratio = 5   # width / height
    wave_height = 300; wave_width = wave_height * wave_im_ratio

    ## orbit plot
    orbit_width = wave_width; orbit_height = orbit_width   # square plot for orbit

    # margins
    left_margin = 8mm; right_margin=8mm; top_margin=8mm; bottom_margin=8mm;

    # font sizes
    xtickfontsize=20; ytickfontsize=20; guidefontsize=30;

    anim = Animation()
        
    for i in 1:25:length(t)
        x_gif = x[1:i]; y_gif = y[1:i]; t_gif = t[1:i]; h_plus_gif = h_plus[1:i];
        
        # waveform
        p1 = plot(t_gif, h_plus_gif,
        color=:blue,
        xlims=xlims,
        legend = :false,
        xticks=(xlims[1]:xTicks:xlims[2], ["$(xlims[1] + n * xTicks)" for n=0:(xlims[2]-xlims[1])÷xTicks]),
        ylims=ylims,
        yticks=(ylims[1]:yTicks:ylims[2], ["$(ylims[1] + n * yTicks)" for n=0:(ylims[2]-ylims[1])÷yTicks]),
        xlabel = L"\textrm{Time\ (M)}",
        ylabel = L"h_{+}",
        size=(wave_width, wave_height),
        left_margin		=  left_margin,
        right_margin	=  right_margin,
        top_margin		=  top_margin,
        bottom_margin	=  bottom_margin,
        xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize,
        guidefontsize = guidefontsize)
        
        # orbit
        p2 = plot(x_gif, y_gif,
        color=:black,
        legend = :false,
        xlabel=L"x",
        xlims=(minimum(x)-1, maximum(x)+1),
        ylabel=L"y", 
        ylims=(minimum(y)-1, maximum(y)+1),
        size=(orbit_width, orbit_height),
        left_margin		=  left_margin,
        right_margin	=  right_margin,
        top_margin		=  top_margin,
        bottom_margin	=  bottom_margin,
        xtickfontsize = xtickfontsize,
        ytickfontsize = ytickfontsize,
        guidefontsize = guidefontsize)

        # plot particle
        scatter!([last(x_gif)], [last(y_gif)], color=:blue)
        l = @layout [
            a{1.0w}
            b{0.2h}
        ]
        subplot= plot(p2, p1, layout = l)
        frame(anim, subplot)

    end
    mkpath(plot_path)
    gif(anim, plot_path * plot_fname, fps=fps)
end
    
end