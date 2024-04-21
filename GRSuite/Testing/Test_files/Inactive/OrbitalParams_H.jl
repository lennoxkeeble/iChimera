# convert trajectories to BL coords
xBL = [Float64[] for i in 1:n_OrbPoints]
xH = [Float64[] for i in 1:n_OrbPoints]

@inbounds Threads.@threads for i in 1:n_OrbPoints
    xBL[i] = Vector{Float64}([t[i], r[i], θ[i], ϕ[i]]);
    xH[i] = HarmonicCoords.xBLtoH(xBL[i][2:4], a, M);
end


# load self force data
SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_q_$(m/M)_tstep_$(saveat_geometric)_tol_$(kerrReltol).txt"
aSF_H = readdlm(SF_filename)

# save "constants" of motion in harmonic coordinates
E_H = zeros(n_OrbPoints); @views E_H[1] = EEi; 
Edot_H = zeros(n_OrbPoints-1);
L_H = zeros(n_OrbPoints); @views L_H[1] = LLi; 
Ldot_H = zeros(n_OrbPoints-1);
C_H = zeros(n_OrbPoints); @views C_H[1] = QQi;    # note that C in the new kludge is equal to Schmidt's Q
Cdot_H = zeros(n_OrbPoints-1);
Q_H = zeros(n_OrbPoints); @views Q_H[1] = C_H[1] + (L_H[1] - a * E_H[1])^2;    # Eq. 17
Qdot_H = zeros(n_OrbPoints-1);

# compute constants of motion using the self-force (Eqs. 30-33)
@inbounds for i=2:n_OrbPoints 
    jBLH = HarmonicCoords.jBLH(xH[i], a, M)
    ζt_α = [g_tt(xBL[i]..., a, M), g_tϕ(xBL[i]..., a, M) * jBLH[3, 1],  g_tϕ(xBL[i]..., a, M) * jBLH[3, 2],  g_tϕ(xBL[i]..., a, M) * jBLH[3, 3]]
    ζϕ_α = [g_tϕ(xBL[i]..., a, M), g_ϕϕ(xBL[i]..., a, M) * jBLH[3, 1],  g_ϕϕ(xBL[i]..., a, M) * jBLH[3, 2],  g_ϕϕ(xBL[i]..., a, M) * jBLH[3, 3]]
    ξ_μν_H = zeros(4, 4)
    ξ_μν = [Kerr.KerrMetric.ξ_μν(xBL[i]..., a, M, α, β) for α=1:4, β=1:4]

    @inbounds Threads.@threads for α=1:4
        @inbounds for β=1:4
            @inbounds for μ=1:4
                @inbounds for ν=1:4
                    @views ξ_μν_H[α, β] += ξ_μν[μ, ν] * jBLH[μ, α] * jBLH[ν, β]
                end
            end
        end
    end

    dE_dτ = 0; dL_dτ = 0; dQ_dτ = 0; dC_dτ = 0

    @inbounds for α=1:4
        dE_dτ += -ζt_α[α] * aSF_H[α]
        dE_dτ += ζϕ_α[α] * aSF_H[α]
        @inbounds for β=1:4
            dQ_dτ += 2 * ξ_μν_H[α, β] * (α==1 ? tdot[i-1] : α==2 ? rdot[i-1] : α==3 ? θdot[i-1] : ϕdot[i-1]) * aSF_H[β]
        end
    end

    # constants of motion
    @views E_H[i] = E_H[i-1] + dE_dτ * saveat_geometric
    @views L_H[i] = L_H[i-1] + dL_dτ * saveat_geometric
    @views Q_H[i] = Q_H[i-1] + dQ_dτ * saveat_geometric
    @views C_H[i] = C_H[i-1] + dC_dτ * saveat_geometric

    # fluxes
    @views Edot_H[i-1] = dE_dτ;
    @views Ldot_H[i-1] = dL_dτ;
    @views Qdot_H[i-1] = dQ_dτ;
    @views Cdot_H[i-1] = dC_dτ;
end

# computing p, e, θmin_H. Note that this may fail, e.g., for the case where e=0 and Q goes negative (in our computation), the function will run into
# square roots of negative numbers
pArray_H = zeros(n_OrbPoints); @views pArray_H[1]=p;                      # semilatus rectum p(t)
ecc_H = zeros(n_OrbPoints); @views ecc_H[1]=e;                            # ecc_Hentricity e(t)
θmin_H = zeros(n_OrbPoints); @views θmin_H[1]=θi;                         # θmin_H(t)
ι_H = zeros(n_OrbPoints); @views ι_H[1] = sign(L_H[1]) * (π/2 - θmin_H[1]);   # inclination ι_H(t)


@inbounds for i=2:n_OrbPoints
    pArray_H[i], ecc_H[i], θmin_H[i] = Kerr.ConstantsOfMotion.peθ(a, E_H[i], L_H[i], Q_H[i], C_H[i], M)
    @views ι_H[i] = sign(L_H[i]) * (π/2 - θmin_H[i])
end

### PLOT FLUXES AND CONSTANTS OF MOTION ###

nPlotTime = Int(1e4 ÷ saveat_secs)
plotτFlux = τ[2:nPlotTime+1] * MtoSecs
plotτOrbParams = τ[1:nPlotTime] * MtoSecs

# margins
left_margin = 8mm; right_margin=8mm; top_margin=8mm; bottom_margin=8mm;

gr()
plotΔE = Plots.plot(plotτFlux, Edot_H[1:nPlotTime], ylabel=L"\dot{E}", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotΔL = Plots.plot(plotτFlux, Ldot_H[1:nPlotTime], ylabel=L"\dot{L}", xlabel=L"τ\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotΔQ = Plots.plot(plotτFlux, Qdot_H[1:nPlotTime], ylabel=L"\dot{Q}", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
# title = plot(title ="p0=$(p)M, e0=$(e), a=$(a)M, ι_H0=$(round(theta_inc_o*180/π; digits=4)), q=$(q)", grid = false, showaxis = false, bottom_margin = -50Plots.px)
title = plot(title ="p0=$(p)M, e0=$(e), a=$(a)M, ι_H0=$(0), q=$(q)", grid = false, showaxis = false, bottom_margin = -50Plots.px)
fluxplot = plot(title, plotΔE, plotΔL, plotΔQ, layout = @layout([A{0.01h}; [B C D]]), size=(1500, 300))
display(fluxplot)

gr()
plotE = Plots.plot(plotτOrbParams, E_H[1:nPlotTime], ylabel=L"E", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotL = Plots.plot(plotτOrbParams, L_H[1:nPlotTime], ylabel=L"L", xlabel=L"τ\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotC = Plots.plot(plotτOrbParams, C_H[1:nPlotTime], ylabel=L"C", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

orbitalParamsPlot=plot(plotE, plotL, plotC, layout=(1, 3), size=(1500, 300))
plotP = Plots.plot(plotτOrbParams, pArray_H[1:nPlotTime], ylabel=L"p", legend=:false, 
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

plotEcc_H = Plots.plot(plotτOrbParams, ecc_H[1:nPlotTime], ylabel=L"e", xlabel=L"τ\ (s)", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)
plotι_H = Plots.plot(plotτOrbParams, ι_H[1:nPlotTime], ylabel=L"ι_H", legend=:false,
    left_margin		=  left_margin,
    right_margin	=  right_margin,
    top_margin		=  top_margin,
    bottom_margin	=  bottom_margin)

orbitalParamsPlot=plot(plotE, plotL, plotC, plotP, plotEcc_H, plotι_H, layout=(2, 3), size=(1500, 600))
display(orbitalParamsPlot)