module EvolveConstants
using StaticArrays
using ..Kerr
using ..CircularNonEquatorial
using ..HarmonicCoords

function Evolve_BL(Δt::Float64, a::Float64, t::Float64, r::Float64, θ::Float64, ϕ::Float64, Γ::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64,
    aSF_BL::AbstractVector{Float64}, EE::AbstractArray, Edot::AbstractArray, LL::AbstractArray, Ldot::AbstractArray, QQ::AbstractArray, Qdot::AbstractArray,
    CC::AbstractArray, Cdot::AbstractArray, pArray::AbstractArray, ecc::AbstractArray, θmin::AbstractArray, M::Float64, nPoints::Int64)
    # first load orbital constants of previous geodesic (recall that we compute updated constants to move to the next geodesic in the inspiral)
    E0 = last(EE); L0 = last(LL); Q0 = last(QQ); C0 = last(CC); p0 = last(pArray); e0 = last(ecc); θmin_0 = last(θmin);
    
    #### update E, L, Q, C ####
    # update E
    dE_dt = (- Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 1, 1) * aSF_BL[1] - Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 4, 1) * aSF_BL[4])/Γ    # Eq. 30
    push!(Edot, dE_dt)

    # update L
    if e0==0.0 && θmin_0==π/2   # circular equatorial
        ang_velocity = 1/(a + M * p * sqrt(p))
        dL_dt = dE_dt / ang_velocity    # page 10 first paragraph on right column
        push!(Ldot, dE_dt / ang_velocity)
    else
        dL_dt = (Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 1, 4) * aSF_BL[1] + Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, 4, 4) * aSF_BL[4])/Γ    # Eq. 31
        push!(Ldot, dL_dt)
    end
    
    # update C, Q
    if θmin_0==π/2  # equatorial
        dC_dt = 0.0
        dQ_dt = - 2 * (a * E0 - L0) * (dL_dt - a * dE_dt)
    elseif e0==0.0 && θmin_0!=π/2   # circular non-equatorial
        r0 = p0 * M
        dC_dt = CircularNonEquatorial.Cdot(r0, dE_dt, dL_dt, a, E0, L0, C0, M)
        dQ_dt = dC_dt - 2 * (a * E0 - L0) * (dL_dt - a * dE_dt)
    else   # generic case
        dQ_dt = 0.0
        @inbounds for α=1:4, β=1:4
            dQ_dt += 2 * Kerr.KerrMetric.ξ_μν(t, r, θ, ϕ, a, M, α, β) * (α==1 ? 1. : α==2 ? rdot : α==3 ? θdot : ϕdot) * aSF_BL[β]    # Eq. 32
        end
        dC_dt = dQ_dt + 2 * (a * E0 - L0) * (dL_dt - a * dE_dt)
    end

    push!(Qdot, dQ_dt)
    push!(Cdot, dC_dt)

    # compute updated E, L, Q, C and store
    E1 = E0 + dE_dt * Δt
    L1 = L0 + dL_dt * Δt
    Q1 = Q0 + dQ_dt * Δt
    C1 = C0 + dC_dt * Δt

    push!(EE, E1)
    push!(LL, L1)
    push!(QQ, Q1)
    push!(CC, C1)

    ### update p, e, θmin ####

    # computing p, e, θmin_BL from updated constants. In the circular non-equatorial case we implement the special case to preserve the circularity of the orbit
    if e0==0.0 && θmin_0!=π/2   # circular non-equatorial
        dr0_dt = CircularNonEquatorial.r0dot(r0, dE_dt, dL_dt, a, E0, L0, C0, M)
        pp = p0 + (dr0_dt / M) * Δt
        ee = 0.0

        ## now calculate θmin
        # coefficients of polynomial in Eq. E33
        c0 = C1 / (a^2 * (1.0 - E1^2))
        c1 = -1.0 - (L1^2 + C1) / (a^2 * (1.0 - E1^2))
        θθ = acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))
    else
        pp, ee, θθ = Kerr.ConstantsOfMotion.peθ_gsl(a, E1, L1, Q1, C1, M)
        # preserve circularity and/or equatorial orbit
        if e0 == 0.0
            ee = 0.0
        end

        if θmin_0 == π/2
            θθ = π/2
        end
    end

    push!(pArray, pp)
    push!(ecc, ee)
    push!(θmin, θθ)
end

Killing_temporal_H(a::Float64, xH::AbstractArray, t::Float64, r::Float64, θ::Float64, ϕ::Float64, M::Float64) = @SVector [Kerr.KerrMetric.g_tt(t, r, θ, ϕ, a, M), Kerr.KerrMetric.g_tϕ(t, r, θ, ϕ, a, M) * HarmonicCoords.∂ϕ_∂xH(xH, a, M),
Kerr.KerrMetric.g_tϕ(t, r, θ, ϕ, a, M) * HarmonicCoords.∂ϕ_∂yH(xH, a, M), Kerr.KerrMetric.g_tϕ(t, r, θ, ϕ, a, M) * HarmonicCoords.∂ϕ_∂zH(xH, a, M)]
Killing_axial_H(a::Float64, xH::AbstractArray, t::Float64, r::Float64, θ::Float64, ϕ::Float64, M::Float64) = @SVector [Kerr.KerrMetric.g_tϕ(t, r, θ, ϕ, a, M), Kerr.KerrMetric.g_ϕϕ(t, r, θ, ϕ, a, M) * HarmonicCoords.∂ϕ_∂xH(xH, a, M),
Kerr.KerrMetric.g_ϕϕ(t, r, θ, ϕ, a, M) * HarmonicCoords.∂ϕ_∂yH(xH, a, M), Kerr.KerrMetric.g_ϕϕ(t, r, θ, ϕ, a, M) * HarmonicCoords.∂ϕ_∂zH(xH, a, M)]
function Killing_tensor_H(a::Float64, xH::AbstractArray, t::Float64, r::Float64, θ::Float64, ϕ::Float64, M::Float64) 
    tensor = zeros(4, 4)
    jBLH = HarmonicCoords.jBLH(xH, a, M)
    ξtt = Kerr.KerrMetric.ξ_tt(t, r, θ, ϕ, a, M) 
    ξtϕ = Kerr.KerrMetric.ξ_tϕ(t, r, θ, ϕ, a, M) 
    ξrr = Kerr.KerrMetric.ξ_rr(t, r, θ, ϕ, a, M) 
    ξθθ = Kerr.KerrMetric.ξ_θθ(t, r, θ, ϕ, a, M) 
    ξϕϕ = Kerr.KerrMetric.ξ_ϕϕ(t, r, θ, ϕ, a, M)

    # time components
    tensor[1, 1] = ξtt
    tensor[1, 2] = ξtϕ * jBLH[3, 1]; tensor[2, 1] = tensor[1, 2]
    tensor[1, 3] = ξtϕ * jBLH[3, 2]; tensor[3, 1] = tensor[1, 3]
    tensor[1, 4] = ξtϕ * jBLH[3, 3]; tensor[4, 1] = tensor[1, 4]

    # spatial components
    tensor[2, 2] = ξrr * jBLH[1, 1] * jBLH[1, 1] + ξθθ * jBLH[2, 1] * jBLH[2, 1] + ξϕϕ * jBLH[3, 1] * jBLH[3, 1]
    tensor[2, 3] = ξrr * jBLH[1, 1] * jBLH[1, 2] + ξθθ * jBLH[2, 1] * jBLH[2, 2] + ξϕϕ * jBLH[3, 1] * jBLH[3, 2]; tensor[3, 2] = tensor[2, 3]
    tensor[2, 4] = ξrr * jBLH[1, 1] * jBLH[1, 3] + ξθθ * jBLH[2, 1] * jBLH[2, 3] + ξϕϕ * jBLH[3, 1] * jBLH[3, 3]; tensor[4, 2] = tensor[2, 4]

    tensor[3, 3] = ξrr * jBLH[1, 2] * jBLH[1, 2] + ξθθ * jBLH[2, 2] * jBLH[2, 2] + ξϕϕ * jBLH[3, 2] * jBLH[3, 2]
    tensor[3, 4] = ξrr * jBLH[1, 2] * jBLH[1, 3] + ξθθ * jBLH[2, 2] * jBLH[2, 3] + ξϕϕ * jBLH[3, 2] * jBLH[3, 3]; tensor[4, 3] = tensor[3, 4]

    tensor[4, 4] = ξrr * jBLH[1, 3] * jBLH[1, 3] + ξθθ * jBLH[2, 3] * jBLH[2, 3] + ξϕϕ * jBLH[3, 3] * jBLH[3, 3]

    return tensor
end

function Evolve_Harm(Δt::Float64, a::Float64, xH::AbstractArray, t::Float64, r::Float64, θ::Float64, ϕ::Float64, Γ::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, aSF_H::AbstractVector{Float64}, EE::AbstractArray, Edot::AbstractArray, LL::AbstractArray, Ldot::AbstractArray, QQ::AbstractArray, Qdot::AbstractArray, CC::AbstractArray, Cdot::AbstractArray, pArray::AbstractArray, ecc::AbstractArray, θmin::AbstractArray, M::Float64, nPoints::Int64)
    E0 = last(EE); L0 = last(LL); Q0 = last(QQ); C0 = last(CC); p0 = last(pArray); e0 = last(ecc); θmin_0 = last(θmin);
    temporal_killing = Killing_temporal_H(a, xH, t, r, θ, ϕ, M)
    axial_killing = Killing_axial_H(a, xH, t, r, θ, ϕ, M)
    tensor_killing = Killing_tensor_H(a, xH, t, r, θ, ϕ, M)

    #### ELQ ####
    dE_dt = -(temporal_killing[1] * aSF_H[1] + temporal_killing[2] * aSF_H[2] + temporal_killing[3] * aSF_H[3] + temporal_killing[4] * aSF_H[4])/Γ    # Eq. 30
    push!(Edot, dE_dt)

    dL_dt = (axial_killing[1] * aSF_H[1] + axial_killing[2] * aSF_H[2] + axial_killing[3] * aSF_H[3] + axial_killing[4] * aSF_H[4])/Γ    # Eq. 31
    push!(Ldot, dL_dt)
    
    dQ_dt = 0
    @inbounds for α=1:4, β=1:4
        dQ_dt += 2 * tensor_killing[α, β] * (α==1 ? 1. : α==2 ? rdot : α==3 ? θdot : ϕdot) * aSF_H[β]    # Eq. 32
    end
    push!(Qdot, dQ_dt)

    dC_dt = dQ_dt + 2 * (a * E0 - L0) * (dL_dt - a * dE_dt)
    push!(Cdot, dC_dt)

    # compute updated E, L, Q, C and store
    E1 = E0 + dE_dt * Δt
    L1 = L0 + dL_dt * Δt
    Q1 = Q0 + dQ_dt * Δt
    C1 = C0 + dC_dt * Δt

    push!(EE, E1)
    push!(LL, L1)
    push!(QQ, Q1)
    push!(CC, C1)


    ### update p, e, θmin ####

    # computing p, e, θmin_BL from updated constants. In the circular non-equatorial case we implement the special case to preserve the circularity of the orbit
    if e0==0.0 && θmin_0!=π/2   # circular non-equatorial
        dr0_dt = CircularNonEquatorial.r0dot(r0, dE_dt, dL_dt, a, E0, L0, C0, M)
        pp = p0 + (dr0_dt / M) * Δt
        ee = 0.0

        ## now calculate θmin
        # coefficients of polynomial in Eq. E33
        c0 = C1 / (a^2 * (1.0 - E1^2))
        c1 = -1.0 - (L1^2 + C1) / (a^2 * (1.0 - E1^2))
        θθ = acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))
    else
        pp, ee, θθ = Kerr.ConstantsOfMotion.peθ_gsl(a, E1, L1, Q1, C1, M)
        # preserve circularity and/or equatorial orbit
        if e0 == 0.0
            ee = 0.0
        end

        if θmin_0 == π/2
            θθ = π/2
        end
    end

    push!(pArray, pp)
    push!(ecc, ee)
    push!(θmin, θθ)
end

end