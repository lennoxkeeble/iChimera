#=

    In this module we write functions which compute the fluxes in the kludge scheme described in arXiv:1109.0572v2. Given a value of the local, post-Minkowskian self-force computed at some point on a geodsic, we compute the fluxes in the orbital 
    constants E, L, Q, C and update their values in-place.

=#

module EvolveConstants
using StaticArrays
using ..Kerr
using ..ConstantsOfMotion
using ..CircularNonEquatorial
using ..HarmonicCoords

"""
    Evolve_BL(args...) -> ReturnType

Short description of what the function does.

# Arguments
- `Δt::Float64`: time elapsed along current geodesi `piece`.
- `a::Float64`: Kerr black hole spin parameter, 0 < a < 1.
- `r::Float64`: Boyer-Lindquist radial coordinate.
- `θ::Float64`: Boyer-Lindquist polar coordinate.
- `ϕ::Float64`: Boyer-Lindquist azimuthal coordinate.
- `Γ::Float64`: relativistic gamma-factor.
- `r_dot::Float64`: Coordinate-time first derivative of the radial coordinate.
- `θ_dot::Float64`: Coordinate-time first derivative of the polar coordinate.
- `ϕ_dot::Float64`: Coordinate-time first derivative of the azimuthal coordinate.
- `aSF_BL::AbstractVector{Float64}`: post-Minkowskian self-force components in Boyer-Lindquist coordinates.
- `EE::AbstractVector{Float64}`: energy per unit mass of each pieacewise geodesic in the inspiral.
- `Edot::AbstractVector{Float64}`: energy flux of each pieacewise geodesic in the inspiral.
- `LL::AbstractVector{Float64}`: axial angular momentum per unit mass of each pieacewise geodesic in the inspiral.
- `Ldot::AbstractVector{Float64}`: angular momentum flux of each pieacewise geodesic in the inspiral.
- `QQ::AbstractVector{Float64}`: alternative Carter constant of each pieacewise geodesic in the inspiral.
- `Qdot::AbstractVector{Float64}`: alternative Carter constant flux of each pieacewise geodesic in the inspiral.
- `CC::AbstractVector{Float64}`: Carter constant of each pieacewise geodesic in the inspiral.
- `Cdot::AbstractVector{Float64}`: Carter constant flux of each pieacewise geodesic in the inspiral.
- `pArray::AbstractVector{Float64}`: semi-latus rectum of each pieacewise geodesic in the inspiral.
- `ecc::AbstractVector{Float64}`: eccentricity of each pieacewise geodesic in the inspiral.
- `θminArray::AbstractVector{Float64}`: minimum polar angle of each pieacewise geodesic in the inspiral.

# Returns
- `nothing`: appends updated values of the orbital constants to the input arrays.

# Notes
- The orbital constants taken as input are time series of their value on each piecewise geodesic that has been computed up to the current point in the inspiral. The function updates the values of the orbital constants, after which the evolution jumps
to the next piecewise geodesic in the inspiral (this is the method of osculating orbits).
- Strictly speaking, these fluxes are local, hence why they are functions of not just the orbital parameters, but also the position on the geodesic. In order to account for this, we take each piecewise geodesic to be ''small'' enough temporally
so that the fluxes can be considered constant over each geodesic. In practice, one can carry out convergence tests to ensure that the temporal length of each geodesic is small enough so that one does not accumulate errors in the evolution by computing
the flux at some point on a given piecewise geodesic versus another.
- The relevant equations are Eqs. 30-33 in arXiv:1109.0572v2.
"""

# updates the orbital constants using the self-force in Boyer-Lindquist coordinates
function Evolve_BL(Δt::Float64, a::Float64, r::Float64, θ::Float64, ϕ::Float64, Γ::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, aSF_BL::AbstractVector{Float64}, 
    EE::AbstractVector{Float64}, Edot::AbstractVector{Float64}, LL::AbstractVector{Float64}, Ldot::AbstractVector{Float64}, QQ::AbstractVector{Float64}, Qdot::AbstractVector{Float64}, CC::AbstractVector{Float64}, Cdot::AbstractVector{Float64},
    pArray::AbstractVector{Float64}, ecc::AbstractVector{Float64}, θminArray::AbstractVector{Float64})
    # first load orbital constants of previous geodesic (recall that we compute updated constants to move to the next geodesic in the inspiral)
    E0 = last(EE); L0 = last(LL); Q0 = last(QQ); C0 = last(CC); p0 = last(pArray); e0 = last(ecc); θmin_0 = last(θminArray);
    
    #### update E, L, Q, C ####
    # update E
    dE_dt = (- Kerr.KerrMetric.g_μν(r, θ, ϕ, a, 1, 1) * aSF_BL[1] - Kerr.KerrMetric.g_μν(r, θ, ϕ, a, 4, 1) * aSF_BL[4])/Γ    # Eq. 30
    push!(Edot, dE_dt)

    # update L
    if e0==0.0 && θmin_0==π/2   # circular equatorial
        ang_velocity = 1/(a + p * sqrt(p))
        dL_dt = dE_dt / ang_velocity    # page 10 first paragraph on right column
        push!(Ldot, dE_dt / ang_velocity)
    else
        dL_dt = (Kerr.KerrMetric.g_μν(r, θ, ϕ, a, 1, 4) * aSF_BL[1] + Kerr.KerrMetric.g_μν(r, θ, ϕ, a, 4, 4) * aSF_BL[4])/Γ    # Eq. 31
        push!(Ldot, dL_dt)
    end
    
    # update C, Q
    if θmin_0==π/2  # equatorial
        dC_dt = 0.0
        dQ_dt = - 2 * (a * E0 - L0) * (dL_dt - a * dE_dt)
    elseif e0==0.0 && θmin_0!=π/2   # circular non-equatorial
        r0 = p0
        dC_dt = CircularNonEquatorial.Cdot(r0, dE_dt, dL_dt, a, E0, L0, C0)
        dQ_dt = dC_dt - 2 * (a * E0 - L0) * (dL_dt - a * dE_dt)
    else   # generic case
        dQ_dt = 0.0
        @inbounds for α=1:4, β=1:4
            dQ_dt += 2 * Kerr.KerrMetric.ξ_μν(r, θ, ϕ, a, α, β) * (α==1 ? 1. : α==2 ? rdot : α==3 ? θdot : ϕdot) * aSF_BL[β]    # Eq. 32
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
        dr0_dt = CircularNonEquatorial.r0dot(r0, dE_dt, dL_dt, a, E0, L0, C0)
        pp = p0 + (dr0_dt) * Δt
        ee = 0.0

        ## now calculate θmin
        # coefficients of polynomial in Eq. E33
        c0 = C1 / (a^2 * (1.0 - E1^2))
        c1 = -1.0 - (L1^2 + C1) / (a^2 * (1.0 - E1^2))
        θθ = acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))
    else
        pp, ee, θθ = ConstantsOfMotion.compute_p_e_θmin(a, E1, L1, Q1, C1)
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
    push!(θminArray, θθ)
end

# functions to update the orbital constants using the self-force in harmonic coordinates---the result is the same as doing it in BL coords
Killing_temporal_H(a::Float64, xH::AbstractArray, t::Float64, r::Float64, θ::Float64, ϕ::Float64) = @SVector [Kerr.KerrMetric.g_tt(r, θ, ϕ, a), Kerr.KerrMetric.g_tϕ(r, θ, ϕ, a) * HarmonicCoords.∂ϕ_∂xH(xH, a),
Kerr.KerrMetric.g_tϕ(r, θ, ϕ, a) * HarmonicCoords.∂ϕ_∂yH(xH, a), Kerr.KerrMetric.g_tϕ(r, θ, ϕ, a) * HarmonicCoords.∂ϕ_∂zH(xH, a)]
Killing_axial_H(a::Float64, xH::AbstractArray, t::Float64, r::Float64, θ::Float64, ϕ::Float64) = @SVector [Kerr.KerrMetric.g_tϕ(r, θ, ϕ, a), Kerr.KerrMetric.g_ϕϕ(r, θ, ϕ, a) * HarmonicCoords.∂ϕ_∂xH(xH, a),
Kerr.KerrMetric.g_ϕϕ(r, θ, ϕ, a) * HarmonicCoords.∂ϕ_∂yH(xH, a), Kerr.KerrMetric.g_ϕϕ(r, θ, ϕ, a) * HarmonicCoords.∂ϕ_∂zH(xH, a)]
function Killing_tensor_H(a::Float64, xH::AbstractArray, t::Float64, r::Float64, θ::Float64, ϕ::Float64) 
    tensor = zeros(4, 4)
    jBLH = HarmonicCoords.jBLH(xH, a)
    ξtt = Kerr.KerrMetric.ξ_tt(r, θ, ϕ, a) 
    ξtϕ = Kerr.KerrMetric.ξ_tϕ(r, θ, ϕ, a) 
    ξrr = Kerr.KerrMetric.ξ_rr(r, θ, ϕ, a) 
    ξθθ = Kerr.KerrMetric.ξ_θθ(r, θ, ϕ, a) 
    ξϕϕ = Kerr.KerrMetric.ξ_ϕϕ(r, θ, ϕ, a)

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

function Evolve_Harm(Δt::Float64, a::Float64, xH::AbstractArray, t::Float64, r::Float64, θ::Float64, ϕ::Float64, Γ::Float64, rdot::Float64, θdot::Float64, ϕdot::Float64, aSF_H::AbstractVector{Float64}, EE::AbstractArray, Edot::AbstractArray, LL::AbstractArray, Ldot::AbstractArray, QQ::AbstractArray, Qdot::AbstractArray, CC::AbstractArray, Cdot::AbstractArray, pArray::AbstractArray, ecc::AbstractArray, θmin::AbstractArray, nPoints::Int64)
    E0 = last(EE); L0 = last(LL); Q0 = last(QQ); C0 = last(CC); p0 = last(pArray); e0 = last(ecc); θmin_0 = last(θminArray);
    temporal_killing = Killing_temporal_H(a, xH, t, r, θ, ϕ)
    axial_killing = Killing_axial_H(a, xH, t, r, θ, ϕ)
    tensor_killing = Killing_tensor_H(a, xH, t, r, θ, ϕ)

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
        dr0_dt = CircularNonEquatorial.r0dot(r0, dE_dt, dL_dt, a, E0, L0, C0)
        pp = p0 + (dr0_dt) * Δt
        ee = 0.0

        ## now calculate θmin
        # coefficients of polynomial in Eq. E33
        c0 = C1 / (a^2 * (1.0 - E1^2))
        c1 = -1.0 - (L1^2 + C1) / (a^2 * (1.0 - E1^2))
        θθ = acos(sqrt((-c1 - sqrt(c1^2 - 4c0))/2))
    else
        pp, ee, θθ = ConstantsOfMotion.compute_p_e_θmin(a, E1, L1, Q1, C1)
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