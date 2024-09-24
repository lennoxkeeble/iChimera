#=

    In this module, we implement Ryan's Flux formulas in arXiv:gr-qc/9511062

=#

module RyanFluxes

r(a::Float64, E::Float64, L::Float64, Q::Float64, e::Float64, ψ::Float64, m::Float64, M::Float64) = (Q + L^2) * (1.0 + a * L * m^3 * M * (6.0 + 2.0 * e * cos(ψ)) / ((Q + L^2)^2)) / ((m^2 * M) * (1.0 + e * cos(ψ)))   # Eq. 8

ψ(a::Float64, E::Float64, L::Float64, Q::Float64, e::Float64, r::Float64, m::Float64, M::Float64) = acos((L^4 + 2*L^2*Q + Q^2 - L^2*m^2*M*r - m^2*M*Q*r + 6.0*L*m^3*M*a)/ (e*m^2*M*(L^2*r + Q*r - 2*L*m*a)))   # invert Eq. 8 for ψ

ψ0(ψ::Float64, θ::Float64, ι::Float64) = asin(cos(θ) / sin(ι)) - ψ

dt_dψ(::Float64) = (Q + L^2)^1.5 * (1.0 + 6.0 * a * L * m^3 * M / ((Q + L^2)^2)) / ((m^3 * M^2) * (1.0 + e * cos(ψ))^2) 

Edot(a::Float64, e::Float64, ι::Float64, ψ0::Float64, p::Float64, m::Float64, M::Float64) = -6.4 * (m^2 / M^2) * (M / p)^5 * (1.0 / (1.0 - e^2))^3.5 * ((1.0 + 73.0 * e^2 / 24.0 + 37.0 * e^4 / 96.0) - 
(a / M^2) * (M / (p * (1.0 - e^2)))^1.5 * cos(ι) * (73.0 / 12.0 + 1211.0 * e^2 / 24.0 + 3143.0 * e^4 / 96.0 + 65.0 * e^6 / 64.0))

Ldot(a::Float64, e::Float64, ι::Float64, ψ0::Float64, p::Float64, m::Float64, M::Float64) = -6.4 * (m^2 / M) * (M / p)^3.5 * (1.0 / (1.0 - e^2))^2 * (cos(ι) * (1.0 + 0.875 * e^2) + (a / M^2) * (M / (p * (1.0 - e^2)))^1.5 * 
((61.0 / 24.0 + 63.0 * e^2 / 8.0 + 95.0 * e^4 / 64.0) - cos(ι)^2 * (61.0 / 8.0 + 109.0 * e^2 / 4.0 + 293.0 * e^4 / 64.0) - cos(2.0 * ψ0) * sin(ι)^2 * (1.25 * e^2 + 13.0 * e^4 / 16.0)))

QplusL2dot(a::Float64, e::Float64, ι::Float64, ψ0::Float64, p::Float64, m::Float64, M::Float64) = -12.8 * m^3 * (M / p)^3 * (1.0 / (1.0 - e^2))^1.5 * ((1.0 + 0.875 * e^2) - (a / M^2) * (M / (p * (1.0 - e^2)))^1.5 * cos(ι) * 
(97.0 / 12.0 + 22.0 * e^2 + 99.0 * e^4 / 32.0))

dι_dt(a::Float64, e::Float64, ι::Float64, ψ0::Float64, p::Float64, m::Float64, M::Float64) = (m * a / M^4) * (M / p)^5.5 * (1.0 / (1.0 - e^2))^4 * sin(ι) * (244.0 / 15.0 + 252.0 * e^2 / 5.0 + 9.5 * e^4 -
cos(2.0ψ0) * (8.0 * e^2 + 26.0 * e^4 / 5.0))

dp_dt(a::Float64, e::Float64, ι::Float64, ψ0::Float64, p::Float64, m::Float64, M::Float64) = -12.8 * (m / M) * (M / p)^3 * (1.0 / (1.0 - e^2))^3.5 * ((1.0 + 73.0 * e^2 / 24.0 + 37.0 * e^4 / 96.0) - (a / M^2) * 
(M / (p * (1.0 - e^2)))^1.5 * cos(ι) * (133.0 / 12.0 + 337.0 * e^2 / 6.0 + 2965.0 * e^4 / 96.0 + 65.0 * e^6 / 64.0))

de_dt(a::Float64, e::Float64, ι::Float64, ψ0::Float64, p::Float64, m::Float64, M::Float64) = -(m / M^2) * (M / p)^4 * (1.0 / (1.0 - e^2))^2.5 * e * ((304.0 + 121.0 * e^2) / 15.0 - (a / M^2) * (M / (p * (1.0 - e^2)))^1.5 * 
cos(ι) * (1364.0 / 5.0 + 5032.0 * e^2 / 15.0 + 263.0 * e^4 / 10.0))

function EvolveConstants(Δt::Float64, a::Float64, r::Float64, θ::Float64, E::AbstractVector{Float64}, Edot::AbstractVector{Float64}, L::AbstractVector{Float64}, 
    Ldot::AbstractVector{Float64}, C::AbstractVector{Float64}, Cdot::AbstractVector{Float64}, p::AbstractVector{Float64}, e::AbstractVector{Float64}, ι::AbstractVector{Float64}, m::Float64, M::Float64, nPoints::Int64)
    EE = last(E); LL = last(L); CC = last(C); ee = last(e); pp = last(p); ιι = last(ι);

    if ee == 0.0
        ψ0 = 0.0
    else
        ψ = RyanFluxes.ψ(a, EE, LL, CC, ee, r, m, M)
        ψ0 = RyanFluxes.ψ0(ψ, θ, ιι)   # all ψ0 terms drop out if e == 0.0
    end

    dE_dt = RyanFluxes.Edot(a, ee, ιι, ψ0, pp, m, M)
    dL_dt = RyanFluxes.Ldot(a, ee, ιι, ψ0, pp, m, M)
    dQplusL2_dt = RyanFluxes.QplusL2dot(a, ee, ιι, ψ0, pp, m, M)
    dι_dt = RyanFluxes.dι_dt(a, ee, ιι, ψ0, pp, m, M)
    dp_dt = RyanFluxes.dp_dt(a, ee, ιι, ψ0, pp, m, M)
    de_dt = RyanFluxes.de_dt(a, ee, ιι, ψ0, pp, m, M)

    # push!(pdot, dp_dt)
    # append!(pdot, zeros(nPoints-1))
    append!(p, ones(nPoints) * (pp + dp_dt * Δt))
    
    # push!(edot, de_dt)
    # append!(edot, zeros(nPoints-1))
    append!(e, ones(nPoints) * (ee + de_dt * Δt))

    # push!(ιdot, dι_dt)
    # append!(ιdot, zeros(nPoints-1))
    append!(ι, ones(nPoints) *(ιι + dι_dt * Δt))

    push!(Edot, dE_dt)
    append!(Edot, zeros(nPoints-1))
    append!(E, ones(nPoints) * (EE + dE_dt * Δt))

    push!(Ldot, dL_dt)
    append!(Ldot, zeros(nPoints-1))
    append!(L, ones(nPoints) * (LL + dL_dt * Δt))
    
    # append!(C, ones(nPoints) * (CC + LL^2 + dQplusL2_dt * Δt - last(L)^2))
    # push!(Cdot, (last(C) - CC) / Δt)
    # append!(Cdot, zeros(nPoints-1))


    dC_dt = dQplusL2_dt - 2 * LL * dL_dt

    push!(Cdot, dC_dt)
    append!(Cdot, zeros(nPoints-1))
    append!(C, ones(nPoints) * (CC + dC_dt * Δt))
end

end