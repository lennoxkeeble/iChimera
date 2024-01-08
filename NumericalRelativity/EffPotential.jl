module EffPotential
# effective potential function
vEff(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, EE::Float64, LL::Float64, m::Float64, gtt::Function, gtϕ::Function, gϕϕ::Function) = (1/2) * ((EE^2 * gϕϕ(t, r, θ, ϕ, a, M) + 2EE * LL * gtϕ(t, r, θ, ϕ, a, M) + gtt(t, r, θ, ϕ, a, M) * LL^2) / (gtt(t, r, θ, ϕ, a, M) * gϕϕ(t, r, θ, ϕ, a, M) - gtϕ(t, r, θ, ϕ, a, M)^2) + m^2)
end