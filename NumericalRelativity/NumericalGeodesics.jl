module NumericalGeodesics
using ForwardDiff
using LinearAlgebra
using FiniteDifferences
using StaticArrays
using DifferentialEquations
using DelimitedFiles

# numerical differentiation method
const fd = central_fdm(5,1) 

# calculate ∂_{μ}g_{αβ} numerically
∂g_∂t(α::Int, β::Int, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = α==1 && β==1  ? fd(t -> g_tt(t, r, θ, ϕ, a, M), t) : α==2 && β==2 ? fd(t -> g_rr(t, r, θ, ϕ, a, M), t) : α==3 && β==3 ? fd(t -> g_θθ(t, r, θ, ϕ, a, M), t) : α==4 && β==4 ? fd(t -> g_ϕϕ(t, r, θ, ϕ, a, M), t) : (α==1 && β==4) || (α==4 && β==1) ? fd(t -> g_tϕ(t, r, θ, ϕ, a, M), t) : 0.0
∂g_∂r(α::Int, β::Int, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = α==1 && β==1  ? fd(r -> g_tt(t, r, θ, ϕ, a, M), r) : α==2 && β==2 ? fd(r -> g_rr(t, r, θ, ϕ, a, M), r) : α==3 && β==3 ? fd(r -> g_θθ(t, r, θ, ϕ, a, M), r) : α==4 && β==4 ? fd(r -> g_ϕϕ(t, r, θ, ϕ, a, M), r) : (α==1 && β==4) || (α==4 && β==1) ? fd(r -> g_tϕ(t, r, θ, ϕ, a, M), r) : 0.0
∂g_∂θ(α::Int, β::Int, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = α==1 && β==1  ? fd(θ -> g_tt(t, r, θ, ϕ, a, M), θ) : α==2 && β==2 ? fd(θ -> g_rr(t, r, θ, ϕ, a, M), θ) : α==3 && β==3 ? fd(θ -> g_θθ(t, r, θ, ϕ, a, M), θ) : α==4 && β==4 ? fd(θ -> g_ϕϕ(t, r, θ, ϕ, a, M), θ) : (α==1 && β==4) || (α==4 && β==1) ? fd(θ -> g_tϕ(t, r, θ, ϕ, a, M), θ) : 0.0
∂g_∂ϕ(α::Int, β::Int, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = α==1 && β==1  ? fd(ϕ -> g_tt(t, r, θ, ϕ, a, M), ϕ) : α==2 && β==2 ? fd(ϕ -> g_rr(t, r, θ, ϕ, a, M), ϕ) : α==3 && β==3 ? fd(ϕ -> g_θθ(t, r, θ, ϕ, a, M), ϕ) : α==4 && β==4 ? fd(ϕ -> g_ϕϕ(t, r, θ, ϕ, a, M), ϕ) : (α==1 && β==4) || (α==4 && β==1) ? fd(ϕ -> g_tϕ(t, r, θ, ϕ, a, M), ϕ) : 0.0
∂g_∂xμ(μ::Int, α::Int, β::Int, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) = μ == 1 ? ∂g_∂t(α, β, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, t, r, θ, ϕ, a, M) : μ == 2 ? ∂g_∂r(α, β, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, t, r, θ, ϕ, a, M) : μ == 3 ? ∂g_∂θ(α, β, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, t, r, θ, ϕ, a, M) : ∂g_∂ϕ(α, β, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, t, r, θ, ϕ, a, M)

# christoffel symbol Γ^μ_{αβ}
function Γμαβ(μ::Int, α::Int, β::Int, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, ginv::Function, t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64) 
    Γ=0.0;
    @inbounds for ρ=1:4
        Γ += 0.5 * ginv(t, r, θ, ϕ, a, M)[ρ, μ] * (∂g_∂xμ(β, α, ρ, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, t, r, θ, ϕ, a, M) + ∂g_∂xμ(α, β, ρ, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, t, r, θ, ϕ, a, M) - ∂g_∂xμ(ρ, α, β, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, t, r, θ, ϕ, a, M))
    end
    return Γ
end

# computes \ddot{x}^μ (which must be pre-allocated) for some given \dot{x}^μ, x^μ
function xddot!(ddx::Vector{Float64}, dx::Vector{Float64}, x::Vector{Float64}, g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, ginv::Function, a::Float64, M::Float64)
    @inbounds Threads.@threads for μ=1:4
        ddx[μ] = 0
        @inbounds for ρ=1:4, σ=1:4
            ddx[μ] += - Γμαβ(μ, ρ, σ, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, ginv, x..., a, M) * dx[ρ] * dx[σ]
        end
    end
    return
end

# evolves geodesic equation where the metric functions take arguments (t, r, θ, ϕ, a, M), and the initial conditions ics = [[dti, dri, dθi, dϕi], [ti, ri, θi, ϕi]]
function compute_geodesic(g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, ginv::Function, ics::AbstractArray, a::Float64, E::Float64, L::Float64, τmax::Float64=3000.0, Δti::Float64=1.0, reltol::Float64=1e-6, abstol::Float64=1e-6, saveat::Float64=0.5; data_path::String="Results/")
    function geodesicEq!(ddu, du, u, params, t) 
        NumericalGeodesics.xddot!(ddu, du, u, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, ginv, params...)
    end

    # orbital parameters
    M = 1.; m=1.;

    # set up ODE
    τspan = (0.0, τmax); params = [a, M];
    prob = SecondOrderODEProblem(geodesicEq!, ics..., τspan, params);

    # numerically solve geodesic equation
    sol = solve(prob, AutoTsit5(RK4()), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat);

    # deconstruct solution
    τ = 0:saveat:τmax |> collect
    tdot = sol[1, :];
    rdot = sol[2, :];
    θdot = sol[3, :];
    ϕdot = sol[4, :];
    t = sol[5, :];
    r = sol[6, :];
    θ = sol[7, :];
    ϕ = sol[8, :];
    sol = transpose(stack([τ, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot]))

    # save trajectory- rows are: τ, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, columns are component values at different times
    # save data
    mkpath(data_path)
    open(data_path * "numerical_ODE_sol_E_$(round(E; digits=3))_L_$(round(L; digits=3))_a_$(a)_tol_$(reltol).txt", "w") do io
        writedlm(io, sol)
    end
    println("ODE saved to: " * data_path * "numerical_ODE_sol_E_$(round(E; digits=3))_L_$(round(L; digits=3))_a_$(a)_tol_$(reltol).txt")
end

# expressions for dt/dτ and dϕ/dτ from Lagrangian
tdot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, EE::Float64, LL::Float64, g_ϕϕ::Function, g_tϕ::Function, g_tt::Function) = (EE * g_ϕϕ(t, r, θ, ϕ, a, M) + LL * g_tϕ(t, r, θ, ϕ, a, M)) / (g_tϕ(t, r, θ, ϕ, a, M)^2 - g_tt(t, r, θ, ϕ, a, M) * g_ϕϕ(t, r, θ, ϕ, a, M))   # Eq. 5.9
ϕdot(t::Float64, r::Float64, θ::Float64, ϕ::Float64, a::Float64, M::Float64, EE::Float64, LL::Float64, g_ϕϕ::Function, g_tϕ::Function, g_tt::Function) = - (EE * g_tϕ(t, r, θ, ϕ, a, M) + LL * g_tt(t, r, θ, ϕ, a, M)) / (g_tϕ(t, r, θ, ϕ, a, M)^2 - g_tt(t, r, θ, ϕ, a, M) * g_ϕϕ(t, r, θ, ϕ, a, M))   # Eq. 5.10


end