# build symbolic functions for E, L, Q that can be differentiated analytically
module ELQFunctions
using Symbolics
using LinearAlgebra

function ELQ(a, p, e, θmin)
    # define turning points rp, ra
    rp = p / (1 + e)
    ra  = p / (1 - e)

    Δ(r) = r^2 - 2r + a^2
    zm = cos(θmin) # z_{-}
    
    # define functions f, g, h, and d as in Eqs. 2.27 - 2.30
    f(r) = r^4 + (a^2) * (r * (r + 2) + (zm^2) * Δ(r))
    g(r) = 2a*r
    h(r) = r * (r-2) + ((zm^2) / (1 - zm^2)) * Δ(r)
    d(r) = (r^2 + (a^2) * zm^2) * Δ(r)

    # define derivatives of f, g, h, and d as in Eqs. 2.32 - 2.35
    fp(r) = 4r^3 + (2a^2) * ((1 + zm(θmin)^2) * r + (1 - zm(θmin)^2))
    gp(r) = 2a
    hp(r) = (2 * (r - 1)) / (1 - zm(θmin)^2)
    dp(r) = 2 * (2r - 3) * r^2 + (2a^2) * ((1 + zm(θmin)^2) * r - zm(θmin)^2)

    # define symbols ε, η, κ, ρ, σ in Eqs. 2.30, 2.31 (current implementation only for e ≠ 0)

    ε = det([d(ra) g(ra); d(rp) g(rp)])
    η = det([f(ra) g(ra); f(rp) g(rp)])
    κ = det([d(ra) h(ra); d(rp) h(rp)])
    ρ = det([f(ra) h(ra); f(rp) h(rp)])
    σ = det([g(ra) h(ra); g(rp) h(rp)])

    # elseif ee == 0
    #     ε = det([d(ra) g(ra); dp(rp) gp(rp)])
    #     η = det([f(ra) g(ra); fp(rp) gp(rp)])
    #     κ = det([d(ra) h(ra); dp(rp) hp(rp)])
    #     ρ = det([f(ra) h(ra); fp(rp) hp(rp)])
    #     σ = det([g(ra) h(ra); gp(rp) hp(rp)])
    # end

    En = sqrt((κ*ρ + 2ε*σ - 2*sqrt(σ * (-η * κ^2 + ε * κ * ρ + σ *ε^2))) / (ρ^2 + 4η * σ)) # Eq. 2.32

    L = sqrt((ε - η * En^2) / σ) # Eq. 2.34

    Q = (zm^2) * ((a^2) * (1 - En^2) + (L^2) / (1 - zm^2)) # Eq. 2.25

    return En, L, Q
end

@variables a, p, e, θmin, m, M

# now obtain dimensionless constants of motion En, L, and Q as a function of m, p, e, θmin, a   
EEE, LLL, QQQ = ELQ(a, p, e, θmin)

# the function ELQ() calculates the dimensionless constants of motion, so we convert them to their dimension-full form
EE = Symbolics.build_function([m * EEE], a, p, e, θmin, m, M, expression = Val{false}) # Eq. 2.1
En(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = EE[1](a, p, e, θmin, m, M)[1]

LL = Symbolics.build_function([m * M * LLL], a, p, e, θmin, m, M, expression = Val{false}) # Eq. 2.1
L(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = LL[1](a, p, e, θmin, m, M)[1]

QQ = Symbolics.build_function([(m^2) * (M^2) * QQQ], a, p, e, θmin, m, M, expression = Val{false}) # Eq. 2.1
Q(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = QQ[1](a, p, e, θmin, m, M)[1]

end


module BumpyFrequencies
using .Kerr
using .ELQFunctions: En, L, Q
using LinearAlgebra
using Symbolics
using QuadGK
using ArbNumerics
using HCubature
using SpecialFunctions
using StaticArrays
using ForwardDiff

# define z_{-}, z_{+}, and k
zm(θmin::Real) = cos(θmin)^2
zp(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (1/((2a^2) * (1 - En(a, p, e, θmin, m, M)^2))) * ((a^2) * (1 - En(a, p, e, θmin, m, M)^2) + L(a, p, e, θmin, m, M)^2 + Q(a, p, e, θmin, m, M) + sqrt((4a^2) * (-1 + En(a, p, e, θmin, m, M)^2) * Q(a, p, e, θmin, m, M) + ((-a^2) * (-1 + En(a, p, e, θmin, m, M)^2) + L(a, p, e, θmin, m, M)^2 + Q(a, p, e, θmin, m, M))^2))
k(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = zm(θmin) / zp(a, p, e, θmin, m, M)

# z = cos(θ)^2
Δ(r::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = r^2 - 2M * r + a^2
Σ(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = r^2 + z * a^2 
R(r::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = ((r^2 + a^2) * En(a, p, e, θmin, m, M) - a * L(a, p, e, θmin, m, M))^2 - Δ(r, a, p, e, θmin, m, M) * ((m^2) * (r^2) + (L(a, p, e, θmin, m, M) - a * En(a, p, e, θmin, m, M))^2 + Q(a, p, e, θmin, m, M))   # Eq. 3.20
Θ(z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = Q(a, p, e, θmin, m, M) - (z * L(a, p, e, θmin, m, M)^2) / (1 - z) - z * (a^2) * (m^2 - En(a, p, e, θmin, m, M)^2)   # Eq. 3.21
Φ(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = L(a, p, e, θmin, m, M) / (1 - z) + a * En(a, p, e, θmin, m, M) * ((r^2 + a^2) / Δ(r, a, p, e, θmin, m, M) - 1) - (a^2) * L(a, p, e, θmin, m, M) / Δ(r, a, p, e, θmin, m, M)   # Eq. 3.22
T(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = En(a, p, e, θmin, m, M) * (((r^2 + a^2)^2) / Δ(r, a, p, e, θmin, m, M) - (a^2) * (1 - z)) + a * L(a, p, e, θmin, m, M) * (1 - (r^2 + a^2) / Δ(r, a, p, e, θmin, m, M))   # Eq. 3.23

# transformation of variables
r(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (p * M) / (1 + e * cos(φ))   # Eq. C.11
z(χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (cos(θmin)^2) * cos(χ)^2   # z = z_{-} cos(χ)^2

# define roots of R- see Eq. B.3
r1(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = p * M / (1 - e)
r2(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = p * M / (1 + e)
AplusB(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = 2M / (1 - En(a, p, e, θmin, m, M)^2) - (r1(a, p, e, θmin, m, M) + r2(a, p, e, θmin, m, M))
AB(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (a^2) * Q(a, p, e, θmin, m, M) / ((1 - En(a, p, e, θmin, m, M)^2) * r1(a, p, e, θmin, m, M) * r2(a, p, e, θmin, m, M)) # Eq. B.3
r3(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (1 / 2) * (AplusB(a, p, e, θmin, m, M) + sqrt(AplusB(a, p, e, θmin, m, M)^2 - 4AB(a, p, e, θmin, m, M)))
r4(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = AB(a, p, e, θmin, m, M) / r3(a, p, e, θmin, m, M)

# define algebraic quantites p3, p4
p3(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = r3(a, p, e, θmin, m, M) * (1 - e) / M   # Eq. C.13
p4(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = r4(a, p, e, θmin, m, M) * (1 + e) / M   # Eq. C.14

# define function P(φ) and a function to calculate its derivative with respect to β, where β ∈ {m, p, e, θmin} throughout this notebook
P(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (1 / (1 - e^2)) * M * sqrt(1 - En(a, p, e, θmin, m, M)^2) * sqrt((p - p3(a, p, e, θmin, m, M)) - e * (p + p3(a, p, e, θmin, m, M) * cos(φ))) * sqrt((p - p4(a, p, e, θmin, m, M)) + e * (p - p4(a, p, e, θmin, m, M) * cos(φ)))   # Eq. C.16
∂P_∂β(φ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ForwardDiff.derivative(m -> P(φ, a, p, e, θmin, m, M), m) : isequal(β, 'p') ? ForwardDiff.derivative(p -> P(φ, a, p, e, θmin, m, M), p) : isequal(β, 'e') ? ForwardDiff.derivative(e -> P(φ, a, p, e, θmin, m, M), e) : ForwardDiff.derivative(θmin -> P(φ, a, p, e, θmin, m, M), θmin) 

# define integrand of Λr and its derivative
ΛrIntegrand(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (1 / P(φ, a, p, e, θmin, m, M))   # Eq. C.20
∂ΛrInt_∂β(φ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ForwardDiff.derivative(m -> ΛrIntegrand(φ, a, p, e, θmin, m, M), m) : isequal(β, 'p') ? ForwardDiff.derivative(p -> ΛrIntegrand(φ, a, p, e, θmin, m, M), p) : isequal(β, 'e') ? ForwardDiff.derivative(e -> ΛrIntegrand(φ, a, p, e, θmin, m, M), e) : ForwardDiff.derivative(θmin -> ΛrIntegrand(φ, a, p, e, θmin, m, M), θmin) 

Λr(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(φ -> ΛrIntegrand(φ, a, p, e, θmin, m, M), 0, 2π, rtol = rtol)[1]   # Eq. C.20

# calculate ∂Λr/∂β
∂Λr_∂m(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(φ -> ∂ΛrInt_∂β(φ, 'm', a, p, e, θmin, m, M), 0, 2π, rtol = rtol)
∂Λr_∂p(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(φ -> ∂ΛrInt_∂β(φ, 'p', a, p, e, θmin, m, M), 0, 2π, rtol = rtol)
∂Λr_∂e(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(φ -> ∂ΛrInt_∂β(φ, 'e', a, p, e, θmin, m, M), 0, 2π, rtol = rtol)
∂Λr_∂θ(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(φ -> ∂ΛrInt_∂β(φ, 'θ', a, p, e, θmin, m, M), 0, 2π, rtol = rtol)

∂Λr_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = isequal(β, 'm') ? ∂Λr_∂m(a, p, e, θmin, m, M, rtol)[1] : isequal(β, 'p') ? ∂Λr_∂p(a, p, e, θmin, m, M, rtol)[1] : isequal(β, 'e') ? ∂Λr_∂e(a, p, e, θmin, m, M, rtol)[1] : ∂Λr_∂θ(a, p, e, θmin, m, M, rtol)[1]


# compute dK(k)/dk where K(k) is the elliptic integral of the first kind
function dEK_dk(x)
    return ellipe(x) / (2x * (1-x)) - ellipk(x) / 2x
end
    
# compute ∂K(x)/∂β
∂EK_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? dEK_dk(k(a, p, e, θmin, m, M)) * ForwardDiff.derivative(m -> k(a, p, e, θmin, m, M), m) : isequal(β, 'p') ? dEK_dk(k(a, p, e, θmin, m, M)) * ForwardDiff.derivative(p -> k(a, p, e, θmin, m, M), p) : isequal(β, 'e') ? dEK_dk(k(a, p, e, θmin, m, M)) * ForwardDiff.derivative(e -> k(a, p, e, θmin, m, M), e) : dEK_dk(k(a, p, e, θmin, m, M)) * ForwardDiff.derivative(θmin -> k(a, p, e, θmin, m, M), θmin)

# define action-angle variables
# define integrand of Jr and its derivative wrt β
JrIntegrand(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (1/π) * (sqrt(R(r(φ, a, p, e, θmin, m, M), a, p, e, θmin, m, M)) / Δ(r(φ, a, p, e, θmin, m, M), a, p, e, θmin, m, M)) * ForwardDiff.derivative(φ -> p * M / (1 + e * cos(φ)), φ)   # Eq. 1.11
∂JrInt_∂β(φ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ForwardDiff.derivative(m -> JrIntegrand(φ, a, p, e, θmin, m, M), m) : isequal(β, 'p') ? ForwardDiff.derivative(p -> JrIntegrand(φ, a, p, e, θmin, m, M), p) : isequal(β, 'e') ? ForwardDiff.derivative(e -> JrIntegrand(φ, a, p, e, θmin, m, M), e) : ForwardDiff.derivative(θmin -> JrIntegrand(φ, a, p, e, θmin, m, M), θmin)

# compute Jr and ∂Jr/∂β
Jr(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(φ -> JrIntegrand(φ, a, p, e, θmin, m, M), 0, π, rtol = rtol)
∂Jr_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(φ -> ∂JrInt_∂β(φ, β, a, p, e, θmin, m, M), 0, π, rtol = rtol)[1]

# define integrand of Jθ and its derivative wrt β
JθIntegrand(χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (2/π) * sqrt(Θ(z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M)) * sqrt(zm(θmin)) * sin(χ) / (sqrt(1 - zm(θmin) * cos(χ)^2))   # Eq. 1.12
∂JθInt_∂β(φ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ForwardDiff.derivative(m -> JθIntegrand(φ, a, p, e, θmin, m, M), m) : isequal(β, 'p') ? ForwardDiff.derivative(p -> JθIntegrand(φ, a, p, e, θmin, m, M), p) : isequal(β, 'e') ? ForwardDiff.derivative(e -> JθIntegrand(φ, a, p, e, θmin, m, M), e) : ForwardDiff.derivative(θmin -> JθIntegrand(φ, a, p, e, θmin, m, M), θmin)

# compute Jθ and ∂Jθ/∂β
Jθ(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(χ -> JθIntegrand(χ, a, p, e, θmin, m, M), 0, π/2, rtol = rtol)
∂Jθ_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = quadgk(φ -> ∂JθInt_∂β(φ, β, a, p, e, θmin, m, M), 0, π/2, rtol = rtol)[1]

# compute Jϕ and ∂Jϕ/∂β
Jϕ(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = L(a, p, e, θmin, m, M)   # Eq. 1.13
∂Jϕ_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ForwardDiff.derivative(m -> Jϕ(a, p, e, θmin, m, M), m) : isequal(β, 'p') ? ForwardDiff.derivative(p -> Jϕ(a, p, e, θmin, m, M), p) : isequal(β, 'e') ? ForwardDiff.derivative(e -> Jϕ(a, p, e, θmin, m, M), e) : ForwardDiff.derivative(θmin -> Jϕ(a, p, e, θmin, m, M), θmin)

# compute Jt and ∂Jt/∂β
Jt(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = -En(a, p, e, θmin, m, M)   # Eq. 3.32
∂Jt_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ForwardDiff.derivative(m -> Jt(a, p, e, θmin, m, M), m) : isequal(β, 'p') ? ForwardDiff.derivative(p -> Jt(a, p, e, θmin, m, M), p) : isequal(β, 'e') ? ForwardDiff.derivative(e -> Jt(a, p, e, θmin, m, M), e) : ForwardDiff.derivative(θmin -> Jt(a, p, e, θmin, m, M), θmin)

# compute jacobian of vector function J = [Jt, Jr, Jθ, Jϕ] wrt β
JacobianVals(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = [∂Jt_∂β.(['m' 'p' 'e' 'θ'], a, p, e, θmin, m, M) 
∂Jr_∂β.(['m' 'p' 'e' 'θ'], a, p, e, θmin, m, M, rtol)
∂Jθ_∂β.(['m' 'p' 'e' 'θ'], a, p, e, θmin, m, M, rtol)
∂Jϕ_∂β.(['m' 'p' 'e' 'θ'], a, p, e, θmin, m, M)]

# compute inverse jacobian of J defined in Eq. 3.55
invJacobianVals(a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = inv(JacobianVals(a, p, e, θmin, m, M, rtol))

# compute frequency shifts (multiplied by p^(7/2)) in bumpy spacetime
function BumpyKerrFreqs(ψ1::Function, γ1::Function, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64)

    k::Float64 = BumpyFrequencies.k(a, p, e, θmin, m, M)
    EK::Float64 = ellipk(k)
    EE::Float64 = ellipe(k)

    # define bumpy spacetime perturbations in Kerr
    b_tt(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = -2 * (1 - 2M * r / Σ(r, z, a, p, e, θmin, m, M)) * ψ1(r, z, a, M)   # Eq. 3.13
    b_tr(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = - γ1(r, z, a, M) * (2a^2) * M * r * (1 - z) / (Δ(r, a, p, e, θmin, m, M) * Σ(r, z, a, p, e, θmin, m, M))   # Eq. 3.14
    b_tϕ(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (γ1(r, z, a, M) - 2 * ψ1(r, z, a, M)) * 2a * M * r * (1 - z) / Σ(r, z, a, p, e, θmin, m, M)   # Eq. 3.15
    b_rr(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = 2 * (γ1(r, z, a, M) - ψ1(r, z, a, M)) * Σ(r, z, a, p, e, θmin, m, M) / Δ(r, a, p, e, θmin, m, M)   # Eq 3.16
    b_rϕ(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = γ1(r, z, a, M) * ((1 - 2M * r / Σ(r, z, a, p, e, θmin, m, M))^(-1) - (4a^2) * (M^2) * (r^2) * (1 - z) / (Δ(r, a, p, e, θmin, m, M) * Σ(r, z, a, p, e, θmin, m, M) * (Σ(r, z, a, p, e, θmin, m, M) - 2M * r))) * a * (1 - z)   # Eq. 3.17
    b_θθ(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = 2 * (γ1(r, z, a, M) - ψ1(r, z, a, M)) * Σ(r, z, a, p, e, θmin, m, M)   # Eq. 3.18
    b_ϕϕ(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = ((γ1(r, a, z, M) - ψ1(r, z, a, M)) * (8a^2) * (M^2) * (r^2) * (1 - z) / (Δ(r, a, p, e, θmin, m, M) * Σ(r, z, a, p, e, θmin, m, M) * (Σ(r, z, a, p, e, θmin, m, M) - 2M * r)) - 2 * ψ1(r, z, a, M) * (1 - 2M * r / Σ(r, z, a, p, e, θmin, m, M))^(-1)) * Δ(r, a, p, e, θmin, m, M) * (1 - z)   # Eq. 3.19

    # define Hamiltonian perturbation and its derivative
    H1(r::Real, z::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (-1 / (2Σ(r, z, a, p, e, θmin, m, M)^2)) * (b_tt(r, z, a, p, e, θmin, m, M) * (T(r, z, a, p, e, θmin, m, M)^2) + b_rr(r, z, a, p, e, θmin, m, M) * R(r, a, p, e, θmin, m, M) + b_θθ(r, z, a, p, e, θmin, m, M) * Θ(z, a, p, e, θmin, m, M) + b_ϕϕ(r, z, a, p, e, θmin, m, M) * (Φ(r, z, a, p, e, θmin, m, M)^2) + 2b_tϕ(r, z, a, p, e, θmin, m, M) * Φ(r, z, a, p, e, θmin, m, M) * T(r, z, a, p, e, θmin, m, M))   # Eq. 3.50
    ∂H1_∂β(φ::Real, χ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ForwardDiff.derivative(m -> H1(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M), m) : isequal(β, 'p') ? ForwardDiff.derivative(p -> H1(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M), p) : isequal(β, 'e') ? ForwardDiff.derivative(e -> H1(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M), e) : ForwardDiff.derivative(θmin -> H1(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M), θmin)
    
    Λr::Float64 = BumpyFrequencies.Λr(a, p, e, θmin, m, M, rtol)

    # define derivatives of ωr and ωθ
    ∂ωr_∂φ(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = 2π / (Λr * P(φ, a, p, e, θmin, m, M))   # Eq. C.20
    ∂ωθ_∂χ(χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (π / 2EK) * (1 / sqrt(1 - k * cos(χ)^2))   # Eq. C.30

    # compute second derivatives of ∂ωr_∂φ wrt β via the chain rule
    ∂2ωr_∂φ_∂m(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = 2π * (-∂Λr_∂β('m', a, p, e, θmin, m, M, rtol)/ ((Λr^2) * P(φ, a, p, e, θmin, m, M)) - ∂P_∂β(φ, 'm', a, p, e, θmin, m, M) / (Λr * (P(φ, a, p, e, θmin, m, M)^2)))
    ∂2ωr_∂φ_∂p(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = 2π * (-∂Λr_∂β('p', a, p, e, θmin, m, M, rtol)/ ((Λr^2) * P(φ, a, p, e, θmin, m, M)) - ∂P_∂β(φ, 'p', a, p, e, θmin, m, M) / (Λr * (P(φ, a, p, e, θmin, m, M)^2)))
    ∂2ωr_∂φ_∂e(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = 2π * (-∂Λr_∂β('e', a, p, e, θmin, m, M, rtol)/ ((Λr^2) * P(φ, a, p, e, θmin, m, M)) - ∂P_∂β(φ, 'e', a, p, e, θmin, m, M) / (Λr * (P(φ, a, p, e, θmin, m, M)^2)))
    ∂2ωr_∂φ_∂θ(φ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = 2π * (-∂Λr_∂β('θ', a, p, e, θmin, m, M, rtol)/ ((Λr^2) * P(φ, a, p, e, θmin, m, M)) - ∂P_∂β(φ, 'θ', a, p, e, θmin, m, M) / (Λr * (P(φ, a, p, e, θmin, m, M)^2)))

    ∂2ωr_∂φ_∂β(φ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = isequal(β, 'm') ? ∂2ωr_∂φ_∂m(φ, a, p, e, θmin, m, M, rtol) : isequal(β, 'p') ? ∂2ωr_∂φ_∂p(φ, a, p, e, θmin, m, M, rtol) : isequal(β, 'e') ? ∂2ωr_∂φ_∂e(φ, a, p, e, θmin, m, M, rtol) : ∂2ωr_∂φ_∂θ(φ, a, p, e, θmin, m, M, rtol)

    # compute second derivatives of ∂ωθ_∂χ wrt β via the chain rule
    ∂2ωθ_∂χ_∂m(χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (π / 2) * (-∂EK_∂β('m', a, p, e, θmin, m, M) / ((EK^2) * sqrt(1 - k * cos(χ)^2)) - ((1/2) * ForwardDiff.derivative(m -> 1 - BumpyFrequencies.k(a, p, e, θmin, m, M) * cos(χ)^2, m)) / (EK * ((1 - k * cos(χ)^2)^(3/2))))
    ∂2ωθ_∂χ_∂p(χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (π / 2) * (-∂EK_∂β('p', a, p, e, θmin, m, M) / ((EK^2) * sqrt(1 - k * cos(χ)^2)) - ((1/2) * ForwardDiff.derivative(p -> 1 - BumpyFrequencies.k(a, p, e, θmin, m, M) * cos(χ)^2, p)) / (EK * ((1 - k * cos(χ)^2)^(3/2))))
    ∂2ωθ_∂χ_∂e(χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (π / 2) * (-∂EK_∂β('e', a, p, e, θmin, m, M) / ((EK^2) * sqrt(1 - k * cos(χ)^2)) - ((1/2) * ForwardDiff.derivative(e -> 1 - BumpyFrequencies.k(a, p, e, θmin, m, M) * cos(χ)^2, e)) / (EK * ((1 - k * cos(χ)^2)^(3/2))))
    ∂2ωθ_∂χ_∂θ(χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (π / 2) * (-∂EK_∂β('θ', a, p, e, θmin, m, M) / ((EK^2) * sqrt(1 - k * cos(χ)^2)) - ((1/2) * ForwardDiff.derivative(θmin -> 1 - BumpyFrequencies.k(a, p, e, θmin, m, M) * cos(χ)^2, θmin)) / (EK * ((1 - k * cos(χ)^2)^(3/2))))

    ∂2ωθ_∂χ_∂β(χ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ∂2ωθ_∂χ_∂m(χ, a, p, e, θmin, m, M) : isequal(β, 'p') ? ∂2ωθ_∂χ_∂p(χ, a, p, e, θmin, m, M) : isequal(β, 'e') ? ∂2ωθ_∂χ_∂e(χ, a, p, e, θmin, m, M) : ∂2ωθ_∂χ_∂θ(χ, a, p, e, θmin, m, M)

    # compute partial derivative of T in Eq. 3.23 wrt β via the chain rule
    ∂T_∂β(φ::Real, χ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = isequal(β, 'm') ? ForwardDiff.derivative(m -> T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M), m) : isequal(β, 'p') ? ForwardDiff.derivative(p -> T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M), p) : isequal(β, 'e') ? ForwardDiff.derivative(e -> T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M), e) : ForwardDiff.derivative(θmin -> T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M), θmin)
    
    # define integrand of Υt and its derivative
    ΥtIntegrand(φ::Real, χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = ∂ωr_∂φ(φ, a, p, e, θmin, m, M) * ∂ωθ_∂χ(χ, a, p, e, θmin, m, M) * T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M)   # Eq. 3.53
    ∂ΥtInt_∂β(φ::Real, χ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = ∂2ωr_∂φ_∂β(φ, β, a, p, e, θmin, m, M, rtol) * ∂ωθ_∂χ(χ, a, p, e, θmin, m, M) * T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) + ∂ωr_∂φ(φ, a, p, e, θmin, m, M) * (∂2ωθ_∂χ_∂β(χ, β, a, p, e, θmin, m, M) * T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) + ∂ωθ_∂χ(χ, a, p, e, θmin, m, M) * ∂T_∂β(φ, χ, β, a, p, e, θmin, m, M))   # using chain rule

    # compute Υt and its derivative
    Υt::Float64 = (1 / ((2π)^2)) * hcubature(x -> ΥtIntegrand(x[1], x[2], a, p, e, θmin, m, M), (0, 0), (2π, 2π), rtol = rtol)[1]
    ∂Υt_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = hcubature(x -> ∂ΥtInt_∂β(x[1], x[2], β, a, p, e, θmin, m, M, rtol), (0, 0), (2π, 2π), rtol = rtol)[1]

    # define integrand of H1′ and its derivative
    H1primeIntegrand(φ::Real, χ::Real, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = ∂ωr_∂φ(φ, a, p, e, θmin, m, M) * ∂ωθ_∂χ(χ, a, p, e, θmin, m, M) * T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) * H1(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M)   # Eq. C.31
    ∂H1primeInt_∂β(φ::Real, χ::Real, β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = ∂2ωr_∂φ_∂β(φ, β, a, p, e, θmin, m, M, rtol) * ∂ωθ_∂χ(χ, a, p, e, θmin, m, M) * T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) * H1(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) + ∂ωr_∂φ(φ, a, p, e, θmin, m, M) * (∂2ωθ_∂χ_∂β(χ, β, a, p, e, θmin, m, M) * T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) * H1(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) + ∂ωθ_∂χ(χ, a, p, e, θmin, m, M) * (∂T_∂β(φ, χ, β, a, p, e, θmin, m, M) * H1(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) + T(r(φ, a, p, e, θmin, m, M), z(χ, a, p, e, θmin, m, M), a, p, e, θmin, m, M) * ∂H1_∂β(φ, χ, β, a, p, e, θmin, m, M)))

    # compute H1′ and its derivative
    H1prime::Float64 = hcubature(x -> H1primeIntegrand(x[1], x[2], a, p, e, θmin, m, M), (0, 0), (2π, 2π), rtol = rtol)[1]   # Eq. C.31
    ∂H1prime_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real, rtol::Float64) = hcubature(x -> ∂H1primeInt_∂β(x[1], x[2], β, a, p, e, θmin, m, M, rtol), (0, 0), (2π, 2π), rtol = rtol)[1]

    # compute ∂H1_∂β
    ∂H1_∂β(β::Char, a::Real, p::Real, e::Real, θmin::Real, m::Real, M::Real) = (1 / ((2π * Υt)^2)) * (Υt * ∂H1prime_∂β(β, a, p, e, θmin, m, M, rtol) - H1prime * (1/((2π)^2) * ∂Υt_∂β(β, a, p, e, θmin, m, M, rtol)))   # Eq. C.33

    # compute ∂H1_∂β for all β ∈ {m, p, e, θmin}  
    H1partials = ∂H1_∂β.(['m', 'p', 'e', 'θ'], a, p, e, θmin, m, M)

    # calculate inverse of the jacobian defined in Eq. 3.55
    invJacobian = invJacobianVals(a, p, e, θmin, m, M, rtol)

    # vector: {δΓ, δωr, δωθ, δωϕ}
    δvector = transpose(invJacobian) * H1partials   # Eq. 3.58
    δΓ = δvector[1]   # Eq. 3.60
    δω = δvector[2:4]   # Eq. 3.60

    backgroundFreqs = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θmin)

    ωhat = backgroundFreqs[1:3]
    Γhat = backgroundFreqs[4]

    δΩ = (δω / Γhat) - ωhat * δΓ / (Γhat^2)   # Eq. 3.48 

    return (p^(7/2)) * δΩ
end 

module QuadrupoleBumps

d1(r, z, a, M) = sqrt(r^2 - 2 * M * r + (M^2 + a^2) * z)   # Eq. 5.6
L1(r, z, a, M) = sqrt((r - M)^2 + (a^2) * z)   # Eq. 5.7 
c20(r, M) = 2 * ((r - M)^4) - 5 * (M^2) * ((r - M)^2) + 3 * (M^4)   # Eq. 5.8 
c22(r, a, M) = 5 * (M^2) * ((r - M)^2) - 3 * (M^4) + (a^2) * (4 * ((r - M)^2) - 5 * (M^2))   # Eq. 5.9 
c24(r, a, M) = (a^2) * (2 * (a^2) + 5 * (M^2))   # Eq. 5.10 
ψ1(r, z, a, M) = ((M^(3)) / 4) *  sqrt(5 / π)  * (1 / (d1(r, z, a, M)^3)) * ((3 * (L1(r, z, a, M)^2) * z) / (d1(r, z, a, M)^2) - 1)   # Eq. 5.4
γ1(r, z, a, M) = sqrt(5 / π) * ((L1(r, z, a, M) / 2) * (c20(r, M) + c22(r, a, M) * z + c24(r, a, M) * (z^2)) / (d1(r, z, a , M)^5) - 1)   # Eq. 5.5

end
end

module NewtonianFrequencies

δΩr(p, e, θmin, M) = (3/8M) * (1/(p^(7/2))) * sqrt(5/π) * (1 - e^2)^2 * (3* sin(θmin)^2 - 1)
δΩr(e, θmin, M) = (3/8M) * sqrt(5/π) * (1 - e^2)^2 * (3* sin(θmin)^2 - 1)

δΩθ(p, e, θmin, M) = (3/8M) * (1/(p^(7/2))) * sqrt(5/π) * (1 - e^2)^(3/2) * ((sin(θmin)^2) * (5 + 3 * sqrt(1 - e^2)) - sqrt(1 - e^2) - 1)
δΩθ(e, θmin, M) = (3/8M) * sqrt(5/π) * (1 - e^2)^(3/2) * ((sin(θmin)^2) * (5 + 3 * sqrt(1 - e^2)) - sqrt(1 - e^2) - 1)

δΩϕ(p, e, θmin, M) = (3/8M) * (1/(p^(7/2))) * sqrt(5/π) * (1 - e^2)^(3/2) * ((sin(θmin)^2) * (5 + 3 * sqrt(1 - e^2)) - 2 * sin(θmin) - sqrt(1 - e^2) - 1)
δΩϕ(e, θmin, M) = (3/8M) * sqrt(5/π) * (1 - e^2)^(3/2) * ((sin(θmin)^2) * (5 + 3 * sqrt(1 - e^2)) - 2 * sin(θmin) - sqrt(1 - e^2) - 1)

end