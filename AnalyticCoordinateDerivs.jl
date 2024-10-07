#=

  In this module we write a function to compute the derivatives of the coordinates with respect to BL time, d^n(x^μ)/dt^n for n = 1, ..., 8, from the derivatives with respect to mino time, d^n(x^μ)/dλ^n.

=#

module AnalyticCoordinateDerivs
using ..MinoDerivs1
using ..MinoDerivs2
using ..MinoDerivs3
using ..MinoDerivs4
using ..MinoDerivs5
using ..MinoDerivs6
using ..MinoDerivs7
using ..MinoDerivs8
using ..ParameterizedDerivs
using ..MinoTimeDerivs

"""
    ComputeDerivs!()

Computes the derivatives of the coordinates r, θ, ϕ with respect to BL time, d^n(x^μ)/dt^n for n = 1, ..., 8, from the derivatives with respect to mino time, d^n(x^μ)/dλ^n.

# Arguments
- `x::AbstractVector{Float64}`: BL coordinates [r, θ, ϕ].
- `sign_dr::Float64`: Sign of dr/dλ-the solution from the goedesic equation introduces a definite sign (in comparison to sign choice if one simply evaluates dr/dλ = sqrt(Θ) in isolation).
- `sign_dθ::Float64`: Sign of dθ/dλ from the geodesic equation.
- `dx_dt::AbstractVector{Float64}`: empty three vector to be filled with coordinate derivatives [dr/dt, dθ/dt, dϕ/dt].
- `d2x_dt::AbstractVector{Float64}`: empty three vector to be filled with second coordinate derivatives [d2r/dt2, d2θ/dt2, d2ϕ/dt2].
- `d3x_dt::AbstractVector{Float64}`: empty three vector to be filled with third coordinate derivatives [d3r/dt3, d3θ/dt3, d3ϕ/dt3].
- `d4x_dt::AbstractVector{Float64}`: empty three vector to be filled with fourth coordinate derivatives [d4r/dt4, d4θ/dt4, d4ϕ/dt4].
- `d5x_dt::AbstractVector{Float64}`: empty three vector to be filled with fifth coordinate derivatives [d5r/dt5, d5θ/dt5, d5ϕ/dt5].
- `d6x_dt::AbstractVector{Float64}`: empty three vector to be filled with sixth coordinate derivatives [d6r/dt6, d6θ/dt6, d6ϕ/dt6].
- `d7x_dt::AbstractVector{Float64}`: empty three vector to be filled with seventh coordinate derivatives [d7r/dt7, d7θ/dt7, d7ϕ/dt7].
- `d8x_dt::AbstractVector{Float64}`: empty three vector to be filled with eighth coordinate derivatives [d8r/dt8, d8θ/dt8, d8ϕ/dt8].
- `dx_dλ::AbstractVector{Float64}`: three vector with values of the Mino time derivatives [dr/dλ, dθ/dλ, dϕ/dλ].
- `d2x_dλ::AbstractVector{Float64}`: three vector with values of the second Mino time derivatives [d2r/dλ2, d2θ/dλ2, d2ϕ/dλ2].
- `d3x_dλ::AbstractVector{Float64}`: three vector with values of the third Mino time derivatives [d3r/dλ3, d3θ/dλ3, d3ϕ/dλ3].
- `d4x_dλ::AbstractVector{Float64}`: three vector with values of the fourth Mino time derivatives [d4r/dλ4, d4θ/dλ4, d4ϕ/dλ4].
- `d5x_dλ::AbstractVector{Float64}`: three vector with values of the fifth Mino time derivatives [d5r/dλ5, d5θ/dλ5, d5ϕ/dλ5].
- `d6x_dλ::AbstractVector{Float64}`: three vector with values of the sixth Mino time derivatives [d6r/dλ6, d6θ/dλ6, d6ϕ/dλ6].
- `d7x_dλ::AbstractVector{Float64}`: three vector with values of the seventh Mino time derivatives [d7r/dλ7, d7θ/dλ7, d7ϕ/dλ7].
- `d8x_dλ::AbstractVector{Float64}`: three vector with values of the eighth Mino time derivatives [d8r/dλ8, d8θ/dλ8, d8ϕ/dλ8].
- `a::Float64`: Spin parameter of the black hole.
- `E::Float64`: Energy of the particle.
- `L::Float64`: Axial angular momentum of the particle.
- `C::Float64`: Carter constant of the particle.

# Returns
- `nothing`: mutates input arrays dnx_dt with the corresponding values of the derivatives.

# Notes
- This function is used in the computation of the radiation reaction fluxes for the Chimera inspiral. In particular, when we solve evolve in Mino time and we compute derivatives of the Multipole moments (either via finite difference or with and
Fourier-fitting procedure), we are left with derivatices with respect to Mino time. We then use this function to convert these derivatives to derivatives with respect to BL time, which are those used in the radiation reaction flux computation.
- See the folder "Mino_derivs" for the expressions behind the functions called below. They were copied from mathematica, in which we did some algebraic manipulations to significantly reduce their complexity and length (as opposed to if one directly
took derivatives of the geodesic equations without any simplifcations, in which case the expressions become unwieldly).
"""
# x = [r, θ, ϕ], while dx_dt, d2x_dt,..., d8x_dt are empty arrays to be filled like dx_dt = [dr/dt, dθ/dt, dϕ/dt], ..., d8x_dt = [d8r/dt8, d8θ/dt8, d8ϕ/dt8] (and similarly for Mino time)
function ComputeDerivs!(x::AbstractVector{Float64}, sign_dr::Float64, sign_dθ::Float64, dx_dt::AbstractVector{Float64}, d2x_dt::AbstractVector{Float64}, d3x_dt::AbstractVector{Float64}, d4x_dt::AbstractVector{Float64},
  d5x_dt::AbstractVector{Float64}, d6x_dt::AbstractVector{Float64}, d7x_dt::AbstractVector{Float64}, d8x_dt::AbstractVector{Float64}, dx_dλ::AbstractVector{Float64}, d2x_dλ::AbstractVector{Float64}, d3x_dλ::AbstractVector{Float64},
  d4x_dλ::AbstractVector{Float64}, d5x_dλ::AbstractVector{Float64}, d6x_dλ::AbstractVector{Float64}, d7x_dλ::AbstractVector{Float64}, d8x_dλ::AbstractVector{Float64}, a::Float64, E::Float64, L::Float64, C::Float64)

  ### COMPUTE DIRST-ORDER SPATIAL DERIVS WRT MINO TIME ###
  dx_dλ[1] = MinoDerivs1.dr_dλ(x, a, E, L, C) * sign_dr;
  dx_dλ[2] = MinoDerivs1.dθ_dλ(x, a, E, L, C) * sign_dθ;
  dx_dλ[3] = MinoDerivs1.dϕ_dλ(x, a, E, L, C);

  ### COMPUTE HIGHER-ORDER DERIVS WRT MINO TIME ###
  d2x_dλ[1] = MinoDerivs2.d2r_dλ(x, dx_dλ, a, E, L, C); 
  d2x_dλ[2] = MinoDerivs2.d2θ_dλ(x, dx_dλ, a, E, L, C);
  d2x_dλ[3] = MinoDerivs2.d2ϕ_dλ(x, dx_dλ, a, E, L, C);
  
  d3x_dλ[1] = MinoDerivs3.d3r_dλ(x, dx_dλ, d2x_dλ, a, E, L, C); 
  d3x_dλ[2] = MinoDerivs3.d3θ_dλ(x, dx_dλ, d2x_dλ, a, E, L, C);
  d3x_dλ[3] = MinoDerivs3.d3ϕ_dλ(x, dx_dλ, d2x_dλ, a, E, L, C);
  
  d4x_dλ[1] = MinoDerivs4.d4r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C); 
  d4x_dλ[2] = MinoDerivs4.d4θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C);
  d4x_dλ[3] = MinoDerivs4.d4ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C);
  
  d5x_dλ[1] = MinoDerivs5.d5r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C); 
  d5x_dλ[2] = MinoDerivs5.d5θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C);
  d5x_dλ[3] = MinoDerivs5.d5ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C);
  
  d6x_dλ[1] = MinoDerivs6.d6r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, E, L, C); 
  d6x_dλ[2] = MinoDerivs6.d6θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, E, L, C);
  d6x_dλ[3] = MinoDerivs6.d6ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, E, L, C);
  
  d7x_dλ[1] = MinoDerivs7.d7r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, a, E, L, C); 
  d7x_dλ[2] = MinoDerivs7.d7θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, a, E, L, C);
  d7x_dλ[3] = MinoDerivs7.d7ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, a, E, L, C);
  
  d8x_dλ[1] = MinoDerivs8.d8r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, a, E, L, C); 
  d8x_dλ[2] = MinoDerivs8.d8θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, a, E, L, C);
  d8x_dλ[3] = MinoDerivs8.d8ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, a, E, L, C);

  ### COMPUTE DERIVATIVES OF COORDINATE TIME WRT MINO TIME ### 
  dt_dλ = MinoDerivs1.dt_dλ(x, a, E, L, C);
  d2t_dλ = MinoDerivs2.d2t_dλ(x, dx_dλ, a, E, L, C);
  d3t_dλ = MinoDerivs3.d3t_dλ(x, dx_dλ, d2x_dλ, a, E, L, C);
  d4t_dλ = MinoDerivs4.d4t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C);
  d5t_dλ = MinoDerivs5.d5t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C);
  d6t_dλ = MinoDerivs6.d6t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, E, L, C);
  d7t_dλ = MinoDerivs7.d7t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, a, E, L, C);
  d8t_dλ = MinoDerivs8.d8t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, a, E, L, C);

  ### COMPUTE DERIVATIVES OF MINO TIME WRT COORDINATE TIME ### 
  dλ_dt = MinoTimeDerivs.dλ_dt(dt_dλ)
  d2λ_dt = MinoTimeDerivs.d2λ_dt(dt_dλ, d2t_dλ)
  d3λ_dt = MinoTimeDerivs.d3λ_dt(dt_dλ, d2t_dλ, d3t_dλ)
  d4λ_dt = MinoTimeDerivs.d4λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ)
  d5λ_dt = MinoTimeDerivs.d5λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ)
  d6λ_dt = MinoTimeDerivs.d6λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ)
  d7λ_dt = MinoTimeDerivs.d7λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ, d7t_dλ)
  d8λ_dt = MinoTimeDerivs.d8λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ, d7t_dλ, d8t_dλ)

  ### COMPUTE d^(n)x/dt^n ###
  dx_dt[1] = ParameterizedDerivs.df_dt(dx_dλ[1], dλ_dt);
  dx_dt[2] = ParameterizedDerivs.df_dt(dx_dλ[2], dλ_dt); 
  dx_dt[3] = ParameterizedDerivs.df_dt(dx_dλ[3], dλ_dt);

  d2x_dt[1] = ParameterizedDerivs.d2f_dt(dx_dλ[1], dλ_dt, d2x_dλ[1], d2λ_dt);
  d2x_dt[2] = ParameterizedDerivs.d2f_dt(dx_dλ[2], dλ_dt, d2x_dλ[2], d2λ_dt);
  d2x_dt[3] = ParameterizedDerivs.d2f_dt(dx_dλ[3], dλ_dt, d2x_dλ[3], d2λ_dt);

  d3x_dt[1] = ParameterizedDerivs.d3f_dt(dx_dλ[1], dλ_dt, d2x_dλ[1], d2λ_dt, d3x_dλ[1], d3λ_dt);
  d3x_dt[2] = ParameterizedDerivs.d3f_dt(dx_dλ[2], dλ_dt, d2x_dλ[2], d2λ_dt, d3x_dλ[2], d3λ_dt);
  d3x_dt[3] = ParameterizedDerivs.d3f_dt(dx_dλ[3], dλ_dt, d2x_dλ[3], d2λ_dt, d3x_dλ[3], d3λ_dt);

  d4x_dt[1] = ParameterizedDerivs.d4f_dt(dx_dλ[1], dλ_dt, d2x_dλ[1], d2λ_dt, d3x_dλ[1], d3λ_dt, d4x_dλ[1], d4λ_dt);
  d4x_dt[2] = ParameterizedDerivs.d4f_dt(dx_dλ[2], dλ_dt, d2x_dλ[2], d2λ_dt, d3x_dλ[2], d3λ_dt, d4x_dλ[2], d4λ_dt);
  d4x_dt[3] = ParameterizedDerivs.d4f_dt(dx_dλ[3], dλ_dt, d2x_dλ[3], d2λ_dt, d3x_dλ[3], d3λ_dt, d4x_dλ[3], d4λ_dt);

  d5x_dt[1] = ParameterizedDerivs.d5f_dt(dx_dλ[1], dλ_dt, d2x_dλ[1], d2λ_dt, d3x_dλ[1], d3λ_dt, d4x_dλ[1], d4λ_dt, d5x_dλ[1], d5λ_dt);
  d5x_dt[2] = ParameterizedDerivs.d5f_dt(dx_dλ[2], dλ_dt, d2x_dλ[2], d2λ_dt, d3x_dλ[2], d3λ_dt, d4x_dλ[2], d4λ_dt, d5x_dλ[2], d5λ_dt);
  d5x_dt[3] = ParameterizedDerivs.d5f_dt(dx_dλ[3], dλ_dt, d2x_dλ[3], d2λ_dt, d3x_dλ[3], d3λ_dt, d4x_dλ[3], d4λ_dt, d5x_dλ[3], d5λ_dt);

  d6x_dt[1] = ParameterizedDerivs.d6f_dt(dx_dλ[1], dλ_dt, d2x_dλ[1], d2λ_dt, d3x_dλ[1], d3λ_dt, d4x_dλ[1], d4λ_dt, d5x_dλ[1], d5λ_dt, d6x_dλ[1], d6λ_dt);
  d6x_dt[2] = ParameterizedDerivs.d6f_dt(dx_dλ[2], dλ_dt, d2x_dλ[2], d2λ_dt, d3x_dλ[2], d3λ_dt, d4x_dλ[2], d4λ_dt, d5x_dλ[2], d5λ_dt, d6x_dλ[2], d6λ_dt);
  d6x_dt[3] = ParameterizedDerivs.d6f_dt(dx_dλ[3], dλ_dt, d2x_dλ[3], d2λ_dt, d3x_dλ[3], d3λ_dt, d4x_dλ[3], d4λ_dt, d5x_dλ[3], d5λ_dt, d6x_dλ[3], d6λ_dt);

  d7x_dt[1] = ParameterizedDerivs.d7f_dt(dx_dλ[1], dλ_dt, d2x_dλ[1], d2λ_dt, d3x_dλ[1], d3λ_dt, d4x_dλ[1], d4λ_dt, d5x_dλ[1], d5λ_dt, d6x_dλ[1], d6λ_dt, d7x_dλ[1], d7λ_dt);
  d7x_dt[2] = ParameterizedDerivs.d7f_dt(dx_dλ[2], dλ_dt, d2x_dλ[2], d2λ_dt, d3x_dλ[2], d3λ_dt, d4x_dλ[2], d4λ_dt, d5x_dλ[2], d5λ_dt, d6x_dλ[2], d6λ_dt, d7x_dλ[2], d7λ_dt);
  d7x_dt[3] = ParameterizedDerivs.d7f_dt(dx_dλ[3], dλ_dt, d2x_dλ[3], d2λ_dt, d3x_dλ[3], d3λ_dt, d4x_dλ[3], d4λ_dt, d5x_dλ[3], d5λ_dt, d6x_dλ[3], d6λ_dt, d7x_dλ[3], d7λ_dt);

  d8x_dt[1] = ParameterizedDerivs.d8f_dt(dx_dλ[1], dλ_dt, d2x_dλ[1], d2λ_dt, d3x_dλ[1], d3λ_dt, d4x_dλ[1], d4λ_dt, d5x_dλ[1], d5λ_dt, d6x_dλ[1], d6λ_dt, d7x_dλ[1], d7λ_dt, d8x_dλ[1], d8λ_dt);
  d8x_dt[2] = ParameterizedDerivs.d8f_dt(dx_dλ[2], dλ_dt, d2x_dλ[2], d2λ_dt, d3x_dλ[2], d3λ_dt, d4x_dλ[2], d4λ_dt, d5x_dλ[2], d5λ_dt, d6x_dλ[2], d6λ_dt, d7x_dλ[2], d7λ_dt, d8x_dλ[2], d8λ_dt);
  d8x_dt[3] = ParameterizedDerivs.d8f_dt(dx_dλ[3], dλ_dt, d2x_dλ[3], d2λ_dt, d3x_dλ[3], d3λ_dt, d4x_dλ[3], d4λ_dt, d5x_dλ[3], d5λ_dt, d6x_dλ[3], d6λ_dt, d7x_dλ[3], d7λ_dt, d8x_dλ[3], d8λ_dt);
end
end