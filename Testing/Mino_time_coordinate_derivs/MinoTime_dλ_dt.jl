#=

    In this module we compute derivatives of Mino time, λ, with respect to coordinate time t, e.g., λ^{(n)}(t), for n = 1, 2, 3, 4, 5, 6, 7, 8, using t^{(n)}(λ) via the chain rule

=#
module MinoTimeDerivs
dλ_dt(dt_dλ::Float64) = 1/dt_dλ

d2λ_dt(dt_dλ::Float64, d2t_dλ::Float64) = -(d2t_dλ/dt_dλ^3)

d3λ_dt(dt_dλ::Float64, d2t_dλ::Float64, d3t_dλ::Float64) = (3*d2t_dλ^2 - dt_dλ*d3t_dλ)/dt_dλ^5

d4λ_dt(dt_dλ::Float64, d2t_dλ::Float64, d3t_dλ::Float64, d4t_dλ::Float64) = -((15*d2t_dλ^3 - 10*dt_dλ*d2t_dλ*d3t_dλ + dt_dλ^2*d4t_dλ)/dt_dλ^7)

d5λ_dt(dt_dλ::Float64, d2t_dλ::Float64, d3t_dλ::Float64, d4t_dλ::Float64, d5t_dλ::Float64) = (105*d2t_dλ^4 - 105*dt_dλ*d2t_dλ^2*d3t_dλ +
5*dt_dλ^2*(2*d3t_dλ^2 + 3*d2t_dλ*d4t_dλ) - dt_dλ^3*d5t_dλ)/dt_dλ^9

d6λ_dt(dt_dλ::Float64, d2t_dλ::Float64, d3t_dλ::Float64, d4t_dλ::Float64, d5t_dλ::Float64, d6t_dλ::Float64) = (7*(-135*d2t_dλ^5 + 180*dt_dλ*d2t_dλ^3*d3t_dλ -
30*dt_dλ^2*d2t_dλ^2*d4t_dλ + 5*dt_dλ^3*d3t_dλ*d4t_dλ + dt_dλ^2*d2t_dλ*(-40*d3t_dλ^2 + 3*dt_dλ*d5t_dλ)) - dt_dλ^4*d6t_dλ)/dt_dλ^11

d7λ_dt(dt_dλ::Float64, d2t_dλ::Float64, d3t_dλ::Float64, d4t_dλ::Float64, d5t_dλ::Float64, d6t_dλ::Float64, d7t_dλ::Float64) = (7*(1485*d2t_dλ^6 - 2475*dt_dλ*d2t_dλ^4*d3t_dλ + 
450*dt_dλ^2*d2t_dλ^3*d4t_dλ + 18*dt_dλ^2*d2t_dλ^2*(50*d3t_dλ^2 - 3*dt_dλ*d5t_dλ) + dt_dλ^3*(-40*d3t_dλ^3 + 5*dt_dλ*d4t_dλ^2 +
8*dt_dλ*d3t_dλ*d5t_dλ) + 4*dt_dλ^3*d2t_dλ*(-45*d3t_dλ*d4t_dλ + dt_dλ*d6t_dλ)) - dt_dλ^5*d7t_dλ)/dt_dλ^13


d8λ_dt(dt_dλ::Float64, d2t_dλ::Float64, d3t_dλ::Float64, d4t_dλ::Float64, d5t_dλ::Float64, d6t_dλ::Float64, d7t_dλ::Float64, d8t_dλ::Float64) = (-135135*d2t_dλ^7 + 270270*dt_dλ*d2t_dλ^5*d3t_dλ -
51975*dt_dλ^2*d2t_dλ^4*d4t_dλ + 6930*dt_dλ^2*d2t_dλ^3*(-20*d3t_dλ^2 + dt_dλ*d5t_dλ) - 630*dt_dλ^3*d2t_dλ^2*(-55*d3t_dλ*d4t_dλ +
dt_dλ*d6t_dλ) + dt_dλ^3*d2t_dλ*(-35*(-440*d3t_dλ^3 + 45*dt_dλ*d4t_dλ^2 + 72*dt_dλ*d3t_dλ*d5t_dλ) + 36*dt_dλ^2*d7t_dλ) + dt_dλ^4*(-2100*d3t_dλ^2*d4t_dλ + 42*dt_dλ*(3*d4t_dλ*d5t_dλ +
2*d3t_dλ*d6t_dλ) - dt_dλ^2*d8t_dλ))/dt_dλ^15
end