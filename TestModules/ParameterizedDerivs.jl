#= 
    Suppose we have some function, f, parameterized by the parameter t. Further, suppose we wish to re-parameterize this function with λ, and we have computed values for f. In this module,
    we compute values of the nth derivative of f wrt t, in terms of derivatives of f wrt λ, and derivatives of λ wrt t.
=#
module ParameterizedDerivs

df_dt(df_dλ::Float64, dλ_dt::Float64) = df_dλ*dλ_dt

d2f_dt(df_dλ::Float64, dλ_dt::Float64, d2f_dλ::Float64, d2λ_dt::Float64) = dλ_dt^2*d2f_dλ + df_dλ*d2λ_dt

d3f_dt(df_dλ::Float64, dλ_dt::Float64, d2f_dλ::Float64, d2λ_dt::Float64, d3f_dλ::Float64, d3λ_dt::Float64) = 3*dλ_dt*d2f_dλ*d2λ_dt + dλ_dt^3*d3f_dλ + df_dλ*d3λ_dt

d4f_dt(df_dλ::Float64, dλ_dt::Float64, d2f_dλ::Float64, d2λ_dt::Float64, d3f_dλ::Float64, d3λ_dt::Float64, d4f_dλ::Float64, d4λ_dt::Float64) = 6*dλ_dt^2*d2λ_dt*d3f_dλ + d2f_dλ*(3*d2λ_dt^2 + 4*dλ_dt*d3λ_dt) + 
dλ_dt^4*d4f_dλ + df_dλ*d4λ_dt

d5f_dt(df_dλ::Float64, dλ_dt::Float64, d2f_dλ::Float64, d2λ_dt::Float64, d3f_dλ::Float64, d3λ_dt::Float64, d4f_dλ::Float64, d4λ_dt::Float64, d5f_dλ::Float64, d5λ_dt::Float64) = 5*(2*d2f_dλ*d2λ_dt*d3λ_dt + 
2*dλ_dt^2*d3f_dλ*d3λ_dt + 2*dλ_dt^3*d2λ_dt*d4f_dλ + dλ_dt*(3*d2λ_dt^2*d3f_dλ + d2f_dλ*d4λ_dt)) + dλ_dt^5*d5f_dλ + df_dλ*d5λ_dt

d6f_dt(df_dλ::Float64, dλ_dt::Float64, d2f_dλ::Float64, d2λ_dt::Float64, d3f_dλ::Float64, d3λ_dt::Float64, d4f_dλ::Float64, d4λ_dt::Float64, d5f_dλ::Float64, d5λ_dt::Float64, d6f_dλ::Float64, d6λ_dt::Float64) = 15*d2λ_dt^3*d3f_dλ + 
45*dλ_dt^2*d2λ_dt^2*d4f_dλ + 20*dλ_dt^3*d3λ_dt*d4f_dλ + 15*dλ_dt^2*d3f_dλ*d4λ_dt + 15*d2λ_dt*(4*dλ_dt*d3f_dλ*d3λ_dt + d2f_dλ*d4λ_dt + dλ_dt^4*d5f_dλ) + 2*d2f_dλ*(5*d3λ_dt^2 + 3*dλ_dt*d5λ_dt) + dλ_dt^6*d6f_dλ + df_dλ*d6λ_dt

d7f_dt(df_dλ::Float64, dλ_dt::Float64, d2f_dλ::Float64, d2λ_dt::Float64, d3f_dλ::Float64, d3λ_dt::Float64, d4f_dλ::Float64, d4λ_dt::Float64, d5f_dλ::Float64, d5λ_dt::Float64, d6f_dλ::Float64, d6λ_dt::Float64, 
d7f_dλ::Float64, d7λ_dt::Float64) = 7*(15*dλ_dt*d2λ_dt^3*d4f_dλ + 5*d2f_dλ*d3λ_dt*d4λ_dt + 5*dλ_dt^3*d4f_dλ*d4λ_dt + 5*dλ_dt^4*d3λ_dt*d5f_dλ + 15*d2λ_dt^2*(d3f_dλ*d3λ_dt + dλ_dt^3*d5f_dλ) + 
3*dλ_dt^2*d3f_dλ*d5λ_dt + 3*d2λ_dt*(10*dλ_dt^2*d3λ_dt*d4f_dλ + 5*dλ_dt*d3f_dλ*d4λ_dt + d2f_dλ*d5λ_dt + dλ_dt^5*d6f_dλ) + dλ_dt*(10*d3f_dλ*d3λ_dt^2 + d2f_dλ*d6λ_dt)) + dλ_dt^7*d7f_dλ + df_dλ*d7λ_dt

d8f_dt(df_dλ::Float64, dλ_dt::Float64, d2f_dλ::Float64, d2λ_dt::Float64, d3f_dλ::Float64, d3λ_dt::Float64, d4f_dλ::Float64, d4λ_dt::Float64, d5f_dλ::Float64, d5λ_dt::Float64, d6f_dλ::Float64, d6λ_dt::Float64, 
d7f_dλ::Float64, d7λ_dt::Float64, d8f_dλ::Float64, d8λ_dt::Float64) = 105*d2λ_dt^4*d4f_dλ + 420*dλ_dt^2*d2λ_dt^3*d5f_dλ + 70*dλ_dt^4*d4λ_dt*d5f_dλ + 56*dλ_dt^3*d4f_dλ*d5λ_dt + 7*d2f_dλ*(5*d4λ_dt^2 + 8*d3λ_dt*d5λ_dt) + 
56*dλ_dt^5*d3λ_dt*d6f_dλ + 210*d2λ_dt^2*(4*dλ_dt*d3λ_dt*d4f_dλ + d3f_dλ*d4λ_dt + dλ_dt^4*d6f_dλ) + 28*dλ_dt^2*(10*d3λ_dt^2*d4f_dλ + d3f_dλ*d6λ_dt) + 28*d2λ_dt*(15*dλ_dt^2*d4f_dλ*d4λ_dt + 20*dλ_dt^3*d3λ_dt*d5f_dλ +
2*d3f_dλ*(5*d3λ_dt^2 + 3*dλ_dt*d5λ_dt) + d2f_dλ*d6λ_dt + dλ_dt^6*d7f_dλ) + 8*dλ_dt*(35*d3f_dλ*d3λ_dt*d4λ_dt + d2f_dλ*d7λ_dt) + dλ_dt^8*d8f_dλ + df_dλ*d8λ_dt

end