#= 
    Suppose we have some function, f, parameterized by the parameter t. Further, suppose we wish to re-parameterize this function with λ(t), and we have computed values for f(λ). In this module,
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

end