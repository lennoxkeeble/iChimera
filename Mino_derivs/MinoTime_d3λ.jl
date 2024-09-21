module MinoDerivs3
d3t_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (2*(a^6*E + 4*a^3*(a*E - L)*M^2 + x[1]*(6*a^3*(-(a*E) + L)*M + x[1]*(3*a^4*E + x[1]*(-2*a*(6*a*E + L)*M + E*x[1]*(3*(a^2 + 4*M^2) - 6*M*x[1] + x[1]^2)))))*dx[1]^2)/(a^2 - 2*M*x[1] + x[1]^2)^3 - 2*a^2*E*cos(2*x[2])*dx[2]^2 + (2*(a^3*(a*E - L)*M + x[1]*(a^4*E + x[1]*(a*(-2*a*E + L)*M + E*x[1]*(2*a^2 - 3*M*x[1] + x[1]^2))))*d2x[1])/(a^2 - 2*M*x[1] + x[1]^2)^2 - a^2*E*sin(2*x[2])*d2x[2]

d3r_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (-C + a^2*(-1 + E^2) - L^2 + 6*x[1]*(M + (-1 + E^2)*x[1]))*dx[1]

d3θ_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (-(a^2*(-1 + E^2)*cos(2*x[2])) - L^2*(2 + cos(2*x[2]))*csc(x[2])^4)*dx[2]

d3ϕ_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = 2*((a*(a^3*L + 4*a*(a*E - L)*M^2 + 6*a*(-(a*E) + L)*M*x[1] - 3*a*L*x[1]^2 + 2*E*M*x[1]^3)*dx[1]^2)/(a^2 - 2*M*x[1] + x[1]^2)^3 + L*(2 +
cos(2*x[2]))*csc(x[2])^4*dx[2]^2 + (a*(a*(a*E - L)*M + a*L*x[1] - E*M*x[1]^2)*d2x[1])/(a^2 - 2*M*x[1] + x[1]^2)^2 - L*cot(x[2])*csc(x[2])^2*d2x[2])
end