module MinoDerivs4
d4t_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, a::Float64, E::Float64, L::Float64, C::Float64) = (12*(a^5*L + 4*a^3*(a*E - L) + 8*a^3*(-(a*E) + L)*x[1] - 6*a^3*L*x[1]^2 + 8*a^2*E*x[1]^3 + (a*L - 4*E)*x[1]^4)*dx[1]^3)/(a^2 - 2*x[1] + x[1]^2)^4 + (6*(a^6*E + 4*a^3*(a*E - L) + x[1]*(6*a^3*(-(a*E) + L) +
x[1]*(3*a^4*E + x[1]*(-2*a*(6*a*E + L) + E*x[1]*(3*(a^2 + 4) - 6*x[1] + x[1]^2)))))*dx[1]*d2x[1])/(a^2 - 2*x[1] + x[1]^2)^3 - 6*a^2*E*cos(2*x[2])*dx[2]*d2x[2] + (2*(a^3*(a*E - L) + x[1]*(a^4*E + x[1]*(a*(-2*a*E + L) + E*x[1]*(2*a^2 - 3*x[1] + x[1]^2))))*d3x[1])/(a^2 - 2*x[1] + x[1]^2)^2 + a^2*E*sin(2*x[2])*(4*dx[2]^3 - d3x[2])

d4r_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, a::Float64, E::Float64, L::Float64, C::Float64) = 6*(1.0 + 2*(-1 + E^2)*x[1])*dx[1]^2 + (-C + a^2*(-1 + E^2) - L^2 + 6*x[1]*(1.0 + (-1 + E^2)*x[1]))*d2x[1]

d4θ_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, a::Float64, E::Float64, L::Float64, C::Float64) = sin(x[2])^2*(2*cot(x[2])*(2*a^2*(-1 + E^2) + L^2*(5 + cos(2*x[2]))*csc(x[2])^6)*dx[2]^2 - (a^2*(-1 + E^2)*(-1 + cot(x[2])^2) + 2*L^2*cot(x[2])^2*csc(x[2])^4 + L^2*csc(x[2])^6)*d2x[2])

d4ϕ_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, a::Float64, E::Float64, L::Float64, C::Float64) = 2*((-6*a*(a^3*(a*E - 2*L) + 4*a*(-(a*E) + L) + x[1]*(2*a^3*L + 8*a*(a*E - L) + 6*a*(-(a*E) + L)*x[1] -
2*a*L*x[1]^2 + E*x[1]^3))*dx[1]^3)/(a^2 - 2*x[1] + x[1]^2)^4 - 2*L*(5 + cos(2*x[2]))*cot(x[2])*csc(x[2])^4*dx[2]^3 + (3*a*(a^3*L + 4*a*(a*E - L) + 6*a*(-(a*E) + L)*x[1] - 3*a*L*x[1]^2 + 2*E*x[1]^3)*dx[1]*d2x[1])/(a^2 - 2*x[1] + x[1]^2)^3 + 3*L*(2 +
cos(2*x[2]))*csc(x[2])^4*dx[2]*d2x[2] + (a*(a*(a*E - L) + a*L*x[1] - E*x[1]^2)*d3x[1])/(a^2 - 2*x[1] + x[1]^2)^2 - L*cot(x[2])*csc(x[2])^2*d3x[2])
end