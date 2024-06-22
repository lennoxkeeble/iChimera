module MinoDeriv3 
d3t_dλ(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (2*(a^6*E + 
4*a^3*(a*E - L)*M^2 + x[1]*(6*a^3*(-(a*E) + L)*M + x[1]*(3*a^4*E + x[1]*(-2*a*(6*a*E + L)*M + E*x[1]*(3*(a^2 + 4*M^2) - 6*M*x[1] + 
x[1]^2)))))*dx[1]^2)/(a^2 - 2*M*x[1] + x[1]^2)^3 - 2*a^2*E*cos(2*x[2])*dx[2]^2 + (2*(a^3*(a*E - L)*M + x[1]*(a^4*E + x[1]*(a*(-2*a*E + L)*M + 
E*x[1]*(2*a^2 - 3*M*x[1] + x[1]^2))))*d2x[1])/(a^2 - 2*M*x[1] + x[1]^2)^2 - a^2*E*sin(2*x[2])*d2x[2]

d3r_dλ(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (-2*((C + (-(a*E) + 
L)^2)*M - (C - a^2*(-1 + E^2) + L^2)*x[1] + 3*M*x[1]^2 + 2*(-1 + E^2)*x[1]^3)^2*dx[1]^2 +
2*(-C + a^2*(-1 + E^2) - L^2 + 6*x[1]*(M + (-1 + E^2)*x[1]))*(-((C + (-(a*E) + L)^2 + x[1]^2)*(a^2 - 2*M*x[1] + x[1]^2)) + 
(a*L - E*(a^2 + x[1]^2))^2)*dx[1]^2 + 2*((C + (-(a*E) + L)^2)*M - (C - a^2*(-1 + E^2) + L^2)*x[1] + 3*M*x[1]^2 + 
2*(-1 + E^2)*x[1]^3)*(-((C + (-(a*E) + L)^2 + x[1]^2)*(a^2 - 2*M*x[1] + x[1]^2)) + (a*L - E*(a^2 + 
x[1]^2))^2)*d2x[1])/(2*(-((C + (-(a*E) + L)^2 + x[1]^2)*(a^2 - 2*M*x[1] + x[1]^2)) + (a*L - E*(a^2 + x[1]^2))^2)^1.5)

d3θ_dλ(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (8*(-3*a^2*(-1 + E^2)*(a^2*(-1 + E^2) + 
16*L^2) - 4*a^2*(-1 + E^2)*(a^2*(-1 + E^2) + 2*(C + L^2))*cos(2*x[2]) - a^4*(-1 +
E^2)^2*cos(4*x[2]) + 8*L^2*csc(x[2])^2*(2*(C + 4*a^2*(-1 + E^2) + L^2) - (3*C + 3*a^2*(-1 + E^2) + 4*L^2)*csc(x[2])^2 + 
2*L^2*csc(x[2])^4))*dx[2]^2 + (a^2 - 4*C - a^2*E^2 + 4*L^2 + 4*(C + L^2)*cos(2*x[2]) + a^2*(-1 + 
E^2)*cos(4*x[2]))*(3*a^2*(-1 + E^2) - 8*L^2 + a^2*(-1 + E^2)*(-4*cos(2*x[2]) + 
cos(4*x[2])))*cot(x[2])*csc(x[2])^4*d2x[2])/(64*(C + a^2*(-1 + E^2)*cos(x[2])^2 - L^2*cot(x[2])^2)^1.5)

d3ϕ_dλ(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = 2*((a*(a^3*L + 4*a*(a*E - L)*M^2 +
6*a*(-(a*E) + L)*M*x[1] - 3*a*L*x[1]^2 + 2*E*M*x[1]^3)*dx[1]^2)/(a^2 - 
2*M*x[1] + x[1]^2)^3 + L*(2 + cos(2*x[2]))*csc(x[2])^4*dx[2]^2 + (a*(a*(a*E - L)*M + a*L*x[1] -
E*M*x[1]^2)*d2x[1])/(a^2 - 2*M*x[1] + x[1]^2)^2 - L*cot(x[2])*csc(x[2])^2*d2x[2])
end