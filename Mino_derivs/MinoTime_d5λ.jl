module MinoDerivs5
d5t_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, d4x::AbstractArray, a::Float64, E::Float64, L::Float64, C::Float64) = (-48*(2*a^5*(a*E - 2*L) + 8*a^3*(-(a*E) + L) + x[1]*(5*a^5*L + 20*a^3*(a*E - L) + 20*a^3*(-(a*E) + L)*x[1] - 10*a^3*L*x[1]^2 +
10*a^2*E*x[1]^3 + (a*L - 4*E)*x[1]^4))*dx[1]^4)/(a^2 - 2*x[1] + x[1]^2)^5 + (72*(a^5*L + 4*a^3*(a*E - L) + 8*a^3*(-(a*E) + L)*x[1] - 6*a^3*L*x[1]^2 + 8*a^2*E*x[1]^3 + (a*L - 4*E)*x[1]^4)*dx[1]^2*d2x[1])/(a^2 - 2*x[1] + x[1]^2)^4 + (8*(a^6*E + 4*a^3*(a*E - L) + x[1]*(6*a^3*(-(a*E) + L) +
x[1]*(3*a^4*E + x[1]*(-2*a*(6*a*E + L) + E*x[1]*(3*(a^2 + 4) - 6*x[1] + x[1]^2)))))*dx[1]*d3x[1])/(a^2 - 2*x[1] + x[1]^2)^3 + 2*a^2*E*cos(2*x[2])*(4*dx[2]^4 - 3*d2x[2]^2 - 4*dx[2]*d3x[2]) + (6*(a^6*E + 4*a^3*(a*E - L) + x[1]*(6*a^3*(-(a*E) + L) + x[1]*(3*a^4*E + x[1]*(-2*a*(6*a*E + L) + E*x[1]*(3*(a^2 + 4) -
6*x[1] + x[1]^2)))))*d2x[1]^2 + 2*(a^2 - 2*x[1] + x[1]^2)*(a^3*(a*E - L) + x[1]*(a^4*E + x[1]*(a*(-2*a*E + L) + E*x[1]*(2*a^2 - 3*x[1] + x[1]^2))))*d4x[1])/(a^2 - 2*x[1] + x[1]^2)^3 + a^2*E*sin(2*x[2])*(24*dx[2]^2*d2x[2] - d4x[2])

d5r_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, d4x::AbstractArray, a::Float64, E::Float64, L::Float64, C::Float64) = 12*(-1 + E^2)*dx[1]^3 + 18*(1.0 + 2*(-1 + E^2)*x[1])*dx[1]*d2x[1] + (-C + a^2*(-1 + E^2) - L^2 + 6*x[1]*(1.0 + (-1 + E^2)*x[1]))*d3x[1]

d5θ_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, d4x::AbstractArray, a::Float64, E::Float64, L::Float64, C::Float64) = 4*(a^2*(-1 + E^2)*cos(2*x[2]) - (L^2*(33 + 26*cos(2*x[2]) + cos(4*x[2]))*csc(x[2])^6)/4.)*dx[2]^3 + 6*(L^2*(5 + cos(2*x[2]))*cot(x[2])*csc(x[2])^4 + a^2*(-1 + E^2)*sin(2*x[2]))*dx[2]*d2x[2] + (-(a^2*(-1 + E^2)*cos(2*x[2])) - L^2*(2 + cos(2*x[2]))*csc(x[2])^4)*d3x[2]

d5ϕ_dλ(x::AbstractArray, dx::AbstractArray, d2x::AbstractArray, d3x::AbstractArray, d4x::AbstractArray, a::Float64, E::Float64, L::Float64, C::Float64) = 2*((-12*a*(a^5*L + 4*a^3*(2*a*E - 3*L) + 16*a*(-(a*E) + L) + x[1]*(-10*a^3*(a*E - 2*L) + 40*a*(a*E - L) -
10*a*(a^2*L + 4*(a*E - L))*x[1] + 20*a*(a*E - L)*x[1]^2 + 5*a*L*x[1]^3 - 2*E*x[1]^4))*dx[1]^4)/(a^2 - 2*x[1] +
x[1]^2)^5 + L*(33 + 26*cos(2*x[2]) + cos(4*x[2]))*csc(x[2])^6*dx[2]^4 - (36*a*(a^3*(a*E - 2*L) + 4*a*(-(a*E) + L) + x[1]*(2*a^3*L + 8*a*(a*E - L) + 6*a*(-(a*E) + L)*x[1] - 2*a*L*x[1]^2 + E*x[1]^3))*dx[1]^2*d2x[1])/(a^2 - 2*x[1] +
x[1]^2)^4 - 12*L*(5 + cos(2*x[2]))*cot(x[2])*csc(x[2])^4*dx[2]^2*d2x[2] - 6*L*csc(x[2])^2*d2x[2]^2 + 9*L*csc(x[2])^4*d2x[2]^2 + (4*a*(a^3*L + 4*a*(a*E - L) + 6*a*(-(a*E) + L)*x[1] - 3*a*L*x[1]^2 + 2*E*x[1]^3)*dx[1]*d3x[1])/(a^2 - 2*x[1] + 
x[1]^2)^3 + 4*L*(2 + cos(2*x[2]))*csc(x[2])^4*dx[2]*d3x[2] + (3*a*(a^3*L + 4*a*(a*E - L) + 6*a*(-(a*E) + L)*x[1] - 3*a*L*x[1]^2 + 2*E*x[1]^3)*d2x[1]^2 + a*(a^2 - 2*x[1] + x[1]^2)*(a*(a*E - L) + a*L*x[1] - E*x[1]^2)*d4x[1])/(a^2 - 2*x[1] +
x[1]^2)^3 - L*cot(x[2])*csc(x[2])^2*d4x[2])
end