module MinoDerivs2
d2t_dλ(x::AbstractArray, dx::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (2*(a^3*(a*E - L)*M + x[1]*(a^4*E + x[1]*(a*(-2*a*E + L)*M + E*x[1]*(2*a^2 - 3*M*x[1] + x[1]^2))))*dx[1])/(a^2 - 2*M*x[1] + x[1]^2)^2 - a^2*E*sin(2*x[2])*dx[2]

d2r_dλ(x::AbstractArray, dx::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (-((-2*M + 2*x[1])*(C + (-(a*E) + L)^2 + x[1]^2)) - 2*x[1]*(a^2 - 2*M*x[1] + x[1]^2) + 4*E*x[1]*(-(a*L) + E*(a^2 + x[1]^2)))/2.

d2θ_dλ(x::AbstractArray, dx::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (2*L^2*cot(x[2])*csc(x[2])^2 + 2*a^2*(1 - E^2)*cos(x[2])*sin(x[2]))/2.

d2ϕ_dλ(x::AbstractArray, dx::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (2*a*(a*(a*E - L)*M + a*L*x[1] - E*M*x[1]^2)*dx[1])/(a^2 - 2*M*x[1] + x[1]^2)^2 - 2*L*cot(x[2])*csc(x[2])^2*dx[2]
end