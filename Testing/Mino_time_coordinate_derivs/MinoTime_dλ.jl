module MinoDeriv1
dt_dλ(x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = (a^4*E*cos(x[2])^2 + (a^2*E*(3 + cos(2*x[2]))*x[1]^2)/2. + 
E*x[1]^4 + 2*a*M*x[1]*(-L + a*E*sin(x[2])^2))/(a^2 - 2*M*x[1] + x[1]^2)

dr_dλ(x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = sqrt(-((C + (-(a*E) + L)^2 + x[1]^2)*(a^2 - 2*M*x[1] + x[1]^2)) + (a*L - E*(a^2 + x[1]^2))^2)

dθ_dλ(x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = sqrt(C + a^2*(-1 + E^2)*cos(x[2])^2 - L^2*cot(x[2])^2)

dϕ_dλ(x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = L*csc(x[2])^2 + (a*(-(a*L) + 2*E*M*x[1]))/(a^2 - 2*M*x[1] + x[1]^2)

end