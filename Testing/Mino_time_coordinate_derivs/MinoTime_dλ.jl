module MinoDerivs1
dt_dλ(x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = a*L*(1 - (a^2 + x[1]^2)/(a^2 - 2*M*x[1] + x[1]^2)) + E*((a^2 + x[1]^2)^2/(a^2 - 2*M*x[1] + x[1]^2) - a^2*sin(x[2])^2)

function dr_dλ(x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64)
    dr_dλ_squared = -((C + (-(a*E) + L)^2 + x[1]^2)*(a^2 - 2*M*x[1] + x[1]^2)) + (-(a*L) + E*(a^2 + x[1]^2))^2

    if abs(dr_dλ_squared) < 1e-6
        return 0.0
    else
        return sqrt(dr_dλ_squared)
    end
end

function dθ_dλ(x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64)
    dθ_dλ_squared = C - a^2*(1 - E^2)*cos(x[2])^2 - L^2*cot(x[2])^2

    if abs(dθ_dλ_squared) < 1e-6
        return 0.0
    else
        return sqrt(dθ_dλ_squared)
    end
end

dϕ_dλ(x::AbstractArray, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64) = L*csc(x[2])^2 + (a*(-(a*L) + 2*E*M*x[1]))/(a^2 - 2*M*x[1] + x[1]^2)
end