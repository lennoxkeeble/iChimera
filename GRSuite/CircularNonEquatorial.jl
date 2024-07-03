#=

    In this module we implement the special case of circular non-equatorial orbits for evolving the constants of motion — this special case ensures the preservation of circularity of the orbit. See
    arXiv:1109.0572v2 for more details. Note that we have corrected a type in Eq. G5, where there was a missing factor of 2 in the coefficient for r0^3

=#

module CircularNonEquatorial

c11(r0::Float64, a::Float64, E::Float64, L::Float64, C::Float64, M::Float64) = 4*a*(C + (a*E - L)^2)*(-(a*E) + L)*M^2 - 4*a^2*E*(C + (a*E - L)^2)*M*r0 + 
2*a*(a^3*E*(1 - E^2) + a*E*(C + L^2) + 6*(a*E - L)*M^2)*r0^2 + 8*(a^2*E^3 + 2*a*(-(a*E) + L) - E*(C + L^2))*M*r0^3 + 
2*E*(a^2*(1 - E^2) + 3*(C + L^2))*r0^4 - 12*E*M*r0^5 + 4*E*(1 - E^2)*r0^6

c12(r0::Float64, a::Float64, E::Float64, L::Float64, C::Float64, M::Float64) = -4*(C + (a*E - L)^2)*(-(a*E) + L)*M^2 + 4*(C + (a*E - L)^2)*L*M*r0 + 2*(-(L*(C + a^2*(1 - E^2) + L^2)) + 
6*(-(a*E) + L)*M^2)*r0^2 - 16*(1 - E^2)*(-(a*E) + L)*M*r0^3 + 4*(1 - E^2)*L*r0^4

c21(r0::Float64, a::Float64, E::Float64, L::Float64, C::Float64, M::Float64) = -2*(-(a^3*(-(a*E) + L)*M) + a^4*E*r0 + a*(-2*a*E + L)*M*r0^2 + 2*a^2*E*r0^3 - 3*E*M*r0^4 + E*r0^5)

c22(r0::Float64, a::Float64, E::Float64, L::Float64, C::Float64, M::Float64) = -2*a*(-(a*(a*E - L)*M) - a*L*r0 + E*M*r0^2)

d(r0::Float64, a::Float64, E::Float64, L::Float64, C::Float64, M::Float64) = a^4*(1 - E^2) + a^2*(C + L^2) - 2*(C + (a*E - L)^2)*M^2 + 2*(C + a^2*(-3 + E^2) - 2*a*E*L + L^2)*M*r0 + 
(-C + 5*a^2*(1 - E^2) - L^2 + 6*M^2)*r0^2 - 8*(1 - E^2)*M*r0^3 + 2*(1 - E^2)*r0^4

Cdot(r0::Float64, Edot::Float64, Ldot::Float64, a::Float64, E::Float64, L::Float64, C::Float64, M::Float64) = -(c11(r0, a, E, L, C, M) * Edot + c12(r0, a, E, L, C, M) * Ldot) / d(r0, a, E, L, C, M)

r0dot(r0::Float64, Edot::Float64, Ldot::Float64, a::Float64, E::Float64, L::Float64, C::Float64, M::Float64) = -(c21(r0, a, E, L, C, M) * Edot + c22(r0, a, E, L, C, M) * Ldot) / d(r0, a, E, L, C, M)

end