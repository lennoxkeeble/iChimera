module BLDeriv3

d3r_dt(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray, a::Float64) = (-4*x[1]*(a^2*cos(x[2])^2 - x[1]^2)*dx[1]^3)/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + ((a^2*cos(x[2])^2 - x[1]^2)*((-2 + x[1])*dx[1] + x[1]*dx[1]))/(a^2*cos(x[2])^2 + x[1]^2)^3 + 
(2*(a^2*cos(x[2])^2 - x[1]^2)*(a^2 + x[1]^2)*dx[1]^2*((-2 + x[1])*dx[1] + x[1]*dx[1]))/((a^2 + (-2 + x[1])*x[1])^2*(a^2*cos(x[2])^2 + x[1]^2)^2) - (2*a^2*sin(2*x[2])*dx[1]^2*dx[2])/(a^2*cos(x[2])^2 + x[1]^2)^2 - (4*a^2*cos(2*x[2])*x[1]*dx[1]*dx[2]^2)/(a^2*cos(x[2])^2 + x[1]^2)^2 +
((a^2 + (-2 + x[1])*x[1])*dx[1]*dx[2]^2)/(a^2*cos(x[2])^2 + x[1]^2) + (2*a^2*sec(x[2])^2*dx[1]*dx[2]^2)/(a^2 + x[1]^2*sec(x[2])^2) + (x[1]*((-2 + x[1])*dx[1] + x[1]*dx[1])*dx[2]^2)/(a^2*cos(x[2])^2 + x[1]^2) + ((a^2 + (-2 + x[1])*x[1])*(-2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/(a^2*cos(x[2])^2 + x[1]^2)^3 - (2*(a^2 + x[1]^2)*dx[1]^2*(-2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) - (3*(a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 - x[1]^2)*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/(a^2*cos(x[2])^2 + x[1]^2)^4 + (4*(a^2*cos(x[2])^2 - x[1]^2)*(a^2 + x[1]^2)*dx[1]^2*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^3) + (4*a^2*x[1]*sin(2*x[2])*dx[1]*dx[2]*(2*x[1]*dx[1] - 
2*a^2*cos(x[2])*sin(x[2])*dx[2]))/(a^2*cos(x[2])^2 + x[1]^2)^3 - (x[1]*(a^2 + (-2 + x[1])*x[1])*dx[2]^2*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/(a^2*cos(x[2])^2 + x[1]^2)^2 - (2*a^2*tan(x[2])*dx[1]*dx[2]*(2*x[1]*sec(x[2])^2*dx[1] + 2*x[1]^2*sec(x[2])^2*tan(x[2])*dx[2]))/(a^2 + x[1]^2*sec(x[2])^2)^2 - dx[1]^2*(-(dx[1]/(a^2 + (-2 + x[1])*x[1])) + dx[1]/(a^2*cos(x[2])^2 + x[1]^2) - ((1 - x[1])*((-2 + x[1])*dx[1] + x[1]*dx[1]))/(a^2 + (-2 + x[1])*x[1])^2 - (x[1]*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/(a^2*cos(x[2])^2 + x[1]^2)^2) - (2*a*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])^2*((-2 + x[1])*dx[1] + x[1]*dx[1])*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 - ((2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[1]^2*((-2 + x[1])*dx[1] + x[1]*dx[1])*dx[3])/((a^2 + (-2 + x[1])*x[1])^2*(a^2*cos(x[2])^2 + x[1]^2)^2) - (4*a*cos(x[2])*(a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])*dx[2]*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 + (2*cos(x[2])*(2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])*dx[1]^2*dx[2]*dx[3])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (4*a^3*cos(x[2])*sin(x[2])^3*dx[1]^2*dx[2]*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^2 + (12*a^3*cos(x[2])^2*x[1]*sin(x[2])^2*dx[1]*dx[2]^2*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^2 - (4*a^3*x[1]*sin(x[2])^4*dx[1]*dx[2]^2*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^2 - (2*a*(a^2 + (-2 + x[1])*x[1])*sin(x[2])^2*(-2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 + (6*a*(a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])^2*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^4 - (2*(2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[1]^2*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^3) - (8*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[1]*dx[2]*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 + (sin(x[2])^2*dx[1]^2*(-2*a^3*(3 + cos(2*x[2]))*x[1]*dx[1] - 24*a*x[1]^3*dx[1] - 4*a^5*cos(x[2])*sin(x[2])*dx[2] + 2*a^3*x[1]^2*sin(2*x[2])*dx[2])*dx[3])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (sin(x[2])^2*(x[1]*(a^2*cos(x[2])^2 + x[1]^2)^2 + a^2*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])^2)*((-2 + x[1])*dx[1] + x[1]*dx[1])*dx[3]^2)/(a^2*cos(x[2])^2 + x[1]^2)^3 + (2*cos(x[2])*(a^2 + (-2 + x[1])*x[1])*sin(x[2])*(x[1]*(a^2*cos(x[2])^2 + x[1]^2)^2 + a^2*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])^2)*dx[2]*dx[3]^2)/(a^2*cos(x[2])^2 + x[1]^2)^3 - (3*(a^2 + (-2 + x[1])*x[1])*sin(x[2])^2*(x[1]*(a^2*cos(x[2])^2 + x[1]^2)^2 + a^2*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])^2)*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3]^2)/(a^2*cos(x[2])^2 + x[1]^2)^4 + ((a^2 + (-2 + x[1])*x[1])*sin(x[2])^2*((a^2*cos(x[2])^2 + x[1]^2)^2*dx[1] + 2*a^2*cos(x[2])*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])*dx[2] + a^2*sin(x[2])^2*(-2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]) + 2*x[1]*(a^2*cos(x[2])^2 + x[1]^2)*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))*dx[3]^2)/(a^2*cos(x[2])^2 + x[1]^2)^3 - (4*(a^2*cos(x[2])^2 - x[1]^2)*(a^2 + x[1]^2)*dx[1]*d2x[1])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) - 2*((1 - x[1])/(a^2 + (-2 + x[1])*x[1]) + x[1]/(a^2*cos(x[2])^2 + x[1]^2))*dx[1]*d2x[1] - (2*a^2*x[1]*sin(2*x[2])*dx[2]*d2x[1])/(a^2*cos(x[2])^2 + x[1]^2)^2 + (2*a^2*tan(x[2])*dx[2]*d2x[1])/(a^2 + x[1]^2*sec(x[2])^2) + (2*(2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[1]*dx[3]*d2x[1])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (4*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[2]*dx[3]*d2x[1])/(a^2*cos(x[2])^2 + x[1]^2)^2 - (2*a^2*x[1]*sin(2*x[2])*dx[1]*d2x[2])/(a^2*cos(x[2])^2 + x[1]^2)^2 + (2*a^2*tan(x[2])*dx[1]*d2x[2])/(a^2 + x[1]^2*sec(x[2])^2) + (2*x[1]*(a^2 + (-2 + x[1])*x[1])*dx[2]*d2x[2])/(a^2*cos(x[2])^2 + x[1]^2) + (4*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[1]*dx[3]*d2x[2])/(a^2*cos(x[2])^2 + x[1]^2)^2 - (2*a*(a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])^2*d2x[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 + ((2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[1]^2*d2x[3])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (4*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[1]*dx[2]*d2x[3])/(a^2*cos(x[2])^2 + x[1]^2)^2 + (2*(a^2 + (-2 + x[1])*x[1])*sin(x[2])^2*(x[1]*(a^2*cos(x[2])^2 + x[1]^2)^2 + a^2*(a^2*cos(x[2])^2 - x[1]^2)*sin(x[2])^2)*dx[3]*d2x[3])/(a^2*cos(x[2])^2 + x[1]^2)^3


d3θ_dt(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray, a::Float64) = (a^2*sin(2*x[2])*dx[1])/(a^2*cos(x[2])^2 + x[1]^2)^3 + (a^2*cos(x[2])*sin(x[2])*dx[1]^2*((-2 + x[1])*dx[1] + x[1]*dx[1]))/((a^2 + (-2 + x[1])*x[1])^2*(a^2*cos(x[2])^2 + x[1]^2)) + (2*a^2*cos(2*x[2])*x[1]*dx[2])/(a^2*cos(x[2])^2 + x[1]^2)^3 - (4*x[1]*(a^2*cos(x[2])^2 - x[1]^2)*dx[1]^2*dx[2])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) - (2*dx[1]^2*dx[2])/(a^2*cos(x[2])^2 + x[1]^2) - (a^2*cos(x[2])^2*dx[1]^2*dx[2])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)) + (a^2*sin(x[2])^2*dx[1]^2*dx[2])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)) + (2*(a^2*cos(x[2])^2 - x[1]^2)*(a^2 + x[1]^2)*dx[1]*((-2 + x[1])*dx[1] + x[1]*dx[1])*dx[2])/((a^2 + (-2 + x[1])*x[1])^2*(a^2*cos(x[2])^2 + x[1]^2)^2) -
(2*a^2*sin(2*x[2])*dx[1]*dx[2]^2)/(a^2*cos(x[2])^2 + x[1]^2)^2 - (4*a^2*cos(2*x[2])*x[1]*dx[2]^3)/(a^2*cos(x[2])^2 + x[1]^2)^2 + (a^2*sec(x[2])^2*dx[2]^3)/(a^2 + x[1]^2*sec(x[2])^2) - (2*(a^2 + x[1]^2)*dx[1]*dx[2]*(-2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) - (3*a^2*x[1]*sin(2*x[2])*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/(a^2*cos(x[2])^2 + x[1]^2)^4 + (a^2*cos(x[2])*sin(x[2])*dx[1]^2*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (4*(a^2*cos(x[2])^2 - x[1]^2)*(a^2 + x[1]^2)*dx[1]*dx[2]*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^3) + (2*x[1]*dx[1]*dx[2]*(2*x[1]*dx[1] - 
2*a^2*cos(x[2])*sin(x[2])*dx[2]))/(a^2*cos(x[2])^2 + x[1]^2)^2 + (4*a^2*x[1]*sin(2*x[2])*dx[2]^2*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))/(a^2*cos(x[2])^2 + x[1]^2)^3 - (a^2*tan(x[2])*dx[2]^2*(2*x[1]*sec(x[2])^2*dx[1] + 2*x[1]^2*sec(x[2])^2*tan(x[2])*dx[2]))/(a^2 + x[1]^2*sec(x[2])^2)^2 - (4*a*x[1]^2*sin(2*x[2])*dx[1]*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 - (2*a*(a^2 + x[1]^2)*sin(2*x[2])*dx[1]*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 - (4*a*cos(2*x[2])*x[1]*(a^2 + x[1]^2)*dx[2]*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 - ((2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[1]*((-2 + x[1])*dx[1] + x[1]*dx[1])*dx[2]*dx[3])/((a^2 + (-2 + x[1])*x[1])^2*(a^2*cos(x[2])^2 + x[1]^2)^2) + (2*cos(x[2])*(2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])*dx[1]*dx[2]^2*dx[3])/((a^2 + 
(-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (4*a^3*cos(x[2])*sin(x[2])^3*dx[1]*dx[2]^2*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^2 + (12*a^3*cos(x[2])^2*x[1]*sin(x[2])^2*dx[2]^3*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^2 - (4*a^3*x[1]*sin(x[2])^4*dx[2]^3*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^2 + (6*a*x[1]*(a^2 + x[1]^2)*sin(2*x[2])*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^4 - (2*(2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[1]*dx[2]*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^3) - (8*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[2]^2*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 + (sin(x[2])^2*dx[1]*dx[2]*(-2*a^3*(3 + cos(2*x[2]))*x[1]*dx[1] - 24*a*x[1]^3*dx[1] - 4*a^5*cos(x[2])*sin(x[2])*dx[2] + 2*a^3*x[1]^2*sin(2*x[2])*dx[2])*dx[3])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (cos(x[2])^2*((a^2 + x[1]^2)*(a^2*cos(x[2])^2 + x[1]^2)^2 + a^2*x[1]*(a^2*(3 + cos(2*x[2])) + 4*x[1]^2)*sin(x[2])^2)*dx[2]*dx[3]^2)/(a^2*cos(x[2])^2 + x[1]^2)^3 - (sin(x[2])^2*((a^2 + x[1]^2)*(a^2*cos(x[2])^2 + x[1]^2)^2 + a^2*x[1]*(a^2*(3 + cos(2*x[2])) + 4*x[1]^2)*sin(x[2])^2)*dx[2]*dx[3]^2)/(a^2*cos(x[2])^2 + x[1]^2)^3 - (3*cos(x[2])*sin(x[2])*((a^2 + x[1]^2)*(a^2*cos(x[2])^2 + x[1]^2)^2 + a^2*x[1]*(a^2*(3 + cos(2*x[2])) + 4*x[1]^2)*sin(x[2])^2)*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*dx[3]^2)/(a^2*cos(x[2])^2 + x[1]^2)^4 + (cos(x[2])*sin(x[2])*(2*x[1]*(a^2*cos(x[2])^2 + x[1]^2)^2*dx[1] + a^2*(a^2*(3 + cos(2*x[2])) + 4*x[1]^2)*sin(x[2])^2*dx[1] + 2*a^2*cos(x[2])*x[1]*(a^2*(3 + cos(2*x[2])) + 4*x[1]^2)*sin(x[2])*dx[2] + 2*(a^2 + x[1]^2)*(a^2*cos(x[2])^2 + x[1]^2)*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]) + a^2*x[1]*sin(x[2])^2*(8*x[1]*dx[1] - 2*a^2*sin(2*x[2])*dx[2]))*dx[3]^2)/(a^2*cos(x[2])^2 + x[1]^2)^3 - (2*a^2*cos(x[2])*sin(x[2])*dx[1]*d2x[1])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)) - (2*(a^2*cos(x[2])^2 - x[1]^2)*(a^2 + x[1]^2)*dx[2]*d2x[1])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) - (2*x[1]*dx[2]*d2x[1])/(a^2*cos(x[2])^2 + x[1]^2) + ((2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[2]*dx[3]*d2x[1])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) - (2*(a^2*cos(x[2])^2 - x[1]^2)*(a^2 + x[1]^2)*dx[1]*d2x[2])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) - (2*x[1]*dx[1]*d2x[2])/(a^2*cos(x[2])^2 + x[1]^2) - (4*a^2*x[1]*sin(2*x[2])*dx[2]*d2x[2])/(a^2*cos(x[2])^2 + x[1]^2)^2 + (2*a^2*tan(x[2])*dx[2]*d2x[2])/(a^2 + x[1]^2*sec(x[2])^2) + ((2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[1]*dx[3]*d2x[2])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (8*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[2]*dx[3]*d2x[2])/(a^2*cos(x[2])^2 + x[1]^2)^2 - (2*a*x[1]*(a^2 + x[1]^2)*sin(2*x[2])*d2x[3])/(a^2*cos(x[2])^2 + x[1]^2)^3 + ((2*a^5*cos(x[2])^2 - a^3*(3 + cos(2*x[2]))*x[1]^2 - 6*a*x[1]^4)*sin(x[2])^2*dx[1]*dx[2]*d2x[3])/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2) + (4*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[2]^2*d2x[3])/(a^2*cos(x[2])^2 + x[1]^2)^2 + (2*cos(x[2])*sin(x[2])*((a^2 + x[1]^2)*(a^2*cos(x[2])^2 + x[1]^2)^2 + a^2*x[1]*(a^2*(3 + cos(2*x[2])) + 4*x[1]^2)*sin(x[2])^2)*dx[3]*d2x[3])/(a^2*cos(x[2])^2 + x[1]^2)^3

d3ϕ_dt(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray, a::Float64) = -((((-2 + x[1])*dx[1] + x[1]*dx[1])*(dx[1]*(2*a^3*cos(x[2])^2 - 2*a*x[1]^2 - 2*(a^4*cos(x[2])^4*x[1] - 2*a^2*x[1]^2 + 2*a^2*cos(x[2])^2*x[1]^3 - 3*x[1]^4 + x[1]^5 + a^4*cos(x[2])^2*(1 + sin(x[2])^2))*dx[3] + a*(2*a^4*cos(x[2])^2 - a^2*(3 + cos(2*x[2]))*x[1]^2 - 6*x[1]^4)*sin(x[2])^2*dx[3]^2) + 2*(a^2 + (-2 + x[1])*x[1])*dx[2]*(2*a*cot(x[2])*x[1] - (cot(x[2])*(a^2*cos(x[2])^2 + x[1]^2)^2 + 2*a^2*x[1]*sin(2*x[2]))*dx[3] + 2*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[3]^2)))/((a^2 + (-2 + x[1])*x[1])^2*(a^2*cos(x[2])^2 + x[1]^2)^2)) - (2*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2])*(dx[1]*(2*a^3*cos(x[2])^2 - 2*a*x[1]^2 - 2*(a^4*cos(x[2])^4*x[1] - 2*a^2*x[1]^2 + 2*a^2*cos(x[2])^2*x[1]^3 - 3*x[1]^4 + x[1]^5 + a^4*cos(x[2])^2*(1 + sin(x[2])^2))*dx[3] + a*(2*a^4*cos(x[2])^2 - a^2*(3 + cos(2*x[2]))*x[1]^2 -
6*x[1]^4)*sin(x[2])^2*dx[3]^2) + 2*(a^2 + (-2 + x[1])*x[1])*dx[2]*(2*a*cot(x[2])*x[1] - (cot(x[2])*(a^2*cos(x[2])^2 + x[1]^2)^2 + 2*a^2*x[1]*sin(2*x[2]))*dx[3] + 2*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[3]^2)))/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^3) + (2*((-2 + x[1])*dx[1] + x[1]*dx[1])*dx[2]*(2*a*cot(x[2])*x[1] - (cot(x[2])*(a^2*cos(x[2])^2 + x[1]^2)^2 + 2*a^2*x[1]*sin(2*x[2]))*dx[3] + 2*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[3]^2) + (2*a^3*cos(x[2])^2 - 2*a*x[1]^2 - 2*(a^4*cos(x[2])^4*x[1] - 2*a^2*x[1]^2 + 2*a^2*cos(x[2])^2*x[1]^3 - 3*x[1]^4 + x[1]^5 + a^4*cos(x[2])^2*(1 + sin(x[2])^2))*dx[3] + a*(2*a^4*cos(x[2])^2 - a^2*(3 + cos(2*x[2]))*x[1]^2 - 6*x[1]^4)*sin(x[2])^2*dx[3]^2)*d2x[1] + 2*(a^2 + (-2 + x[1])*x[1])*(2*a*cot(x[2])*x[1] - (cot(x[2])*(a^2*cos(x[2])^2 + x[1]^2)^2 + 2*a^2*x[1]*sin(2*x[2]))*dx[3] + 2*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[3]^2)*d2x[2] + dx[1]*(-4*a*x[1]*dx[1] -
4*a^3*cos(x[2])*sin(x[2])*dx[2] - 2*(a^4*cos(x[2])^4*dx[1] - 4*a^2*x[1]*dx[1] + 6*a^2*cos(x[2])^2*x[1]^2*dx[1] - 12*x[1]^3*dx[1] + 5*x[1]^4*dx[1] + 2*a^4*cos(x[2])^3*sin(x[2])*dx[2] - 4*a^4*cos(x[2])^3*x[1]*sin(x[2])*dx[2] - 4*a^2*cos(x[2])*x[1]^3*sin(x[2])*dx[2] - 2*a^4*cos(x[2])*sin(x[2])*(1 + sin(x[2])^2)*dx[2])*dx[3] + 2*a*cos(x[2])*(2*a^4*cos(x[2])^2 - a^2*(3 + cos(2*x[2]))*x[1]^2 - 6*x[1]^4)*sin(x[2])*dx[2]*dx[3]^2 + a*sin(x[2])^2*(-2*a^2*(3 + cos(2*x[2]))*x[1]*dx[1] - 24*x[1]^3*dx[1] - 4*a^4*cos(x[2])*sin(x[2])*dx[2] + 2*a^2*x[1]^2*sin(2*x[2])*dx[2])*dx[3]^2 - 2*(a^4*cos(x[2])^4*x[1] - 2*a^2*x[1]^2 + 2*a^2*cos(x[2])^2*x[1]^3 - 3*x[1]^4 + x[1]^5 + a^4*cos(x[2])^2*(1 + sin(x[2])^2))*d2x[3] + 2*a*(2*a^4*cos(x[2])^2 - a^2*(3 + cos(2*x[2]))*x[1]^2 - 6*x[1]^4)*sin(x[2])^2*dx[3]*d2x[3]) + 2*(a^2 + (-2 + x[1])*x[1])*dx[2]*(2*a*cot(x[2])*dx[1] - 
2*a*csc(x[2])^2*x[1]*dx[2] - (2*a^2*sin(2*x[2])*dx[1] + 4*a^2*cos(2*x[2])*x[1]*dx[2] - csc(x[2])^2*(a^2*cos(x[2])^2 + x[1]^2)^2*dx[2] + 2*cot(x[2])*(a^2*cos(x[2])^2 + x[1]^2)*(2*x[1]*dx[1] - 2*a^2*cos(x[2])*sin(x[2])*dx[2]))*dx[3] + 2*a^3*cos(x[2])*sin(x[2])^3*dx[1]*dx[3]^2 + 6*a^3*cos(x[2])^2*x[1]*sin(x[2])^2*dx[2]*dx[3]^2 - 2*a^3*x[1]*sin(x[2])^4*dx[2]*dx[3]^2 - (cot(x[2])*(a^2*cos(x[2])^2 + x[1]^2)^2 + 2*a^2*x[1]*sin(2*x[2]))*d2x[3] + 4*a^3*cos(x[2])*x[1]*sin(x[2])^3*dx[3]*d2x[3]))/((a^2 + (-2 + x[1])*x[1])*(a^2*cos(x[2])^2 + x[1]^2)^2)

end