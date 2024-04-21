include("d2x.jl")
include("d3x.jl")
include("d4x.jl")
include("d5x.jl")
include("d6x.jl")

module Function_1

f(x::AbstractArray) = cos(x[2])*x[1]*sin(x[3])

df_dt(dx::AbstractArray, x::AbstractArray) = cos(x[2])*sin(x[3])*dx[1] - x[1]*sin(x[2])*sin(x[3])*dx[2] + 
cos(x[2])*cos(x[3])*x[1]*dx[3]

d2f_dt(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) = 2*cos(x[3])*(cos(x[2])*dx[1] - x[1]*sin(x[2])*dx[2])*dx[3] + 
sin(x[3])*(-2*sin(x[2])*dx[1]*dx[2] + cos(x[2])*d2x[1] + 
x[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2])) + 
cos(x[2])*x[1]*(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3])

d3f_dt(d3x::AbstractArray, d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) = 3*cos(x[3])*dx[3]*(-2*sin(x[2])*dx[1]*dx[2] + 
cos(x[2])*d2x[1] + x[1]*
(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2])) + 
3*(cos(x[2])*dx[1] - x[1]*sin(x[2])*dx[2])*
(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3]) + 
sin(x[3])*(-3*sin(x[2])*dx[2]*d2x[1] + 
3*dx[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]) + 
cos(x[2])*d3x[1] + x[1]*
(sin(x[2])*dx[2]^3 - 3*cos(x[2])*dx[2]*d2x[2] - 
sin(x[2])*d3x[2])) + 
cos(x[2])*x[1]*(-(cos(x[3])*dx[3]^3) - 
3*sin(x[3])*dx[3]*d2x[3] + cos(x[3])*d3x[3])

d4f_dt(d4x::AbstractArray, d3x::AbstractArray, d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) = 6*(-2*sin(x[2])*dx[1]*dx[2] + cos(x[2])*d2x[1] + 
x[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]))*
(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3]) + 
4*cos(x[3])*dx[3]*(-3*sin(x[2])*dx[2]*d2x[1] + 
3*dx[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]) + 
cos(x[2])*d3x[1] + x[1]*
(sin(x[2])*dx[2]^3 - 3*cos(x[2])*dx[2]*d2x[2] - 
sin(x[2])*d3x[2])) + 
4*(cos(x[2])*dx[1] - x[1]*sin(x[2])*dx[2])*
(-(cos(x[3])*dx[3]^3) - 3*sin(x[3])*dx[3]*d2x[3] + 
cos(x[3])*d3x[3]) + sin(x[3])*
(6*d2x[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]) - 
4*sin(x[2])*dx[2]*d3x[1] + 
4*dx[1]*(sin(x[2])*dx[2]^3 - 
3*cos(x[2])*dx[2]*d2x[2] - sin(x[2])*d3x[2]) + 
cos(x[2])*d4x[1] + x[1]*
(cos(x[2])*dx[2]^4 + 6*sin(x[2])*dx[2]^2*d2x[2] - 
3*cos(x[2])*d2x[2]^2 - 4*cos(x[2])*dx[2]*d3x[2] - 
sin(x[2])*d4x[2])) + 
cos(x[2])*x[1]*(sin(x[3])*dx[3]^4 - 
6*cos(x[3])*dx[3]^2*d2x[3] - 3*sin(x[3])*d2x[3]^2 - 
4*sin(x[3])*dx[3]*d3x[3] + cos(x[3])*d4x[3])

d5f_dt(d5x::AbstractArray, d4x::AbstractArray, d3x::AbstractArray, d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) =10*(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3])*
(-3*sin(x[2])*dx[2]*d2x[1] + 
3*dx[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]) + 
cos(x[2])*d3x[1] + x[1]*
(sin(x[2])*dx[2]^3 - 3*cos(x[2])*dx[2]*d2x[2] - 
sin(x[2])*d3x[2])) + 
10*(-2*sin(x[2])*dx[1]*dx[2] + cos(x[2])*d2x[1] + 
x[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]))*
(-(cos(x[3])*dx[3]^3) - 3*sin(x[3])*dx[3]*d2x[3] + 
cos(x[3])*d3x[3]) + 5*cos(x[3])*dx[3]*
(6*d2x[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]) - 
4*sin(x[2])*dx[2]*d3x[1] + 
4*dx[1]*(sin(x[2])*dx[2]^3 - 
3*cos(x[2])*dx[2]*d2x[2] - sin(x[2])*d3x[2]) + 
cos(x[2])*d4x[1] + x[1]*
(cos(x[2])*dx[2]^4 + 6*sin(x[2])*dx[2]^2*d2x[2] - 
3*cos(x[2])*d2x[2]^2 - 4*cos(x[2])*dx[2]*d3x[2] - 
sin(x[2])*d4x[2])) + 
5*(cos(x[2])*dx[1] - x[1]*sin(x[2])*dx[2])*
(sin(x[3])*dx[3]^4 - 6*cos(x[3])*dx[3]^2*d2x[3] - 
3*sin(x[3])*d2x[3]^2 - 4*sin(x[3])*dx[3]*d3x[3] + 
cos(x[3])*d4x[3]) + sin(x[3])*
(10*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2])*d3x[1] + 
10*d2x[1]*(sin(x[2])*dx[2]^3 - 
3*cos(x[2])*dx[2]*d2x[2] - sin(x[2])*d3x[2]) - 
5*sin(x[2])*dx[2]*d4x[1] + 
5*dx[1]*(cos(x[2])*dx[2]^4 + 
6*sin(x[2])*dx[2]^2*d2x[2] - 3*cos(x[2])*d2x[2]^2 - 
4*cos(x[2])*dx[2]*d3x[2] - sin(x[2])*d4x[2]) + 
cos(x[2])*d5x[1] + x[1]*
(-(sin(x[2])*dx[2]^5) + 10*cos(x[2])*dx[2]^3*d2x[2] + 
15*sin(x[2])*dx[2]*d2x[2]^2 + 
10*sin(x[2])*dx[2]^2*d3x[2] - 
10*cos(x[2])*d2x[2]*d3x[2] - 
5*cos(x[2])*dx[2]*d4x[2] - sin(x[2])*d5x[2])) + 
cos(x[2])*x[1]*(cos(x[3])*dx[3]^5 + 
10*sin(x[3])*dx[3]^3*d2x[3] - 
15*cos(x[3])*dx[3]*d2x[3]^2 - 
10*cos(x[3])*dx[3]^2*d3x[3] - 
10*sin(x[3])*d2x[3]*d3x[3] - 
5*sin(x[3])*dx[3]*d4x[3] + cos(x[3])*d5x[3])

d6f_dt(d6x::AbstractArray, d5x::AbstractArray, d4x::AbstractArray, d3x::AbstractArray, d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) = 20*(-3*sin(x[2])*dx[2]*d2x[1] + 
3*dx[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]) + 
cos(x[2])*d3x[1] + x[1]*
(sin(x[2])*dx[2]^3 - 3*cos(x[2])*dx[2]*d2x[2] - 
sin(x[2])*d3x[2]))*
(-(cos(x[3])*dx[3]^3) - 3*sin(x[3])*dx[3]*d2x[3] + 
cos(x[3])*d3x[3]) + 15*
(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3])*
(6*d2x[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]) - 
4*sin(x[2])*dx[2]*d3x[1] + 
4*dx[1]*(sin(x[2])*dx[2]^3 - 
3*cos(x[2])*dx[2]*d2x[2] - sin(x[2])*d3x[2]) + 
cos(x[2])*d4x[1] + x[1]*
(cos(x[2])*dx[2]^4 + 6*sin(x[2])*dx[2]^2*d2x[2] - 
3*cos(x[2])*d2x[2]^2 - 4*cos(x[2])*dx[2]*d3x[2] - 
sin(x[2])*d4x[2])) + 
15*(-2*sin(x[2])*dx[1]*dx[2] + cos(x[2])*d2x[1] + 
x[1]*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2]))*
(sin(x[3])*dx[3]^4 - 6*cos(x[3])*dx[3]^2*d2x[3] - 
3*sin(x[3])*d2x[3]^2 - 4*sin(x[3])*dx[3]*d3x[3] + 
cos(x[3])*d4x[3]) + 6*cos(x[3])*dx[3]*
(10*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2])*d3x[1] + 
10*d2x[1]*(sin(x[2])*dx[2]^3 - 
3*cos(x[2])*dx[2]*d2x[2] - sin(x[2])*d3x[2]) - 
5*sin(x[2])*dx[2]*d4x[1] + 
5*dx[1]*(cos(x[2])*dx[2]^4 + 
6*sin(x[2])*dx[2]^2*d2x[2] - 3*cos(x[2])*d2x[2]^2 - 
4*cos(x[2])*dx[2]*d3x[2] - sin(x[2])*d4x[2]) + 
cos(x[2])*d5x[1] + x[1]*
(-(sin(x[2])*dx[2]^5) + 10*cos(x[2])*dx[2]^3*d2x[2] + 
15*sin(x[2])*dx[2]*d2x[2]^2 + 
10*sin(x[2])*dx[2]^2*d3x[2] - 
10*cos(x[2])*d2x[2]*d3x[2] - 
5*cos(x[2])*dx[2]*d4x[2] - sin(x[2])*d5x[2])) + 
6*(cos(x[2])*dx[1] - x[1]*sin(x[2])*dx[2])*
(cos(x[3])*dx[3]^5 + 10*sin(x[3])*dx[3]^3*d2x[3] - 
15*cos(x[3])*dx[3]*d2x[3]^2 - 
10*cos(x[3])*dx[3]^2*d3x[3] - 
10*sin(x[3])*d2x[3]*d3x[3] - 
5*sin(x[3])*dx[3]*d4x[3] + cos(x[3])*d5x[3]) + 
sin(x[3])*(20*d3x[1]*(sin(x[2])*dx[2]^3 - 
3*cos(x[2])*dx[2]*d2x[2] - sin(x[2])*d3x[2]) + 
15*(-(cos(x[2])*dx[2]^2) - sin(x[2])*d2x[2])*d4x[1] + 
15*d2x[1]*(cos(x[2])*dx[2]^4 + 
6*sin(x[2])*dx[2]^2*d2x[2] - 3*cos(x[2])*d2x[2]^2 - 
4*cos(x[2])*dx[2]*d3x[2] - sin(x[2])*d4x[2]) - 
6*sin(x[2])*dx[2]*d5x[1] + 
6*dx[1]*(-(sin(x[2])*dx[2]^5) + 
10*cos(x[2])*dx[2]^3*d2x[2] + 
15*sin(x[2])*dx[2]*d2x[2]^2 + 
10*sin(x[2])*dx[2]^2*d3x[2] - 
10*cos(x[2])*d2x[2]*d3x[2] - 
5*cos(x[2])*dx[2]*d4x[2] - sin(x[2])*d5x[2]) + 
cos(x[2])*d6x[1] + x[1]*
(-(cos(x[2])*dx[2]^6) - 15*sin(x[2])*dx[2]^4*d2x[2] + 
45*cos(x[2])*dx[2]^2*d2x[2]^2 + 15*sin(x[2])*d2x[2]^3 + 
20*cos(x[2])*dx[2]^3*d3x[2] + 
60*sin(x[2])*dx[2]*d2x[2]*d3x[2] - 
10*cos(x[2])*d3x[2]^2 + 15*sin(x[2])*dx[2]^2*d4x[2] - 
15*cos(x[2])*d2x[2]*d4x[2] - 
6*cos(x[2])*dx[2]*d5x[2] - sin(x[2])*d6x[2])) + 
cos(x[2])*x[1]*(-(sin(x[3])*dx[3]^6) + 
15*cos(x[3])*dx[3]^4*d2x[3] + 
45*sin(x[3])*dx[3]^2*d2x[3]^2 - 15*cos(x[3])*d2x[3]^3 + 
20*sin(x[3])*dx[3]^3*d3x[3] - 
60*cos(x[3])*dx[3]*d2x[3]*d3x[3] - 
10*sin(x[3])*d3x[3]^2 - 15*cos(x[3])*dx[3]^2*d4x[3] - 
15*sin(x[3])*d2x[3]*d4x[3] - 
6*sin(x[3])*dx[3]*d5x[3] + cos(x[3])*d6x[3])

end

module Function_2

f(x::AbstractArray) = (sin(x[2])*sin(x[3]))/x[1]

df_dt(dx::AbstractArray, x::AbstractArray) = -((sin(x[2])*sin(x[3])*dx[1])/x[1]^2) + (cos(x[2])*sin(x[3])*dx[2])/x[1] + 
(cos(x[3])*sin(x[2])*dx[3])/x[1]

d2f_dt(d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) = 2*cos(x[3])*(-((sin(x[2])*dx[1])/x[1]^2) + (cos(x[2])*dx[2])/x[1])*
dx[3] + sin(x[3])*((-2*cos(x[2])*dx[1]*dx[2])/x[1]^2 + 
sin(x[2])*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) + 
(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2])/x[1]) + 
(sin(x[2])*(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3]))/x[1]

d3f_dt(d3x::AbstractArray, d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) = 3*cos(x[3])*dx[3]*((-2*cos(x[2])*dx[1]*dx[2])/x[1]^2 + 
sin(x[2])*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) + 
(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2])/x[1]) + 
3*(-((sin(x[2])*dx[1])/x[1]^2) + (cos(x[2])*dx[2])/x[1])*
(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3]) + 
sin(x[3])*(3*cos(x[2])*dx[2]*
((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) - 
(3*dx[1]*(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2]))/
x[1]^2 + sin(x[2])*((-6*dx[1]^3)/x[1]^4 + 
(6*dx[1]*d2x[1])/x[1]^3 - d3x[1]/x[1]^2) + 
(-(cos(x[2])*dx[2]^3) - 3*sin(x[2])*dx[2]*d2x[2] + 
cos(x[2])*d3x[2])/x[1]) + 
(sin(x[2])*(-(cos(x[3])*dx[3]^3) - 3*sin(x[3])*dx[3]*d2x[3] + 
cos(x[3])*d3x[3]))/x[1]

d4f_dt(d4x::AbstractArray, d3x::AbstractArray, d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) =6*((-2*cos(x[2])*dx[1]*dx[2])/x[1]^2 + 
sin(x[2])*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) + 
(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2])/x[1])*
(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3]) + 
4*cos(x[3])*dx[3]*(3*cos(x[2])*dx[2]*
((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) - 
(3*dx[1]*(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2]))/
x[1]^2 + sin(x[2])*((-6*dx[1]^3)/x[1]^4 + 
(6*dx[1]*d2x[1])/x[1]^3 - d3x[1]/x[1]^2) + 
(-(cos(x[2])*dx[2]^3) - 3*sin(x[2])*dx[2]*d2x[2] + 
cos(x[2])*d3x[2])/x[1]) + 
4*(-((sin(x[2])*dx[1])/x[1]^2) + (cos(x[2])*dx[2])/x[1])*
(-(cos(x[3])*dx[3]^3) - 3*sin(x[3])*dx[3]*d2x[3] + 
cos(x[3])*d3x[3]) + sin(x[3])*
(6*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2)*
(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2]) + 
4*cos(x[2])*dx[2]*((-6*dx[1]^3)/x[1]^4 + 
(6*dx[1]*d2x[1])/x[1]^3 - d3x[1]/x[1]^2) - 
(4*dx[1]*(-(cos(x[2])*dx[2]^3) - 
3*sin(x[2])*dx[2]*d2x[2] + cos(x[2])*d3x[2]))/x[1]^2 + 
sin(x[2])*((24*dx[1]^4)/x[1]^5 - 
(36*dx[1]^2*d2x[1])/x[1]^4 + (6*d2x[1]^2)/x[1]^3 + 
(8*dx[1]*d3x[1])/x[1]^3 - d4x[1]/x[1]^2) + 
(sin(x[2])*dx[2]^4 - 6*cos(x[2])*dx[2]^2*d2x[2] - 
3*sin(x[2])*d2x[2]^2 - 4*sin(x[2])*dx[2]*d3x[2] + 
cos(x[2])*d4x[2])/x[1]) + 
(sin(x[2])*(sin(x[3])*dx[3]^4 - 6*cos(x[3])*dx[3]^2*d2x[3] - 
3*sin(x[3])*d2x[3]^2 - 4*sin(x[3])*dx[3]*d3x[3] + 
cos(x[3])*d4x[3]))/x[1]

d5f_dt(d5x::AbstractArray, d4x::AbstractArray, d3x::AbstractArray, d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) =  10*(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3])*
(3*cos(x[2])*dx[2]*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) - 
(3*dx[1]*(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2]))/
x[1]^2 + sin(x[2])*((-6*dx[1]^3)/x[1]^4 + 
(6*dx[1]*d2x[1])/x[1]^3 - d3x[1]/x[1]^2) + 
(-(cos(x[2])*dx[2]^3) - 3*sin(x[2])*dx[2]*d2x[2] + 
cos(x[2])*d3x[2])/x[1]) + 
10*((-2*cos(x[2])*dx[1]*dx[2])/x[1]^2 + 
sin(x[2])*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) + 
(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2])/x[1])*
(-(cos(x[3])*dx[3]^3) - 3*sin(x[3])*dx[3]*d2x[3] + 
cos(x[3])*d3x[3]) + 5*cos(x[3])*dx[3]*
(6*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2)*
(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2]) + 
4*cos(x[2])*dx[2]*((-6*dx[1]^3)/x[1]^4 + 
(6*dx[1]*d2x[1])/x[1]^3 - d3x[1]/x[1]^2) - 
(4*dx[1]*(-(cos(x[2])*dx[2]^3) - 
3*sin(x[2])*dx[2]*d2x[2] + cos(x[2])*d3x[2]))/x[1]^2 + 
sin(x[2])*((24*dx[1]^4)/x[1]^5 - 
(36*dx[1]^2*d2x[1])/x[1]^4 + (6*d2x[1]^2)/x[1]^3 + 
(8*dx[1]*d3x[1])/x[1]^3 - d4x[1]/x[1]^2) + 
(sin(x[2])*dx[2]^4 - 6*cos(x[2])*dx[2]^2*d2x[2] - 
3*sin(x[2])*d2x[2]^2 - 4*sin(x[2])*dx[2]*d3x[2] + 
cos(x[2])*d4x[2])/x[1]) + 
5*(-((sin(x[2])*dx[1])/x[1]^2) + (cos(x[2])*dx[2])/x[1])*
(sin(x[3])*dx[3]^4 - 6*cos(x[3])*dx[3]^2*d2x[3] - 
3*sin(x[3])*d2x[3]^2 - 4*sin(x[3])*dx[3]*d3x[3] + 
cos(x[3])*d4x[3]) + sin(x[3])*
(10*(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2])*
((-6*dx[1]^3)/x[1]^4 + (6*dx[1]*d2x[1])/x[1]^3 - 
d3x[1]/x[1]^2) + 10*
((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2)*
(-(cos(x[2])*dx[2]^3) - 3*sin(x[2])*dx[2]*d2x[2] + 
cos(x[2])*d3x[2]) + 
5*cos(x[2])*dx[2]*((24*dx[1]^4)/x[1]^5 - 
(36*dx[1]^2*d2x[1])/x[1]^4 + (6*d2x[1]^2)/x[1]^3 + 
(8*dx[1]*d3x[1])/x[1]^3 - d4x[1]/x[1]^2) - 
(5*dx[1]*(sin(x[2])*dx[2]^4 - 
6*cos(x[2])*dx[2]^2*d2x[2] - 3*sin(x[2])*d2x[2]^2 - 
4*sin(x[2])*dx[2]*d3x[2] + cos(x[2])*d4x[2]))/x[1]^2 + 
sin(x[2])*((-120*dx[1]^5)/x[1]^6 + 
(240*dx[1]^3*d2x[1])/x[1]^5 - 
(90*dx[1]*d2x[1]^2)/x[1]^4 - 
(60*dx[1]^2*d3x[1])/x[1]^4 + 
(20*d2x[1]*d3x[1])/x[1]^3 + 
(10*dx[1]*d4x[1])/x[1]^3 - d5x[1]/x[1]^2) + 
(cos(x[2])*dx[2]^5 + 10*sin(x[2])*dx[2]^3*d2x[2] - 
15*cos(x[2])*dx[2]*d2x[2]^2 - 
10*cos(x[2])*dx[2]^2*d3x[2] - 
10*sin(x[2])*d2x[2]*d3x[2] - 
5*sin(x[2])*dx[2]*d4x[2] + cos(x[2])*d5x[2])/x[1]) + 
(sin(x[2])*(cos(x[3])*dx[3]^5 + 10*sin(x[3])*dx[3]^3*d2x[3] - 
15*cos(x[3])*dx[3]*d2x[3]^2 - 
10*cos(x[3])*dx[3]^2*d3x[3] - 
10*sin(x[3])*d2x[3]*d3x[3] - 
5*sin(x[3])*dx[3]*d4x[3] + cos(x[3])*d5x[3]))/x[1]

d6f_dt(d6x::AbstractArray, d5x::AbstractArray, d4x::AbstractArray, d3x::AbstractArray, d2x::AbstractArray, dx::AbstractArray, x::AbstractArray) =20*(3*cos(x[2])*dx[2]*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) - 
(3*dx[1]*(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2]))/
x[1]^2 + sin(x[2])*((-6*dx[1]^3)/x[1]^4 + 
(6*dx[1]*d2x[1])/x[1]^3 - d3x[1]/x[1]^2) + 
(-(cos(x[2])*dx[2]^3) - 3*sin(x[2])*dx[2]*d2x[2] + 
cos(x[2])*d3x[2])/x[1])*
(-(cos(x[3])*dx[3]^3) - 3*sin(x[3])*dx[3]*d2x[3] + 
cos(x[3])*d3x[3]) + 15*
(-(sin(x[3])*dx[3]^2) + cos(x[3])*d2x[3])*
(6*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2)*
(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2]) + 
4*cos(x[2])*dx[2]*((-6*dx[1]^3)/x[1]^4 + 
(6*dx[1]*d2x[1])/x[1]^3 - d3x[1]/x[1]^2) - 
(4*dx[1]*(-(cos(x[2])*dx[2]^3) - 
3*sin(x[2])*dx[2]*d2x[2] + cos(x[2])*d3x[2]))/x[1]^2 + 
sin(x[2])*((24*dx[1]^4)/x[1]^5 - 
(36*dx[1]^2*d2x[1])/x[1]^4 + (6*d2x[1]^2)/x[1]^3 + 
(8*dx[1]*d3x[1])/x[1]^3 - d4x[1]/x[1]^2) + 
(sin(x[2])*dx[2]^4 - 6*cos(x[2])*dx[2]^2*d2x[2] - 
3*sin(x[2])*d2x[2]^2 - 4*sin(x[2])*dx[2]*d3x[2] + 
cos(x[2])*d4x[2])/x[1]) + 
15*((-2*cos(x[2])*dx[1]*dx[2])/x[1]^2 + 
sin(x[2])*((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2) + 
(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2])/x[1])*
(sin(x[3])*dx[3]^4 - 6*cos(x[3])*dx[3]^2*d2x[3] - 
3*sin(x[3])*d2x[3]^2 - 4*sin(x[3])*dx[3]*d3x[3] + 
cos(x[3])*d4x[3]) + 6*cos(x[3])*dx[3]*
(10*(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2])*
((-6*dx[1]^3)/x[1]^4 + (6*dx[1]*d2x[1])/x[1]^3 - 
d3x[1]/x[1]^2) + 10*
((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2)*
(-(cos(x[2])*dx[2]^3) - 3*sin(x[2])*dx[2]*d2x[2] + 
cos(x[2])*d3x[2]) + 
5*cos(x[2])*dx[2]*((24*dx[1]^4)/x[1]^5 - 
(36*dx[1]^2*d2x[1])/x[1]^4 + (6*d2x[1]^2)/x[1]^3 + 
(8*dx[1]*d3x[1])/x[1]^3 - d4x[1]/x[1]^2) - 
(5*dx[1]*(sin(x[2])*dx[2]^4 - 
6*cos(x[2])*dx[2]^2*d2x[2] - 3*sin(x[2])*d2x[2]^2 - 
4*sin(x[2])*dx[2]*d3x[2] + cos(x[2])*d4x[2]))/x[1]^2 + 
sin(x[2])*((-120*dx[1]^5)/x[1]^6 + 
(240*dx[1]^3*d2x[1])/x[1]^5 - 
(90*dx[1]*d2x[1]^2)/x[1]^4 - 
(60*dx[1]^2*d3x[1])/x[1]^4 + 
(20*d2x[1]*d3x[1])/x[1]^3 + 
(10*dx[1]*d4x[1])/x[1]^3 - d5x[1]/x[1]^2) + 
(cos(x[2])*dx[2]^5 + 10*sin(x[2])*dx[2]^3*d2x[2] - 
15*cos(x[2])*dx[2]*d2x[2]^2 - 
10*cos(x[2])*dx[2]^2*d3x[2] - 
10*sin(x[2])*d2x[2]*d3x[2] - 
5*sin(x[2])*dx[2]*d4x[2] + cos(x[2])*d5x[2])/x[1]) + 
6*(-((sin(x[2])*dx[1])/x[1]^2) + (cos(x[2])*dx[2])/x[1])*
(cos(x[3])*dx[3]^5 + 10*sin(x[3])*dx[3]^3*d2x[3] - 
15*cos(x[3])*dx[3]*d2x[3]^2 - 
10*cos(x[3])*dx[3]^2*d3x[3] - 
10*sin(x[3])*d2x[3]*d3x[3] - 
5*sin(x[3])*dx[3]*d4x[3] + cos(x[3])*d5x[3]) + 
sin(x[3])*(20*((-6*dx[1]^3)/x[1]^4 + 
(6*dx[1]*d2x[1])/x[1]^3 - d3x[1]/x[1]^2)*
(-(cos(x[2])*dx[2]^3) - 3*sin(x[2])*dx[2]*d2x[2] + 
cos(x[2])*d3x[2]) + 
15*(-(sin(x[2])*dx[2]^2) + cos(x[2])*d2x[2])*
((24*dx[1]^4)/x[1]^5 - (36*dx[1]^2*d2x[1])/x[1]^4 + 
(6*d2x[1]^2)/x[1]^3 + (8*dx[1]*d3x[1])/x[1]^3 - 
d4x[1]/x[1]^2) + 15*
((2*dx[1]^2)/x[1]^3 - d2x[1]/x[1]^2)*
(sin(x[2])*dx[2]^4 - 6*cos(x[2])*dx[2]^2*d2x[2] - 
3*sin(x[2])*d2x[2]^2 - 4*sin(x[2])*dx[2]*d3x[2] + 
cos(x[2])*d4x[2]) + 
6*cos(x[2])*dx[2]*((-120*dx[1]^5)/x[1]^6 + 
(240*dx[1]^3*d2x[1])/x[1]^5 - 
(90*dx[1]*d2x[1]^2)/x[1]^4 - 
(60*dx[1]^2*d3x[1])/x[1]^4 + 
(20*d2x[1]*d3x[1])/x[1]^3 + 
(10*dx[1]*d4x[1])/x[1]^3 - d5x[1]/x[1]^2) - 
(6*dx[1]*(cos(x[2])*dx[2]^5 + 
10*sin(x[2])*dx[2]^3*d2x[2] - 
15*cos(x[2])*dx[2]*d2x[2]^2 - 
10*cos(x[2])*dx[2]^2*d3x[2] - 
10*sin(x[2])*d2x[2]*d3x[2] - 
5*sin(x[2])*dx[2]*d4x[2] + cos(x[2])*d5x[2]))/x[1]^2 + 
sin(x[2])*((720*dx[1]^6)/x[1]^7 - 
(1800*dx[1]^4*d2x[1])/x[1]^6 + 
(1080*dx[1]^2*d2x[1]^2)/x[1]^5 - 
(90*d2x[1]^3)/x[1]^4 + (480*dx[1]^3*d3x[1])/x[1]^5 - 
(360*dx[1]*d2x[1]*d3x[1])/x[1]^4 + 
(20*d3x[1]^2)/x[1]^3 - (90*dx[1]^2*d4x[1])/x[1]^4 + 
(30*d2x[1]*d4x[1])/x[1]^3 + 
(12*dx[1]*d5x[1])/x[1]^3 - d6x[1]/x[1]^2) + 
(-(sin(x[2])*dx[2]^6) + 15*cos(x[2])*dx[2]^4*d2x[2] + 
45*sin(x[2])*dx[2]^2*d2x[2]^2 - 15*cos(x[2])*d2x[2]^3 + 
20*sin(x[2])*dx[2]^3*d3x[2] - 
60*cos(x[2])*dx[2]*d2x[2]*d3x[2] - 
10*sin(x[2])*d3x[2]^2 - 15*cos(x[2])*dx[2]^2*d4x[2] - 
15*sin(x[2])*d2x[2]*d4x[2] - 
6*sin(x[2])*dx[2]*d5x[2] + cos(x[2])*d6x[2])/x[1]) + 
(sin(x[2])*(-(sin(x[3])*dx[3]^6) + 
15*cos(x[3])*dx[3]^4*d2x[3] + 
45*sin(x[3])*dx[3]^2*d2x[3]^2 - 15*cos(x[3])*d2x[3]^3 + 
20*sin(x[3])*dx[3]^3*d3x[3] - 
60*cos(x[3])*dx[3]*d2x[3]*d3x[3] - 
10*sin(x[3])*d3x[3]^2 - 15*cos(x[3])*dx[3]^2*d4x[3] - 
15*sin(x[3])*d2x[3]*d4x[3] - 
6*sin(x[3])*dx[3]*d5x[3] + cos(x[3])*d6x[3]))/x[1]

end