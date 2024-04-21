include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/Geodesics/Mathematica/BL_time_derivs/d2x.jl")
include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/Geodesics/Mathematica/BL_time_derivs/d3x.jl")
include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/Geodesics/Mathematica/BL_time_derivs/d4x.jl")
include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/Geodesics/Mathematica/BL_time_derivs/d5x.jl")
include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/Geodesics/Mathematica/BL_time_derivs/d6x.jl")
include("/Users/lennoxkeeble/Downloads/GRSuite/main.jl")
include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/Geodesics/Mathematica/BL_time_derivs/Test_functions.jl")

using .Deriv2, .Deriv3, .Deriv4, .Deriv5, .Deriv6, .HJEvolution, .Function_1, .Function_2

x = [13.1, π/4, 1.5π];
dx = [0.9, 2.2, 0.5];
a=0.435;

HJd2x = [HJEvolution.dr2_dt2(0.0, x..., dx..., a, 1.0), HJEvolution.dθ2_dt2(0.0, x..., dx..., a, 1.0), HJEvolution.dϕ2_dt2(0.0, x..., dx..., a, 1.0)]

d2x = [Deriv2.d2r_dt(dx, x, a), Deriv2.d2θ_dt(dx, x, a), Deriv2.d2ϕ_dt(dx, x, a)]
d3x = [Deriv3.d3r_dt(d2x, dx, x, a), Deriv3.d3θ_dt(d2x, dx, x, a), Deriv3.d3ϕ_dt(d2x, dx, x, a)]
d4x = [Deriv4.d4r_dt(d3x, d2x, dx, x, a), Deriv4.d4θ_dt(d3x, d2x, dx, x, a), Deriv4.d4ϕ_dt(d3x, d2x, dx, x, a)]
d5x = [Deriv5.d5r_dt(d4x, d3x, d2x, dx, x, a), Deriv5.d5θ_dt(d4x, d3x, d2x, dx, x, a), Deriv5.d5ϕ_dt(d4x, d3x, d2x, dx, x, a)]
d6x = [Deriv6.d6r_dt(d5x, d4x, d3x, d2x, dx, x, a), Deriv6.d6θ_dt(d5x, d4x, d3x, d2x, dx, x, a), Deriv6.d6ϕ_dt(d5x, d4x, d3x, d2x, dx, x, a)]

Function_1.f(x)
Function_1.df_dt(dx, x)
Function_1.d2f_dt(d2x, dx, x)
Function_1.d3f_dt(d3x, d2x, dx, x)
Function_1.d4f_dt(d4x, d3x, d2x, dx, x)
Function_1.d5f_dt(d5x, d4x, d3x, d2x, dx, x)
Function_1.d6f_dt(d6x, d5x, d4x, d3x, d2x, dx, x)

Function_2.f(x)
Function_2.df_dt(dx, x)
Function_2.d2f_dt(d2x, dx, x)
Function_2.d3f_dt(d3x, d2x, dx, x)
Function_2.d4f_dt(d4x, d3x, d2x, dx, x)
Function_2.d5f_dt(d5x, d4x, d3x, d2x, dx, x)
Function_2.d6f_dt(d6x, d5x, d4x, d3x, d2x, dx, x)