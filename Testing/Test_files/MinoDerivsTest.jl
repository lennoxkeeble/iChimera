include("/home/lkeeble/GRSuite/main.jl");
include("/home/lkeeble/GRSuite/Testing/Test_modules/ParameterizedDerivs.jl");
include("/home/lkeeble/GRSuite/Testing/Test_modules/MinoTimeBLTimeDerivs.jl");
include("/home/lkeeble/GRSuite/Testing/BL_time_derivs/d2x.jl")
include("/home/lkeeble/GRSuite/Testing/BL_time_derivs/d3x.jl")
include("/home/lkeeble/GRSuite/Testing/BL_time_derivs/d4x.jl")
include("/home/lkeeble/GRSuite/Testing/BL_time_derivs/d5x.jl")
include("/home/lkeeble/GRSuite/Testing/BL_time_derivs/d6x.jl")
using .ParameterizedDerivs, .MinoTimeDerivs
using .Deriv2, .Deriv3, .Deriv4, .Deriv5, .Deriv6, .HJEvolution
using .Kerr

p=7.0; e=0.6; a=0.98; θmin=0.570798; M=1.0;

E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θmin, M);

x = [13.1, π/4, 1.5π];
dx = [0.9, 2.2, 0.5];

d2x = [Deriv2.d2r_dt(dx, x, a), Deriv2.d2θ_dt(dx, x, a), Deriv2.d2ϕ_dt(dx, x, a)]
d3x = [Deriv3.d3r_dt(d2x, dx, x, a), Deriv3.d3θ_dt(d2x, dx, x, a), Deriv3.d3ϕ_dt(d2x, dx, x, a)]
d4x = [Deriv4.d4r_dt(d3x, d2x, dx, x, a), Deriv4.d4θ_dt(d3x, d2x, dx, x, a), Deriv4.d4ϕ_dt(d3x, d2x, dx, x, a)]
d5x = [Deriv5.d5r_dt(d4x, d3x, d2x, dx, x, a), Deriv5.d5θ_dt(d4x, d3x, d2x, dx, x, a), Deriv5.d5ϕ_dt(d4x, d3x, d2x, dx, x, a)]
d6x = [Deriv6.d6r_dt(d5x, d4x, d3x, d2x, dx, x, a), Deriv6.d6θ_dt(d5x, d4x, d3x, d2x, dx, x, a), Deriv6.d6ϕ_dt(d5x, d4x, d3x, d2x, dx, x, a)]

dλdt = MinoTimeDerivs.dλ_dt(x, a, M, E, L)

d2λdt = MinoTimeDerivs.d2λ_dt(dx, x, a, M, E, L)

d3λdt = MinoTimeDerivs.d3λ_dt(d2x, dx, x, a, M, E, L)

d4λdt = MinoTimeDerivs.d4λ_dt(d3x, d2x, dx, x, a, M, E, L)

d5λdt = MinoTimeDerivs.d5λ_dt(d4x, d3x, d2x, dx, x, a, M, E, L)

d6λdt = MinoTimeDerivs.d6λ_dt(d5x, d4x, d3x, d2x, dx, x, a, M, E, L)

println(dλdt)
println(d2λdt)
println(d3λdt)
println(d4λdt)
println(d5λdt)
println(d6λdt)

df_dλ=1.1; d2f_dλ=2.2;  d3f_dλ=3.3;  d4f_dλ=4.4; d5f_dλ=5.5;  d6f_dλ=6.6;

dfdt = ParameterizedDerivs.df_dt(df_dλ, dλdt)

d2fdt = ParameterizedDerivs.d2f_dt(df_dλ, dλdt, d2f_dλ, d2λdt)

d3fdt = ParameterizedDerivs.d3f_dt(df_dλ, dλdt, d2f_dλ, d2λdt, d3f_dλ, d3λdt)

d4fdt = ParameterizedDerivs.d4f_dt(df_dλ, dλdt, d2f_dλ, d2λdt, d3f_dλ, d3λdt, d4f_dλ, d4λdt)

d5fdt = ParameterizedDerivs.d5f_dt(df_dλ, dλdt, d2f_dλ, d2λdt, d3f_dλ, d3λdt, d4f_dλ, d4λdt, d5f_dλ, d5λdt)

d6fdt = ParameterizedDerivs.d6f_dt(df_dλ, dλdt, d2f_dλ, d2λdt, d3f_dλ, d3λdt, d4f_dλ, d4λdt, d5f_dλ, d5λdt, d6f_dλ, d6λdt)