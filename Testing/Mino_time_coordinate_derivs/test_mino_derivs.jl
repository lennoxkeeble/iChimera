include("MinoTime_dλ.jl"); include("MinoTime_d2λ.jl");
include("MinoTime_d3λ.jl"); include("MinoTime_d4λ.jl");
include("MinoTime_d5λ.jl"); include("MinoTime_d6λ.jl");
include("MinoTime_d7λ.jl"); include("MinoTime_d8λ.jl");
include("MinoTime_dλ_dt.jl");

M=1.

# case 1
a=0.98; E=0.9575515155935412; L=1.7345010496294588; C=7.353612189722577; x = [13.1, π/4, 1.5π]

dx = [MinoDerivs1.dr_dλ(x, a, M, E, L, C), MinoDerivs1.dθ_dλ(x, a, M, E, L, C), MinoDerivs1.dϕ_dλ(x, a, M, E, L, C)]
d2x = [MinoDerivs2.d2r_dλ(x, dx, a, M, E, L, C), MinoDerivs2.d2θ_dλ(x, dx, a, M, E, L, C), MinoDerivs2.d2ϕ_dλ(x, dx, a, M, E, L, C)]
d3x = [MinoDerivs3.d3r_dλ(x, dx, d2x, a, M, E, L, C), MinoDerivs3.d3θ_dλ(x, dx, d2x, a, M, E, L, C), MinoDerivs3.d3ϕ_dλ(x, dx, d2x, a, M, E, L, C)]
d4x = [MinoDerivs4.d4r_dλ(x, dx, d2x, d3x, a, M, E, L, C), MinoDerivs4.d4θ_dλ(x, dx, d2x, d3x, a, M, E, L, C), MinoDerivs4.d4ϕ_dλ(x, dx, d2x, d3x, a, M, E, L, C)]
d5x = [MinoDerivs5.d5r_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C), MinoDerivs5.d5θ_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C), MinoDerivs5.d5ϕ_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C)]
d6x = [MinoDerivs6.d6r_dλ(x, dx, d2x, d3x, d4x, d5x, a, M, E, L, C), MinoDerivs6.d6θ_dλ(x, dx, d2x, d3x, d4x, d5x, a, M, E, L, C), MinoDerivs6.d6ϕ_dλ(x, dx, d2x, d3x, d4x, d5x, a, M, E, L, C)]
d7x = [MinoDerivs7.d7r_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, a, M, E, L, C), MinoDerivs7.d7θ_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, a, M, E, L, C), MinoDerivs7.d7ϕ_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, a, M, E, L, C)]
d8x = [MinoDerivs8.d8r_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, a, M, E, L, C), MinoDerivs8.d8θ_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, a, M, E, L, C), MinoDerivs8.d8ϕ_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, a, M, E, L, C)]


isapprox(dx,[21.431355384669573, 2.0749014516225395, 3.6255930350377064])
isapprox(d2x, [12.417688948164141, 6.056890046942951, -14.675729874329178]) 
isapprox(d3x, [-372.9400048473996, -49.93862673581793, 78.29533597436837])
isapprox(d4x, [-3459.937598779571, 371.6245604078169, 154.4983422367948])
isapprox(d5x, [-8964.213519897563, -1146.89608946532, -16279.991403583128])
isapprox(d6x, [248611.09051473954, -57168.65919593128, 390973.20164298656])
isapprox(d7x, [4.760141708552408*1e6, 2.0675229185474059*1e6, -5.133498324026765*1e6])
isapprox(d8x, [2.822195828733162*1e7, -4.162423400305682*1e7, -5.342686238559523*1e7])

# case 2
a = 0.8; E = 0.966389906261774; L = 1.8622387328496137; C = 10.435522757932183; x = [21.0, 0.5235987755982987, 0.0];

dx = [MinoDerivs1.dr_dλ(x, a, M, E, L, C), MinoDerivs1.dθ_dλ(x, a, M, E, L, C), MinoDerivs1.dϕ_dλ(x, a, M, E, L, C)]
d2x = [MinoDerivs2.d2r_dλ(x, dx, a, M, E, L, C), MinoDerivs2.d2θ_dλ(x, dx, a, M, E, L, C), MinoDerivs2.d2ϕ_dλ(x, dx, a, M, E, L, C)]
d3x = [MinoDerivs3.d3r_dλ(x, dx, d2x, a, M, E, L, C), MinoDerivs3.d3θ_dλ(x, dx, d2x, a, M, E, L, C), MinoDerivs3.d3ϕ_dλ(x, dx, d2x, a, M, E, L, C)]
d4x = [MinoDerivs4.d4r_dλ(x, dx, d2x, d3x, a, M, E, L, C), MinoDerivs4.d4θ_dλ(x, dx, d2x, d3x, a, M, E, L, C), MinoDerivs4.d4ϕ_dλ(x, dx, d2x, d3x, a, M, E, L, C)]
d5x = [MinoDerivs5.d5r_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C), MinoDerivs5.d5θ_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C), MinoDerivs5.d5ϕ_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C)]
d6x = [MinoDerivs6.d6r_dλ(x, dx, d2x, d3x, d4x, d5x, a, M, E, L, C), MinoDerivs6.d6θ_dλ(x, dx, d2x, d3x, d4x, d5x, a, M, E, L, C), MinoDerivs6.d6ϕ_dλ(x, dx, d2x, d3x, d4x, d5x, a, M, E, L, C)]
d7x = [MinoDerivs7.d7r_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, a, M, E, L, C), MinoDerivs7.d7θ_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, a, M, E, L, C), MinoDerivs7.d7ϕ_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, a, M, E, L, C)]
d8x = [MinoDerivs8.d8r_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, a, M, E, L, C), MinoDerivs8.d8θ_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, a, M, E, L, C), MinoDerivs8.d8ϕ_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, a, M, E, L, C)]

isapprox(dx,[0., 0., 7.527222542401794])
isapprox(d2x, [-182.36825996910557, 24.044860843035114, 0.]) 
isapprox(d3x, [0., 0., -619.72902160979])
isapprox(d4x, [11456.619679458343, -3334.9302261174007, 0.])
isapprox(d5x, [0., 0., 344448.1942127126])
isapprox(d6x, [-1.7827999495283351*1e6, 2.296036934052782*1e6, 0.])
isapprox(d7x, [0., 0., -4.751953100346559*1e8])
isapprox(d8x, [5.1807224678446645*1e8, -3.6957774632972255*1e9, 0.])

dt_dλ = MinoDerivs1.dt_dλ(x, a, M, E, L, C)
d2t_dλ = MinoDerivs2.d2t_dλ(x, dx, a, M, E, L, C)
d3t_dλ = MinoDerivs3.d3t_dλ(x, dx, d2x, a, M, E, L, C)
d4t_dλ = MinoDerivs4.d4t_dλ(x, dx, d2x, d3x, a, M, E, L, C)
d5t_dλ = MinoDerivs5.d5t_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C)
d6t_dλ = MinoDerivs6.d6t_dλ(x, dx, d2x, d3x, d4x, d5x, a, M, E, L, C)
d7t_dλ = MinoDerivs7.d7t_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, a, M, E, L, C)
d8t_dλ = MinoDerivs8.d8t_dλ(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, a, M, E, L, C)

dλ_dt = MinoTimeDerivs.dλ_dt(dt_dλ)
d2λ_dt = MinoTimeDerivs.d2λ_dt(dt_dλ, d2t_dλ)
d3λ_dt = MinoTimeDerivs.d3λ_dt(dt_dλ, d2t_dλ, d3t_dλ)
d4λ_dt = MinoTimeDerivs.d4λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ)
d5λ_dt = MinoTimeDerivs.d5λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ)
d6λ_dt = MinoTimeDerivs.d6λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ)
d7λ_dt = MinoTimeDerivs.d7λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ, d7t_dλ)
d8λ_dt = MinoTimeDerivs.d8λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ, d7t_dλ, d8t_dλ)

dnλ_dt = [dλ_dt, d2λ_dt, d3λ_dt, d4λ_dt, d5λ_dt, d6λ_dt, d7λ_dt, d8λ_dt]

# isapprox(dnλ_dt, [0.002121614141246884, 0., 1.573306493512927*1e-7, 0., 5.45909234404972*1e-11, 0., 4.04534732937671*1e-14, 0.])
isapprox(dnλ_dt, [0.005153058932112476, -0.00007883898475740231, 2.764491178571364*1e-6, -1.1540137762710911*1e-7,
5.890033180483987*1e-9, -3.2481270203887455*1e-10, 1.7889069848136207*1e-11, -7.672180664020724*1e-13])

d3x = [Deriv3.d3r_dt(d2x, dx, x, a), Deriv3.d3θ_dt(d2x, dx, x, a), Deriv3.d3ϕ_dt(d2x, dx, x, a)]
d4x = [Deriv4.d4r_dt(d3x, d2x, dx, x, a), Deriv4.d4θ_dt(d3x, d2x, dx, x, a), Deriv4.d4ϕ_dt(d3x, d2x, dx, x, a)]
d5x = [Deriv5.d5r_dt(d4x, d3x, d2x, dx, x, a), Deriv5.d5θ_dt(d4x, d3x, d2x, dx, x, a), Deriv5.d5ϕ_dt(d4x, d3x, d2x, dx, x, a)]
# d6x = [Deriv6.d6r_dt(d5x, d4x, d3x, d2x, dx, x, a), Deriv6.d6θ_dt(d5x, d4x, d3x, d2x, dx, x, a), Deriv6.d6ϕ_dt(d5x, d4x, d3x, d2x, dx, x, a)]

dλ_dt = MinoTimeDerivs.dλ_dt(x, a, M, E, L)
d2λ_dt = MinoTimeDerivs.d2λ_dt(dx, x, a, M, E, L)
d3λ_dt = MinoTimeDerivs.d3λ_dt(d2x, dx, x, a, M, E, L)
d4λ_dt = MinoTimeDerivs.d4λ_dt(d3x, d2x, dx, x, a, M, E, L)
d5λ_dt = MinoTimeDerivs.d5λ_dt(d4x, d3x, d2x, dx, x, a, M, E, L)
d6λ_dt = MinoTimeDerivs.d6λ_dt(d5x, d4x, d3x, d2x, dx, x, a, M, E, L)
