include("MinoTimeBLTimeDerivs.jl");

M=1.

# case 1
a=0.98; E=0.9575515155935412; L=1.7345010496294588; C=7.353612189722577; x = [13.1, π/4, 1.5π]

dx = [10., π/3., 0.0];
d2x = [1.3, 1.1, 1.01];
d3x = [5.5, 6.6, 7.7];
d4x = [8.8, 9.9, 11.];
d5x = [12.1, 13.2, 14.3];

MinoTimeDerivs.dλ_dt(x, a, M, E, L)
MinoTimeDerivs.d2λ_dt(dx, x, a, M, E, L)
MinoTimeDerivs.d3λ_dt(d2x, dx, x, a, M, E, L)
MinoTimeDerivs.d4λ_dt(d3x, d2x, dx, x, a, M, E, L)
MinoTimeDerivs.d5λ_dt(d4x, d3x, d2x, dx, x, a, M, E, L)
MinoTimeDerivs.d6λ_dt(d5x, d4x, d3x, d2x, dx, x, a, M, E, L)

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