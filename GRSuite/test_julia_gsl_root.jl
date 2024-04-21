include("main.jl")
using .Kerr

p1=7.0; e1=0.6; a=0.98; θi1=0.570798

E1, L1, Q1, C1 = Kerr.ConstantsOfMotion.ELQ(a, p1, e1, θi1, M)

p2_gsl, e2_gsl, θmin2_gsl = Kerr.ConstantsOfMotion.peθ_gsl(a, E1, L1, Q1, C1, M)

p2_jul, e2_jul, θmin2_jul = Kerr.ConstantsOfMotion.peθ(a, E1, L1, Q1, C1, M)

cos(θi)

cos(θmin2_gsl)