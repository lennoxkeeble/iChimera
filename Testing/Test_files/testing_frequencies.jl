include("/home/lkeeble/GRSuite/main.jl")
include("/home/lkeeble/GRSuite/FujitaFrequencies.jl")
using .Kerr, .FujitaFrequencies

p=7.0; e=0.6; a=0.98; θmin=0.570798; M=1.0; rplus = Kerr.KerrMetric.rplus(a, M); rminus = Kerr.KerrMetric.rminus(a, M);

E, L, Q, C = Kerr.ConstantsOfMotion.ELQ(a, p, e, θmin, M)
EE, LL, QQ = Kerr.ConstantsOfMotion.SchmidtELQ(a, p, e, θmin)

p2, e2, θmin2 = Kerr.ConstantsOfMotion.peθ(a, E, L, Q, C, M)

println("p-p2 = $(p-p2); e-e2 = $(e-e2); θmin-θmin2 = $(θmin-θmin2)")

Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θmin, E, L, Q, C, rplus, rminus, M)
FujitaFrequencies.compute_frequencies(a, p, e, θmin, E, L, C, rplus, rminus, M)

γNK = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θmin, E, L, Q, C, rplus, rminus, M)
γFuj = FujitaFrequencies.compute_frequencies(a, p, e, θmin, E, L, C, rplus, rminus, M)


Ω_mino = γNK[1:3]/γNK[4]

ω_prop = Kerr.ConstantsOfMotion.SchmidtKerrFreqs(a, p, e, θmin); Ω_prop = ω_prop[1:3] / ω_prop[4]

Ω_prop .- Ω_mino