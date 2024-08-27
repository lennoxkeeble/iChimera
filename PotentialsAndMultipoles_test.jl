include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/GRSuite/main.jl")
include("/Users/lennoxkeeble/Library/Mobile Documents/com~apple~CloudDocs/Physics_Research/EMRIs/GRSuite/PotentialsAndMultipoles.jl")
using BenchmarkTools

Γαμν(t, r, θ, ϕ, a, M, α, μ, ν) = Kerr.KerrMetric.Γαμν(t, r, θ, ϕ, a, M, α, μ, ν);   # Christoffel symbols
# covariant metric components
g_tt(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_tt(t, r, θ, ϕ, a, M);
g_tϕ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_tϕ(t, r, θ, ϕ, a, M);
g_rr(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_rr(t, r, θ, ϕ, a, M);
g_θθ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_θθ(t, r, θ, ϕ, a, M);
g_ϕϕ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.g_ϕϕ(t, r, θ, ϕ, a, M);
g_μν(t, r, θ, ϕ, a, M, μ, ν) = Kerr.KerrMetric.g_μν(t, r, θ, ϕ, a, M, μ, ν); 
# contravariant metric components
gTT(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gTT(t, r, θ, ϕ, a, M);
gTΦ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gTΦ(t, r, θ, ϕ, a, M);
gRR(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gRR(t, r, θ, ϕ, a, M);
gThTh(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gThTh(t, r, θ, ϕ, a, M);
gΦΦ(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.gΦΦ(t, r, θ, ϕ, a, M);
ginv(t, r, θ, ϕ, a, M) = Kerr.KerrMetric.ginv(t, r, θ, ϕ, a, M);

xH = [123.321, 3245.543, 65765.67879687]; 
rH = sqrt(xH[1]^2 + xH[2]^2 + xH[3]^2);
xBL = [0974839803.432, 412973.3243, 8796059454.3245];
jBLH = [4239084.423 423435.657 76768.465; 324.532 5654762.973 76543.423; 124.352 456754.35 21453412.53];
HessBLH = [HarmonicCoords.HessBLH(xH, rH, a, M, m) for m=1:3]
a=1.321; M=213.213;

## function 1 ##
# Equality
test1 = [SelfAcceleration.∂K_∂xk(xH, xBL, jBLH, HessBLH, a, M, g_μν, Γαμν, k) for k in 1:3]
test2 = [PotentialsAndMultipoles.∂K_∂xk(xH, xBL, jBLH, a, M, g_μν, Γαμν, k) for k in 1:3]
isequal(test1, test2)

# timing 
@btime SelfAcceleration.∂K_∂xk(xH, xBL, jBLH, HessBLH, a, M, g_μν, Γαμν, 2)
@btime PotentialsAndMultipoles.∂K_∂xk(xH, xBL, jBLH, a, M, g_μν, Γαμν, 2)

## Function 2 ##
# Equality
test3 = [SelfAcceleration.∂Ki_∂xk(xH, rH, xBL, jBLH, HessBLH, a, M, g_μν, Γαμν, i, k) for i=1:3, k=1:3]
test4 = [PotentialsAndMultipoles.∂Ki_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, i, k) for i=1:3, k=1:3]
isequal(test3, test4)

# timing
@btime SelfAcceleration.∂Ki_∂xk(xH, rH, xBL, jBLH, HessBLH, a, M, g_μν, Γαμν, 2, 3)
@btime PotentialsAndMultipoles.∂Ki_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, 2, 3)

## Function 3 ##
test5 = [SelfAcceleration.∂Kij_∂xk(xH, rH, xBL, jBLH, HessBLH, a, M, g_μν, Γαμν, k, i, j) for i=1:3, j=1:3, k=1:3]
test6 = [PotentialsAndMultipoles.∂Kij_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, k, i, j) for i=1:3, j=1:3, k=1:3]
isequal(test5, test6)

# timing
@btime SelfAcceleration.∂Kij_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, 2, 3, 1)
@btime PotentialsAndMultipoles.∂Kij_∂xk(xH, rH, xBL, jBLH, a, M, g_μν, Γαμν, 2, 3, 1)