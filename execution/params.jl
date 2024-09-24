include("../main.jl");

### CONSTANTS ###
c = 2.99792458 * 1e8; Grav_Newton = 6.67430 * 1e-11; Msol = (1.988) * 1e30; year = 365 * 24 * 60 * 60;

### (initial) orbital parameters ###
Mass_MBH = 1e6 * Msol;   # mass of the MBH — sets evolution time
a = 0.98;   # spin
e = 0.6;   # eccentricity
q = 1e-5;   # mass ratio m/M
# ι = 57.39 * π / 180;   # iota
p = 7.0;   # semi-latus rectum
# θmin = SemiRelativisticKludge.Inspiral.iota_to_theta_min(a, p, e, ι);  # θmin
θmin = 0.570798

### evolution time ###
MtoSecs = Mass_MBH * Grav_Newton / c^3;   # conversion from t(M) -> t(s)
t_max_secs = (10^-3) * year / 3.   # seconds
t_max_M = t_max_secs / MtoSecs;   # units of M

### fourier fit parameters ###
gsl_fit = "GSL";
nPointsFitGSL = 501;   # number of points in each piecewise geodesic  
nHarmGSL = 2;    # number of harmonics in the fourier series expansion

julia_fit = "Julia"
nPointsFitJulia = 101;   # number of points in each piecewise geodesic
nHarmJulia = 3;    # number of harmonics in the fourier series expansion

### Boyer-Lindquist fourier fit params ###
t_range_factor_BL = 0.5;   # determines the "length", ΔΤ, of the piecewise geodesics: ΔT = t_range_factor_BL * (2π / min(Ω)), where Ω are the fundamental frequencies

### Mino time fourier fit params ###
t_range_factor_Mino_FF = 0.05;   # tests seem to indicate 0.05 consistently yields sufficiently accurate fits

### Mino time finite differences params ###
h=0.001;

### geodesic solver parameters ###
nPointsGeodesic = 500;
kerrReltol = 1e-14;   # relative tolerance
kerrAbstol = 1e-14;   # absolute tolerance

### file paths ###
# Boyer-Lindquist fourier fit
results_path = "../Results";
data_path=results_path * "/Data/";
plot_path=results_path * "/Plots/";

mkpath(data_path);
mkpath(plot_path);

# metric components — currently limited to Kerr
metric = "Kerr";
if metric=="Kerr"
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
end

MBH_geometric_mass = 1.; SCO_geometric_mass = q;   # perform evolution with M = 1, which means the mass of the small compact object is equal to the mass ratio

## compute frequency of flux computation to match Chimera ##
EE, LL, QQ, CC, ι = Kerr.ConstantsOfMotion.ELQ(a, p, e, θmin, MBH_geometric_mass);
rplus = Kerr.KerrMetric.rplus(a, MBH_geometric_mass); rminus = Kerr.KerrMetric.rminus(a, MBH_geometric_mass);
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θmin, EE, LL, QQ, CC, rplus, rminus, MBH_geometric_mass);    # Mino time frequencies
Ω = ω[1:3]/ω[4]; Ωr, Ωθ, Ωϕ = Ω;  # BL time frequencies

compute_SF_frac = 0.05;
compute_fluxes_BL = e!=0.0 ? compute_SF_frac * minimum(@. 2π /Ω) : compute_SF_frac * minimum(@. 2π /Ω[2:3]);   # in units of M (not seconds)
# compute_fluxes_Mino = compute_SF_frac * minimum(@. 2π /ω);   # in units of M (not seconds)
compute_fluxes_Mino = e!= 0.0 ? compute_SF_frac * minimum(@. 2π /ω[1:3]) : compute_SF_frac * minimum(@. 2π /ω[2:3]);   # in units of M (not seconds)

## number of points in Finite Difference geodesic ##
nPointsFDM = Int(floor(compute_fluxes_Mino / h))

## WAVEFORMS ##
obs_distance = 1.;
Θ=π/4; Φ=0.;   # observer latitude and azimuth