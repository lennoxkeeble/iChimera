include("/home/lkeeble/GRSuite/main.jl");
### CONSTANTS ###
c = 2.99792458 * 1e8; Grav_Newton = 6.67430 * 1e-11; Msol = (1.988) * 1e30; year = 365 * 24 * 60 * 60;

### (initial) orbital parameters ###
Mass_MBH = 1e6 * Msol;   # mass of the MBH — sets evolution time
a = 0.98;   # spin
p = 7.0;   # semi-latus rectum
e = 0.6;   # eccentricity
q = 1e-5;   # mass ratio m/M
θmin = 0.570798;   # θmin

### evolution time ###
MtoSecs = Mass_MBH * Grav_Newton / c^3;   # conversion from t(M) -> t(s)
t_max_secs = (10^-3) * year / 3.   # seconds
t_max_M = t_max_secs / MtoSecs;   # units of M

compute_SF_frac = 0.05;   # intervals in time (in unitws of M) between successive self-force computations are given by a fraction of the smallest time period (e.g., the inverse of the fundamental frequencies of motion)

### fourier fit parameters ###
nPointsFit = 501;   # number of points in each piecewise geodesic  
nHarm = 2;    # number of harmonics in the fourier series expansion — must be odd

### Boyer-Lindquist fourier fit params ###
t_range_factor_BL = 0.5;   # determines the "length", ΔΤ, of the piecewise geodesics: ΔT = t_range_factor_BL * (2π / min(Ω)), where Ω are the fundamental frequencies

### Mino time fourier fit params ###
t_range_factor_Mino_FF = 0.05;   # tests seem to indicate 0.05 yields sufficiently accurate fits

### Mino time finite differences params ###
h=0.001;

### geodesic solver parameters ###
nPointsGeodesic = 500;
kerrReltol = 1e-10;   # relative tolerance
kerrAbstol = 1e-10;   # absolute tolerance

### file paths ###
# Boyer-Lindquist fourier fit
results_path = "/home/lkeeble/GRSuite/Results";
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