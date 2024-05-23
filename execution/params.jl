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

### fourier fit parameters ###
nPointsFit = 500;   # number of points in each piecewise geodesic
nHarm = 2;    # number of harmonics in the fourier series expansion

### Boyer-Lindquist fourier fit params ###
t_range_factor_BL = 0.05;   # determines the "length", ΔΤ, of the piecewise geodesics: ΔT = t_range_factor_BL * (2π / min(Ω)), where Ω are the fundamental frequencies

### Mino time fourier fit params ###
t_range_factor_Mino_FF = 0.05;   # tests seem to indicate 0.5 is not accurate enough. Need 0.05 but this takes a long time (e.g., ~10 times longer than with 0.5)

### Mino time finite differences params ###
h=0.01;   # tests indicate h = 0.01 seems to be the optimal value. Larger h seems to not have converged, while h=0.005 seems to be slightly less accurate
nPointsFDM = 20;    # update the self-acceleration every 20 points, i.e., each piecewise geodesic consists of 20 points

### geodesic solver parameters ###
kerrReltol = 1e-10;   # relative tolerance
kerrAbstol = 1e-10;   # absolute tolerance

### file paths ###
# Boyer-Lindquist fourier fit
results_path_BL = "/home/lkeeble/GRSuite/Results/FourierFit/BoyerLindquist";
data_path_BL=results_path_BL * "/Data/t_range_factor_$(t_range_factor_BL)/";
plot_path_BL=results_path_BL * "/Plots/t_range_factor_$(t_range_factor_BL)/";

# Mino time fourier fit
results_path_Mino_FF = "/home/lkeeble/GRSuite/Results/FourierFit/Mino";
data_path_Mino_FF=results_path_Mino_FF * "/Data/t_range_factor_$(t_range_factor_Mino_FF)/";
plot_path_Mino_FF=results_path_Mino_FF * "/Plots/t_range_factor_$(t_range_factor_Mino_FF)/";

# Mino time finite differences
results_path_Mino_FDM = "/home/lkeeble/GRSuite/Results/FiniteDifferences/Mino";
data_path_Mino_FDM=results_path_Mino_FDM * "/Data/h_$(h)/";
plot_path_Mino_FDM=results_path_Mino_FDM * "/Plots/h_$(h)/";

mkpath(data_path_BL);
mkpath(plot_path_BL);
mkpath(data_path_Mino_FF);
mkpath(plot_path_Mino_FF);
mkpath(data_path_Mino_FDM);
mkpath(plot_path_Mino_FDM);

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

M = 1.; m = q;   # perform evolution with M = 1, which means the mass of the small compact object is equal to the mass ratio