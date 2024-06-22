# compute time interval between successive self force computations
EE, LL, QQ, CC, ι = Kerr.ConstantsOfMotion.ELQ(a, p, e, θmin, MBH_geometric_mass);
rplus = Kerr.KerrMetric.rplus(a, MBH_geometric_mass); rminus = Kerr.KerrMetric.rminus(a, MBH_geometric_mass);
ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θmin, EE, LL, QQ, CC, rplus, rminus, MBH_geometric_mass);    # Mino time frequencies
Ω = ω[1:3]/ω[4]; Ωr, Ωθ, Ωϕ = Ω;  # BL time frequencies
compute_SF = compute_SF_frac * minimum(@. 2π /Ω)   # in units of M (not seconds)

@time InspiralEvolution.FourierFit.BLTime.compute_inspiral_HJE!(t_max_M, compute_SF, t_range_factor_BL, nPointsGeodesic, nPointsFit, MBH_geometric_mass, SCO_geometric_mass,
a, p, e, θmin, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ, nHarm, kerrReltol, kerrAbstol; data_path=data_path)