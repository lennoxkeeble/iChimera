# inspiral evolved in Mino time and using Julia's base least squares to fit the multipole moments to their Fourier series expansion in order to estimate their high order time derivatives. If use_FDM=true, finite differences are used to compute
# the (lower order) time derivatives of the multipole moments required for waveform generation. If use_FDM=false, these time derivatives are also estimated using Fourier fits.
use_FDM=true
@time ChimeraInspiral.FourierFit.MinoTime.compute_inspiral(a, p, e, θmin, sign_Lz, q, psi0, chi0, phi0, nPointsFitJulia, nHarmJulia, t_range_factor_Mino_FF, compute_fluxes_Mino, t_max_M, use_FDM, julia_fit, reltol, abstol;
        h=h, nPointsGeodesic=nPointsGeodesic, data_path=data_path)