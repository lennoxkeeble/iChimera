module KludgeWaveforms

module SemiRelativistic
using DelimitedFiles
using Dierckx
## mass quadrupole contribution
# returns array of values for I^{jk}
function Ijk(x, m, j, k)
    return @. m * x[j] * x[k]   # Eq. 6.20
end

# returns array of values for \dot{I}^{jk}
function dotIjk(x, v, m, j, k)
    return @. m * (x[k] * v[j] + x[j] * v[k])   # Eq. D.10
end

# returns array of values for \ddot{I}^{jk}
function ddotIjk(x, v, a, m, j, k)
    return @. m * (x[k] * a[j] + 2v[j] * v[k] + x[j] * a[k])   # Eq. D.11
end

# returns array of values for \dddot{I}^{jk}
function dddotIjk(x, v, a, jerk, m, j, k)
    return @. m * (jerk[j] * x[k] + 3a[j] * v[k] + 3v[j] * a[k] + x[j] * jerk[k])   # Eq. D.12
end

# returns mass quadrupole matrix
Iddot(x, v, a, m) = collect(Vector{Float64}(ddotIjk(x, v, a, m, j, k)) for j=1:3, k=1:3)

## current quadrupole contribution

# returns array of values for \ddot{S}^{ijk}
function ddotSijk(x, v, a, jerk, m, i, j, k)
    IJK = Ijk(x, m, j, k)
    dotIJK = dotIjk(x, v, m, j, k)
    ddotIJK = ddotIjk(x, v, a, m, j, k)
    return @. IJK * jerk[i] + 2dotIJK * a[i] + v[i] * ddotIJK   # Eq. D.13
end

# returns array of values for n_{i}\ddot{S}^{ijk}
function ddotSjk(x, v, a, jerk, Θ, Φ, m, j, k)
    nx = sin(Θ) * cos(Φ)   # Eq. D.15
    ny = sin(Θ) * sin(Φ)   # Eq. D.15
    nz = cos(Θ)   # Eq. D.15
    n = [nx, ny, nz]
    return n[1] * ddotSijk(x, v, a, jerk, m, 1, j, k) .+ n[2] * ddotSijk(x, v, a, jerk, m, 2, j, k) .+ n[3] * ddotSijk(x, v, a, jerk, m, 3, j, k)
end

# returns matrix of n_{i}\ddot{S}^{ijk}
Sddot(x, v, a, jerk, Θ, Φ, m) = collect(Vector{Float64}(ddotSjk(x, v, a, jerk, Θ, Φ, m, j, k)) for j=1:3, k=1:3)

## mass octupole contribution

# returns array of values for \dddot{M}^{ijk}
function dddotMijk(x, v, a, jerk, m, i, j, k)
    IJK = Ijk(x, m, j, k)
    dotIJK = dotIjk(x, v, m, j, k)
    ddotIJK = ddotIjk(x, v, a, m, j, k)
    dddotIJK = dddotIjk(x, v, a, jerk, m, j, k)
    return @. jerk[i] * IJK + 3dotIJK * a[i] + 3v[i] * ddotIJK + x[i] * dddotIJK   # Eq. D.14
end

# returns array of values for n_{i}\dddot{M}^{ijk}
function dddotMjk(x, v, a, jerk, Θ, Φ, m, j, k)
    nx = sin(Θ) * cos(Φ)   # Eq. D.15
    ny = sin(Θ) * sin(Φ)   # Eq. D.15
    nz = cos(Θ)   # Eq. D.15
    n = [nx, ny, nz]
    return n[1] * dddotMijk(x, v, a, jerk, m, 1, j, k) .+ n[2] * dddotMijk(x, v, a, jerk, m, 2, j, k) .+ n[3] * dddotMijk(x, v, a, jerk, m, 3, j, k)
end

# returns matrix of n_{i}\dddot{M}^{ijk}
Mdddot(x, v, a, jerk, Θ, Φ, m) = collect(Vector{Float64}(dddotMjk(x, v, a, jerk, Θ, Φ, m, j, k)) for j=1:3, k=1:3)

### metric perturbation

# perturbation matrix 
h(x, v, a, jerk, Θ, Φ, obs_distance, m) = (2 / obs_distance) * (Iddot(x, v, a, m) .- 2Sddot(x, v, a, jerk, Θ, Φ, m) .+  Mdddot(x, v, a, jerk, Θ, Φ, m))   # Eq. 6.9

# project h into TT gauge
hΘΘ(h, Θ, Φ) = (cos(Θ)^2) * (h[1, 1] * cos(Φ)^2 + h[1, 2] * sin(2Φ) + h[2, 2] * sin(Φ)^2) + h[3, 3] * sin(Θ)^2 - sin(2Θ) * (h[1, 3] * cos(Φ) + h[2, 3] * sin(Φ))   # Eq. 6.15
hΘΦ(h, Θ, Φ) = cos(Θ) * (((-1/2) * h[1, 1] * sin(2Φ)) + h[1, 2] * cos(2Φ) + (1/2) * h[2, 2] * sin(2Φ)) + sin(Θ) * (h[1, 3] * sin(Φ) - h[2, 3] * cos(Φ))   # Eq. 6.16
hΦΦ(h, θ, Φ) = h[1, 1] * sin(Φ)^2 - h[1, 2] * sin(2Φ) + h[2, 2] * cos(Φ)^2   # Eq. 6.17

# define h_{+} and h_{×} components of GW
hplus(h, Θ, Φ) = (1/2) *  (hΘΘ(h, Θ, Φ) - hΦΦ(h, Θ, Φ))
hcross(h, Θ, Φ) = hΘΦ(h, Θ, Φ)

# compute kerr semi-relativistic waveform
function Kerr_waveform(kerr_ode_sol_fname::String, waveform_filename::String, Θ::Float64, Φ::Float64, obs_distance::Float64)
    m=1.0;
    # load solution
    sol = readdlm(kerr_ode_sol_fname)
    τ = sol[1, :]; t = sol[2, :]; r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :]; tdot = sol[6, :]; rdot = sol[7, :]; θdot = sol[8, :]; ϕdot = sol[9, :]; tddot = sol[10, :]; rddot = sol[11, :]; θddot = sol[12, :]; ϕddot = sol[13, :];

    # numerically obtain third order derivatives
    # interpolators
    tddot_sp = Spline1D(τ, tddot, bc="nearest")  # use Dierckx
    rddot_sp = Spline1D(τ, rddot, bc="nearest")  # use Dierckx
    θddot_sp = Spline1D(τ, θddot, bc="nearest")  # use Dierckx
    ϕddot_sp = Spline1D(τ, ϕddot, bc="nearest")  # use Dierckx

    # first derivatives of interpolators
    t_ddot_1(x) = derivative(tddot_sp, x)
    r_ddot_1(x) = derivative(rddot_sp, x)
    θ_ddot_1(x) = derivative(θddot_sp, x)
    ϕ_ddot_1(x) = derivative(ϕddot_sp, x)

    # compute Jerk
    tdddot = t_ddot_1.(τ)
    rdddot = r_ddot_1.(τ)
    θdddot = θ_ddot_1.(τ)
    ϕdddot = ϕ_ddot_1.(τ)


    # project onto cartesian coordinates in flat space
    x = @. r * sin(θ) * cos(ϕ);   # Eq. 6.3
    y = @. r * sin(θ) * sin(ϕ);   # Eq. 6.4
    z = @. r * cos(θ);   # Eq. 6.5

    # compute various derivatives of x_{p}^{μ} wrt τ
    dx = @. (cos(ϕ) * (sin(θ) * rdot + cos(θ) * r * θdot) - r * sin(θ) * sin(ϕ) * ϕdot) / tdot ;   # Eq. D.1
    d2x = @. (cos(ϕ) * sin(θ) * rddot + 2rdot * (cos(θ) * cos(ϕ) * θdot - sin(θ) * sin(ϕ) * ϕdot) + r * (cos(θ) * (-2sin(ϕ) * θdot * ϕdot + cos(ϕ) * θddot) - sin(θ) * (cos(ϕ) * (θdot^2 + ϕdot^2) + sin(ϕ) * ϕddot))) / (tdot^2);   # Eq. D.2
    d3x = @. (3 * (θddot *  cos(θ) - sin(θ) * θdot^2) * (rdot * cos(ϕ) - r * ϕdot  * sin(ϕ)) - 3θdot * cos(θ) * (cos(ϕ) * (r * ϕdot^2 - rddot) + sin(ϕ) * (2rdot * ϕdot + r * ϕddot)) + sin(θ) * (cos(ϕ) * (rdddot - 3ϕdot * (rdot * ϕdot + r * ϕddot)) + sin(ϕ) * (r * (ϕdot^3 - ϕdddot) - 3 * (rddot * ϕdot + rdot * ϕddot))) + r * cos(ϕ) * ((θdddot - θdot^3) * cos(θ) - 3θdot * θddot * sin(θ))) / (tdot^3);   # Eq. D.3

    dy = @. (sin(ϕ) * (sin(θ) * rdot + cos(θ) * r * θdot) + cos(ϕ) * r * sin(θ) * ϕdot) / tdot;   # Eq. D.4
    d2y = @. (sin(ϕ) * (sin(θ) * (rddot - r * (θdot^2 + ϕdot^2)) + cos(θ) * r * θddot) + 2rdot * (cos(θ) * sin(ϕ) * θdot + cos(ϕ) * sin(θ) * ϕdot) + cos(ϕ) * r * (2cos(θ) * θdot * ϕdot + sin(θ) * ϕddot)) / (tdot^2);   # Eq. D.5
    d3y = @. (3 * (rdot * sin(θ) + r * θdot * cos(θ)) * (ϕddot * cos(ϕ) - sin(ϕ) * ϕdot^2) + 3ϕdot * cos(ϕ) * (rddot * sin(θ) + 2θdot * rdot * cos(θ) + r * (θddot * cos(θ) - sin(θ) * θdot^2)) + sin(ϕ) * (rdddot * sin(θ) + 3rddot * θdot * cos(θ) + 3rdot * (θddot * cos(θ) - sin(θ) * θdot^2) + r * ((θdddot - θdot^3) * cos(θ) - 3θdot * θddot * sin(θ))) + r * sin(θ) * ((ϕdddot - ϕdot^3) * cos(ϕ) - 3ϕdot * ϕddot * sin(ϕ))) / (tdot^3);   # Eq. D.6

    dz = @. (cos(θ) * rdot - r * sin(θ) * θdot) / tdot;   # Eq. D.7
    d2z = @. (cos(θ) * (-r * θdot^2 + rddot) - sin(θ) * (2rdot * θdot + r * θddot)) / (tdot^2);   # Eq. D.8
    d3z = @. (rdddot * cos(θ) - 3rddot * θdot * sin(θ) - 3rdot * (cos(θ) * θdot^2 + θddot * sin(θ)) + r * ((θdot^3 - θdddot) * sin(θ) - 3θdot * θddot * cos(θ))) / (tdot^3);   # Eq. D.9

    xμ = [x, y, z];
    vμ = [dx, dy, dz];
    aμ = [d2x, d2y, d2z];
    jerkμ = [d3x, d3y, d3z];

    # calculate waveform
    hMatrix = h(xμ, vμ, aμ, jerkμ, Θ, Φ, obs_distance, m);   # perturbation matrix

    h_plus = hplus(hMatrix, Θ, Φ);
    h_cross = hcross(hMatrix, Θ, Φ);

    waveform = transpose(stack([t, h_plus, h_cross]))
    open(waveform_filename, "w") do io
        writedlm(io, waveform)
    end
    println("Waveform saved to: " * waveform_filename)
end

# compute semi-relativistic waveform from generic metric geodesic
function waveform(ode_sol_fname::String, waveform_filename::String, Θ::Float64, Φ::Float64, obs_distance::Float64)
    m=1.0;
    # load solution
    sol = readdlm(ode_sol_fname)
    τ = sol[1, :]; t = sol[2, :]; r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :]; tdot = sol[6, :]; rdot = sol[7, :]; θdot = sol[8, :]; ϕdot = sol[9, :];

    # numerically compute acceleration and jerk
    tdot_sp = Spline1D(τ, tdot, bc="nearest")  # use Dierckx
    rdot_sp = Spline1D(τ, rdot, bc="nearest")  # use Dierckx
    θdot_sp = Spline1D(τ, θdot, bc="nearest")  # use Dierckx
    ϕdot_sp = Spline1D(τ, ϕdot, bc="nearest")  # use Dierckx

    # acceleration
    t_dot_1(x) = derivative(tdot_sp, x)
    r_dot_1(x) = derivative(rdot_sp, x)
    θ_dot_1(x) = derivative(θdot_sp, x)
    ϕ_dot_1(x) = derivative(ϕdot_sp, x)

    tddot = t_dot_1.(τ)
    rddot = r_dot_1.(τ)
    θddot = θ_dot_1.(τ)
    ϕddot = ϕ_dot_1.(τ)

    # jerk 
    t_dot_2(x) = derivative(tdot_sp, x, nu=2)
    r_dot_2(x) = derivative(rdot_sp, x, nu=2)
    θ_dot_2(x) = derivative(θdot_sp, x, nu=2)
    ϕ_dot_2(x) = derivative(ϕdot_sp, x, nu=2)

    tdddot = t_dot_2.(τ)
    rdddot = r_dot_2.(τ)
    θdddot = θ_dot_2.(τ)
    ϕdddot = ϕ_dot_2.(τ)

    # project onto cartesian coordinates in flat space
    x = @. r * sin(θ) * cos(ϕ);   # Eq. 6.3
    y = @. r * sin(θ) * sin(ϕ);   # Eq. 6.4
    z = @. r * cos(θ);   # Eq. 6.5

    # compute various derivatives of x_{p}^{μ} wrt τ
    dx = @. (cos(ϕ) * (sin(θ) * rdot + cos(θ) * r * θdot) - r * sin(θ) * sin(ϕ) * ϕdot) / tdot ;   # Eq. D.1
    d2x = @. (cos(ϕ) * sin(θ) * rddot + 2rdot * (cos(θ) * cos(ϕ) * θdot - sin(θ) * sin(ϕ) * ϕdot) + r * (cos(θ) * (-2sin(ϕ) * θdot * ϕdot + cos(ϕ) * θddot) - sin(θ) * (cos(ϕ) * (θdot^2 + ϕdot^2) + sin(ϕ) * ϕddot))) / (tdot^2);   # Eq. D.2
    d3x = @. (3 * (θddot *  cos(θ) - sin(θ) * θdot^2) * (rdot * cos(ϕ) - r * ϕdot  * sin(ϕ)) - 3θdot * cos(θ) * (cos(ϕ) * (r * ϕdot^2 - rddot) + sin(ϕ) * (2rdot * ϕdot + r * ϕddot)) + sin(θ) * (cos(ϕ) * (rdddot - 3ϕdot * (rdot * ϕdot + r * ϕddot)) + sin(ϕ) * (r * (ϕdot^3 - ϕdddot) - 3 * (rddot * ϕdot + rdot * ϕddot))) + r * cos(ϕ) * ((θdddot - θdot^3) * cos(θ) - 3θdot * θddot * sin(θ))) / (tdot^3);   # Eq. D.3

    dy = @. (sin(ϕ) * (sin(θ) * rdot + cos(θ) * r * θdot) + cos(ϕ) * r * sin(θ) * ϕdot) / tdot;   # Eq. D.4
    d2y = @. (sin(ϕ) * (sin(θ) * (rddot - r * (θdot^2 + ϕdot^2)) + cos(θ) * r * θddot) + 2rdot * (cos(θ) * sin(ϕ) * θdot + cos(ϕ) * sin(θ) * ϕdot) + cos(ϕ) * r * (2cos(θ) * θdot * ϕdot + sin(θ) * ϕddot)) / (tdot^2);   # Eq. D.5
    d3y = @. (3 * (rdot * sin(θ) + r * θdot * cos(θ)) * (ϕddot * cos(ϕ) - sin(ϕ) * ϕdot^2) + 3ϕdot * cos(ϕ) * (rddot * sin(θ) + 2θdot * rdot * cos(θ) + r * (θddot * cos(θ) - sin(θ) * θdot^2)) + sin(ϕ) * (rdddot * sin(θ) + 3rddot * θdot * cos(θ) + 3rdot * (θddot * cos(θ) - sin(θ) * θdot^2) + r * ((θdddot - θdot^3) * cos(θ) - 3θdot * θddot * sin(θ))) + r * sin(θ) * ((ϕdddot - ϕdot^3) * cos(ϕ) - 3ϕdot * ϕddot * sin(ϕ))) / (tdot^3);   # Eq. D.6

    dz = @. (cos(θ) * rdot - r * sin(θ) * θdot) / tdot;   # Eq. D.7
    d2z = @. (cos(θ) * (-r * θdot^2 + rddot) - sin(θ) * (2rdot * θdot + r * θddot)) / (tdot^2);   # Eq. D.8
    d3z = @. (rdddot * cos(θ) - 3rddot * θdot * sin(θ) - 3rdot * (cos(θ) * θdot^2 + θddot * sin(θ)) + r * ((θdot^3 - θdddot) * sin(θ) - 3θdot * θddot * cos(θ))) / (tdot^3);   # Eq. D.9

    xμ = [x, y, z];
    vμ = [dx, dy, dz];
    aμ = [d2x, d2y, d2z];
    jerkμ = [d3x, d3y, d3z];

    # calculate waveform
    hMatrix = h(xμ, vμ, aμ, jerkμ, Θ, Φ, obs_distance, m);   # perturbation matrix

    h_plus = hplus(hMatrix, Θ, Φ);
    h_cross = hcross(hMatrix, Θ, Φ);

    waveform = transpose(stack([t, h_plus, h_cross]))
    open(waveform_filename, "w") do io
        writedlm(io, waveform)
    end
    println("Waveform saved to: " * waveform_filename)
end
end

module NewKludge
using DelimitedFiles
using ...HarmonicCoords
using ...SelfForce

# project h into TT gauge using projection in Babak et al (arXiv:gr-qc/0607007v2) Eqs. 23a - 23c
hΘΘ(h, Θ, Φ) = @views (cos(Θ)^2) * (h[1, 1, :] * cos(Φ)^2 + h[1, 2, :] * sin(2Φ) + h[2, 2, :] * sin(Φ)^2) + h[3, 3, :] * sin(Θ)^2 - sin(2Θ) * (h[1, 3, :] * cos(Φ) + h[2, 3, :] * sin(Φ))
hΘΦ(h, Θ, Φ) = @views cos(Θ) * (((-1/2) * h[1, 1, :] * sin(2Φ)) + h[1, 2, :] * cos(2Φ) + (1/2) * h[2, 2, :] * sin(2Φ)) + sin(Θ) * (h[1, 3, :] * sin(Φ) - h[2, 3, :] * cos(Φ))
hΦΦ(h, θ, Φ) = @views h[1, 1, :] * sin(Φ)^2 - h[1, 2, :] * sin(2Φ) + h[2, 2, :] * cos(Φ)^2

# define h_{+} and h_{×} components of GW
hplus(h, Θ, Φ) = (1/2) *  (hΘΘ(h, Θ, Φ) - hΦΦ(h, Θ, Φ))
hcross(h, Θ, Φ) = hΘΦ(h, Θ, Φ)

# compute kerr waveform in the new kludge scheme
function Kerr_waveform(a::Float64, M::Float64, m::Float64, kerr_ode_sol_fname::String, waveform_filename::String, Θ::Float64, Φ::Float64, obs_distance::Float64)
    
    # load solution
    sol = readdlm(kerr_ode_sol_fname)
    t = sol[2, :]; r = sol[3, :]; θ = sol[4, :]; ϕ = sol[5, :]; tdot = sol[6, :]; rdot = sol[7, :]; θdot = sol[8, :]; ϕdot = sol[9, :]; tddot = sol[10, :]; rddot = sol[11, :]; θddot = sol[12, :]; ϕddot = sol[13, :];

    # number of points in the trajectory
    nPoints = length(t)

    # initialize data arrays for waveform computation
    Mijkl2_data = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    Sijk1_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    xBL = [Float64[] for i in 1:nPoints]
    vBL = [Float64[] for i in 1:nPoints]
    aBL = [Float64[] for i in 1:nPoints]
    xH = [Float64[] for i in 1:nPoints]
    x_H = [Float64[] for i in 1:nPoints]
    vH = [Float64[] for i in 1:nPoints]
    v_H = [Float64[] for i in 1:nPoints]
    v = zeros(nPoints)
    rH = zeros(nPoints)
    aH = [Float64[] for i in 1:nPoints]
    a_H = [Float64[] for i in 1:nPoints]
    Mij2 = zeros(3, 3, nPoints)
    Mijk3 = zeros(3, 3, 3, nPoints)
    Mijkl4 = zeros(3, 3, 3, 3, nPoints)
    Sij2 = zeros(3, 3, nPoints)
    Sijk3 = zeros(3, 3, 3, nPoints)
    hij = zeros(3, 3, nPoints)

    # define object that will be iterated over
    n = 1:length(t) |> collect

    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in n
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]) / tdot[i];             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]) / (tdot[i]^2);      # divide by (dt/dτ)² to get accelerations wrt BL time
    end

    @inbounds Threads.@threads for i in n
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
        x_H[i] = xH[i]
        rH[i] = HarmonicCoords.norm_3d(xH[i]);
    end
    @inbounds Threads.@threads for i in n
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        v_H[i] = vH[i]; 
        v[i] = HarmonicCoords.norm_3d(vH[i]);
    end
    @inbounds Threads.@threads for i in n
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
        a_H[i] = aH[i]
    end

    # calculate ddotMijk, ddotMijk, dotSij "analytically" (that is, by using analytial expressions that take as argument the numerical solution of the geodesic equation)
    SelfForce.moments_wf!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data)

    # calculate moment derivatives dddotMijk, ddotSij, ddddotMijkl, and dddotSijk via numerical differentiation
    SelfForce.moment_derivs_wf!(t, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    # calculate metric perturbations (Eq. 84) using moment derivatives
    SelfForce.hij!(hij, nPoints, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    h_plus = NewKludge.hplus(hij, Θ, Φ)
    h_cross = NewKludge.hcross(hij, Θ, Φ);

    waveform = transpose(stack([t, h_plus, h_cross]))

    open(waveform_filename, "w") do io
        writedlm(io, waveform)
    end

    end
end

end