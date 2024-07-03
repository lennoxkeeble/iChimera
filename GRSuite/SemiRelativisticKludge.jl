module SemiRelativisticKludge
using DelimitedFiles, ..Deriv3, ..HJEvolution

# mass quadrupole
function Ijk(x::Vector{Float64}, m::Float64, j::Int64, k::Int64)
    return m * x[j] * x[k]   # Eq. 6.20
end

# first time derivative of mass quadrupole
function dotIjk(x::Vector{Float64}, v::Vector{Float64}, m::Float64, j::Int64, k::Int64)
    return m * (x[k] * v[j] + x[j] * v[k])   # Eq. D.10
end

# second time derivative of mass quadrupole
function ddotIjk(x::Vector{Float64}, v::Vector{Float64}, a::Vector{Float64}, m::Float64, j::Int64, k::Int64)
    return m * (x[k] * a[j] + 2.0 * v[j] * v[k] + x[j] * a[k])   # Eq. D.11
end

# third time derivative of mass quadrupole
function dddotIjk(x::Vector{Float64}, v::Vector{Float64}, a::Vector{Float64}, jerk, m::Float64, j::Int64, k::Int64)
    return m * (jerk[j] * x[k] + 3.0 * a[j] * v[k] + 3.0 * v[j] * a[k] + x[j] * jerk[k])   # Eq. D.12
end

# second time derivative of current quadrupole
function ddotSijk(x::Vector{Float64}, v::Vector{Float64}, a::Vector{Float64}, jerk::Vector{Float64}, m::Float64, i::Int64, j::Int64, k::Int64)
    return Ijk(x, m, j, k) * jerk[i] + 2.0 * dotIjk(x, v, m, j, k) * a[i] + v[i] * ddotIjk(x, v, a, m, j, k)   # Eq. D.13
end

# returns n_{i}\ddot{S}^{ijk}
function ddotSjk(x::Vector{Float64}, v::Vector{Float64}, a::Vector{Float64}, jerk::Vector{Float64}, nx::Float64, ny::Float64, nz::Float64, m::Float64, j::Int64, k::Int64)
    return nx * ddotSijk(x, v, a, jerk, m, 1, j, k) + ny * ddotSijk(x, v, a, jerk, m, 2, j, k) + nz * ddotSijk(x, v, a, jerk, m, 3, j, k)
end

# second time derivative of mass octupole
function dddotMijk(x::Vector{Float64}, v::Vector{Float64}, a::Vector{Float64}, jerk::Vector{Float64}, m::Float64, i::Int64, j::Int64, k::Int64)
    return jerk[i] * Ijk(x, m, j, k) + 3.0 * dotIjk(x, v, m, j, k) * a[i] + 3.0 * v[i] * ddotIjk(x, v, a, m, j, k) + x[i] * dddotIjk(x, v, a, jerk, m, j, k)   # Eq. D.14
end

# returns n_{i}\dddot{M}^{ijk}
function dddotMjk(x::Vector{Float64}, v::Vector{Float64}, a::Vector{Float64}, jerk::Vector{Float64}, nx::Float64, ny::Float64, nz::Float64, m::Float64, j::Int64, k::Int64)
    return nx * dddotMijk(x, v, a, jerk, m, 1, j, k) + ny * dddotMijk(x, v, a, jerk, m, 2, j, k) + nz * dddotMijk(x, v, a, jerk, m, 3, j, k)
end

# compute_metric perturbation
@views function compute_metric_perturbation!(hij::AbstractArray, h_plus::Vector{Float64}, h_cross::Vector{Float64}, x::AbstractArray, v::AbstractArray, a::AbstractArray, jerk::AbstractArray, m::Float64, Θ::Float64, Φ::Float64, obs_distance::Float64)
    nx = sin(Θ) * cos(Φ)   # Eq. D.15
    ny = sin(Θ) * sin(Φ)   # Eq. D.15
    nz = cos(Θ)   # Eq. D.15

    @inbounds for i=1:3, j=1:3
        @inbounds for t in eachindex(x)
            hij[i, j][t] = (2 / obs_distance) * (ddotIjk(x[t], v[t], a[t], m, i, j) - 2.0 * ddotSjk(x[t], v[t], a[t], jerk[t], nx, ny, nz, m, i, j) + dddotMjk(x[t], v[t], a[t], jerk[t], nx, ny, nz, m, i, j))   # Eq. 6.9
            h_plus[t] = hplus(hij, Θ, Φ, t)
            h_cross[t] = hcross(hij, Θ, Φ, t)
        end
    end
end

# project h into TT gauge
@views hΘΘ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = (cos(Θ)^2) * (h[1, 1][t] * cos(Φ)^2 + h[1, 2][t] * sin(2Φ) + h[2, 2][t] * sin(Φ)^2) + h[3, 3][t] * sin(Θ)^2 - sin(2Θ) * (h[1, 3][t] * cos(Φ) + h[2, 3][t] * sin(Φ))   # Eq. 6.15
@views hΘΦ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = cos(Θ) * (-0.5 * h[1, 1][t] * sin(2Φ) + h[1, 2][t] * cos(2Φ) + 0.5 * h[2, 2][t] * sin(2Φ)) + sin(Θ) * (h[1, 3][t] * sin(Φ) - h[2, 3][t] * cos(Φ))   # Eq. 6.16
@views hΦΦ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = h[1, 1][t] * sin(Φ)^2 - h[1, 2][t] * sin(2Φ) + h[2, 2][t] * cos(Φ)^2   # Eq. 6.17

# define h_{+} and h_{×} components of GW
hplus(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = 0.5 *  (hΘΘ(h, Θ, Φ, t) - hΦΦ(h, Θ, Φ, t))
hcross(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = hΘΦ(h, Θ, Φ, t)

# compute kerr semi-relativistic waveform
function kerr_waveform(a::Float64, p::Float64, e::Float64, θmin::Float64, t_max_M::Float64, m::Float64, saveat::Float64, Θ::Float64, Φ::Float64, obs_distance::Float64; kerrReltol::Float64=1e-10, kerrAbstol::Float64=1e-10)
    # compute geodesic
    specify_params = false;
    specify_ics = false; 
    num_points_geodesic = Int(ceil(t_max_M / saveat));
    Δti = saveat/10.0;
    sol = HJEvolution.compute_kerr_geodesic(a, p, e, θmin, num_points_geodesic, specify_ics, specify_params, t_max_M, Δti, kerrReltol, kerrAbstol; save_to_file=false, inspiral = false)
    
    # load solution
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; dr_dt=sol[5,:];
    dθ_dt=sol[6,:]; dϕ_dt=sol[7,:]; d2r_dt2=sol[8,:]; d2θ_dt2=sol[9,:]; d2ϕ_dt2=sol[10,:];

    total_num_points = length(t)
    # initialize BL and cartesian data arrays
    x_cart = [Float64[] for i in 1:total_num_points];
    v_cart = [Float64[] for i in 1:total_num_points];
    a_cart = [Float64[] for i in 1:total_num_points];
    jerk_cart = [Float64[] for i in 1:total_num_points];

    # initialize waveform arrarys
    hij = [zeros(total_num_points) for i=1:3, j=1:3];
    h_plus = zeros(total_num_points);
    h_cross = zeros(total_num_points);

    @inbounds for i in 1:total_num_points
        r_i = r[i]; θ_i = θ[i]; ϕ_i = ϕ[i];
        r_dot_i = dr_dt[i]; θ_dot_i = dθ_dt[i]; ϕ_dot_i = dϕ_dt[i];
        r_ddot_i = d2r_dt2[i]; θ_ddot_i = d2θ_dt2[i]; ϕ_ddot_i = d2ϕ_dt2[i];

        # Boyer-Lindquist
        x_BL = [r_i, θ_i, ϕ_i];
        v_BL = [r_dot_i, θ_dot_i, ϕ_dot_i];
        a_BL = [r_ddot_i, θ_ddot_i, ϕ_ddot_i];
        r_dddot_i = Deriv3.d3r_dt(a_BL, v_BL, x_BL, a); θ_dddot_i = Deriv3.d3θ_dt(a_BL, v_BL, x_BL, a); ϕ_dddot_i = Deriv3.d3ϕ_dt(a_BL, v_BL, x_BL, a);

        # project onto cartesian coordinates in flat space
        sinθ = sin(θ_i); cosθ = cos(θ_i);
        sinϕ = sin(ϕ_i); cosϕ = cos(ϕ_i);

        x = r_i * sinθ * cosϕ;   # Eq. 6.3
        y = r_i * sinθ * sinϕ;   # Eq. 6.4
        z = r_i * cosθ;   # Eq. 6.5

        # compute various derivatives of cartesian coordinates x, y, z wrt t
        dx = cosϕ * (sinθ * r_dot_i + cosθ * r_i * θ_dot_i) - r_i * sinθ * sinϕ * ϕ_dot_i;   # Eq. D.1
        d2x = cosϕ * sinθ * r_ddot_i + 2.0 * r_dot_i * (cosθ * cosϕ * θ_dot_i - sinθ * sinϕ * ϕ_dot_i) + r_i * (cosθ * (-2.0 * sinϕ * θ_dot_i * ϕ_dot_i + cosϕ * θ_ddot_i) - sinθ * (cosϕ * (θ_dot_i^2 + ϕ_dot_i^2) + sinϕ * ϕ_ddot_i));   # Eq. D.2
        d3x = 3.0 * (θ_ddot_i * cosθ - sinθ * θ_dot_i^2) * (r_dot_i * cosϕ - r_i * ϕ_dot_i  * sinϕ) - 3.0 * θ_dot_i * cosθ * (cosϕ * (r_i * ϕ_dot_i^2 - r_ddot_i) + sinϕ * (2.0 * r_dot_i * ϕ_dot_i + r_i * ϕ_ddot_i)) +
            sinθ * (cosϕ * (r_dddot_i - 3.0 * ϕ_dot_i * (r_dot_i * ϕ_dot_i + r_i * ϕ_ddot_i)) + sinϕ * (r_i * (ϕ_dot_i^3 - ϕ_dddot_i) - 3.0 * (r_ddot_i * ϕ_dot_i + r_dot_i * ϕ_ddot_i))) + r_i * cosϕ * ((θ_dddot_i - θ_dot_i^3) * cosθ - 3.0 * θ_dot_i * θ_ddot_i * sinθ);   # Eq. D.3

        dy = sinϕ * (sinθ * r_dot_i + cosθ * r_i * θ_dot_i) + cosϕ * r_i * sinθ * ϕ_dot_i;   # Eq. D.4
        d2y = sinϕ * (sinθ * (r_ddot_i - r_i * (θ_dot_i^2 + ϕ_dot_i^2)) + cosθ * r_i * θ_ddot_i) + 2.0 * r_dot_i * (cosθ * sinϕ * θ_dot_i + cosϕ * sinθ * ϕ_dot_i) + cosϕ * r_i * (2.0 * cosθ * θ_dot_i * ϕ_dot_i + sinθ * ϕ_ddot_i);   # Eq. D.5
        d3y = 3.0 * (r_dot_i * sinθ + r_i * θ_dot_i * cosθ) * (ϕ_ddot_i * cosϕ - sinϕ * ϕ_dot_i^2) + 3.0 * ϕ_dot_i * cosϕ * (r_ddot_i * sinθ + 2.0 * θ_dot_i * r_dot_i * cosθ + r_i * (θ_ddot_i * cosθ - sinθ * θ_dot_i^2)) + 
            sinϕ * (r_dddot_i * sinθ + 3.0 * r_ddot_i * θ_dot_i * cosθ + 3.0 * r_dot_i * (θ_ddot_i * cosθ - sinθ * θ_dot_i^2) + r_i * ((θ_dddot_i - θ_dot_i^3) * cosθ - 3.0 * θ_dot_i * θ_ddot_i * sinθ)) + r_i * sinθ * ((ϕ_dddot_i - ϕ_dot_i^3) * cosϕ - 3.0 * ϕ_dot_i * ϕ_ddot_i * sinϕ);   # Eq. D.6

        dz = cosθ * r_dot_i - r_i * sinθ * θ_dot_i;   # Eq. D.7
        d2z = cosθ * (-r_i * θ_dot_i^2 + r_ddot_i) - sinθ * (2.0 * r_dot_i * θ_dot_i + r_i * θ_ddot_i);   # Eq. D.8
        d3z = r_dddot_i * cosθ - 3.0 * r_ddot_i * θ_dot_i * sinθ - 3.0 * r_dot_i * (cosθ * θ_dot_i^2 + θ_ddot_i * sinθ) + r_i * ((θ_dot_i^3 - θ_dddot_i) * sinθ - 3.0 * θ_dot_i * θ_ddot_i * cosθ);   # Eq. D.9

        x_cart[i] = [x, y, z];
        v_cart[i] = [dx, dy, dz];
        a_cart[i] = [d2x, d2y, d2z];
        jerk_cart[i] = [d3x, d3y, d3z];
    end

    # calculate waveform
    compute_metric_perturbation!(hij, h_plus, h_cross, x_cart, v_cart, a_cart, jerk_cart, m, Θ, Φ, obs_distance)
    return t, h_plus, h_cross
end

end