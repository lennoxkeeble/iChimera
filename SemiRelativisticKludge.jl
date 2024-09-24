module SemiRelativisticKludge
using DelimitedFiles, ..BLDeriv3, ..BLTimeEvolution

module Waveform
# mass quadrupole
function Ijk(x::AbstractVector{Float64}, m::Float64, j::Int64, k::Int64)
    return m * x[j] * x[k]   # Eq. 6.20
end

# first time derivative of mass quadrupole
function dotIjk(x::AbstractVector{Float64}, v::AbstractVector{Float64}, m::Float64, j::Int64, k::Int64)
    return m * (x[k] * v[j] + x[j] * v[k])   # Eq. D.10
end

# second time derivative of mass quadrupole
function ddotIjk(x::AbstractVector{Float64}, v::AbstractVector{Float64}, a::AbstractVector{Float64}, m::Float64, j::Int64, k::Int64)
    return m * (x[k] * a[j] + 2.0 * v[j] * v[k] + x[j] * a[k])   # Eq. D.11
end

# third time derivative of mass quadrupole
function dddotIjk(x::AbstractVector{Float64}, v::AbstractVector{Float64}, a::AbstractVector{Float64}, jerk, m::Float64, j::Int64, k::Int64)
    return m * (jerk[j] * x[k] + 3.0 * a[j] * v[k] + 3.0 * v[j] * a[k] + x[j] * jerk[k])   # Eq. D.12
end

# second time derivative of current quadrupole
function ddotSijk(x::AbstractVector{Float64}, v::AbstractVector{Float64}, a::AbstractVector{Float64}, jerk::AbstractVector{Float64}, m::Float64, i::Int64, j::Int64, k::Int64)
    return Ijk(x, m, j, k) * jerk[i] + 2.0 * dotIjk(x, v, m, j, k) * a[i] + v[i] * ddotIjk(x, v, a, m, j, k)   # Eq. D.13
end

# returns n_{i}\ddot{S}^{ijk}
function ddotSjk(x::AbstractVector{Float64}, v::AbstractVector{Float64}, a::AbstractVector{Float64}, jerk::AbstractVector{Float64}, nx::Float64, ny::Float64, nz::Float64, m::Float64, j::Int64, k::Int64)
    return nx * ddotSijk(x, v, a, jerk, m, 1, j, k) + ny * ddotSijk(x, v, a, jerk, m, 2, j, k) + nz * ddotSijk(x, v, a, jerk, m, 3, j, k)
end

# second time derivative of mass octupole
function dddotMijk(x::AbstractVector{Float64}, v::AbstractVector{Float64}, a::AbstractVector{Float64}, jerk::AbstractVector{Float64}, m::Float64, i::Int64, j::Int64, k::Int64)
    return jerk[i] * Ijk(x, m, j, k) + 3.0 * dotIjk(x, v, m, j, k) * a[i] + 3.0 * v[i] * ddotIjk(x, v, a, m, j, k) + x[i] * dddotIjk(x, v, a, jerk, m, j, k)   # Eq. D.14
end

# returns n_{i}\dddot{M}^{ijk}
function dddotMjk(x::AbstractVector{Float64}, v::AbstractVector{Float64}, a::AbstractVector{Float64}, jerk::AbstractVector{Float64}, nx::Float64, ny::Float64, nz::Float64, m::Float64, j::Int64, k::Int64)
    return nx * dddotMijk(x, v, a, jerk, m, 1, j, k) + ny * dddotMijk(x, v, a, jerk, m, 2, j, k) + nz * dddotMijk(x, v, a, jerk, m, 3, j, k)
end

# compute_metric perturbation
@views function compute_metric_perturbation!(hij::AbstractArray, h_plus::AbstractVector{Float64}, h_cross::AbstractVector{Float64}, x::AbstractArray, v::AbstractArray, a::AbstractArray, jerk::AbstractArray, m::Float64, Θ::Float64, Φ::Float64, obs_distance::Float64)
    nx = sin(Θ) * cos(Φ)   # Eq. D.15
    ny = sin(Θ) * sin(Φ)   # Eq. D.15
    nz = cos(Θ)   # Eq. D.15
    # @inbounds for i=1:3, j=1:3
    #     @inbounds for t in eachindex(x)
    #         hij[i, j][t] = (2 / obs_distance) * (ddotIjk(x[t], v[t], a[t], m, i, j) - 2.0 * ddotSjk(x[t], v[t], a[t], jerk[t], nx, ny, nz, m, i, j) + dddotMjk(x[t], v[t], a[t], jerk[t], nx, ny, nz, m, i, j))   # Eq. 6.9
    #         h_plus[t] = hplus(hij, Θ, Φ, t)
    #         h_cross[t] = hcross(hij, Θ, Φ, t)
    #     end
    # end

    @inbounds Threads.@threads for t in eachindex(x)
        @inbounds for i=1:3, j=1:3
            hij[i, j][t] = (2 / obs_distance) * (ddotIjk(x[t], v[t], a[t], m, i, j) - 2.0 * ddotSjk(x[t], v[t], a[t], jerk[t], nx, ny, nz, m, i, j) + dddotMjk(x[t], v[t], a[t], jerk[t], nx, ny, nz, m, i, j))   # Eq. 6.9
        end
        h_plus[t] = hplus(hij, Θ, Φ, t)
        h_cross[t] = hcross(hij, Θ, Φ, t)
    end
end

# project h into TT gauge
@views hΘΘ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = (cos(Θ)^2) * (h[1, 1][t] * cos(Φ)^2 + h[1, 2][t] * sin(2Φ) + h[2, 2][t] * sin(Φ)^2) + h[3, 3][t] * sin(Θ)^2 - sin(2Θ) * (h[1, 3][t] * cos(Φ) + h[2, 3][t] * sin(Φ))   # Eq. 6.15
@views hΘΦ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = cos(Θ) * (-0.5 * h[1, 1][t] * sin(2Φ) + h[1, 2][t] * cos(2Φ) + 0.5 * h[2, 2][t] * sin(2Φ)) + sin(Θ) * (h[1, 3][t] * sin(Φ) - h[2, 3][t] * cos(Φ))   # Eq. 6.16
@views hΦΦ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = h[1, 1][t] * sin(Φ)^2 - h[1, 2][t] * sin(2Φ) + h[2, 2][t] * cos(Φ)^2   # Eq. 6.17

# define h_{+} and h_{×} components of GW
hplus(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = 0.5 *  (hΘΘ(h, Θ, Φ, t) - hΦΦ(h, Θ, Φ, t))
hcross(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = hΘΦ(h, Θ, Φ, t)

end 

# compute example kerr semi-relativistic waveform
function kerr_waveform(a::Float64, p::Float64, e::Float64, θmin::Float64, t_max_M::Float64, m::Float64, saveat::Float64, Θ::Float64, Φ::Float64, obs_distance::Float64; kerrReltol::Float64=1e-10, kerrAbstol::Float64=1e-10)
    # compute geodesic
    specify_params = false;
    specify_ics = false; 
    num_points_geodesic = Int(ceil(t_max_M / saveat));
    Δti = saveat/10.0;
    t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, psi, chi = BLTimeEvolution.compute_kerr_geodesic(a, p, e, θmin, num_points_geodesic, specify_ics, specify_params, t_max_M, Δti, kerrReltol, kerrAbstol; save_to_file=false)
    
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
        r_dddot_i = BLDeriv3.d3r_dt(a_BL, v_BL, x_BL, a); θ_dddot_i = BLDeriv3.d3θ_dt(a_BL, v_BL, x_BL, a); ϕ_dddot_i = BLDeriv3.d3ϕ_dt(a_BL, v_BL, x_BL, a);

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
    Waveform.compute_metric_perturbation!(hij, h_plus, h_cross, x_cart, v_cart, a_cart, jerk_cart, m, Θ, Φ, obs_distance)
    return t, h_plus, h_cross
end

module Fluxes
# in this module we write the fluxes \dot{E}, \dot{L}, dot{Q} in Eqs. 44, 45, and 56 in PhysRevD.73.064037. We set M = 1 so that q = a.

# define g functions
function g1(e::Float64)::Float64
    return 1.0 + (73.0/24.0)*e^2 + (37.0/96.0)*e^4
end

function g2(e::Float64)::Float64
    return (73.0/12.0) + (823.0/24.0)*e^2 + (949.0/32.0)*e^4 + (491.0/192.0)*e^6
end

function g3(e::Float64)::Float64
    return (1247.0/336.0) + (9181.0/672.0)*e^2
end

function g4(e::Float64)::Float64
    return 4.0 + (1375.0/48.0)*e^2
end

function g5(e::Float64)::Float64
    return (44711.0/9072.0) + (172157.0/2592.0)*e^2
end

function g6(e::Float64)::Float64
    return (33.0/16.0) + (359.0/32.0)*e^2
end

function g7(e::Float64)::Float64
    return (8191.0/672.0) + (44531.0/336.0)*e^2
end

function g8(e::Float64)::Float64
    return (3749.0/336.0) - (5143.0/168.0)*e^2
end

function g9(e::Float64)::Float64
    return 1.0 + (7.0/8.0)*e^2
end

function g10a(e::Float64)::Float64
    return (61.0/24.0) + (63.0/8.0) * e^2 + (95.0/64.0) * e^4
end

function g10b(e::Float64)::Float64
    return (61.0/8.0) + (91.0/4.0) * e^2 + (461.0/64.0) * e^4
end

function g11(e::Float64)::Float64
    return (1247.0/336.0) + (425.0/336.0)*e^2
end

function g12(e::Float64)::Float64
    return 4.0 + (97.0/8.0)*e^2
end

function g13(e::Float64)::Float64
    return (44711.0/9072.0) + (302893.0/6048.0)*e^2
end

function g14(e::Float64)::Float64
    return (33.0/16.0) + (95.0/16.0)*e^2
end

function g15(e::Float64)::Float64
    return (8191.0/672.0) + (48361.0/1344.0)*e^2
end

function g16(e::Float64)::Float64
    return (417.0/56.0) - (37241.0/672.0)*e^2
end

# define \dot{E}
function E_dot(a::Float64, p::Float64, e::Float64, ι::Float64, mu::Float64)::Float64
    term1 = -(32.0/5.0)*(mu^2)*(1.0/p)^5*(1.0 - e^2)^(3/2)
    term2 = g1(e) - a*(1/p)^(3/2)*cos(ι)*g2(e) - (1.0/p)*g3(e) + π*(1.0/p)^(3/2)*g4(e)
    term3 = -(1.0/p)^2*g5(e) + a^2*(1.0/p)^2*g6(e)
    term4 = -(527.0/96.0)*a^2*(1.0/p)^2*sin(ι)^2
    
    return term1 * (term2 + term3 + term4)
end

# define \dot{L_z}
function L_dot(a::Float64, p::Float64, e::Float64, ι::Float64, mu::Float64)::Float64
    term1 = -(32.0/5.0)*(mu^2)*(1.0/p)^(7/2)*(1.0 - e^2)^(3/2)
    term2 = g9(e)*cos(ι) + a*(1.0/p)^(3/2)*(g10a(e) - cos(ι)^2*g10b(e)) - (1.0/p)*g11(e)*cos(ι)
    term3 = π*(1.0/p)^(3/2)*g12(e)*cos(ι) - (1.0/p)^2*g13(e)*cos(ι)
    term4 = a^2*(1.0/p)^2*cos(ι)*(g14(e) - (45.0/8.0)*sin(ι)^2)
    
    return term1 * (term2 + term3 + term4)
end

# define \dot{Q}
function Q_dot(a::Float64, p::Float64, e::Float64, ι::Float64, mu::Float64, Q::Float64)::Float64
    term1 = -(64.0/5.0)*(mu^2)*(1.0/p)^(7/2)*sqrt(Q)*sin(ι)*(1.0 - e^2)^(3/2)
    term2 = g9(e) - a*(1.0/p)^(3/2)*cos(ι)*g10b(e) - (1.0/p)*g11(e)
    term3 = π*(1.0/p)^(3/2)*g12(e) - (1.0/p)^2*g13(e)
    term4 = a^2*(1.0/p)^2*(g14(e) - (45.0/8.0)*sin(ι)^2)
    
    return  term1 * (term2 + term3 + term4)
end

end

module Inspiral
using LinearAlgebra
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using ...Kerr
using ...BLTimeEvolution
using ..Fluxes
using JLD2
using FileIO
using ..Waveform
using ..BLDeriv3
using Roots

Z_1(a::Float64, M::Float64) = 1 + (1 - a^2 / M^2)^(1/3) * ((1 + a / M)^(1/3) + (1 - a / M)^(1/3))
Z_2(a::Float64, M::Float64) = sqrt(3 * (a / M)^2 + Z_1(a, M)^2)
LSO_r(a::Float64, M::Float64) = M * (3 + Z_2(a, M) - sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # retrograde LSO
LSO_p(a::Float64, M::Float64) = M * (3 + Z_2(a, M) + sqrt((3 - Z_1(a, M)) * (3 + Z_1(a, M) * 2 * Z_2(a, M))))   # prograde LSO

# coefficients of polynomial in E, L (Eq. E3)
αI(a::Float64, M::Float64, rI::Float64, zm::Float64) = (rI^2 + a^2) * (rI^2 + a^2 * zm) + 2.0M * rI * a^2 * (1.0 - zm)    # Eq. E4
βI(a::Float64, M::Float64, rI::Float64, zm::Float64) = - 2.0M * rI * a    # Eq. E5
γI(a::Float64, M::Float64, rI::Float64, zm::Float64) = -(1.0 / (1.0 - zm)) * (rI^2 + a^2 * zm - 2.0 * M * rI)    # Eq. E6
λI(a::Float64, M::Float64, rI::Float64, zm::Float64) = -(rI^2 + a^2 * zm) * (rI^2 - 2.0M * rI + a^2)    # Eq. E7

# for circular orbits
α_2(a::Float64, M::Float64, r0::Float64, zm::Float64) = 2.0r0 * (r0^2 + a^2) - a^2 * (r0 - M) * (1.0 - zm)    # Eq. E8
β_2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -a * M    # Eq. E9
γ_2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -(r0 - M) / (1.0 - zm)    # Eq. E10
λ_2(a::Float64, M::Float64, r0::Float64, zm::Float64) = -r0 * (r0^2 - 2.0M * r0 + a^2) - (r0 - M) * (r0^2 + a^2 * zm)    # Eq. E11

# define [*, *] operation in Eq. E3
commute(Πa::Float64, Πp::Float64, Ωa::Float64, Ωp::Float64) = Πa * Ωp - Πp * Ωa

# test prograde / retrograde orbits
function compute_ELQ(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64)
    M=1.0;
    
    ### COMPUTE ELQ ###
    zm = cos(θmin)^2
    if e==0.0
        r0 = p * M
        α1 = αI(a, M, r0, zm)
        α2 = α_2(a, M, r0, zm)
        β1 = βI(a, M, r0, zm)
        β2 = β_2(a, M, r0, zm)
        γ1 = γI(a, M, r0, zm)
        γ2 = γ_2(a, M, r0, zm)
        λ1 = λI(a, M, r0, zm)
        λ2 = λ_2(a, M, r0, zm)
    else
        rp = p * M / (1 + e)
        ra = p * M / (1 - e)
        α1 = αI(a, M, ra, zm)
        α2 = αI(a, M, rp, zm)
        β1 = βI(a, M, ra, zm)
        β2 = βI(a, M, rp, zm)
        γ1 = γI(a, M, ra, zm)
        γ2 = γI(a, M, rp, zm)
        λ1 = λI(a, M, ra, zm)
        λ2 = λI(a, M, rp, zm)
    end
    
    # write out coefficients of Eq. E12 in the form ax^2 + bx + c
    aa = (commute(α1, α2, γ1, γ2)^2 + 4.0 * commute(α1, α2, β1, β2) * commute(γ1, γ2, β1, β2))
    b = 2.0 * (commute(α1, α2, γ1, γ2) * commute(λ1, λ2, γ1, γ2) + 2.0 * commute(γ1, γ2, β1, β2) * commute(λ1, λ2, β1, β2))
    c = commute(λ1, λ2, γ1, γ2)^2

    # prograde
    if sign_Lz>0
        # prograge energy (Eq. E12) - retrograde is other root
        E = sqrt((-b - sqrt(b^2 - 4.0aa * c))/ 2.0aa)
        # prograde z-component of angular momentum (Eq. E14) - retrograde is negative root
        L = sqrt((commute(α1, α2, β1, β2) * E^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))
    else
        # retrograde
        E = sqrt((-b + sqrt(b^2 - 4.0aa * c))/ 2.0aa)
        L = -sqrt((commute(α1, α2, β1, β2) * E^2 + commute(λ1, λ2, β1, β2)) / commute(β1, β2, γ1, γ2))
    end


    if θmin==0.0
        C = 0.0
    else
        C = zm * (L^2 / (1.0 - zm) + a^2 * (1.0 - E^2))    # Eq. E2
    end

    Q = C + (L - a * E)^2    # Eq. 17
    
    return E, L, Q, C
end

function compute_iota(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64)
    E, L, Q, C = compute_ELQ(a, p, e, θmin, sign_Lz)
    return acos(L / sqrt(L^2 + C))
end

# compute iota corresponding to θmin
function iota_to_theta_min(a::Float64, p::Float64, e::Float64, ι::Float64)
    if ι < 0. || ι > π
        throw(DomainError("ι must be in the range [0, π]"))
    else
        sign_Lz = ι < π/2 ? +1 : -1;
        iota_theta(θmin::Float64) = compute_iota(a, p, e, θmin, sign_Lz) - ι
        θmin = find_zeros(iota_theta, 0.001, π/2-0.001)
    end
    return length(θmin) == 1 ? θmin[1] : throw(DomainError())
end


function EvolveConstants(Δt::Float64, a::Float64, EE::AbstractArray, Edot::AbstractArray, LL::AbstractArray, Ldot::AbstractArray, QQ::AbstractArray, CC::AbstractArray, Cdot::AbstractArray, pArray::AbstractArray, ecc::AbstractArray, θmin::AbstractArray, iota::AbstractArray, mu::Float64, nPoints::Int64)
    M=1.0
    # first load orbital constants of previous geodesic (recall that we compute updated constants to move to the next geodesic in the inspiral)
    E0 = last(EE); L0 = last(LL); C0 = last(CC); p0 = last(pArray); e0 = last(ecc); ι0 = last(iota);
    
    #### update E, L, Q, C ####
    # update E
    dE_dt = Fluxes.E_dot(a, p0, e0, ι0, mu) / mu
    push!(Edot, dE_dt)

    # update L
    dL_dt = Fluxes.L_dot(a, p0, e0, ι0, mu) / mu
    push!(Ldot, dL_dt)

    # update C
    dC_dt = Fluxes.Q_dot(a, p0, e0, ι0, mu, C0) / mu   # function is "Qdot" due to differing convention
    push!(Cdot, dC_dt)

    # compute updated E, L, Q, C and store
    E1 = E0 + dE_dt * Δt
    L1 = L0 + dL_dt * Δt
    C1 = C0 + dC_dt * Δt
    Q1 = C1 + (L1 - a * E1)^2   # Eq. 17 in arxiv 1109.0572

    push!(EE, E1)
    push!(LL, L1)
    push!(QQ, Q1)
    push!(CC, C1)

    #### update p, e, θmin ####
    pp, ee, θθ = Kerr.ConstantsOfMotion.peθ_gsl(a, E1, L1, Q1, C1, M)
    ιι = acos(L1 / sqrt(L1^2 + C1))
    # preserve circularity and/or equatorial orbit
    if e0 == 0.0
        ee = 0.0
    end

    if ι0 == 0.0
        ιι = 0.0
    end


    push!(pArray, pp)
    push!(ecc, ee)
    push!(θmin, θθ)
    push!(iota, ιι)
end

# for consistency with the chimera code, we write Babak et al's "Q" as Yunes & Sopuerta's "C"
function compute_inspiral_HJE!(a::Float64, p::Float64, e::Float64, ι::Float64,  mu::Float64, tOrbit::Float64, dt_Fluxes::Float64, nPointsGeodesic::Int64, reltol::Float64=1e-10, abstol::Float64=1e-10; data_path::String="Data/")
    M=1.;
    # create arrays for trajectory
    t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[];


    # calculate integrals of motion from orbital parameters
    θi = iota_to_theta_min(a, p, e, ι); sign_Lz = ι < π/2 ? +1 : -1;
    EEi, LLi, QQi, CCi = compute_ELQ(a, p, e, θi, sign_Lz)  

    # store orbital params in arrays
    EE = ones(1) * EEi; 
    Edot = zeros(1);
    LL = ones(1) * LLi; 
    Ldot = zeros(1);
    CC = ones(1) * CCi
    Cdot = zeros(1);
    QQ = ones(1) * QQi
    pArray = ones(1) * p;
    ecc = ones(1) * e;
    iota = ones(1) * ι;
    θmin = ones(1) * θi;
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0

    # time of flux computations
    t_Fluxes = ones(1) * t0

    # compute apastron for initial conditions
    ra = p * M / (1 - e);
    geodesic_ics = BLTimeEvolution.HJ_ics(ra, p, e, M);

    rLSO = LSO_p(a, M)

    use_custom_ics = true; use_specified_params = true;
    save_at_trajectory = dt_Fluxes / (nPointsGeodesic - 1); Δti=save_at_trajectory;    # initial time step for geodesic integration

    # in the code, we will want to compute the geodesic with an additional time step at the end so that these coordinate values can be used as initial conditions for the
    # subsequent geodesic
    geodesic_time_length = dt_Fluxes + save_at_trajectory;
    num_points_geodesic = nPointsGeodesic + 1;


    while tOrbit > t0
        print("Completion: $(round(100 * t0/tOrbit; digits=5))%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # orbital parameters
        E_t = last(EE); L_t = last(LL); Q_t = last(QQ); C_t = last(CC); p_t = last(pArray); θmin_t = last(θmin); e_t = last(ecc);  

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t * M / (1.0 - e_t); rp=p_t * M / (1.0 + e_t);
        A = M / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t) / M; p4 = r4 * (1.0 + e_t) / M    # Above Eq. 96

        # geodesic
        tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi = BLTimeEvolution.compute_kerr_geodesic(a, p_t, e_t, θmin_t, num_points_geodesic, use_custom_ics, use_specified_params, geodesic_time_length, Δti, reltol, abstol;
        ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)

        tt = tt .+ t0   # tt from the above function call starts from zero

        # check that geodesic output is as expected
        if (length(tt) != num_points_geodesic) || !isapprox(tt[nPointsGeodesic], t0 + dt_Fluxes)
            println("Integration terminated at t = $(last(t))")
            println("total_num_points - len(sol) = $(num_points_geodesic-length(tt))")
            println("tt[nPointsGeodesic] = $(tt[nPointsGeodesic])")
            println("t0 + dt_Fluxes = $(t0 + dt_Fluxes)")
            break
        end

        # extract initial conditions for next geodesic, then remove these points from the data array
        t0 = last(tt); geodesic_ics = @SArray [last(psi), last(chi), last(ϕϕ)];
        pop!(tt); pop!(rr); pop!(θθ); pop!(ϕϕ); pop!(r_dot); pop!(θ_dot); pop!(ϕ_dot);
        pop!(r_ddot); pop!(θ_ddot); pop!(ϕ_ddot); pop!(Γ); pop!(psi); pop!(chi);

        # store physical trajectory
        append!(t, tt); append!(dt_dτ, Γ); append!(r, rr); append!(dr_dt, r_dot); append!(d2r_dt2, r_ddot); 
        append!(θ, θθ); append!(dθ_dt, θ_dot); append!(d2θ_dt2, θ_ddot); append!(ϕ, ϕϕ); 
        append!(dϕ_dt, ϕ_dot); append!(d2ϕ_dt2, ϕ_ddot);

        # evolve orbital parameters using fluxes
        EvolveConstants(dt_Fluxes, a, EE, Edot, LL, Ldot, QQ, CC, Cdot, pArray, ecc, θmin, iota, mu, nPointsGeodesic)
        push!(t_Fluxes, last(tt))
    end

    print("Completion: 100%")

    # delete final "extra" energies and fluxes
    pop!(EE)
    pop!(LL)
    pop!(QQ)
    pop!(CC)
    pop!(pArray)
    pop!(ecc)
    pop!(θmin)
    pop!(iota)
    pop!(Edot)
    pop!(Ldot)
    pop!(Cdot)
    pop!(t_Fluxes)

    # save data 
    mkpath(data_path)

    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_iota0_$(round(ι; digits=3))_q_$(mu/M)_tol_$(reltol)_nGeo_$(nPointsGeodesic)_Numerical_Kludge.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save params
    constants = (t_Fluxes, EE, LL, QQ, CC, pArray, ecc, iota, θmin)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_iota0_$(round(ι; digits=3))_q_$(mu/M)_tol_$(reltol)_nGeo_$(nPointsGeodesic)_Numerical_Kludge.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_iota0_$(round(ι; digits=3))_q_$(mu/M)_tol_$(reltol)_nGeo_$(nPointsGeodesic)_Numerical_Kludge.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

function load_trajectory(a::Float64, p::Float64, e::Float64, ι::Float64, q::Float64, nPointsGeodesic::Int64, reltol::Float64, data_path::String)
    # load ODE solution
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_iota0_$(round(ι; digits=3))_q_$(q)_tol_$(reltol)_nGeo_$(nPointsGeodesic)_Numerical_Kludge.txt"
    sol = readdlm(ODE_filename)
    t=sol[1,:]; r=sol[2,:]; θ=sol[3,:]; ϕ=sol[4,:]; dr_dt=sol[5,:]; dθ_dt=sol[6,:]; dϕ_dt=sol[7,:]; d2r_dt2=sol[8,:]; d2θ_dt2=sol[9,:]; d2ϕ_dt2=sol[10,:]; dt_dτ=sol[11,:]
    return t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ
end

@views function load_constants_of_motion(a::Float64, p::Float64, e::Float64, ι::Float64, q::Float64, nPointsGeodesic::Int64, reltol::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_iota0_$(round(ι; digits=3))_q_$(q)_tol_$(reltol)_nGeo_$(nPointsGeodesic)_Numerical_Kludge.txt"
    constants=readdlm(constants_filename)
    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_iota0_$(round(ι; digits=3))_q_$(q)_tol_$(reltol)_nGeo_$(nPointsGeodesic)_Numerical_Kludge.txt"
    constants_derivs = readdlm(constants_derivs_filename)
    t_Fluxes, EE, LL, QQ, CC, pArray, ecc, ι, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :], constants[8, :], constants[9, :]
    Edot, Ldot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :]
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, CC, Cdot, pArray, ecc, ι, θmin
end

# compute inspiral waveform
function compute_inspiral_waveform(a::Float64, p::Float64, e::Float64, ι::Float64,  mu::Float64, nPointsGeodesic::Int64, Θ::Float64, Φ::Float64, obs_distance::Float64, kerrReltol::Float64=1e-10, data_path::String="Data/")
    
    # load inspiral waveform
    t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ = load_trajectory(a, p, e, ι, mu, nPointsGeodesic, kerrReltol, data_path)

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
        r_dddot_i = BLDeriv3.d3r_dt(a_BL, v_BL, x_BL, a); θ_dddot_i = BLDeriv3.d3θ_dt(a_BL, v_BL, x_BL, a); ϕ_dddot_i = BLDeriv3.d3ϕ_dt(a_BL, v_BL, x_BL, a);

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
    Waveform.compute_metric_perturbation!(hij, h_plus, h_cross, x_cart, v_cart, a_cart, jerk_cart, mu, Θ, Φ, obs_distance)
    return t, h_plus, h_cross
end

end

end