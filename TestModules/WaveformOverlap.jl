#=
    In this module we write code which computes the inner product between two (normalized) waveforms in order to measure how much they "overlap".
    See Babak et al (arXiv:gr-qc/0607007) for further detail
=#

module WaveformOverlap

R(u::Float64) = ((1. + cos(u)^2) * (1.0 / 3.0 - 2.0 / u^2) + sin(u)^2 + 4.0 * sin(u) * cos(u) / u^3) / u^2   # Eq. 38

const u_trans::Float64 = 0.25
const τ::Float64 = 50.0 / 3.0   # light travel time down one of LISA's arms

# LISA noise response (Eq. 36 - 37)
# function S_h(f::Float64)
#     u = 2π * f * τ
#     if u < u_trans
#         return 8.08 * 1e-48 / (2π * f)^4 + 5.52 * 1e-41
#     else
#         return (2.88 * 1e-48 / (2π * f)^4 + 5.52 * 1e-41) / R(u)
#     end
# end

S_h(f::Float64) = 1.

NIntegrate_trapezium_rule(y::Vector{ComplexF64}, dx::Float64, num_points::Int64) = 0.5 * dx * (2 * sum(y) - y[1] - y[num_points])

function NIntegrate_trapezium_rule(y::Vector{ComplexF64}, dx::Vector{Float64}, num_points::Int64)
    sum = 0.0
    @inbounds for i in 1:(num_points-1)
        sum += 0.5 * dx[i] * (y[i] + y[i+1])
    end
    return sum
end

OverLapIntegrand(x::Vector{ComplexF64}, x_conj::Vector{ComplexF64}, h::Vector{ComplexF64}, h_conj::Vector{ComplexF64}, freqs::Vector{Float64}) = @. 2 * (x * h_conj + x_conj * h) / S_h(freqs)
OverLapIntegrandMisMatch(x::Vector{ComplexF64}, x_conj::Vector{ComplexF64}, h::Vector{ComplexF64}, h_conj::Vector{ComplexF64}, freqs::Vector{Float64}, t::Float64) = @. 2 * (x * h_conj * (cos(2π * freqs * t) - im * sin(2π * freqs * t)) + x_conj * h * (cos(2π * freqs * t) + im * sin(2π * freqs * t))) / S_h(freqs)

function compute_overlap(x::Vector{ComplexF64}, h::Vector{ComplexF64}, freqs::Vector{Float64})
    df = diff(freqs)
    num_points = length(x)
    overlap_integrand = OverLapIntegrand(x, conj(x), h, conj(h), freqs)
    return real(NIntegrate_trapezium_rule(overlap_integrand, df, num_points))
end

function compute_overlaps_mismatch(x::Vector{ComplexF64}, h::Vector{ComplexF64}, freqs::Vector{Float64}, t::Vector{Float64})
    overlaps = zeros(length(t))
    df = diff(freqs)
    num_points = length(x)
    @inbounds Threads.@threads for i in eachindex(t)
        overlap_integrand = OverLapIntegrandMisMatch(x, conj(x), h, conj(h), freqs, t[i])
        overlaps[i] = real(NIntegrate_trapezium_rule(overlap_integrand, df, num_points))
    end
    overlap, time_offset_index = findmax(overlaps) 
    return t[time_offset_index], overlap
    # return t, overlaps
end

# OverLapIntegrand(x::Vector{ComplexF64}, conj_h::Vector{ComplexF64}, freqs::Vector{Float64}) = @. 4 * (x * conj_h) / S_h(freqs)
# OverLapIntegrandMisMatch(x::Vector{ComplexF64}, conj_h::Vector{ComplexF64}, freqs::Vector{Float64}, t::Float64) = @. 4 * (x * conj_h) * (cos(2π * freqs * t) + im * sin(2π * freqs * t))/ S_h(freqs)


# function compute_overlap(x::Vector{ComplexF64}, h::Vector{ComplexF64}, freqs::Vector{Float64})
#     df = diff(freqs)
#     num_points = length(x)
#     overlap_integrand = OverLapIntegrand(x, conj(h), freqs)
#     return real(NIntegrate_trapezium_rule(overlap_integrand, df, num_points))
# end

# function compute_overlaps_mismatch(x::Vector{ComplexF64}, h::Vector{ComplexF64}, freqs::Vector{Float64}, t::Vector{Float64})
#     overlaps = zeros(length(t))
#     df = diff(freqs)
#     num_points = length(x)
#     conj_h = conj(h)
#     @inbounds Threads.@threads for i in eachindex(t)
#         overlap_integrand = OverLapIntegrandMisMatch(x, conj_h, freqs, t[i])
#         overlaps[i] = real(NIntegrate_trapezium_rule(overlap_integrand, df, num_points))
#     end
#     return overlaps
# end


end