#=

    In this module we write the metric perturbation in the TT gauge. See Eq. 84 in arXiv:1109.0572v2

=#

module Waveform
using Combinatorics

const spatial_indices_3::Array = [[x, y, z] for x=1:3, y=1:3, z=1:3]
const εkl::Array{Vector} = [[levicivita(spatial_indices_3[k, l, i]) for i = 1:3] for k=1:3, l=1:3]

# returns hij array at some time t specified as an index (rather than a time in seconds)
function hij!(hij::AbstractArray, nPoints::Int, r::Float64, Θ::Float64, Φ::Float64, Mij2::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray)
    # n ≡ unit vector pointing in direction of far away observer
    nx = sin(Θ) * cos(Φ)
    ny = sin(Θ) * sin(Φ)
    nz = cos(Θ)
    n = [nx, ny, nz]

    # calculate perturbations in TT gauge (Eq. 84)
    # @inbounds Threads.@threads for t=1:nPoints
    @inbounds for t=1:nPoints
        for i=1:3
            @inbounds for j=1:3

                @views hij[i, j][t] = 0.    # set all entries to zero

                @views hij[i, j][t] += 2.0 * Mij2[i, j][t] / r    # first term in Eq. 84 

                @inbounds for k=1:3
                    @views hij[i, j][t] += 2.0 * Mijk3[i, j, k][t] * n[k] / (3.0r)    # second term in Eq. 84
    
                    @inbounds for l=1:3
                        @views hij[i, j][t] += 4.0 * (εkl[k, l][i] * Sij2[j, k][t] * n[l] + εkl[k, l][j] * Sij2[i, k][t] * n[l]) / (3.0r) + Mijkl4[i, j, k, l][t] * n[k] * n[l] / (6.0r)    # third and fourth terms in Eq. 84
        
                        @inbounds for m=1:3
                            @views hij[i, j][t] += (εkl[k, l][i] * Sijk3[j, k, m][t] * n[l] * n[m] + εkl[k, l][j] * Sijk3[i, k, m][t] * n[l] * n[m]) / (2.0r)
                        end
                    end
                end
            end
        end
    end
end


# project h into TT gauge (Reference: https://arxiv.org/pdf/gr-qc/0607007)
hΘΘ(h::AbstractArray, Θ::Float64, Φ::Float64) = @. (cos(Θ)^2) * (h[1, 1] * cos(Φ)^2 + h[1, 2] * sin(2Φ) + h[2, 2] * sin(Φ)^2) + h[3, 3] * sin(Θ)^2 - sin(2Θ) * (h[1, 3] * cos(Φ) + h[2, 3] * sin(Φ))   # Eq. 6.15
hΘΦ(h::AbstractArray, Θ::Float64, Φ::Float64) = @. cos(Θ) * (((-1/2) * h[1, 1] * sin(2Φ)) + h[1, 2] * cos(2Φ) + (1/2) * h[2, 2] * sin(2Φ)) + sin(Θ) * (h[1, 3] * sin(Φ) - h[2, 3] * cos(Φ))   # Eq. 6.16
hΦΦ(h::AbstractArray, Θ::Float64, Φ::Float64) = @. h[1, 1] * sin(Φ)^2 - h[1, 2] * sin(2Φ) + h[2, 2] * cos(Φ)^2   # Eq. 6.17

# define h_{+} and h_{×} components of GW (https://arxiv.org/pdf/gr-qc/0607007)
hplus(h::AbstractArray, Θ::Float64, Φ::Float64) = (1/2) *  (hΘΘ(h, Θ, Φ) .- hΦΦ(h, Θ, Φ))
hcross(h::AbstractArray, Θ::Float64, Φ::Float64) = hΘΦ(h, Θ, Φ)

end