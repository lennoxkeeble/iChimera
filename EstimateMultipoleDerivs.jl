#=

    In this module we write functions which estimate the higher order derivatives of the multipole moments. This is done via two methods: finite differences (see `MultipolFDM.jl`) and fitting to a fourier series expansion (see `FourierFitGSL.jl`,
    `FourierFitJuliaBase.jl` and `MultipoleFitting.jl`).
    
=#

module EstimateMultipoleDerivs
using ..MultipoleMoments

"""
# Common Arguments in this module
- `r::Float64`: Boyer-Lindquist radial coordinate.
- `θ::Float64`: Boyer-Lindquist polar coordinate.
- `ϕ::Float64`: Boyer-Lindquist azimuthal coordinate.
- `xBL::AbstractVector{Float64}`: Boyer-Lindquist coordinates, xBL = [r, θ, ϕ].
- `vBL::AbstractVector{Float64}`: Boyer-Lindquist velocity wrt coordinate time, vBL = [dr_dt, dθ_dt, dϕ_dt].
- `aBL::AbstractVector{Float64}`: Boyer-Lindquist acceleration wrt coordinate time, aBL = [d^2r_dt^2, d^2θ_dt^2, d^2ϕ_dt^2].
- `xH::AbstractVector{Float64}`: Harmonic coordinates, xH = [x, y, z].
- `vH::AbstractVector{Float64}`: Harmonic velocity wrt coordinate time, vH = [dx_dt, dy_dt, dz_dt].
- `aH::AbstractVector{Float64}`: Harmonic acceleration wrt coordinate time, aH = [d^2x_dt^2, d^2y_dt^2, d^2z_dt^2].
- `rH::Float64`: rH = sqrt(xH^2 + yH^2 + zH^2).
- `dr_dt::Float64`: Coordinate-time first derivative of the radial coordinate.
- `d2r_dt2::Float64`: Coordinate-time second derivative of the radial coordinate.
- `dθ_dt::Float64`: Coordinate-time first derivative of the polar coordinate.
- `d2θ_dt2::Float64`: Coordinate-time second derivative of the polar coordinate.
- `dϕ_dt::Float64`: Coordinate-time first derivative of the azimuthal coordinate.
- `d2ϕ_dt2::Float64`: Coordinate-time second derivative of the azimuthal coordinate.
- `Mij2::AbstractArray`: second derivative of the mass quadrupole (Eq. 48, arXiv:1109.0572v2).
- `Mij5::AbstractArray`: fifth derivative of the mass quadrupole.
- `Mij6::AbstractArray`: sixth derivative of the mass quadrupole.
- `Mij7::AbstractArray`: seventh derivative of the mass quadrupole.
- `Mij8::AbstractArray`: eighth derivative of the mass quadrupole.
- `Mijk2::AbstractArray`: second derivative of the mass octupole (Eq. 48).
- `Mijk3::AbstractArray`: third derivative of the mass octupole.
- `Mijk7::AbstractArray`: seventh derivative of the mass octupole.
- `Mijk8::AbstractArray`: eighth derivative of the mass octupole.
- `Mijkl2::AbstractArray`: second derivative of the mass hexadecapole (Eq. 85).
- `Mijkl4::AbstractArray`: fourth derivative of the mass hexadecapole.
- `Sij1::AbstractArray`: first derivative of the current quadrupole (Eq. 49).
- `Sij5::AbstractArray`: fifth derivative of the current quadrupole.
- `Sij6::AbstractArray`: sixth derivative of the current quadrupole.
- `Sijk1::AbstractArray`: first derivative of the current octupole (Eq. 86).
- `Sijk3::AbstractArray`: third derivative of the current octupole.
- `a::Float64`: Kerr black hole spin parameter.
- `q::Float64`: mass ratio.
- `E::Float64`: energy per unit mass (Eq. 14).
- `L::Float64`: axial (i.e., z-component of the) angular momentum per unit mass (Eq. 15).
- `C::Float64`: Carter constant---note that this C is what is commonly referred to as 'Q' elsewhere (Eq. 17).
- `nHarm::Int64`: Number of radial harmonic frequencies in the fourier fit.
"""

norm2_3d(u::AbstractVector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
norm_3d(u::AbstractVector{Float64}) = sqrt(norm2_3d(u))

# compute first and second derivatives of the mass and current multipole moments for waveform computation from analytic expressions
function analytic_moment_derivs_wf!(aH::AbstractArray, vH::AbstractArray,  xH::AbstractArray, q::Float64, Mij2::AbstractArray, Mijk2::AbstractArray, Mijkl2::AbstractArray, Sij1::AbstractArray, Sijk1::AbstractArray)
    @inbounds for i=1:3
        for j=1:3
            Mij2[i, j] = MultipoleMoments.ddotMij.(aH, vH, xH, q, i, j)
            Sij1[i, j] = MultipoleMoments.dotSij.(aH, vH, xH, q, i, j)
            @inbounds for k=1:3
                Mijk2[i, j, k] = MultipoleMoments.ddotMijk.(aH, vH, xH, q, i, j, k)
                Sijk1[i, j, k] = MultipoleMoments.dotSijk.(aH, vH, xH, q, i, j, k)
                @inbounds for l=1:3
                    Mijkl2[i, j, k, l] = MultipoleMoments.ddotMijkl.(aH, vH, xH, q, i, j, k, l)
                end
            end
        end
    end
end

# compute first and second derivatives of the mass and current multipole moments for self-force and flux computation from analytic expressions
function analytic_moment_derivs_tr!(aH::AbstractArray, vH::AbstractArray,  xH::AbstractArray, q::Float64, Mij2::AbstractArray, Mijk2::AbstractArray, Sij1::AbstractArray)
    @inbounds for i=1:3
        for j=1:3
            Mij2[i, j] = MultipoleMoments.ddotMij.(aH, vH, xH, q, i, j)
            Sij1[i, j] = MultipoleMoments.dotSij.(aH, vH, xH, q, i, j)
            @inbounds for k=1:3
                Mijk2[i, j, k] = MultipoleMoments.ddotMijk.(aH, vH, xH, q, i, j, k)
            end
        end
    end
end


module FiniteDifferences
using ..EstimateMultipoleDerivs
using ...MinoTimeDerivs
using ...MinoDerivs1
using ...MinoDerivs2
using ...MinoDerivs3
using ...MinoDerivs4
using ...MinoDerivs5
using ...MinoDerivs6
using ...HarmonicCoords
using ...ParameterizedDerivs
using ...SymmetricTensors
using ...FiniteDiffOrder5

# compute time derivatives of the multipole moments using finite differences. The resulting derivatives are wrt Mino time λ, so we convert them to derivatives wrt BL coordinate time t
function moment_derivs_wf_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, x::AbstractArray, dr_dt::AbstractArray, dθ_dt::AbstractArray, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, Sijk1data::AbstractArray,
    Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nPoints::Int64, h::Float64)
    
    dx = [Float64[] for i in eachindex(x)]
    d2x = [Float64[] for i in eachindex(x)]
    dt_dλ = zeros(nPoints)
    d2t_dλ = zeros(nPoints)
    dλ_dt = zeros(nPoints)
    d2λ_dt = zeros(nPoints)

    @inbounds for i in eachindex(x)
        # compute derivatives of coordinates wrt to lambda
        dx[i] = [MinoDerivs1.dr_dλ(x[i], a, E, L, C) * sign(dr_dt[i]), MinoDerivs1.dθ_dλ(x[i], a, E, L, C) * sign(dθ_dt[i]), MinoDerivs1.dϕ_dλ(x[i], a, E, L, C)]
        d2x[i] = [MinoDerivs2.d2r_dλ(x[i], dx[i], a, E, L, C), MinoDerivs2.d2θ_dλ(x[i], dx[i], a, E, L, C), MinoDerivs2.d2ϕ_dλ(x[i], dx[i], a, E, L, C)]

        # compute derivatives of coordinate time wrt lambda
        dt_dλ[i] = MinoDerivs1.dt_dλ(x[i], a, E, L, C);
        d2t_dλ[i] = MinoDerivs2.d2t_dλ(x[i], dx[i], a, E, L, C);

        # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
        dλ_dt[i] = MinoTimeDerivs.dλ_dt(dt_dλ[i])
        d2λ_dt[i] = MinoTimeDerivs.d2λ_dt(dt_dλ[i], d2t_dλ[i])
    end

    @inbounds Threads.@threads for compute_at in 1:nPoints
        @inbounds for indices in SymmetricTensors.waveform_indices
            if length(indices)==2
                i, j = indices
                # current quadrupole
                dSij1_dλ = FiniteDiffOrder5.compute_first_derivative(compute_at,  Sij1data[i, j], h, nPoints)
                Sij2[i, j][compute_at] = ParameterizedDerivs.df_dt(dSij1_dλ, dλ_dt[compute_at])
            elseif length(indices)==3
                # mass octupole
                i, j, k = indices
                dMijk2_dλ = FiniteDiffOrder5.compute_first_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
                Mijk3[i, j, k][compute_at] = ParameterizedDerivs.df_dt(dMijk2_dλ, dλ_dt[compute_at])

                dSijk1_dλ = FiniteDiffOrder5.compute_first_derivative(compute_at,  Sijk1data[i, j, k], h, nPoints)
                dSijk1_d2λ = FiniteDiffOrder5.compute_second_derivative(compute_at,  Sijk1data[i, j, k], h, nPoints)
                Sijk3[i, j, k][compute_at] = ParameterizedDerivs.d2f_dt(dSijk1_dλ, dλ_dt[compute_at], dSijk1_d2λ, d2λ_dt[compute_at])
            else
                i, j, k, l = indices
                dMijkl2_dλ = FiniteDiffOrder5.compute_first_derivative(compute_at,  Mijkl2data[i, j, k, l], h, nPoints)
                dMijkl2_d2λ = FiniteDiffOrder5.compute_second_derivative(compute_at,  Mijkl2data[i, j, k, l], h, nPoints)
                Mijkl4[i, j, k, l][compute_at] = ParameterizedDerivs.d2f_dt(dMijkl2_dλ, dλ_dt[compute_at], dMijkl2_d2λ, d2λ_dt[compute_at])
            end
        end
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3); SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);
    SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
end


function compute_waveform_moments_and_derivs_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, q::Float64, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, 
    xH::AbstractArray, rH::AbstractArray, vH::AbstractArray,  aH::AbstractArray, v::AbstractArray, 
    t::AbstractArray, r::AbstractArray, dr_dt::AbstractArray, d2r_dt2::AbstractArray, θ::AbstractArray, dθ_dt::AbstractArray, d2θ_dt2::AbstractArray, ϕ::AbstractArray,
    dϕ_dt::AbstractArray, d2ϕ_dt2::AbstractArray, Mij2data::AbstractArray, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, 
    Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nPoints::Int64, h::Float64)

    # convert trajectories to BL coords
    @inbounds for i=1:nPoints
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
        aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);

        HarmonicCoords.xBLtoH!(xH[i], xBL[i], a);
        HarmonicCoords.vBLtoH!(vH[i], xH[i], vBL[i], a); 
        HarmonicCoords.aBLtoH!(aH[i], xH[i], vBL[i], aBL[i], a);

        rH[i] = EstimateMultipoleDerivs.norm_3d(xH[i]);
        v[i] = EstimateMultipoleDerivs.norm_3d(vH[i]);
    end

    EstimateMultipoleDerivs.analytic_moment_derivs_wf!(aH, vH, xH, q, Mij2data, Mijk2data, Mijkl2data, Sij1data, Sijk1data)
    EstimateMultipoleDerivs.FiniteDifferences.moment_derivs_wf_Mino!(a, E, L, C, xBL, dr_dt, dθ_dt, Mijk2data, Mijkl2data, Sij1data, Sijk1data, Mijk3, Mijkl4, Sij2, Sijk3, nPoints, h)
end

end

module FourierFit
using ..EstimateMultipoleDerivs
using ...MinoTimeDerivs
using ...MinoDerivs1
using ...MinoDerivs2
using ...MinoDerivs3
using ...MinoDerivs4
using ...MinoDerivs5
using ...MinoDerivs6
using ...HarmonicCoords
using ...FourierFitGSL
using ...FourierFitJuliaBase
using ...ParameterizedDerivs
using ...SymmetricTensors
using ...MultipoleFitting

@views function compute_waveform_moments_and_derivs!(a::Float64, q::Float64, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, rH::AbstractArray,
    vH::AbstractArray,  aH::AbstractArray, v::AbstractArray, t::Vector{Float64}, r::Vector{Float64}, dr_dt::Vector{Float64}, d2r_dt2::Vector{Float64}, θ::Vector{Float64},
    dθ_dt::Vector{Float64}, d2θ_dt2::Vector{Float64}, ϕ::Vector{Float64}, dϕ_dt::Vector{Float64}, d2ϕ_dt2::Vector{Float64}, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Mijkl2_data::AbstractArray,
    Sij1_data::AbstractArray, Sijk1_data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64,
    nPoints::Int64, n_freqs::Int64, chisq::Vector{Float64}, fit::String)

    @inbounds for i=1:nPoints
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
        aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);

        HarmonicCoords.xBLtoH!(xH[i], xBL[i], a);
        HarmonicCoords.vBLtoH!(vH[i], xH[i], vBL[i], a); 
        HarmonicCoords.aBLtoH!(aH[i], xH[i], vBL[i], aBL[i], a);

        rH[i] = EstimateMultipoleDerivs.norm_3d(xH[i]);
        v[i] = EstimateMultipoleDerivs.norm_3d(vH[i]);
    end

    # compute first and second derivatives of the mass and current multipole moments for waveform computation from analytic expressions
    EstimateMultipoleDerivs.analytic_moment_derivs_wf!(aH, vH, xH, q, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data)
    # estimate higher order derivative moments using fourier fits in BL time
    MultipoleFitting.fit_moments_wf_BL!(t, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3, Mijkl4, Sij2, Sijk3, nHarm, Ωr, Ωθ, Ωϕ, nPoints, n_freqs, chisq, fit)
end

@views function compute_waveform_moments_and_derivs_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, q::Float64, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, 
    xH::AbstractArray, rH::AbstractArray, vH::AbstractArray,  aH::AbstractArray, v::AbstractArray, λ::Vector{Float64}, r::Vector{Float64}, dr_dt::Vector{Float64},
    d2r_dt2::Vector{Float64}, θ::Vector{Float64}, dθ_dt::Vector{Float64}, d2θ_dt2::Vector{Float64}, ϕ::Vector{Float64},
    dϕ_dt::Vector{Float64}, d2ϕ_dt2::Vector{Float64}, Mij2_data::AbstractArray, Mijk2_data::AbstractArray, Mijkl2_data::AbstractArray, Sij1_data::AbstractArray, 
    Sijk1_data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nHarm::Int64, γr::Float64, γθ::Float64, γϕ::Float64, 
    nPoints::Int64, n_freqs::Int64, chisq::Vector{Float64}, fit::String)

    @inbounds for i=1:nPoints
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([dr_dt[i], dθ_dt[i], dϕ_dt[i]]);
        aBL[i] = Vector{Float64}([d2r_dt2[i], d2θ_dt2[i], d2ϕ_dt2[i]]);

        HarmonicCoords.xBLtoH!(xH[i], xBL[i], a);
        HarmonicCoords.vBLtoH!(vH[i], xH[i], vBL[i], a); 
        HarmonicCoords.aBLtoH!(aH[i], xH[i], vBL[i], aBL[i], a);

        rH[i] = EstimateMultipoleDerivs.norm_3d(xH[i]);
        v[i] = EstimateMultipoleDerivs.norm_3d(vH[i]);
    end

    # compute first and second derivatives of the mass and current multipole moments for waveform computation from analytic expressions
    EstimateMultipoleDerivs.analytic_moment_derivs_wf!(aH, vH, xH, q, Mij2_data, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data)
    # estimate higher order derivative moments using fourier fits in Mino time
    MultipoleFitting.fit_moments_wf_Mino!(a, E, L, C, λ, xBL, dr_dt, dθ_dt, Mijk2_data, Mijkl2_data, Sij1_data, Sijk1_data, Mijk3, Mijkl4, Sij2, Sijk3, nHarm, γr, γθ, γϕ, nPoints, n_freqs, chisq, fit)
end

end

end