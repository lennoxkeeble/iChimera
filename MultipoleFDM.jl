#=

    In this module we write code which estimates the high-order derivatives of the multipole moments using finite differences in Mino time.

=#

module MultipoleFDM
using StaticArrays
using ...MinoTimeDerivs
using ...MinoDerivs1
using ...MinoDerivs2
using ...MinoDerivs3
using ...MinoDerivs4
using ...MinoDerivs5
using ...MinoDerivs6
using ...HarmonicCoords
using ...FiniteDiffOrder5
using ...ParameterizedDerivs
using ...SymmetricTensors

"""
# Common Arguments in this module
- `r::Float64`: Boyer-Lindquist radial coordinate.
- `θ::Float64`: Boyer-Lindquist polar coordinate.
- `ϕ::Float64`: Boyer-Lindquist azimuthal coordinate.
- `xBL::AbstractArray`: array of arrays of Boyer-Lindquist coordinates, xBL = [[r[1], θ[1], ϕ[1]], [r[2], θ[2], ϕ[2]],....] at each time the multipole derivatives are to be approximated.
- `rH::Float64`: rH = sqrt(xH^2 + yH^2 + zH^2).
- `dr_dt::Float64`: Coordinate-time first derivative of the radial coordinate.
- `sign_dr::Float64`: sign of the dr_dt.
- `sign_dθ::Float64`: sign of the dθ_dt.
- `dθ_dt::Float64`: Coordinate-time first derivative of the polar coordinate.
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
- `nPoints::Int64`: number of points at which the derivatives are to be approximated.
- `h::Float64`: step size for finite differencing.
- `compute_at::Int64`: index (in the stencil) at which the derivatives are to be approximated.
"""

# multipole moments
const multipole_moments = ["MassQuad", "MassOct", "MassHex", "CurrentQuad", "CurrentOct"]

# independent components of two, three, and four index tensors
const two_index_components::Vector{Tuple{Int64, Int64}} = [(1, 2), (1, 3), (2, 3), (1, 1), (2, 2), (3, 3)];
const three_index_components::Vector{Tuple{Int64, Int64, Int64}} = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 3, 3), (1, 2, 3), (2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3)];
const four_index_components::Vector{Tuple{Int64, Int64, Int64, Int64}} = [(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2), (1, 1, 1, 3), (1, 1, 3, 3), (1, 3, 3, 3), (1, 1, 2, 3), (1, 2, 2, 3),
(1, 2, 3, 3), (2, 2, 2, 2), (2, 2, 2, 3), (2, 2, 3, 3), (2, 3, 3, 3), (3, 3, 3, 3)];

# multipole moments and their independent components
const mass_quad_moments = SVector{length(two_index_components)}(["MassQuad", indices] for indices in two_index_components)
const mass_oct_moments = SVector{length(three_index_components)}(["MassOct", indices] for indices in three_index_components)
const mass_hex_moments = SVector{length(four_index_components)}(["MassHex", indices] for indices in four_index_components)
const current_quad_moments = SVector{length(two_index_components)}(["CurrentQuad", indices] for indices in two_index_components)
const current_oct_moments = SVector{length(three_index_components)}(["MassOct", indices] for indices in three_index_components)

# waveform moments and trajectory (flux) moments. This construction is so that we can use threads to parallelize the computation of the derivatives.
const moments_tr = SVector(vcat(mass_quad_moments, mass_oct_moments, current_quad_moments)...)
const moments_wf = SVector(vcat(mass_oct_moments, mass_hex_moments, current_quad_moments, current_oct_moments)...)

# compute derivatives of the multipole moments wrt to Mino time and converts to BL time. This function is for computing the moment derivatives necessary for the fluxes. This is computed at a single point (whereas the waveform moments are computed
# at each point in the piecewise geodesic)
function diff_moments_tr_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, xBL::AbstractArray, sign_dr::Float64, sign_dθ::Float64, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray,
    Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray,
    compute_at::Int64, nPoints::Int64, h::Float64)

    # compute derivatives of coordinates wrt to lambda
    dx_dλ = [MinoDerivs1.dr_dλ(xBL, a, E, L, C) * sign_dr, MinoDerivs1.dθ_dλ(xBL, a, E, L, C) * sign_dθ, MinoDerivs1.dϕ_dλ(xBL, a, E, L, C)]
    d2x_dλ = [MinoDerivs2.d2r_dλ(xBL, dx_dλ, a, E, L, C), MinoDerivs2.d2θ_dλ(xBL, dx_dλ, a, E, L, C), MinoDerivs2.d2ϕ_dλ(xBL, dx_dλ, a, E, L, C)]
    d3x_dλ = [MinoDerivs3.d3r_dλ(xBL, dx_dλ, d2x_dλ, a, E, L, C), MinoDerivs3.d3θ_dλ(xBL, dx_dλ, d2x_dλ, a, E, L, C), MinoDerivs3.d3ϕ_dλ(xBL, dx_dλ, d2x_dλ, a, E, L, C)]
    d4x_dλ = [MinoDerivs4.d4r_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C), MinoDerivs4.d4θ_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C), MinoDerivs4.d4ϕ_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C)]
    d5x_dλ = [MinoDerivs5.d5r_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C), MinoDerivs5.d5θ_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C), MinoDerivs5.d5ϕ_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C)]

    # compute derivatives of coordinate time wrt lambda
    dt_dλ = MinoDerivs1.dt_dλ(xBL, a, E, L, C);
    d2t_dλ = MinoDerivs2.d2t_dλ(xBL, dx_dλ, a, E, L, C);
    d3t_dλ = MinoDerivs3.d3t_dλ(xBL, dx_dλ, d2x_dλ, a, E, L, C);
    d4t_dλ = MinoDerivs4.d4t_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C);
    d5t_dλ = MinoDerivs5.d5t_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C);
    d6t_dλ = MinoDerivs6.d6t_dλ(xBL, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, E, L, C);

    # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
    dλ_dt = MinoTimeDerivs.dλ_dt(dt_dλ)
    d2λ_dt = MinoTimeDerivs.d2λ_dt(dt_dλ, d2t_dλ)
    d3λ_dt = MinoTimeDerivs.d3λ_dt(dt_dλ, d2t_dλ, d3t_dλ)
    d4λ_dt = MinoTimeDerivs.d4λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ)
    d5λ_dt = MinoTimeDerivs.d5λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ)
    d6λ_dt = MinoTimeDerivs.d6λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ)

    @inbounds Threads.@threads for multipole_moment in moments_tr
        type = multipole_moment[1];
        if isequal(type, "MassQuad")
            i1, i2 = multipole_moment[2];
            df_dλ = FiniteDiffOrder5.compute_first_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d2f_dλ = FiniteDiffOrder5.compute_second_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d3f_dλ = FiniteDiffOrder5.compute_third_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d4f_dλ = FiniteDiffOrder5.compute_fourth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d5f_dλ = FiniteDiffOrder5.compute_fifth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d6f_dλ = FiniteDiffOrder5.compute_sixth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)

            d3f_dt = ParameterizedDerivs.d3f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt)
            d4f_dt = ParameterizedDerivs.d4f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt)
            d5f_dt = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
            d6f_dt = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)
            @views Mij5[i1, i2] = d3f_dt
            @views Mij6[i1, i2] = d4f_dt
            @views Mij7[i1, i2] = d5f_dt
            @views Mij8[i1, i2] = d6f_dt

        elseif isequal(type, "MassOct")
            i1, i2, i3 = multipole_moment[2];
            df_dλ = FiniteDiffOrder5.compute_first_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d2f_dλ = FiniteDiffOrder5.compute_second_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d3f_dλ = FiniteDiffOrder5.compute_third_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d4f_dλ = FiniteDiffOrder5.compute_fourth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d5f_dλ = FiniteDiffOrder5.compute_fifth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d6f_dλ = FiniteDiffOrder5.compute_sixth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)

            @views Mijk7[i1, i2, i3] = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
            @views Mijk8[i1, i2, i3] = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)

        elseif isequal(type, "CurrentQuad")
            i1, i2 = multipole_moment[2];
            df_dλ = FiniteDiffOrder5.compute_first_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d2f_dλ = FiniteDiffOrder5.compute_second_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d3f_dλ = FiniteDiffOrder5.compute_third_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d4f_dλ = FiniteDiffOrder5.compute_fourth_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d5f_dλ = FiniteDiffOrder5.compute_fifth_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
    
            @views Sij5[i1, i2] = ParameterizedDerivs.d4f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt)
            @views Sij6[i1, i2] = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
        end
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij5); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij6);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij7); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij8); 
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk7); SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk8);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij5); SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij6);  
end

# compute derivatives of the multipole moments wrt to Mino time and convert to BL time. This function is for computing the moment derivatives necessary for the waveform. This is computed at each point in the piecewise geodesic.
function diff_moments_Mino_wf!(a::Float64, E::Float64, L::Float64, C::Float64, xBL::AbstractArray, dr_dt::AbstractVector{Float64}, dθ_dt::AbstractVector{Float64}, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, Sijk1data::AbstractArray,
    Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nPoints::Int64, h::Float64)

    dx = [Float64[] for i in eachindex(xBL)]
    d2x = [Float64[] for i in eachindex(xBL)]
    dt_dλ = zeros(nPoints)
    d2t_dλ = zeros(nPoints)
    dλ_dt = zeros(nPoints)
    d2λ_dt = zeros(nPoints)

    @inbounds Threads.@threads for i in 1:nPoints
        # compute derivatives of coordinates wrt to lambda
        dx[i] = [MinoDerivs1.dr_dλ(xBL[i], a, E, L, C) * sign(dr_dt[i]), MinoDerivs1.dθ_dλ(xBL[i], a, E, L, C) * sign(dθ_dt[i]), MinoDerivs1.dϕ_dλ(xBL[i], a, E, L, C)]
        d2x[i] = [MinoDerivs2.d2r_dλ(xBL[i], dx[i], a, E, L, C), MinoDerivs2.d2θ_dλ(xBL[i], dx[i], a, E, L, C), MinoDerivs2.d2ϕ_dλ(xBL[i], dx[i], a, E, L, C)]

        # compute derivatives of coordinate time wrt lambda
        dt_dλ[i] = MinoDerivs1.dt_dλ(xBL[i], a, E, L, C);
        d2t_dλ[i] = MinoDerivs2.d2t_dλ(xBL[i], dx[i], a, E, L, C);

        # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
        dλ_dt[i] = MinoTimeDerivs.dλ_dt(dt_dλ[i])
        d2λ_dt[i] = MinoTimeDerivs.d2λ_dt(dt_dλ[i], d2t_dλ[i])

        @inbounds Threads.@threads for multipole_moment in moments_wf
            type = multipole_moment[1];
            if isequal(type, "MassOct")
                i1, i2, i3 = multipole_moment[2];
                df_dλ = FiniteDiffOrder5.compute_first_derivative(i,  Mijk2data[i1, i2, i3], h, nPoints)
                @views Mijk3[i1, i2, i3][i] = ParameterizedDerivs.df_dt(df_dλ, dλ_dt[i])            
            elseif isequal(type, "MassHex")
                i1, i2, i3, i4 = multipole_moment[2];
                df_dλ = FiniteDiffOrder5.compute_first_derivative(i,  Mijkl2data[i1, i2, i3, i4], h, nPoints)
                d2f_dλ = FiniteDiffOrder5.compute_second_derivative(i,  Mijkl2data[i1, i2, i3, i4], h, nPoints)
                @views Mijkl4[i1, i2, i3, i4][i] = ParameterizedDerivs.d2f_dt(df_dλ, dλ_dt[i], d2f_dλ, d2λ_dt)

            elseif isequal(type, "CurrentQuad")
                i1, i2 = multipole_moment[2];
                df_dλ = FiniteDiffOrder5.compute_first_derivative(i,  Sij1data[i1, i2], h, nPoints)
                d2f_dλ = FiniteDiffOrder5.compute_second_derivative(i,  Sij1data[i1, i2], h, nPoints)    
                @views Sij2[i1, i2][i] = ParameterizedDerivs.df_dt(df_dλ, dλ_dt[i])
            elseif isequal(type, "CurrentOct")
                i1, i2, i3 = multipole_moment[2];
                df_dλ = FiniteDiffOrder5.compute_first_derivative(i,  Sijk1data[i1, i2, i3], h, nPoints)
                d2f_dλ = FiniteDiffOrder5.compute_second_derivative(i,  Sijk1data[i1, i2, i3], h, nPoints)
                @views Sijk3[i1, i2, i3][i] = ParameterizedDerivs.d2f_dt(df_dλ, dλ_dt[i], d2f_dλ, d2λ_dt)
            end
        end
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3); SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2); SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);
end


end