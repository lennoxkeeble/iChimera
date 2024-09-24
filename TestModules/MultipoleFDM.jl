module MultipoleFDM
using StaticArrays
using ...MinoTimeDerivs, ...MinoDerivs1, ...MinoDerivs2, ...MinoDerivs3, ...MinoDerivs4, ...MinoDerivs5, ...MinoDerivs6
using ...HarmonicCoords
using ...FiniteDiff_5
using ...ParameterizedDerivs
using ...SymmetricTensors

# multipole moments
const multipole_moments = ["MassQuad", "MassOct", "MassHex", "CurrentQuad", "CurrentOct"]

# independent components of two, three, and four index tensors
const two_index_components::Vector{Tuple{Int64, Int64}} = [(1, 2), (1, 3), (2, 3), (1, 1), (2, 2), (3, 3)];
const three_index_components::Vector{Tuple{Int64, Int64, Int64}} = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 3, 3), (1, 2, 3), (2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3)];
const four_index_components::Vector{Tuple{Int64, Int64, Int64, Int64}} = [(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2), (1, 1, 1, 3), (1, 1, 3, 3), (1, 3, 3, 3), (1, 1, 2, 3), (1, 2, 2, 3),
(1, 2, 3, 3), (2, 2, 2, 2), (2, 2, 2, 3), (2, 2, 3, 3), (2, 3, 3, 3), (3, 3, 3, 3)];

const mass_quad_moments = SVector{length(two_index_components)}(["MassQuad", indices] for indices in two_index_components)
const mass_oct_moments = SVector{length(three_index_components)}(["MassOct", indices] for indices in three_index_components)
const mass_hex_moments = SVector{length(four_index_components)}(["MassHex", indices] for indices in four_index_components)
const current_quad_moments = SVector{length(two_index_components)}(["CurrentQuad", indices] for indices in two_index_components)
const current_oct_moments = SVector{length(three_index_components)}(["MassOct", indices] for indices in three_index_components)

const moments = SVector(vcat(mass_quad_moments, mass_oct_moments, mass_hex_moments, current_quad_moments, current_oct_moments)...)

function diff_moments_tr_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, M::Float64, x::AbstractArray, sign_dr::Float64, sign_dθ::Float64, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray,
    Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray,
    compute_at::Int64, nPoints::Int64, h::Float64)

    # compute derivatives of coordinates wrt to lambda
    dx_dλ = [MinoDerivs1.dr_dλ(x, a, M, E, L, C) * sign_dr, MinoDerivs1.dθ_dλ(x, a, M, E, L, C) * sign_dθ, MinoDerivs1.dϕ_dλ(x, a, M, E, L, C)]
    d2x_dλ = [MinoDerivs2.d2r_dλ(x, dx_dλ, a, M, E, L, C), MinoDerivs2.d2θ_dλ(x, dx_dλ, a, M, E, L, C), MinoDerivs2.d2ϕ_dλ(x, dx_dλ, a, M, E, L, C)]
    d3x_dλ = [MinoDerivs3.d3r_dλ(x, dx_dλ, d2x_dλ, a, M, E, L, C), MinoDerivs3.d3θ_dλ(x, dx_dλ, d2x_dλ, a, M, E, L, C), MinoDerivs3.d3ϕ_dλ(x, dx_dλ, d2x_dλ, a, M, E, L, C)]
    d4x_dλ = [MinoDerivs4.d4r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, M, E, L, C), MinoDerivs4.d4θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, M, E, L, C), MinoDerivs4.d4ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, M, E, L, C)]
    d5x_dλ = [MinoDerivs5.d5r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, M, E, L, C), MinoDerivs5.d5θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, M, E, L, C), MinoDerivs5.d5ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, M, E, L, C)]

    # compute derivatives of coordinate time wrt lambda
    dt_dλ = MinoDerivs1.dt_dλ(x, a, M, E, L, C);
    d2t_dλ = MinoDerivs2.d2t_dλ(x, dx_dλ, a, M, E, L, C);
    d3t_dλ = MinoDerivs3.d3t_dλ(x, dx_dλ, d2x_dλ, a, M, E, L, C);
    d4t_dλ = MinoDerivs4.d4t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, M, E, L, C);
    d5t_dλ = MinoDerivs5.d5t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, M, E, L, C);
    d6t_dλ = MinoDerivs6.d6t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, M, E, L, C);

    # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
    dλ_dt = MinoTimeDerivs.dλ_dt(dt_dλ)
    d2λ_dt = MinoTimeDerivs.d2λ_dt(dt_dλ, d2t_dλ)
    d3λ_dt = MinoTimeDerivs.d3λ_dt(dt_dλ, d2t_dλ, d3t_dλ)
    d4λ_dt = MinoTimeDerivs.d4λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ)
    d5λ_dt = MinoTimeDerivs.d5λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ)
    d6λ_dt = MinoTimeDerivs.d6λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ)

    @inbounds Threads.@threads for multipole_moment in moments
        type = multipole_moment[1];
        if isequal(type, "MassQuad")
            i1, i2 = multipole_moment[2];
            df_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d2f_dλ = FiniteDiff_5.compute_second_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d3f_dλ = FiniteDiff_5.compute_third_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d4f_dλ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d5f_dλ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d6f_dλ = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)

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
            df_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d2f_dλ = FiniteDiff_5.compute_second_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d3f_dλ = FiniteDiff_5.compute_third_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d4f_dλ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d5f_dλ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d6f_dλ = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)

            @views Mijk7[i1, i2, i3] = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
            @views Mijk8[i1, i2, i3] = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)

        elseif isequal(type, "CurrentQuad")
            i1, i2 = multipole_moment[2];
            df_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d2f_dλ = FiniteDiff_5.compute_second_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d3f_dλ = FiniteDiff_5.compute_third_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d4f_dλ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d5f_dλ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
    
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


function diff_moments_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, M::Float64, x::AbstractArray, sign_dr::Float64, sign_dθ::Float64, Mij2data::AbstractArray, Mijk2data::AbstractArray,
    Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray,
    compute_at::Int64, nPoints::Int64, h::Float64)

    # compute derivatives of coordinates wrt to lambda
    dx_dλ = [MinoDerivs1.dr_dλ(x, a, M, E, L, C) * sign_dr, MinoDerivs1.dθ_dλ(x, a, M, E, L, C) * sign_dθ, MinoDerivs1.dϕ_dλ(x, a, M, E, L, C)]
    d2x_dλ = [MinoDerivs2.d2r_dλ(x, dx_dλ, a, M, E, L, C), MinoDerivs2.d2θ_dλ(x, dx_dλ, a, M, E, L, C), MinoDerivs2.d2ϕ_dλ(x, dx_dλ, a, M, E, L, C)]
    d3x_dλ = [MinoDerivs3.d3r_dλ(x, dx_dλ, d2x_dλ, a, M, E, L, C), MinoDerivs3.d3θ_dλ(x, dx_dλ, d2x_dλ, a, M, E, L, C), MinoDerivs3.d3ϕ_dλ(x, dx_dλ, d2x_dλ, a, M, E, L, C)]
    d4x_dλ = [MinoDerivs4.d4r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, M, E, L, C), MinoDerivs4.d4θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, M, E, L, C), MinoDerivs4.d4ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, M, E, L, C)]
    d5x_dλ = [MinoDerivs5.d5r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, M, E, L, C), MinoDerivs5.d5θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, M, E, L, C), MinoDerivs5.d5ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, M, E, L, C)]

    # compute derivatives of coordinate time wrt lambda
    dt_dλ = MinoDerivs1.dt_dλ(x, a, M, E, L, C);
    d2t_dλ = MinoDerivs2.d2t_dλ(x, dx_dλ, a, M, E, L, C);
    d3t_dλ = MinoDerivs3.d3t_dλ(x, dx_dλ, d2x_dλ, a, M, E, L, C);
    d4t_dλ = MinoDerivs4.d4t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, M, E, L, C);
    d5t_dλ = MinoDerivs5.d5t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, M, E, L, C);
    d6t_dλ = MinoDerivs6.d6t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, M, E, L, C);

    # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
    dλ_dt = MinoTimeDerivs.dλ_dt(dt_dλ)
    d2λ_dt = MinoTimeDerivs.d2λ_dt(dt_dλ, d2t_dλ)
    d3λ_dt = MinoTimeDerivs.d3λ_dt(dt_dλ, d2t_dλ, d3t_dλ)
    d4λ_dt = MinoTimeDerivs.d4λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ)
    d5λ_dt = MinoTimeDerivs.d5λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ)
    d6λ_dt = MinoTimeDerivs.d6λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ)

    @inbounds Threads.@threads for multipole_moment in moments
        type = multipole_moment[1];
        if isequal(type, "MassQuad")
            i1, i2 = multipole_moment[2];
            df_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d2f_dλ = FiniteDiff_5.compute_second_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d3f_dλ = FiniteDiff_5.compute_third_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d4f_dλ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d5f_dλ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)
            d6f_dλ = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mij2data[i1, i2], h, nPoints)

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
            df_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d2f_dλ = FiniteDiff_5.compute_second_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d3f_dλ = FiniteDiff_5.compute_third_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d4f_dλ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d5f_dλ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)
            d6f_dλ = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mijk2data[i1, i2, i3], h, nPoints)

            @views Mijk3[i1, i2, i3] = ParameterizedDerivs.df_dt(df_dλ, dλ_dt)
            @views Mijk7[i1, i2, i3] = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
            @views Mijk8[i1, i2, i3] = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)
            
        elseif isequal(type, "MassHex")
            i1, i2, i3, i4 = multipole_moment[2];
            df_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijkl2data[i1, i2, i3, i4], h, nPoints)
            d2f_dλ = FiniteDiff_5.compute_second_derivative(compute_at,  Mijkl2data[i1, i2, i3, i4], h, nPoints)
            @views Mijkl4[i1, i2, i3, i4] = ParameterizedDerivs.d2f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt)

        elseif isequal(type, "CurrentQuad")
            i1, i2 = multipole_moment[2];
            df_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d2f_dλ = FiniteDiff_5.compute_second_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d3f_dλ = FiniteDiff_5.compute_third_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d4f_dλ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
            d5f_dλ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Sij1data[i1, i2], h, nPoints)
    
            @views Sij2[i1, i2] = ParameterizedDerivs.df_dt(df_dλ, dλ_dt)
            @views Sij5[i1, i2] = ParameterizedDerivs.d4f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt)
            @views Sij6[i1, i2] = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
        elseif isequal(type, "CurrentOct")
            i1, i2, i3 = multipole_moment[2];
            df_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sijk1data[i1, i2, i3], h, nPoints)
            d2f_dλ = FiniteDiff_5.compute_second_derivative(compute_at,  Sijk1data[i1, i2, i3], h, nPoints)

            @views Sijk3[i1, i2, i3] = ParameterizedDerivs.d2f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt)
        end
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij5); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij6);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij7); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij8);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3); SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk7);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk8); SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2); SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij5);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij6);  SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);
end


end