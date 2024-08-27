#=

    In this module we write functions which estimate the higher order derivatives of the multipole moments. This is done via two methods: finite
    differences and fitting to a fourier series expansion

=#

module EstimateMultipoleDerivs
using LinearAlgebra
using Combinatorics
using ..Multipoles

# define some useful functions
otimes(a::Vector, b::Vector) = [a[i] * b[j] for i=1:size(a, 1), j=1:size(b, 1)]    # tensor product of two vectors
otimes(a::Vector) = [a[i] * a[j] for i=1:size(a, 1), j=1:size(a, 1)]    # tensor product of a vector with itself
dot3d(u::Vector{Float64}, v::Vector{Float64}) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3]
norm2_3d(u::Vector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3]
norm_3d(u::Vector{Float64}) = sqrt(EstimateMultipoleDerivs.norm2_3d(u))
dot4d(u::Vector{Float64}, v::Vector{Float64}) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4]
norm2_4d(u::Vector{Float64}) = u[1] * u[1] + u[2] * u[2] + u[3] * u[3] + u[4] * u[4]
norm_4d(u::Vector{Float64}) = sqrt(EstimateMultipoleDerivs.norm2_4d(u))

# define STF projections 
STF(u::Vector, i::Int, j::Int) = u[i] * u[j] - dot(u, u) * δ(i, j) /3.0                                                                     # STF projection x^{<ij>}
STF(u::Vector, v::Vector, i::Int, j::Int) = (u[i] * v[j] + u[j] * v[i])/2.0 - dot(u, v)* δ(i, j) /3.0                                       # STF projection of two distinct vectors
STF(u::Vector, i::Int, j::Int, k::Int) = u[i] * u[j] * u[k] - (1.0/5.0) * dot(u, u) * (δ(i, j) * u[k] + δ(j, k) * u[i] + δ(k, i) * u[j])    # STF projection x^{<ijk>} (Eq. 46)

# define some objects useful for efficient calculation of current quadrupole and its derivatives
const ρ::Vector{Int} = [1, 2, 3]   # spacial indices
const spatial_indices_3::Array = [[x, y, z] for x=1:3, y=1:3, z=1:3]   # array where each element kl = [[k, l, i] for i=1:3]
const εkl::Array{Vector} = [[levicivita(spatial_indices_3[k, l, i]) for i = 1:3] for k=1:3, l=1:3]   # array where each element kl = [e_{kli} for i=1:3]
const index_pairs::Matrix{Tuple{Int64, Int64}} = [(i, j) for i=1:3, j=1:3];   # array of index pairs
const multipoles_tr::Vector{String} = ["mass_quad_2nd", "mass_oct_2nd", "current_quad_1st"];
const two_index_multipoles_tr::Vector{String} = ["mass_quad_2nd", "current_quad_1st"];
const three_index_multipoles_wf::Vector{String} = ["mass_oct_2nd", "current_oct_1st"];

# fill pre-allocated arrays with the appropriate derivatives of the mass and current moments for trajectory evolution, i.e., to compute self-force
function multipole_moments_tr!(vH::AbstractArray, xH::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, Mij::AbstractArray, Mijk::AbstractArray, Sij::AbstractArray)
    @inbounds for i=1:3
        for j=1:3
            Mij[i, j] = Multipoles.Mij.(x_H, m, M, i, j)
            Sij[i, j] = Multipoles.Sij.(x_H, xH, vH, m, M, i, j)
            @inbounds for k=1:3
                Mijk[i, j, k] = Multipoles.Mijk.(x_H, m, M, i, j, k)
            end
        end
    end
end

# fill pre-allocated arrays with the appropriate derivatives of the mass and current moments for trajectory evolution, i.e., to compute self-force
function moments_tr!(aH::AbstractArray, a_H::AbstractArray, vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, Mij2::AbstractArray, Mijk2::AbstractArray, Sij1::AbstractArray)
    @inbounds for i=1:3
        for j=1:3
            Mij2[i, j] = Multipoles.ddotMij.(a_H, v_H, x_H, m, M, i, j)
            Sij1[i, j] = Multipoles.dotSij.(aH, v_H, vH, x_H, xH, m, M, i, j)
            @inbounds for k=1:3
                Mijk2[i, j, k] = Multipoles.ddotMijk.(a_H, v_H, x_H, m, M, i, j, k)
            end
        end
    end
end

# fill pre-allocated arrays with the appropriate derivatives of the mass and current moments for waveform computation
function moments_wf!(aH::AbstractArray, a_H::AbstractArray, vH::AbstractArray, v_H::AbstractArray, xH::AbstractArray, x_H::AbstractArray, m::Float64, M::Float64, Mij2::AbstractArray, Mijk2::AbstractArray, Mijkl2::AbstractArray, Sij1::AbstractArray, Sijk1::AbstractArray)
    @inbounds for i=1:3
        for j=1:3
            Mij2[i, j] = Multipoles.ddotMij.(a_H, v_H, x_H, m, M, i, j)
            Sij1[i, j] = Multipoles.dotSij.(aH, v_H, vH, x_H, xH, m, M, i, j)
            @inbounds for k=1:3
                Mijk2[i, j, k] = Multipoles.ddotMijk.(a_H, v_H, x_H, m, M, i, j, k)
                Sijk1[i, j, k] = Multipoles.dotSijk.(a_H, v_H, x_H, m, M, i, j, k)
                @inbounds for l=1:3
                    Mijkl2[i, j, k, l] = Multipoles.ddotMijkl.(a_H, v_H, x_H, m, M, i, j, k, l)
                end
            end
        end
    end
end

module FiniteDifferences
using ..EstimateMultipoleDerivs
using ...MinoTimeDerivs, ...MinoDerivs1, ...MinoDerivs2, ...MinoDerivs3, ...MinoDerivs4, ...MinoDerivs5, ...MinoDerivs6
using ...HarmonicCoords
using ...ParameterizedDerivs
using ...ConstructSymmetricArrays
using ...FiniteDiff_5

# calculate time derivatives of the mass and current moments for trajectory evolution in BL time, i.e., to compute self-force
function moment_derivs_tr_BL!(h::Float64, compute_at::Int64, nPoints::Int64, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray)
    @inbounds for i=1:3
        @inbounds for j=1:3
            Mij5[i, j] = FiniteDiff_5.compute_third_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            Mij6[i, j] = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            Mij7[i, j] = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            Mij8[i, j] = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mij2data[i, j], h, nPoints)

            Sij5[i, j] = FiniteDiff_5.compute_fourth_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            Sij6[i, j] = FiniteDiff_5.compute_fifth_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            @inbounds for k=1:3
                Mijk7[i, j, k] = FiniteDiff_5.compute_fifth_derivative(compute_at, Mijk2data[i, j, k], h, nPoints)
                Mijk8[i, j, k] = FiniteDiff_5.compute_sixth_derivative(compute_at, Mijk2data[i, j, k], h, nPoints)
            end 
        end
    end
end


# calculate time derivatives of the mass and current moments for trajectory evolution in mino time, i.e., to compute self-force
function moment_derivs_tr_Mino!(h::Float64, compute_at::Int64, nPoints::Int64, a::Float64, M::Float64, E::Float64, L::Float64, C::Float64, x::Vector{Float64}, sign_dr::Float64, sign_dθ::Float64, Mij2data::AbstractArray,
    Mijk2data::AbstractArray, Sij1data::AbstractArray, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray,
    Sij6::AbstractArray)
    ### to make faster will want these arrays pre-allocated
    # compute first-order derivatives of coordinates wrt to lambda
    dx = [MinoDerivs1.dr_dλ(x, a, M, E, L, C) * sign_dr, MinoDerivs1.dθ_dλ(x, a, M, E, L, C) * sign_dθ, MinoDerivs1.dϕ_dλ(x, a, M, E, L, C)]

    # compute first-order derivatives of coordinates wrt to lambda
    d2x = [MinoDerivs2.d2r_dλ(x, dx, a, M, E, L, C), MinoDerivs2.d2θ_dλ(x, dx, a, M, E, L, C), MinoDerivs2.d2ϕ_dλ(x, dx, a, M, E, L, C)]
    d3x = [MinoDerivs3.d3r_dλ(x, dx, d2x, a, M, E, L, C), MinoDerivs3.d3θ_dλ(x, dx, d2x, a, M, E, L, C), MinoDerivs3.d3ϕ_dλ(x, dx, d2x, a, M, E, L, C)]
    d4x = [MinoDerivs4.d4r_dλ(x, dx, d2x, d3x, a, M, E, L, C), MinoDerivs4.d4θ_dλ(x, dx, d2x, d3x, a, M, E, L, C), MinoDerivs4.d4ϕ_dλ(x, dx, d2x, d3x, a, M, E, L, C)]
    d5x = [MinoDerivs5.d5r_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C), MinoDerivs5.d5θ_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C), MinoDerivs5.d5ϕ_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C)]

    # compute derivatives of coordinate time wrt lambda
    dt_dλ = MinoDerivs1.dt_dλ(x, a, M, E, L, C);
    d2t_dλ = MinoDerivs2.d2t_dλ(x, dx, a, M, E, L, C);
    d3t_dλ = MinoDerivs3.d3t_dλ(x, dx, d2x, a, M, E, L, C);
    d4t_dλ = MinoDerivs4.d4t_dλ(x, dx, d2x, d3x, a, M, E, L, C);
    d5t_dλ = MinoDerivs5.d5t_dλ(x, dx, d2x, d3x, d4x, a, M, E, L, C);
    d6t_dλ = MinoDerivs6.d6t_dλ(x, dx, d2x, d3x, d4x, d5x, a, M, E, L, C);

    # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
    dλ_dt = MinoTimeDerivs.dλ_dt(dt_dλ)
    d2λ_dt = MinoTimeDerivs.d2λ_dt(dt_dλ, d2t_dλ)
    d3λ_dt = MinoTimeDerivs.d3λ_dt(dt_dλ, d2t_dλ, d3t_dλ)
    d4λ_dt = MinoTimeDerivs.d4λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ)
    d5λ_dt = MinoTimeDerivs.d5λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ)
    d6λ_dt = MinoTimeDerivs.d6λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ)

    @inbounds Threads.@threads for indices in ConstructSymmetricArrays.traj_indices
        if length(indices)==2
            i, j = indices
            # the naming is a bit confusing in the code block below. For example take f(t) = Mij2 = d2Mij_dt2. We now wish to compute higher order derivatives of f(t) from the re-parameterized function 
            # f(λ(t)). We do this by first computing d^{n}f/dλ^{n} numerically from finite difference formulas, and d^{n}λ/dt^{n} from analytix expressions. We can then use these to compute d^{n}f_dt^{n} 
            # via the chain rule (e.g., see the file "ParameterizedDerivs.jl")

            # mass quadrupole
            dMij2_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d3λ = FiniteDiff_5.compute_third_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d4λ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d5λ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mij2data[i, j], h, nPoints)
            dMij2_d6λ = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mij2data[i, j], h, nPoints)

            Mij5[i, j] = ParameterizedDerivs.d3f_dt(dMij2_dλ, dλ_dt, dMij2_d2λ, d2λ_dt, dMij2_d3λ, d3λ_dt)
            Mij6[i, j] = ParameterizedDerivs.d4f_dt(dMij2_dλ, dλ_dt, dMij2_d2λ, d2λ_dt, dMij2_d3λ, d3λ_dt, dMij2_d4λ, d4λ_dt)
            Mij7[i, j] = ParameterizedDerivs.d5f_dt(dMij2_dλ, dλ_dt, dMij2_d2λ, d2λ_dt, dMij2_d3λ, d3λ_dt, dMij2_d4λ, d4λ_dt, dMij2_d5λ, d5λ_dt)
            Mij8[i, j] = ParameterizedDerivs.d6f_dt(dMij2_dλ, dλ_dt, dMij2_d2λ, d2λ_dt, dMij2_d3λ, d3λ_dt, dMij2_d4λ, d4λ_dt, dMij2_d5λ, d5λ_dt, dMij2_d6λ, d6λ_dt)

            # current quadrupole
            dSij1_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            dSij1_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            dSij1_d3λ = FiniteDiff_5.compute_third_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            dSij1_d4λ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Sij1data[i, j], h, nPoints)
            dSij1_d5λ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Sij1data[i, j], h, nPoints)

            Sij5[i, j] = ParameterizedDerivs.d4f_dt(dSij1_dλ, dλ_dt, dSij1_d2λ, d2λ_dt, dSij1_d3λ, d3λ_dt, dSij1_d4λ, d4λ_dt)
            Sij6[i, j] = ParameterizedDerivs.d5f_dt(dSij1_dλ, dλ_dt, dSij1_d2λ, d2λ_dt, dSij1_d3λ, d3λ_dt, dSij1_d4λ, d4λ_dt, dSij1_d5λ, d5λ_dt)
        else
            # mass octupole
            i, j, k = indices
            dMijk2_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d3λ = FiniteDiff_5.compute_third_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d4λ = FiniteDiff_5.compute_fourth_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d5λ = FiniteDiff_5.compute_fifth_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
            dMijk2_d6λ = FiniteDiff_5.compute_sixth_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)

            Mijk7[i, j, k] = ParameterizedDerivs.d5f_dt(dMijk2_dλ, dλ_dt, dMijk2_d2λ, d2λ_dt, dMijk2_d3λ, d3λ_dt, dMijk2_d4λ, d4λ_dt, dMijk2_d5λ, d5λ_dt)
            Mijk8[i, j, k] = ParameterizedDerivs.d6f_dt(dMijk2_dλ, dλ_dt, dMijk2_d2λ, d2λ_dt, dMijk2_d3λ, d3λ_dt, dMijk2_d4λ, d4λ_dt, dMijk2_d5λ, d5λ_dt, dMijk2_d6λ, d6λ_dt)
        end
    end

    # symmetrize moments
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij5); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij6);
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij7); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij8);
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij5); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij6);
    ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk7); ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk8);
end

@views function moment_derivs_wf_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, M::Float64, x::AbstractArray, dr::AbstractArray, dθ::AbstractArray, 
    Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray,
    Sij2::AbstractArray, Sijk3::AbstractArray, nPoints::Int64, h::Float64)
    
    dx = [Float64[] for i in eachindex(x)]
    d2x = [Float64[] for i in eachindex(x)]
    dt_dλ = zeros(nPoints)
    d2t_dλ = zeros(nPoints)
    dλ_dt = zeros(nPoints)
    d2λ_dt = zeros(nPoints)

    for i in eachindex(x)
        # compute derivatives of coordinates wrt to lambda
        dx[i] = [MinoDerivs1.dr_dλ(x[i], a, M, E, L, C) * sign(dr[i]), MinoDerivs1.dθ_dλ(x[i], a, M, E, L, C) * sign(dθ[i]), MinoDerivs1.dϕ_dλ(x[i], a, M, E, L, C)]
        d2x[i] = [MinoDerivs2.d2r_dλ(x[i], dx[i], a, M, E, L, C), MinoDerivs2.d2θ_dλ(x[i], dx[i], a, M, E, L, C), MinoDerivs2.d2ϕ_dλ(x[i], dx[i], a, M, E, L, C)]

        # compute derivatives of coordinate time wrt lambda
        dt_dλ[i] = MinoDerivs1.dt_dλ(x[i], a, M, E, L, C);
        d2t_dλ[i] = MinoDerivs2.d2t_dλ(x[i], dx[i], a, M, E, L, C);

        # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
        dλ_dt[i] = MinoTimeDerivs.dλ_dt(dt_dλ[i])
        d2λ_dt[i] = MinoTimeDerivs.d2λ_dt(dt_dλ[i], d2t_dλ[i])
    end

    @inbounds Threads.@threads for indices in ConstructSymmetricArrays.waveform_indices
        for compute_at in 1:nPoints
            if length(indices)==2
                i, j = indices
                # current quadrupole
                dSij1_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sij1data[i, j], h, nPoints)
                Sij2[i, j][compute_at] = ParameterizedDerivs.df_dt(dSij1_dλ, dλ_dt[compute_at])
            elseif length(indices)==3
                # mass octupole
                i, j, k = indices
                dMijk2_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijk2data[i, j, k], h, nPoints)
                Mijk3[i, j, k][compute_at] = ParameterizedDerivs.df_dt(dMijk2_dλ, dλ_dt[compute_at])

                dSijk1_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Sijk1data[i, j, k], h, nPoints)
                dSijk1_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Sijk1data[i, j, k], h, nPoints)
                Sijk3[i, j, k][compute_at] = ParameterizedDerivs.d2f_dt(dSijk1_dλ, dλ_dt[compute_at], dSijk1_d2λ, d2λ_dt[compute_at])
            else
                i, j, k, l = indices
                dMijkl2_dλ = FiniteDiff_5.compute_first_derivative(compute_at,  Mijkl2data[i, j, k, l], h, nPoints)
                dMijkl2_d2λ = FiniteDiff_5.compute_second_derivative(compute_at,  Mijkl2data[i, j, k, l], h, nPoints)
                Mijkl4[i, j, k, l][compute_at] = ParameterizedDerivs.d2f_dt(dMijkl2_dλ, dλ_dt[compute_at], dMijkl2_d2λ, d2λ_dt[compute_at])
            end
        end
    end

    # # symmetrize moments
    # ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij2);
    # ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk3); ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Sijk3);
    # ConstructSymmetricArrays.SymmetrizeFourIndexTensor!(Mijkl4);
end


@views function compute_waveform_moments_and_derivs_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, m::Float64, M::Float64, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, 
    xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray, v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, 
    t::Vector{Float64}, r::Vector{Float64}, rdot::Vector{Float64}, rddot::Vector{Float64}, θ::Vector{Float64}, θdot::Vector{Float64}, θddot::Vector{Float64}, ϕ::Vector{Float64},
    ϕdot::Vector{Float64}, ϕddot::Vector{Float64}, Mij2data::AbstractArray, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, 
    Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nPoints::Int64, h::Float64)

    # convert trajectories to BL coords
    @inbounds for i=1:nPoints
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]);
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]);

        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M);
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M);

        rH[i] = EstimateMultipoleDerivs.norm_3d(xH[i]);
        v[i] = EstimateMultipoleDerivs.norm_3d(vH[i]);

        x_H[i] = xH[i];
        v_H[i] = vH[i];
        a_H[i] = aH[i];

    end

    EstimateMultipoleDerivs.moments_wf!(aH[1:nPoints], a_H[1:nPoints], vH[1:nPoints], v_H[1:nPoints], xH[1:nPoints], x_H[1:nPoints], m, M, Mij2data, Mijk2data, Mijkl2data, Sij1data, Sijk1data)
    EstimateMultipoleDerivs.FiniteDifferences.moment_derivs_wf_Mino!(a, E, L, C, M, xBL, rdot, θdot, Mijk2data, Mijkl2data, Sij1data, Sijk1data, Mijk3, Mijkl4, Sij2, Sijk3, nPoints, h)
end

end

module FourierFit
using ..EstimateMultipoleDerivs
using ...MinoTimeDerivs, ...MinoDerivs1, ...MinoDerivs2, ...MinoDerivs3, ...MinoDerivs4, ...MinoDerivs5, ...MinoDerivs6
using ...HarmonicCoords
using ...FourierFitGSL
using ...ParameterizedDerivs
using ...ConstructSymmetricArrays

function moment_derivs_tr_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, M::Float64, λ::AbstractArray, x::AbstractArray, sign_dr::Float64, sign_dθ::Float64, Mij2data::AbstractArray, Mijk2data::AbstractArray,
    Sij1data::AbstractArray, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray,
    compute_at::Int64, nHarm::Int64, γr::Float64, γθ::Float64, γϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::Vector{Float64})
    
    γ = [γr, γθ, γϕ];

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

    @inbounds Threads.@threads for indices in ConstructSymmetricArrays.traj_indices
        if length(indices)==2
            i1, i2 = indices
            for multipole in EstimateMultipoleDerivs.two_index_multipoles_tr
                fit_params = zeros(2 * n_freqs + 1);
                if isequal(multipole, "mass_quad_2nd")
                    Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mij2data[i1, i2], nPoints, nHarm, chisq,  γ, fit_params)
                    df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)[compute_at]
                    d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)[compute_at]
                    d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
                    d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                    d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                    d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]

                    d3f_dt = ParameterizedDerivs.d3f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt)
                    d4f_dt = ParameterizedDerivs.d4f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt)
                    d5f_dt = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
                    d6f_dt = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)

                    @views Mij5[i1, i2] = d3f_dt
                    @views Mij6[i1, i2] = d4f_dt
                    @views Mij7[i1, i2] = d5f_dt
                    @views Mij8[i1, i2] = d6f_dt
                elseif isequal(multipole, "current_quad_1st")
                    Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Sij1data[i1, i2], nPoints, nHarm, chisq,  γ, fit_params)           
                    df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)[compute_at]
                    d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)[compute_at]
                    d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
                    d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                    d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                    d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]

                    d4f_dt = ParameterizedDerivs.d4f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt)
                    d5f_dt = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)

                    @views Sij5[i1, i2] = d4f_dt
                    @views Sij6[i1, i2] = d5f_dt
                end
            end
        else
            i1, i2, i3 = indices
            fit_params = zeros(2 * n_freqs + 1);
            Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mijk2data[i1, i2, i3], nPoints, nHarm, chisq,  γ, fit_params)
            df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)[compute_at]
            d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)[compute_at]
            d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
            d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
            d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
            d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]

            d5f_dt = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
            d6f_dt = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)
            
            @views Mijk7[i1, i2, i3] = d5f_dt
            @views Mijk8[i1, i2, i3] = d6f_dt
        end
    end

    # symmetrize moments
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij5); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij6);
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij7); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij8);
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij5); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij6);
    ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk7); ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk8);
end


function moment_derivs_wf_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, M::Float64, λ::AbstractArray, x::AbstractArray, dr::AbstractArray, dθ::AbstractArray,
    Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray,
    Sij2::AbstractArray, Sijk3::AbstractArray, nHarm::Int64, γr::Float64, γθ::Float64, γϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::Vector{Float64})
    γ = [γr, γθ, γϕ];

    dx = [Float64[] for i in eachindex(x)]
    d2x = [Float64[] for i in eachindex(x)]
    dt_dλ = zeros(nPoints)
    d2t_dλ = zeros(nPoints)
    dλ_dt = zeros(nPoints)
    d2λ_dt = zeros(nPoints)

    for i in eachindex(x)
        # compute derivatives of coordinates wrt to lambda
        dx[i] = [MinoDerivs1.dr_dλ(x[i], a, M, E, L, C) * sign(dr[i]), MinoDerivs1.dθ_dλ(x[i], a, M, E, L, C) * sign(dθ[i]), MinoDerivs1.dϕ_dλ(x[i], a, M, E, L, C)]
        d2x[i] = [MinoDerivs2.d2r_dλ(x[i], dx[i], a, M, E, L, C), MinoDerivs2.d2θ_dλ(x[i], dx[i], a, M, E, L, C), MinoDerivs2.d2ϕ_dλ(x[i], dx[i], a, M, E, L, C)]

        # compute derivatives of coordinate time wrt lambda
        dt_dλ[i] = MinoDerivs1.dt_dλ(x[i], a, M, E, L, C);
        d2t_dλ[i] = MinoDerivs2.d2t_dλ(x[i], dx[i], a, M, E, L, C);

        # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
        dλ_dt[i] = MinoTimeDerivs.dλ_dt(dt_dλ[i])
        d2λ_dt[i] = MinoTimeDerivs.d2λ_dt(dt_dλ[i], d2t_dλ[i])
    end

    @inbounds Threads.@threads for indices in ConstructSymmetricArrays.waveform_indices
        if length(indices)==2
            i1, i2 = indices
            fit_params = zeros(2 * n_freqs + 1);
            Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Sij1data[i1, i2], nPoints, nHarm, chisq,  γ, fit_params)  
            df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)
            df_dt = ParameterizedDerivs.df_dt.(df_dλ, dλ_dt)
            @views Sij2[i1, i2] = df_dt
        elseif length(indices)==3
            i1, i2, i3 = indices
            for multipole in EstimateMultipoleDerivs.three_index_multipoles_wf
                fit_params = zeros(2 * n_freqs + 1);
                if isequal(multipole, "mass_oct_2nd")
                    Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mijk2data[i1, i2, i3], nPoints, nHarm, chisq,  γ, fit_params) 
                    df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)
                    df_dt = ParameterizedDerivs.df_dt.(df_dλ, dλ_dt)
                    @views Mijk3[i1, i2, i3] = df_dt
                elseif isequal(multipole, "current_oct_1st")
                    Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Sijk1data[i1, i2, i3], nPoints, nHarm, chisq,  γ, fit_params)
                    df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)
                    d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)
                    d2f_dt = ParameterizedDerivs.d2f_dt.(df_dλ, dλ_dt, d2f_dλ, d2λ_dt)
                    @views Sijk3[i1, i2, i3] = d2f_dt
                end
            end
        else
            i1, i2, i3, i4 = indices
            fit_params = zeros(2 * n_freqs + 1);
            Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mijkl2data[i1, i2, i3, i4], nPoints, nHarm, chisq,  γ, fit_params) 
            df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)
            d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)
            d2f_dt = ParameterizedDerivs.d2f_dt.(df_dλ, dλ_dt, d2f_dλ, d2λ_dt)
            @views Mijkl4[i1, i2, i3, i4] = d2f_dt
        end
    end

    # # symmetrize moments
    # ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij2);
    # ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk3); ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Sijk3);
    # ConstructSymmetricArrays.SymmetrizeFourIndexTensor!(Mijkl4);
end


function moment_derivs_tr!(tdata::AbstractArray, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, compute_at::Int64, nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::Vector{Float64})
    Ω = [Ωr, Ωθ, Ωϕ];
    @inbounds Threads.@threads for indices in ConstructSymmetricArrays.traj_indices
        if length(indices)==2
            i1, i2 = indices
            for multipole in EstimateMultipoleDerivs.two_index_multipoles_tr
                fit_params = zeros(2 * n_freqs + 1);
                if isequal(multipole, "mass_quad_2nd")
                    Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Mij2data[i1, i2], nPoints, nHarm, chisq,  Ω, fit_params)
                    @views Mij5[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
                    @views Mij6[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                    @views Mij7[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                    @views Mij8[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]

                elseif isequal(multipole, "current_quad_1st")
                    Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Sij1data[i1, i2], nPoints, nHarm, chisq,  Ω, fit_params)                 
                    @views Sij5[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                    @views Sij6[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                end
            end
        else
            i1, i2, i3 = indices
            fit_params = zeros(2 * n_freqs + 1);
            Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Mijk2data[i1, i2, i3], nPoints, nHarm, chisq,  Ω, fit_params) 
            @views Mijk7[i1, i2, i3] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
            @views Mijk8[i1, i2, i3] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]
        end
    end

    # symmetrize moments
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij5); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij6);
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij7); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Mij8);
    ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij5); ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij6);
    ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk7); ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk8);
end


function moment_derivs_wf!(tdata::AbstractArray, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, 
    Sij2::AbstractArray, Sijk3::AbstractArray, nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::Vector{Float64})
    Ω = [Ωr, Ωθ, Ωϕ];
    @inbounds Threads.@threads for indices in ConstructSymmetricArrays.waveform_indices
        if length(indices)==2
            i1, i2 = indices
            fit_params = zeros(2 * n_freqs + 1);
            Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Sij1data[i1, i2], nPoints, nHarm, chisq,  Ω, fit_params)                 
            @views Sij2[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 1)
        elseif length(indices)==3
            i1, i2, i3 = indices
            for multipole in EstimateMultipoleDerivs.three_index_multipoles_wf
                fit_params = zeros(2 * n_freqs + 1);
                if isequal(multipole, "mass_oct_2nd")
                    Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Mijk2data[i1, i2, i3], nPoints, nHarm, chisq,  Ω, fit_params) 
                    @views Mijk3[i1, i2, i3] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 1)
                elseif isequal(multipole, "current_oct_1st")
                    Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Sijk1data[i1, i2, i3], nPoints, nHarm, chisq,  Ω, fit_params) 
                    @views Sijk3[i1, i2, i3] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 2)
                end
            end
        else
            i1, i2, i3, i4 = indices
            fit_params = zeros(2 * n_freqs + 1);
            Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Mijkl2data[i1, i2, i3, i4], nPoints, nHarm, chisq,  Ω, fit_params) 
            @views Mijkl4[i1, i2, i3, i4] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 2)
        end
    end

    # # symmetrize moments
    # ConstructSymmetricArrays.SymmetrizeTwoIndexTensor!(Sij2);
    # ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Mijk3); ConstructSymmetricArrays.SymmetrizeThreeIndexTensor!(Sijk3);
    # ConstructSymmetricArrays.SymmetrizeFourIndexTensor!(Mijkl4);
end


@views function compute_waveform_moments_and_derivs!(a::Float64, m::Float64, M::Float64, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray,
    vH::AbstractArray, v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, t::Vector{Float64}, r::Vector{Float64}, rdot::Vector{Float64}, rddot::Vector{Float64}, θ::Vector{Float64},
    θdot::Vector{Float64}, θddot::Vector{Float64}, ϕ::Vector{Float64}, ϕdot::Vector{Float64}, ϕddot::Vector{Float64}, Mij2data::AbstractArray, Mijk2data::AbstractArray, Mijkl2data::AbstractArray,
    Sij1data::AbstractArray, Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64,
    nPoints::Int64, n_freqs::Int64, chisq::Vector{Float64})

    @inbounds for i=1:nPoints
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]);
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]);

        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M);
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M);

        rH[i] = EstimateMultipoleDerivs.norm_3d(xH[i]);
        v[i] = EstimateMultipoleDerivs.norm_3d(vH[i]);

        x_H[i] = xH[i];
        v_H[i] = vH[i];
        a_H[i] = aH[i];

    end

    EstimateMultipoleDerivs.moments_wf!(aH[1:nPoints], a_H[1:nPoints], vH[1:nPoints], v_H[1:nPoints], xH[1:nPoints], x_H[1:nPoints], m, M, Mij2data, Mijk2data, Mijkl2data, Sij1data, Sijk1data)
    EstimateMultipoleDerivs.FourierFit.moment_derivs_wf!(t, Mijk2data, Mijkl2data, Sij1data, Sijk1data, Mijk3, Mijkl4, Sij2, Sijk3, nHarm, Ωr, Ωθ, Ωϕ, nPoints, n_freqs, chisq)
end

@views function compute_waveform_moments_and_derivs_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, m::Float64, M::Float64, xBL::AbstractArray, vBL::AbstractArray, aBL::AbstractArray, 
    xH::AbstractArray, x_H::AbstractArray, rH::AbstractArray, vH::AbstractArray, v_H::AbstractArray, aH::AbstractArray, a_H::AbstractArray, v::AbstractArray, 
    λ::Vector{Float64}, r::Vector{Float64}, rdot::Vector{Float64}, rddot::Vector{Float64}, θ::Vector{Float64}, θdot::Vector{Float64}, θddot::Vector{Float64}, ϕ::Vector{Float64},
    ϕdot::Vector{Float64}, ϕddot::Vector{Float64}, Mij2data::AbstractArray, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, 
    Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nHarm::Int64, γr::Float64, γθ::Float64, γϕ::Float64, 
    nPoints::Int64, n_freqs::Int64, chisq::Vector{Float64})

    @inbounds for i=1:nPoints
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]);
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]);

        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M);
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M);

        rH[i] = EstimateMultipoleDerivs.norm_3d(xH[i]);
        v[i] = EstimateMultipoleDerivs.norm_3d(vH[i]);

        x_H[i] = xH[i];
        v_H[i] = vH[i];
        a_H[i] = aH[i];

    end

    EstimateMultipoleDerivs.moments_wf!(aH[1:nPoints], a_H[1:nPoints], vH[1:nPoints], v_H[1:nPoints], xH[1:nPoints], x_H[1:nPoints], m, M, Mij2data, Mijk2data, Mijkl2data, Sij1data, Sijk1data)
    EstimateMultipoleDerivs.FourierFit.moment_derivs_wf_Mino!(a, E, L, C, M, λ, xBL, rdot, θdot, Mijk2data, Mijkl2data, Sij1data, Sijk1data, Mijk3, Mijkl4, Sij2, Sijk3, nHarm, γr, γθ, γϕ, nPoints, n_freqs, chisq)
end

end

end