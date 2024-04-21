module MatrixFitting
using LinearAlgebra
using Combinatorics
using StaticArrays
using DelimitedFiles
using DifferentialEquations
using Statistics
using LsqFit
using ..Kerr
using ..SelfForce
import ..HarmonicCoords: norm_3d, g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ..HarmonicCoords
using ..FourierFit

# returns percentaage deviation
function deviation(y_true::Float64, y_approx::Float64)
    return abs(100 * (y_true-y_approx)/y_true)
end

# prints maximum, minimum and average error of derivative N
function print_errors(data::Vector{Float64}, fitted_data::Vector{Float64}, N::Int)
    # compute percentage difference in real data and best-fit data 
    deviations = @. deviation(data, fitted_data)

    println("Error in fit to function f, derivative order $(N)")
    # println("Minimum deviation =$(minimum(abs.(deviations)))%")
    println("Maxmium deviation =$(maximum(abs.(deviations)))%")
    println("Average deviation =$(mean(abs.(deviations)))%")
end

function compute_multipoles(nPoints::Int64, n_freqs::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, saveat::Float64=0.5, Δti::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10)    
    
    # initialize ydata arrays
    Mijk_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij_data = [Float64[] for i=1:3, j=1:3]
    Sij_data = [Float64[] for i=1:3, j=1:3]

    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    fit_array_length = n_freqs
    xBL = [Float64[] for i in 1:fit_array_length]
    vBL = [Float64[] for i in 1:fit_array_length]
    aBL = [Float64[] for i in 1:fit_array_length]
    xH = [Float64[] for i in 1:fit_array_length]
    x_H = [Float64[] for i in 1:fit_array_length]
    vH = [Float64[] for i in 1:fit_array_length]
    v_H = [Float64[] for i in 1:fit_array_length]
    v = zeros(fit_array_length)
    rH = zeros(fit_array_length)
    aH = [Float64[] for i in 1:fit_array_length]
    a_H = [Float64[] for i in 1:fit_array_length]

    function geodesicEq(du, u, params, t)
        ddt = Kerr.KerrGeodesics.tddot(u..., du..., params...)
        ddr = Kerr.KerrGeodesics.rddot(u..., du..., params...)
        ddθ = Kerr.KerrGeodesics.θddot(u..., du..., params...)
        ddϕ = Kerr.KerrGeodesics.ϕddot(u..., du..., params...)
        @SArray [ddt, ddr, ddθ, ddϕ]
    end

    # orbital parameters
    params = @SArray [a, M];

    # define periastron and apastron
    ra = p * M / (1 - e);
    EEi, LLi, QQi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi)   # dimensionless constants

    # initial conditions for Kerr geodesic trajectory
    ri = ra;
    ics = Kerr.KerrGeodesics.boundKerr_ics(a, M, EEi, LLi, ri, θi, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ);
    τ0 = 0.0; Δτ = nPoints * saveat     ### note that his gives (n+1) points in the geodesic since it starts at t=0
    τF = τ0 + Δτ; params = [a, M];

    τspan = (τ0, τF)   ## overshoot into the future for fit

    # numerically solve for geodesic motion
    prob = SecondOrderODEProblem(geodesicEq, ics..., τspan, params);
    integrator=RK4()
    # integrator = OwrenZen5()
    # integrator = DP8()
    
    sol = solve(prob, AutoTsit5(integrator), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat);

    # deconstruct solution
    tdot = sol[1,(nPoints+1-n_freqs):nPoints];
    rdot = sol[2,(nPoints+1-n_freqs):nPoints];
    θdot = sol[3,(nPoints+1-n_freqs):nPoints];
    ϕdot = sol[4,(nPoints+1-n_freqs):nPoints];
    t = sol[5,(nPoints+1-n_freqs):nPoints];
    r = sol[6,(nPoints+1-n_freqs):nPoints];
    θ = sol[7,(nPoints+1-n_freqs):nPoints];
    ϕ = sol[8,(nPoints+1-n_freqs):nPoints];

    tddot = Kerr.KerrGeodesics.tddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    rddot = Kerr.KerrGeodesics.rddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    θddot = Kerr.KerrGeodesics.θddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    ϕddot = Kerr.KerrGeodesics.ϕddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);

    ###### COMPUTE MULTIPOLE MOMENTS ######

    # we first compute the fundamental frequencies in order to determine over what interval of time a fit needs to be carried out
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi); Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];
    
    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]) / tdot[i];             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]) / (tdot[i]^2);      # divide by (dt/dτ)² to get accelerations wrt BL time
    end
    @inbounds Threads.@threads for i in eachindex(t)
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
        x_H[i] = xH[i]
        rH[i] = norm_3d(xH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        v_H[i] = vH[i]; 
        v[i] = norm_3d(vH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
        a_H[i] = aH[i]
    end

    SelfForce.multipole_moments_tr!(vH, xH, x_H, m/M, M, Mij_data, Mijk_data, Sij_data)
    SelfForce.moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)


    return Mij_data, Mijk_data, Sij_data, Mij2_data, Mijk2_data, Sij1_data, Ωr, Ωθ, Ωϕ, t
end


function compute_multipoles_tau(nPoints::Int64, nPointsMultipoleFit::Int64, M::Float64, m::Float64, a::Float64, p::Float64, e::Float64, θi::Float64,  g_tt::Function, g_tϕ::Function, g_rr::Function, g_θθ::Function, g_ϕϕ::Function, saveat::Float64=0.5, Δti::Float64=1.0, reltol::Float64=1e-10, abstol::Float64=1e-10)    
    
    # initialize ydata arrays
    Mijk_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij_data = [Float64[] for i=1:3, j=1:3]
    Sij_data = [Float64[] for i=1:3, j=1:3]

    Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mij2_data = [Float64[] for i=1:3, j=1:3]
    Sij1_data = [Float64[] for i=1:3, j=1:3]
    # length of arrays for trajectory: we fit into the "past" and "future", so the arrays will have an odd size (see later code)
    fit_array_length = iseven(nPointsMultipoleFit) ? nPointsMultipoleFit+1 : nPointsMultipoleFit
    xBL = [Float64[] for i in 1:fit_array_length]
    vBL = [Float64[] for i in 1:fit_array_length]
    aBL = [Float64[] for i in 1:fit_array_length]
    xH = [Float64[] for i in 1:fit_array_length]
    x_H = [Float64[] for i in 1:fit_array_length]
    vH = [Float64[] for i in 1:fit_array_length]
    v_H = [Float64[] for i in 1:fit_array_length]
    v = zeros(fit_array_length)
    rH = zeros(fit_array_length)
    aH = [Float64[] for i in 1:fit_array_length]
    a_H = [Float64[] for i in 1:fit_array_length]

    function geodesicEq(du, u, params, t)
        ddt = Kerr.KerrGeodesics.tddot(u..., du..., params...)
        ddr = Kerr.KerrGeodesics.rddot(u..., du..., params...)
        ddθ = Kerr.KerrGeodesics.θddot(u..., du..., params...)
        ddϕ = Kerr.KerrGeodesics.ϕddot(u..., du..., params...)
        @SArray [ddt, ddr, ddθ, ddϕ]
    end

    # orbital parameters
    params = @SArray [a, M];

    # define periastron and apastron
    ra = p * M / (1 - e);
    EEi, LLi, QQi = Kerr.ConstantsOfMotion.ELQ(a, p, e, θi)   # dimensionless constants

    # initial conditions for Kerr geodesic trajectory
    ri = ra;
    ics = Kerr.KerrGeodesics.boundKerr_ics(a, M, EEi, LLi, ri, θi, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ);
    τ0 = 0.0; Δτ = nPoints * saveat     ### note that his gives (n+1) points in the geodesic since it starts at t=0
    τF = τ0 + Δτ; params = [a, M];

    τspan = (τ0, τF)   ## overshoot into the future for fit

    # numerically solve for geodesic motion
    prob = SecondOrderODEProblem(geodesicEq, ics..., τspan, params);
    integrator=RK4()
    # integrator = OwrenZen5()
    # integrator = DP8()
    
    sol = solve(prob, AutoTsit5(integrator), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat);

    # deconstruct solution
    tdot = sol[1,:];
    rdot = sol[2,:];
    θdot = sol[3,:];
    ϕdot = sol[4,:];
    tt = sol[5,:];
    rr = sol[6,:];
    θθ = sol[7,:];
    ϕϕ= sol[8,:];

    ###### COMPUTE MULTIPOLE MOMENTS ######

    # we first compute the fundamental frequencies in order to determine over what interval of time a fit needs to be carried out
    ω = Kerr.ConstantsOfMotion.KerrFreqs(a, p, e, θi); Ωr, Ωθ, Ωϕ = ω[1:3]/ω[4];
    τFit = 0.05 * minimum(@. 2π/ω[1:3]);

    # Our fitting method is as follows: we want to perform our fit over the 'future' and 'past' of the point at which we wish to compute the self-force. In other words, we would like 
    # to perform a fit to ydata, and take the values of the fit at the center of the arrays (this has obvious benefits since interpolation/numerical differentiation schemes often
    # have unwieldly "edge" effects. To achieve this centering, we first evolve a geodesic from τ0 to τF-τFit/2, in order to obtain initial condiditions of the trajectory at the 
    # time τ=τF-τFit/2. We then use these initial conditions to evolve a geodesic from τF-τFit/2 to τF+τFit/2, which places the point at which we want to compute the self force 
    # at the center of the ydata array. Then we use this ydata to carry out a "fourier fit". Note that all the geodesic ydata we compute is solely for the computation of the self-force,
    # and will be discarded thereafter.
    
    # begin by carrying out fit from τ0 to τF-τFit/2
    saveat_multipole_fit = (τFit)/(nPointsMultipoleFit-1)
    τSpanMultipoleFit = (τspan[2] - saveat_multipole_fit * (nPointsMultipoleFit ÷ 2), τspan[2] + saveat_multipole_fit * (nPointsMultipoleFit ÷ 2))    # this range ensures that τF is the center point
    compute_at = 1 + (nPointsMultipoleFit÷2)    # this will be the index of τF in the trajectory ydata arrays

    τspanFit0 = (τ0, τSpanMultipoleFit[1])
    prob = SecondOrderODEProblem(geodesicEq, ics..., τspanFit0, params);
    sol = solve(prob, AutoTsit5(integrator), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat);

    tdot = sol[1, :];
    rdot = sol[2, :];
    θdot = sol[3, :];
    ϕdot = sol[4, :];
    t = sol[5, :];
    r = sol[6, :];
    θ = sol[7, :];
    ϕ= sol[8, :];

    # carry out fit from τF-τFit/2 to τF+τFit/2

    icsMultipoleFit = [@SArray[last(tdot), last(rdot), last(θdot), last(ϕdot)], @SArray[last(t), last(r), last(θ), last(ϕ)]];    # initial conditions at τ=τF

    prob = SecondOrderODEProblem(geodesicEq, icsMultipoleFit..., τSpanMultipoleFit, params);
    
    sol = solve(prob, AutoTsit5(integrator), adaptive=true, dt=Δti, reltol = reltol, abstol = abstol, saveat=saveat_multipole_fit);

    tdot = sol[1, 1:fit_array_length];
    rdot = sol[2, 1:fit_array_length];
    θdot = sol[3, 1:fit_array_length];
    ϕdot = sol[4, 1:fit_array_length];
    t = sol[5, 1:fit_array_length];
    r = sol[6, 1:fit_array_length];
    θ = sol[7, 1:fit_array_length];
    ϕ= sol[8, 1:fit_array_length];

    tddot = Kerr.KerrGeodesics.tddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    rddot = Kerr.KerrGeodesics.rddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    θddot = Kerr.KerrGeodesics.θddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);
    ϕddot = Kerr.KerrGeodesics.ϕddot.(t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, params...);

    println("Check that we are computing the multipole moments at the right value:\n τF=$(τspan[2]), compute_at=$(sol.t[compute_at])")

    # convert trajectories to BL coords
    @inbounds Threads.@threads for i in eachindex(t)
        xBL[i] = Vector{Float64}([r[i], θ[i], ϕ[i]]);
        vBL[i] = Vector{Float64}([rdot[i], θdot[i], ϕdot[i]]) / tdot[i];             # Eq. 27: divide by dt/dτ to get velocity wrt BL time
        aBL[i] = Vector{Float64}([rddot[i], θddot[i], ϕddot[i]]) / (tdot[i]^2);      # divide by (dt/dτ)² to get accelerations wrt BL time
    end
    @inbounds Threads.@threads for i in eachindex(t)
        xH[i] = HarmonicCoords.xBLtoH(xBL[i], a, M)
        x_H[i] = xH[i]
        rH[i] = norm_3d(xH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        vH[i] = HarmonicCoords.vBLtoH(xH[i], vBL[i], a, M); 
        v_H[i] = vH[i]; 
        v[i] = norm_3d(vH[i]);
    end
    @inbounds Threads.@threads for i in eachindex(t)
        aH[i] = HarmonicCoords.aBLtoH(xH[i], vBL[i], aBL[i], a, M); 
        a_H[i] = aH[i]
    end

    SelfForce.multipole_moments_tr!(vH, xH, x_H, m/M, M, Mij_data, Mijk_data, Sij_data)
    SelfForce.moments_tr!(aH, a_H, vH, v_H, xH, x_H, m, M, Mij2_data, Mijk2_data, Sij1_data)
    Mij2_data = Mij2_data / tdot[compute_at]^2
    Sij1_data = Sij1_data / tdot[compute_at]^1
    Mijk2_data = Mijk2_data / tdot[compute_at]^2

    return Mij_data, Mijk_data, Sij_data, Mij2_data, Mijk2_data, Sij1_data, Ωr, Ωθ, Ωϕ, t, ω[1:3]..., sol.t
end

function compute_fourier_fit_Mij(Mij_data::AbstractArray, xdata::Vector{Float64}, index1::Int64, index2::Int64, nHarm::Int64, nPoints::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, a::Float64, p::Float64, e::Float64, θi::Float64, data_path::String)
    fit_fname_save=data_path * "Mij_$(index1)_$(index2)_fit_params_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θi; digits=3))_nHarm_$(nHarm)_nPoints_$(nPoints)_tmin_$(round(first(xdata); digits=3))_tmax_$(round(last(xdata))).txt"

    # load fit (if it exists)
    isfile(fit_fname_save) ? p0 = readdlm(fit_fname_save)[:] : p0 = Float64[];

    # carry out fit
    Ω, ffit, fitted_data = FourierFit.fourier_fit(xdata, Mij_data[index1, index2], Ωr, Ωθ, Ωϕ, nHarm, p0=p0)

    fit_params=coef(ffit)
    fitted_data = [FourierFit.curve_fit_functional_derivs(xdata, Ω, fit_params, N) for N=0:2]

    # save fit #
    open(fit_fname_save, "w") do io
        writedlm(io, fit_params)
    end

    return fitted_data
end

function compute_n_frequencies(nHarm::Int64)
    i=0
    # implement method used in Chimera code (e.g., the way of constructing the fitting frequencies)
    @inbounds for i_r in 0:nHarm
        @inbounds for i_θ in 0:(nHarm+i_r)
            @inbounds for i_ϕ in 0:(nHarm+i_r+i_θ)
                i+=1
            end
        end
    end
    return i
end

function compute_fitting_frequencies(Ω::Vector{Float64}, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, nHarm::Int64)
    # implement method used in Chimera code (e.g., the way of constructing the fitting frequencies)
    @inbounds for i_r in 0:nHarm
        @inbounds for i_θ in 0:(nHarm+i_r)
            @inbounds for i_ϕ in 0:(nHarm+i_r+i_θ)
                append!(Ω, abs(i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ))
            end
        end
    end
end

function fill_matrix!(t::Vector{Float64}, matrix::Matrix{ComplexF64}, Ω::Vector{Float64}, dim1::Int64)
    @inbounds for i in 1:dim1
        @views matrix[:, i] = @. exp(-im * Ω[i] * t)
    end
end

function compute_derivative(t::Vector{Float64}, fourier_coeff::Vector{ComplexF64}, Ω::Vector{Float64}, dim1::Int64, N::Int64)
    auxiliary_matrix = im * zeros(dim1, dim1)
    @inbounds for i in 1:dim1
        @views auxiliary_matrix[:, i] = @. (-im * Ω[i])^N * exp(-im * Ω[i] * t)
    end
    return  auxiliary_matrix * fourier_coeff
end

function compute_fourier_coeffs(matrix::Matrix{ComplexF64}, fitted_data::Vector{ComplexF64}, data::Vector{Float64}, fourier_coeff::Vector{ComplexF64})
    @views fourier_coeff[:] = inv(matrix) * data
    @views fitted_data[:] = matrix * fourier_coeff
end

end
