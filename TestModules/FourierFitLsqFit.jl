#=

    In this module we write code to perform a multi-parameter linear least squares fit using the LsqFit library, where the functional form to which we fit is specifically the 
    fundamental-frequency Fourier series expansion

=#

module FourierFitLsqFit
using LsqFit

# compute Float64 of fitting frequencies for fits with one, two, and three fundamental frequencies for a given harmonic (-1 since we don't count constant term)
compute_num_fitting_freqs_1(nHarm::Int64) = nHarm
compute_num_fitting_freqs_2(nHarm::Int64) = Int((nHarm * (5 + 3 * nHarm) / 2))
compute_num_fitting_freqs_3(nHarm::Int64) = Int( nHarm * (13 + 2 * nHarm * (9 + 4 * nHarm)) / 3)

function compute_num_fitting_freqs_master(nHarm::Int64, Ω::Vector{Float64})
    num_freqs = sum(Ω .< 1e9)
    if num_freqs==3
        compute_num_fitting_freqs_3(nHarm)
    elseif num_freqs==2
        compute_num_fitting_freqs_2(nHarm)
    elseif num_freqs==1
        compute_num_fitting_freqs_1(nHarm)
    end
end


# construct fitting frequencies for one fundamental frequency
function compute_fitting_frequencies_1(nHarm::Int64, Ωr::Float64)
    Ω = Float64[]
    @inbounds for i_r in 1:nHarm
        append!(Ω, i_r * Ωr)
    end
    return Ω
end

# chimera construct fitting frequencies
function compute_fitting_frequencies_2(nHarm::Int64, Ωr::Float64, Ωθ::Float64)
    Ω = Float64[]
    @inbounds for i_r in 0:nHarm
        @inbounds for i_θ in -i_r:nHarm
            (i_r==0 && i_θ==0) ? nothing : append!(Ω, abs(i_r * Ωr + i_θ * Ωθ))
        end
    end
    return Ω
end


# chimera construct fitting frequencies
function compute_fitting_frequencies_3(nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64)
    Ω = Float64[]
    @inbounds for i_r in 0:nHarm
        @inbounds for i_θ in -i_r:nHarm
            @inbounds for i_ϕ in -(i_r+i_θ):nHarm
                (i_r==0 && i_θ==0 && i_ϕ==0) ? nothing : append!(Ω, abs(i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ))
            end
        end
    end
    return Ω
end

function compute_fitting_frequencies_master(nHarm::Int64, Ω::Vector{Float64})
    freqs = Ω[Ω .< 1e9];
    num_freqs = length(freqs);
    if num_freqs==1
        compute_fitting_frequencies_1(nHarm, freqs...)
    elseif num_freqs==2
        compute_fitting_frequencies_2(nHarm, freqs...)
    elseif num_freqs==3
        compute_fitting_frequencies_3(nHarm, freqs...)
    end
end

# functional form to which we fit data
function curve_fit_functional(params::AbstractVector{T}, fit_freqs::AbstractVector{T}, t::AbstractVector{T}) where T<:Real
    f = params[1] * ones(length(t))
    @inbounds for i in eachindex(t)
        @inbounds @views for j in eachindex(fit_freqs)
            f[i] += params[2*j]*cos(fit_freqs[j] * t[i]) + params[2*j+1]*sin(fit_freqs[j]*t[i])
        end
    end
    return f
end

function curve_fit_functional_with_first_deriv(params::AbstractVector{T}, fit_freqs::AbstractVector{T}, t::AbstractVector{T}) where T<:Real
    f = params[1] * ones(div(length(t), 2))
    g = zeros(div(length(t), 2))
    @inbounds for i in eachindex(f)
        @inbounds @views for j in eachindex(fit_freqs)
            f[i] += params[2*j]*cos(fit_freqs[j] * t[i]) + params[2*j+1]*sin(fit_freqs[j]*t[i])
            g[i] += (fit_freqs[j]) * (params[2*j]*cos(fit_freqs[j] * t[i] + π/2) + params[2*j+1]*sin(fit_freqs[j]*t[i] + π/2))
        end
    end
    return [f; g]
end

function curve_fit_functional_with_second_deriv(params::AbstractVector{T}, fit_freqs::AbstractVector{T}, t::AbstractVector{T}) where T<:Real
    f = params[1] * ones(div(length(t), 3))
    g = zeros(div(length(t), 3))
    h = zeros(div(length(t), 3))
    @inbounds for i in eachindex(f)
        @inbounds @views for j in eachindex(fit_freqs)
            f[i] += params[2*j]*cos(fit_freqs[j] * t[i]) + params[2*j+1]*sin(fit_freqs[j]*t[i])
            g[i] += (fit_freqs[j]) * (params[2*j]*cos(fit_freqs[j] * t[i] + π/2) + params[2*j+1]*sin(fit_freqs[j]*t[i] + π/2))
            h[i] += (fit_freqs[j])^2 * (params[2*j]*cos(fit_freqs[j] * t[i] + π) + params[2*j+1]*sin(fit_freqs[j]*t[i] + π))
        end
    end
    return [f; g; h]
end

# compute Nth derivative
function curve_fit_functional_derivs(params::AbstractVector{T}, fit_freqs::AbstractVector{T}, t::AbstractVector{T}, N::Int) where T<:Real
    f = N == 0 ? params[1] * ones(length(t)) : zeros(length(t))
    @inbounds for i in eachindex(t)
        @inbounds for j in eachindex(fit_freqs)
            f[i] += (fit_freqs[j] ^ N) * (params[2*j]*cos(fit_freqs[j] * t[i] + N * π / 2) + params[2*j+1]*sin(fit_freqs[j]*t[i] + N * π / 2))
        end
    end
    return f
end

# master functions for carrying out fit with one, two or three fundamental frequencies
function LsqFit_1!(xdata::Vector{Float64}, ydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_1(nHarm, Ω1)
    n_freqs = compute_num_fitting_freqs_1(nHarm)
    
    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, xdata, ydata, p0)
    @views fit_params[:] = fit.param

    return Ω_fit
end

function LsqFit_2!(xdata::Vector{Float64}, ydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, Ω2::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_2(nHarm, Ω1, Ω2)
    n_freqs = compute_num_fitting_freqs_2(nHarm)

    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, xdata, ydata, p0)
    @views fit_params[:] = fit.param
    
    return Ω_fit
end

function LsqFit_3!(xdata::Vector{Float64}, ydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, Ω2::Float64, Ω3::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_3(nHarm, Ω1, Ω2, Ω3)
    n_freqs = compute_num_fitting_freqs_3(nHarm)

    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, xdata, ydata, p0)
    @views fit_params[:] = fit.param
    
    return Ω_fit
end

function LsqFit_master!(xdata::Vector{Float64}, ydata::Vector{Float64}, nHarm::Int64,  Ω::Vector{Float64}, fit_params::Vector{Float64})
    freqs = Ω[Ω .< 1e9];
    num_freqs = length(freqs);
    if num_freqs==1
        LsqFit_1!(xdata, ydata, nHarm, freqs..., fit_params)
    elseif num_freqs==2
        LsqFit_2!(xdata, ydata, nHarm, freqs..., fit_params)
    elseif num_freqs==3
        LsqFit_3!(xdata, ydata, nHarm, freqs..., fit_params)
    end
end

#### fitting functions which simultaneously fit the data and its first (and possibly second) derivative
# master functions for carrying out fit with one, two or three fundamental frequencies
function LsqFit_first_deriv_1!(xdata::Vector{Float64}, ydata::Vector{Float64}, dydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_1(nHarm, Ω1)
    n_freqs = compute_num_fitting_freqs_1(nHarm)
    
    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional_with_first_deriv(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, [xdata; xdata], [ydata; dydata], p0)
    @views fit_params[:] = fit.param

    return Ω_fit
end

function LsqFit_first_deriv_2!(xdata::Vector{Float64}, ydata::Vector{Float64}, dydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, Ω2::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_2(nHarm, Ω1, Ω2)
    n_freqs = compute_num_fitting_freqs_2(nHarm)

    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional_with_first_deriv(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, [xdata; xdata], [ydata; dydata], p0)
    @views fit_params[:] = fit.param
    
    return Ω_fit
end

function LsqFit_first_deriv_3!(xdata::Vector{Float64}, ydata::Vector{Float64}, dydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, Ω2::Float64, Ω3::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_3(nHarm, Ω1, Ω2, Ω3)
    n_freqs = compute_num_fitting_freqs_3(nHarm)

    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional_with_first_deriv(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, [xdata; xdata], [ydata; dydata], p0)
    @views fit_params[:] = fit.param
    
    return Ω_fit
end

function LsqFit_first_deriv_master!(xdata::Vector{Float64}, ydata::Vector{Float64}, dydata::Vector{Float64}, nHarm::Int64,  Ω::Vector{Float64}, fit_params::Vector{Float64})
    freqs = Ω[Ω .< 1e9];
    num_freqs = length(freqs);
    if num_freqs==1
        LsqFit_first_deriv_1!(xdata, ydata, dydata, nHarm, freqs..., fit_params)
    elseif num_freqs==2
        LsqFit_first_deriv_2!(xdata, ydata, dydata, nHarm, freqs..., fit_params)
    elseif num_freqs==3
        LsqFit_first_deriv_3!(xdata, ydata, dydata, nHarm, freqs..., fit_params)
    end
end

function LsqFit_second_deriv_1!(xdata::Vector{Float64}, ydata::Vector{Float64}, dydata::Vector{Float64}, d2ydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_1(nHarm, Ω1)
    n_freqs = compute_num_fitting_freqs_1(nHarm)
    
    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional_with_second_deriv(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, [xdata; xdata; xdata], [ydata; dydata; d2ydata], p0)
    @views fit_params[:] = fit.param

    return Ω_fit
end

function LsqFit_second_deriv_2!(xdata::Vector{Float64}, ydata::Vector{Float64}, dydata::Vector{Float64}, d2ydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, Ω2::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_2(nHarm, Ω1, Ω2)
    n_freqs = compute_num_fitting_freqs_2(nHarm)

    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional_with_second_deriv(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, [xdata; xdata; xdata], [ydata; dydata; d2ydata], p0)
    @views fit_params[:] = fit.param
    
    return Ω_fit
end

function LsqFit_second_deriv_3!(xdata::Vector{Float64}, ydata::Vector{Float64}, dydata::Vector{Float64}, d2ydata::Vector{Float64}, nHarm::Int64,  Ω1::Float64, Ω2::Float64, Ω3::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitLsqFit.compute_fitting_frequencies_3(nHarm, Ω1, Ω2, Ω3)
    n_freqs = compute_num_fitting_freqs_3(nHarm)

    model(t::AbstractVector{T}, p::AbstractVector{T}) where T<:Real = curve_fit_functional_with_second_deriv(p, Ω_fit, t)
    p0 = zeros(2 * n_freqs + 1)
    fit = curve_fit(model, [xdata; xdata; xdata], [ydata; dydata; d2ydata], p0)
    @views fit_params[:] = fit.param
    
    return Ω_fit
end

function LsqFit_second_deriv_master!(xdata::Vector{Float64}, ydata::Vector{Float64}, dydata::Vector{Float64}, d2ydata::Vector{Float64}, nHarm::Int64,  Ω::Vector{Float64}, fit_params::Vector{Float64})
    freqs = Ω[Ω .< 1e9];
    num_freqs = length(freqs);
    if num_freqs==1
        LsqFit_second_deriv_1!(xdata, ydata, dydata, d2ydata, nHarm, freqs..., fit_params)
    elseif num_freqs==2
        LsqFit_second_deriv_2!(xdata, ydata, dydata, d2ydata, nHarm, freqs..., fit_params)
    elseif num_freqs==3
        LsqFit_second_deriv_3!(xdata, ydata, dydata, d2ydata, nHarm, freqs..., fit_params)
    end
end

end