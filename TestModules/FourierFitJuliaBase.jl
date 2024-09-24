#=

    In this module write the functions that fit orbital functional time series to their fourier expansion using Julias base least squares solver.

=#

module FourierFitJuliaBase

# compute number of fitting frequencies for fits with one, two, and three fundamental frequencies for a given harmonic (-1 since we don't count constant term)
compute_num_fitting_freqs_1(nHarm::Int)::Int = nHarm
compute_num_fitting_freqs_2(nHarm::Int)::Int = Int((nHarm * (5 + 3 * nHarm) / 2))
compute_num_fitting_freqs_3(nHarm::Int)::Int = Int( nHarm * (13 + 2 * nHarm * (9 + 4 * nHarm)) / 3)

function compute_num_fitting_freqs_master(nHarm::Int, Ω::Vector{<:Number})::Int
    num_freqs = sum(Ω .< 1e9)
    if num_freqs==3
        return compute_num_fitting_freqs_3(nHarm)
    elseif num_freqs==2
        return compute_num_fitting_freqs_2(nHarm)
    elseif num_freqs==1
        return compute_num_fitting_freqs_1(nHarm)
    end
end

# construct fitting frequencies for one fundamental frequency
function compute_fitting_frequencies_1(nHarm::Int, Ωr::Number)::Vector{<:Number}
    Ω = Float64[]
    @inbounds for i_r in 1:nHarm
        append!(Ω, i_r * Ωr)
    end
    return Ω
end

# chimera construct fitting frequencies
function compute_fitting_frequencies_2(nHarm::Int, Ωr::Number, Ωθ::Number)::Vector{<:Number}
    Ω = Float64[]
    @inbounds for i_r in 0:nHarm
        @inbounds for i_θ in -i_r:nHarm
            (i_r==0 && i_θ==0) ? nothing : append!(Ω, abs(i_r * Ωr + i_θ * Ωθ))
        end
    end
    return Ω
end


# chimera construct fitting frequencies
function compute_fitting_frequencies_3(nHarm::Int, Ωr::Number, Ωθ::Number, Ωϕ::Number)::Vector{<:Number}
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

function compute_fitting_frequencies_master(nHarm::Int, Ω::Vector{<:Number})::Vector{<:Number}
    freqs = Ω[Ω .< 1e9];
    num_freqs = length(freqs);
    if num_freqs==1
        return compute_fitting_frequencies_1(nHarm, freqs...)
    elseif num_freqs==2
        return compute_fitting_frequencies_2(nHarm, freqs...)
    elseif num_freqs==3
        return compute_fitting_frequencies_3(nHarm, freqs...)
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

# compute coefficient matrix for linear system
function compute_coefficient_matrix(nFreqs::Int, nParams::Int, nPoints::Int, xdata::Vector{<:Number}, fit_freqs::Vector{<:Number})::AbstractArray
    A = ones(nPoints, nParams)
    @inbounds for i in 1:nPoints
        @inbounds for j in 1:nFreqs
            A[i, 2*j] = cos(fit_freqs[j] * xdata[i])
            A[i, 2*j + 1] = sin(fit_freqs[j] * xdata[i])
        end
    end
    return A
end

# master functions for carrying out fit with one, two or three fundamental frequencies
function Fit_1_freqs!(xdata::Vector{<:Number}, ydata::Vector{<:Number}, nPoints::Int, nHarm::Int,  Ω1::Number, fit_params::Vector{<:Number})::Vector{<:Number}
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitJuliaBase.compute_fitting_frequencies_1(nHarm, Ω1)
    n_freqs = compute_num_fitting_freqs_1(nHarm)
    n_coeffs = 2 * n_freqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency

    # allocate memory and fill GSL vectors and matrices
    coeff_matrix = compute_coefficient_matrix(n_freqs, n_coeffs, nPoints, xdata, Ω_fit)
    @views fit_params[:] = coeff_matrix \ ydata
    return Ω_fit
end

function Fit_2_freqs!(xdata::Vector{<:Number}, ydata::Vector{<:Number}, nPoints::Int, nHarm::Int, Ω1::Number, Ω2::Number, fit_params::Vector{<:Number})::Vector{<:Number}
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitJuliaBase.compute_fitting_frequencies_2(nHarm, Ω1, Ω2)
    n_freqs = compute_num_fitting_freqs_2(nHarm)
    n_coeffs = 2 * n_freqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency

    # allocate memory and fill GSL vectors and matrices
    coeff_matrix = compute_coefficient_matrix(n_freqs, n_coeffs, nPoints, xdata, Ω_fit)
    @views fit_params[:] = coeff_matrix \ ydata
    return Ω_fit
end

function Fit_3_freqs!(xdata::Vector{<:Number}, ydata::Vector{<:Number}, nPoints::Int, nHarm::Int, Ω1::Number, Ω2::Number, Ω3::Number, fit_params::Vector{<:Number})::Vector{<:Number}
    # compute fitting frequncies and their Float64
    Ω_fit = FourierFitJuliaBase.compute_fitting_frequencies_3(nHarm, Ω1, Ω2, Ω3)
    n_freqs = compute_num_fitting_freqs_3(nHarm)
    n_coeffs = 2 * n_freqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency

    # allocate memory and fill GSL vectors and matrices
    coeff_matrix = compute_coefficient_matrix(n_freqs, n_coeffs, nPoints, xdata, Ω_fit)
    @views fit_params[:] = coeff_matrix \ ydata
    return Ω_fit
end

function Fit_master!(xdata::Vector{<:Number}, ydata::Vector{<:Number}, nPoints::Int, nHarm::Int, Ω::Vector{<:Number}, fit_params::Vector{<:Number})::Vector{<:Number}
    freqs = Ω[Ω .< 1e9];
    num_freqs = length(freqs);
    if num_freqs==1
        return Fit_1_freqs!(xdata, ydata, nPoints, nHarm, freqs..., fit_params)
    elseif num_freqs==2
        return Fit_2_freqs!(xdata, ydata, nPoints, nHarm, freqs..., fit_params)
    elseif num_freqs==3
        return Fit_3_freqs!(xdata, ydata, nPoints, nHarm, freqs..., fit_params)
    end
end

end