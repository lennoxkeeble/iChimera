#=

    In this module we write code to perform a multi-parameter linear least squares fit using a julia's base least squares solver. See the commentary at the beginning of "FourierFitGSL.jl" for some useful notes. We have found that for a certain range
    of test orbital functionals, this more accurately estimates the derivatives (1 part in 10^{5} in the sixth derivative for GSL versus roughly 1 part in 10^7 for this implementation), and can do so with less points in the fit, and hence either just
    as quickly or even faster. 
=#


module FourierFitJuliaBase
"""
# Common Arguments in this module
- `nHarm::Int64`: Number of harmonic frequencies in the fit. This is the number of coefficients of the radial frequency in the fit (see Notes).
- `Ω::Vector{<:Number}`: three-vector of fundamental frequencies in the fit. For fits in BL time, this should be [Ωr, Ωθ, Ωϕ], and in Mino time, this should be [ωr, ωθ, ωϕ].
- `nPoints::Int64`: Number of data points to be fit.
- `nFreqs::Int64`: Number of fitting frequencies---this is set when one chooses the number of harmonic frequencies.
- `fit_params::Vector{<:Number}`: Vector of fit parameters. The first element is the constant term, and the rest are the coefficients of the sin and cos terms for each fitting frequency.
- `nParams::Int64`: Number of fit parameters. This is 2 * nFreqs + 1, where the +1 is for the constant term, and the factor of 2 is for the sin and cos terms for each fitting frequency.
- `xdata::Vector{<:Number}`: Vector of x-values for the data to be fit (this should be a vector of coordinate time or Mino time).
- `ydata::Vector{<:Number}`: Vector of y-values for the data to be fit (this should be a vector of the orbital functional data).

# Notes
- The relevant equations in arXiv:1109.0572v2 are Eqs. 98-99, though we perform the fit here in terms of sines and cosines, so that there are two coefficients for each fitting frequency.
- The number of harmonics, nHarm, is the number of coefficients of the radial frequency in the fit. The fitting frequencies are then constructed by summing over integer multiples of the the fundamental frequencies, with multiples of the radial
frequency ranging from k = 0 to nHarm, multiples of the polar frequency ranging from m = -k to nHarm, and multiples of the azimuthal frequency ranging from l = -(k+m) to nHarm. This is the convention followed in (arXiv:1109.0572v2), but, of course, this
construction is not unique.
- We provide master functions that allow one to perform the fits without specifying the number of fundamental frequencies explicitly. For example, for an equatorial orbit, the polar frequency is infinite, and the functions we provide for computing the
fundamental frequencies will set their value to 1e12. The master functions select out the frequencies which are <1e9, and thus only those that are well defined. They then proceed to fit with only these fundamental frequencies.
"""

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
function curve_fit_functional(fit_params::AbstractVector{T}, fit_freqs::AbstractVector{T}, t::AbstractVector{T}) where T<:Real
    f = fit_params[1] * ones(length(t))
    @inbounds for i in eachindex(t)
        @inbounds @views for j in eachindex(fit_freqs)
            f[i] += fit_params[2*j]*cos(fit_freqs[j] * t[i]) + fit_params[2*j+1]*sin(fit_freqs[j]*t[i])
        end
    end
    return f
end


# compute Nth derivative from fourier series expansion (see)
function curve_fit_functional_derivs(fit_params::AbstractVector{T}, fit_freqs::AbstractVector{T}, t::AbstractVector{T}, N::Int) where T<:Real
    f = N == 0 ? fit_params[1] * ones(length(t)) : zeros(length(t))
    @inbounds for i in eachindex(t)
        @inbounds for j in eachindex(fit_freqs)
            f[i] += (fit_freqs[j] ^ N) * (fit_params[2*j]*cos(fit_freqs[j] * t[i] + N * π / 2) + fit_params[2*j+1]*sin(fit_freqs[j]*t[i] + N * π / 2))
        end
    end
    return f
end

# compute coefficient matrix for linear system---note that the convention here for the coefficients is to have the constant term first, followed by the sin and cos terms for each frequency, which are adjacent to each other (as oppoed
# to the convention in the GSL fit, where the parameters are ordered in terms of cosines first and sines last, so that the coefficients for a given frequency are not adjacent to each other and are separated by 'n_freqs' elements)
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