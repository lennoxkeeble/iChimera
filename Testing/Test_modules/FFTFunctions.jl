#=

    In this module we write several functions useful for computing fourier transforms, and extracting useful things from them

=#

module  FourierFunctions
using FFTW, ..FourierFitGSL, ..FourierFitGSL_Derivs, GSL

# find local maxima of a signal
function findlocalmaxima(signal::Vector{Float64})
    inds = Int[]
    if length(signal)>1
        if signal[1]>signal[2]
            push!(inds,1)
        end
        for i=2:length(signal)-1
            if signal[i-1]<signal[i]>signal[i+1]
                push!(inds,i)
            end
        end
        if signal[end]>signal[end-1]
            push!(inds,length(signal))
        end
    end
    inds
end

# computes FFT of a real signal
function compute_real_FFT(signal::Vector{Float64}, fs::Float64, nPoints::Int64)
    F = rfft(signal)
    F_freqs = rfftfreq(nPoints, fs)
    return F, F_freqs
end


# this function takes as input some real FFT, and outputs the N most dominant harmonic frequenices — nFreqs==-1 for all peak frequencies
function extract_frequencies!(F::Vector{ComplexF64}, F_freqs::Frequencies{Float64}, nFreqs::Int64, exclude_zero::Bool)
    # finding local maxima of the fourier transform
    dominant_indices = findlocalmaxima(abs.(F))

    # extract peaks and the corresponding frequencies
    peaks_freqs = F_freqs[dominant_indices]
    peaks_F_vals = abs.(F[dominant_indices])

    # now sort frequencies in decreasing order of the height of their respective peaks
    perm = sortperm(peaks_F_vals, rev=true); 
    peaks_freqs .= peaks_freqs[perm]
    peaks_F_vals .= peaks_F_vals[perm]

    # extract most dominant frequencies
    if nFreqs==-1
        ordered_freqs = peaks_freqs;  ordered_F_vals = peaks_F_vals;
    else
        ordered_freqs = peaks_freqs[1:nFreqs];  ordered_F_vals = peaks_F_vals[1:nFreqs];
    end

    # exclude zero frequency
    if exclude_zero==true
        for i in eachindex(ordered_freqs)
            if ordered_freqs[i]==0.0
                println(i)
                deleteat!(ordered_freqs, i); 
                deleteat!(ordered_F_vals, i);
                
                if nFreqs!=-1
                    append!(ordered_freqs, peaks_freqs[nFreqs+1])
                    append!(ordered_F_vals, peaks_F_vals[nFreqs+1])
                end
                break
            end
        end
    end

    return 2π .* ordered_freqs, ordered_F_vals
end

#= 
    In the function below we implement a different method for fitting orbital functionals to their fourier series expansion. Instead of computing the fitting frequencies from sums of
    the fundamental frequencies, e.g., Ωkmn = kΩr + mΩθ + nΩϕ, we instead take in the signal to which we wish to fit, fourier transform it and extract the dominant frequencies.
    We then use these fourier-extracted frequencies for Ωkmn.
=#

# TO-DO: deal with zero frequencies and/or possible repeated frequencies?
function GSL_fit!(xdata::Vector{Float64}, ydata::Vector{Float64}, peak_freqs::Vector{Float64}, orbital_freqs::Vector{Float64}, nPoints::Int64, nFittingFreqs::Int64, nFFTFreqs::Int64, chisq::Vector{Float64}, fit_params::Vector{Float64})    
    # fitting frequencies
    fitting_freqs = zeros(nFittingFreqs)
    total_num_FFT_freqs = length(peak_freqs)
    additional_freqs = orbital_freqs

    if nFFTFreqs > total_num_FFT_freqs &&  nFFTFreqs < nFittingFreqs
        FFT_freqs = peak_freqs
    elseif nFFTFreqs < total_num_FFT_freqs
        FFT_freqs = peak_freqs[1:nFFTFreqs]
    elseif nFFTFreqs > total_num_FFT_freqs &&  nFFTFreqs > nFittingFreqs
        throw(DomainError("There are only $(nFittingFreqs) harmonic frequencies, but $(nFFTFreqs) frequencies were requested from the FFT, which has $(total_num_FFT_freqs) frequencies"))
    end
    # println("There are $(nFittingFreqs) harmonic frequencies and $(nFFTFreqs) frequencies were requested from the FFT, which has $(total_num_FFT_freqs) frequencies")
    # construct fitting frequencies with the selected Float64 of FFT frequenices, and the remaining given by Ωkmn given from the fundamental frequencies
   
    if nFFTFreqs < nFittingFreqs
        for freq in FFT_freqs
            closest_fund_freq = argmin(additional_freqs .- freq)
            deleteat!(additional_freqs, closest_fund_freq)
        end
        fitting_freqs = vcat(FFT_freqs, additional_freqs)
    else
        fitting_freqs = FFT_freqs
    end


    n_coeffs = 2 * nFittingFreqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency
    # allocate memory and fill GSL vectors and matrices
    x, y, X, c, cov, work = FourierFitGSL.allocate_memory(nPoints, n_coeffs)
    FourierFitGSL.fill_gsl_vectors!(x, y, xdata, ydata, nPoints)
    FourierFitGSL.fill_predictor_matrix!(X, xdata, nPoints, nFittingFreqs, n_coeffs, fitting_freqs)

    # carry out fit and store best fit params
    # println("Carrying out fit")
    FourierFitGSL.curve_fit!(y, X, c, cov, work, chisq)
    @views fit_params[:] = GSL.wrap_gsl_vector(c)

    # free memory
    FourierFitGSL.free_memory!(x, y, X, c, cov, work)
    return fitting_freqs, FFT_freqs
end


# TO-DO: deal with zero frequencies and/or possible repeated frequencies?
function GSL_fit!(xdata::Vector{Float64}, ydata::Vector{Float64}, ydata_1::Vector{Float64}, ydata_2::Vector{Float64}, peak_freqs::Vector{Float64}, orbital_freqs::Vector{Float64}, nPoints::Int64, nFittingFreqs::Int64, nFFTFreqs::Int64, chisq::Vector{Float64}, fit_params::Vector{Float64})    
    # # fitting frequencies
    # fitting_freqs = zeros(nFittingFreqs)
    # total_num_FFT_freqs = length(peak_freqs)
    # additional_freqs = orbital_freqs

    # if nFFTFreqs > total_num_FFT_freqs &&  nFFTFreqs < nFittingFreqs
    #     FFT_freqs = peak_freqs
    # elseif nFFTFreqs < total_num_FFT_freqs
    #     FFT_freqs = peak_freqs[1:nFFTFreqs]
    # elseif nFFTFreqs > total_num_FFT_freqs &&  nFFTFreqs > nFittingFreqs
    #     throw(DomainError("There are only $(nFittingFreqs) harmonic frequencies, but $(nFFTFreqs) frequencies were requested from the FFT, which has $(total_num_FFT_freqs) frequencies"))
    # end
    # # println("There are $(nFittingFreqs) harmonic frequencies and $(nFFTFreqs) frequencies were requested from the FFT, which has $(total_num_FFT_freqs) frequencies")
    # # construct fitting frequencies with the selected Float64 of FFT frequenices, and the remaining given by Ωkmn given from the fundamental frequencies
   
    # if nFFTFreqs < nFittingFreqs
    #     for freq in FFT_freqs
    #         closest_fund_freq = argmin(additional_freqs .- freq)
    #         deleteat!(additional_freqs, closest_fund_freq)
    #     end
    #     fitting_freqs = vcat(FFT_freqs, additional_freqs)
    # else
    #     fitting_freqs = FFT_freqs
    # end

    fitting_freqs = nFFTFreqs < length(peak_freqs) ? peak_freqs[1:nFFTFreqs] : peak_freqs
    nFittingFreqs = length(fitting_freqs)


    n_coeffs = 2 * nFittingFreqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency

    # allocate memory and fill GSL vectors and matrices
    x, y, X, c, cov, work = FourierFitGSL_Derivs.allocate_memory(nPoints, n_coeffs)
    FourierFitGSL_Derivs.fill_gsl_vectors!(x, y, xdata, ydata, ydata_1, ydata_2, nPoints)
    FourierFitGSL_Derivs.fill_predictor_matrix!(X, xdata, nPoints, nFittingFreqs, n_coeffs, fitting_freqs)

    # carry out fit and store best fit params
    # println("Carrying out fit")
    FourierFitGSL_Derivs.curve_fit!(y, X, c, cov, work, chisq)
    @views fit_params[:] = GSL.wrap_gsl_vector(c)

    # free memory
    FourierFitGSL_Derivs.free_memory!(x, y, X, c, cov, work)
    return fitting_freqs, fitting_freqs
end


end