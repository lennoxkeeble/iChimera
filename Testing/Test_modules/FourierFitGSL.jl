#=

    In this module we write code to perform a multi-parameter linear least squares fit using the GSL library, where the functional form to which we fit is specifically the 
    fundamental-frequency Fourier series expansion

=#

module FourierFitGSL
using GSL

# compute number of fitting frequencies for a given harmonic (-1 since we don't count constant term)
compute_num_freqs(nHarm::Int64)= Int((1 + nHarm) * (2 + 8 * nHarm + 7 * nHarm^2) / 2 - 1)

# functional form to which we fit data
function curve_fit_functional(f::Vector{Float64}, tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, n_freqs::Int64)
    @inbounds Threads.@threads for i in eachindex(tdata)
        f[i]=params[1]    # first parameter is the constant term
        @inbounds for j in eachindex(Ω)
            f[i] += params[j+1] * cos(Ω[j] * tdata[i]) + params[n_freqs + j+1] * sin(Ω[j] * tdata[i])
        end
    end
    return f
end


# compute Nth derivative
function curve_fit_functional_derivs(tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, n_freqs::Int64, n_points::Int64, N::Int64)
    # if computing the functional form (i.e., zeroth derivative), then must include constant term
    f = N==0 ? params[1] * ones(n_points) : zeros(n_points)
    @inbounds for i in eachindex(tdata)
        for j in eachindex(Ω)  
            f[i] += (Ω[j]^N) * (params[j+1] * cos(Ω[j] * tdata[i] + N*π/2) + params[n_freqs + j+1] * sin(Ω[j] * tdata[i] + N*π/2))
        end
    end
    return f
end

# row constructor for predictor matrix in GSL fit
function GSL_fourier_model(t::Vector{Float64}, Ω::Vector{Float64}, n_freqs::Int64)
    f=ones(n_freqs+1)
    @inbounds for i in 1:n_freqs
        f[i+1] = cos(Ω[i] * t)
        f[i+1+n_freqs] = sin(Ω[i] * t)
    end
    return f
end

# construct fitting frequencies
function compute_fitting_frequenices(nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64)
    Ω = Float64[]
    @inbounds for i_r in 0:nHarm
        @inbounds for i_θ in 0:(nHarm+i_r)
            @inbounds for i_ϕ in 0:(nHarm+i_r+i_θ)
                (i_r==0 && i_θ==0 && i_ϕ==0) ? nothing : append!(Ω, i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ)
            end
        end
    end
    return Ω
end

# allocate memory for fitting method input
function allocate_memory(n_p::Int64, n_coeffs::Int64)
    x = GSL.vector_alloc(n_p)
    y = GSL.vector_alloc(n_p)
    X = GSL.matrix_alloc(n_p, n_coeffs)
    c = GSL.vector_alloc(n_coeffs)
    cov = GSL.matrix_alloc(n_coeffs, n_coeffs)
    work = GSL.multifit_linear_alloc(n_p, n_coeffs)
    return x, y, X, c, cov, work
end

# free memory used for fitting
function free_memory!(x::Ptr{gsl_vector}, y::Ptr{gsl_vector}, X::Ptr{gsl_matrix}, c::Ptr{gsl_vector}, cov::Ptr{gsl_matrix}, work::Ptr{gsl_multifit_linear_workspace})
    GSL.vector_free(x)
    GSL.vector_free(y)
    GSL.matrix_free(X)
    GSL.vector_free(c)
    GSL.matrix_free(cov)
    GSL.multifit_linear_free(work)
end

# fill GSL vectors 'xdata' and 'ydata' for the fit
function fill_gsl_vectors!(xGSL::Ptr{gsl_vector}, yGSL::Ptr{gsl_vector}, x::Vector{Float64}, y::Vector{Float64}, n_p::Int64)
    @inbounds Threads.@threads for i=0:(n_p-1)
        GSL.vector_set(xGSL, i, x[i+1])
        GSL.vector_set(yGSL, i, y[i+1])
    end
end

#=

    The predictor matrix X has a number of rows equal to the number of y-values to which we are fitting. Each row, therefore, consists of the functional form to which we 
    are fitting evaluated at each element of the x vector. The input function 'model' must output a vector whose elements are the componenets of the functional form. In other words,
    if the function form we are fitting to is f(x), then we must have f(x) = sum(model(x))

=#

# row constructor for predictor matrix in GSL fit
function GSL_fourier_model(t::Float64, Ω::Vector{Float64}, n_freqs::Int64,  n_coeffs::Int64)
    f=ones(n_coeffs)    # a constant term, and then a cosine and sine term for each fitting frequency
    @inbounds for i in 1:n_freqs
        f[i+1] = cos(Ω[i] * t)
        f[i+1+n_freqs] = sin(Ω[i] * t)
    end
    return f
end

# fill predictor matrix X: N is number of harmonics, and Ω_fit is the fitting frequencies
function fill_predictor_matrix!(X::Ptr{gsl_matrix}, x::Vector{Float64}, n_p::Int64, n_freqs::Int64,  n_coeffs::Int64, Ω_fit::Vector{Float64})
    # construct the fit matrix X 
    @inbounds Threads.@threads for i=0:n_p-1
        Xij = FourierFitGSL.GSL_fourier_model(x[i+1], Ω_fit, n_freqs, n_coeffs)
        # fill in row i of X 
        for j=0:(2 * n_freqs)
            GSL.matrix_set(X, i, j, Xij[j+1])
        end
    end
end

# call multilinear fit
function curve_fit!(y::Ptr{gsl_vector}, X::Ptr{gsl_matrix}, c::Ptr{gsl_vector}, cov::Ptr{gsl_matrix}, work::Ptr{gsl_multifit_linear_workspace}, chisq::Vector{Float64})
    GSL.multifit_linear(X, y, c, cov, chisq, work)
end

# master function for carrying out fit
function GSL_fit!(xdata::Vector{Float64}, ydata::Vector{Float64}, n_p::Int64, nHarm::Int64, chisq::Vector{Float64},  Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their number
    Ω_fit = FourierFitGSL.compute_fitting_frequenices(nHarm, Ωr, Ωθ, Ωϕ)
    n_freqs = compute_num_freqs(nHarm)
    n_coeffs = 2 * n_freqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency

    # allocate memory and fill GSL vectors and matrices
    x, y, X, c, cov, work = FourierFitGSL.allocate_memory(n_p, n_coeffs)
    FourierFitGSL.fill_gsl_vectors!(x, y, xdata, ydata, n_p)
    FourierFitGSL.fill_predictor_matrix!(X, xdata, n_p, n_freqs, n_coeffs, Ω_fit)

    # carry out fit and store best fit params
    # println("Carrying out fit")
    FourierFitGSL.curve_fit!(y, X, c, cov, work, chisq)
    @views fit_params[:] = GSL.wrap_gsl_vector(c)

    # free memory
    FourierFitGSL.free_memory!(x, y, X, c, cov, work)
    return Ω_fit
end

end