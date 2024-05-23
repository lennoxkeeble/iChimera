#=

    In this module we write code to perform a multi-parameter linear least squares fit using the GSL library, where the functional form to which we fit is specifically the 
    fundamental-frequency Fourier series expansion

=#

module FourierFitGSL_Derivs
using GSL

# compute number of fitting frequencies for fits with one, two, and three fundamental frequencies for a given harmonic (-1 since we don't count constant term)
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
function compute_fitting_frequenices_1(nHarm::Int64, Ωr::Float64)
    Ω = Float64[]
    @inbounds for i_r in 1:nHarm
        append!(Ω, i_r * Ωr)
    end
    return Ω
end

# chimera construct fitting frequencies
function compute_fitting_frequenices_2(nHarm::Int64, Ωr::Float64, Ωθ::Float64)
    Ω = Float64[]
    @inbounds for i_r in 0:nHarm
        @inbounds for i_θ in -i_r:nHarm
            (i_r==0 && i_θ==0) ? nothing : append!(Ω, abs(i_r * Ωr + i_θ * Ωθ))
        end
    end
    return Ω
end


# chimera construct fitting frequencies
function compute_fitting_frequenices_3(nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64)
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
        compute_fitting_frequenices_1(nHarm, freqs...)
    elseif num_freqs==2
        compute_fitting_frequenices_2(nHarm, freqs...)
    elseif num_freqs==3
        compute_fitting_frequenices_3(nHarm, freqs...)
    end
end

# fit data and its derivatives to the fourier series expansion
function curve_fit_functional(f::Vector{Float64}, tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, n_freqs::Int64, nPoints::Int64)
    N_1=1   # first derivative
    N_2=2   # second derivative
    # nPoints = number of points in time array, noting that the time array will be concatenated with itself in order to fit to multiple functions simultaneously
    @inbounds Threads.@threads for i=1:nPoints
        f[i]=params[1]    # first parameter is the constant term
        @inbounds for j in eachindex(Ω)
            # zeroth derivative
            f[i] += params[j+1] * cos(Ω[j] * tdata[i]) + params[n_freqs + j+1] * sin(Ω[j] * tdata[i])
            # first derivative
            f[i+ N_1 * nPoints] += (Ω[j]^N_1) * (params[j+1] * cos(Ω[j] * tdata[i+ N_1 * nPoints] + N_1*π/2) + 
            params[n_freqs + j+1] * sin(Ω[j] * tdata[i+ N_1 * nPoints] + N_1*π/2))

            # second derivative
            f[i+ N_2 * nPoints] += (Ω[j]^N_2) * (params[j+1] * cos(Ω[j] * tdata[i+ N_2 * nPoints] + N_2*π/2) + 
            params[n_freqs + j+1] * sin(Ω[j] * tdata[i+ N_2 * nPoints] + N_2*π/2))
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


# allocate memory for fitting method input
function allocate_memory(n_p::Int64, n_coeffs::Int64)
    x = GSL.vector_alloc(3 * n_p)
    y = GSL.vector_alloc(3 * n_p)
    X = GSL.matrix_alloc(3 * n_p, n_coeffs)
    c = GSL.vector_alloc(n_coeffs)
    cov = GSL.matrix_alloc(n_coeffs, n_coeffs)
    work = GSL.multifit_linear_alloc(3 * n_p, n_coeffs)
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
function fill_gsl_vectors!(xGSL::Ptr{gsl_vector}, yGSL::Ptr{gsl_vector}, x::Vector{Float64}, y::Vector{Float64}, y_1::Vector{Float64}, y_2::Vector{Float64}, n_p::Int64)
    @inbounds Threads.@threads for i=0:(n_p-1)
        # zeroth deriv
        GSL.vector_set(xGSL, i, x[i+1])
        GSL.vector_set(yGSL, i, y[i+1])
        # first deriv
        GSL.vector_set(xGSL, i+n_p, x[i+1])
        GSL.vector_set(yGSL, i+n_p, y_1[i+1])

        # second deriv
        GSL.vector_set(xGSL, i+ 2 * n_p, x[i+1])
        GSL.vector_set(yGSL, i+ 2 * n_p, y_2[i+1])
    end
end

#=

    The predictor matrix X has a number of rows equal to the number of y-values to which we are fitting. Each row, therefore, consists of the functional form to which we 
    are fitting evaluated at each element of the x vector. The input function 'model' must output a vector whose elements are the componenets of the functional form. In other words,
    if the function form we are fitting to is f(x), then we must have f(x) = sum(model(x))

=#

# row constructor for predictor matrix in GSL fit
function GSL_fourier_model_zero_deriv(t::Float64, Ω::Vector{Float64}, n_freqs::Int64,  n_coeffs::Int64)
    f=ones(n_coeffs)    # a constant term, and then a cosine and sine term for each fitting frequency
    @inbounds for i in 1:n_freqs
        f[i+1] = cos(Ω[i] * t)
        f[i+1+n_freqs] = sin(Ω[i] * t)
    end
    return f
end

function GSL_fourier_model_first_deriv(t::Float64, Ω::Vector{Float64}, n_freqs::Int64,  n_coeffs::Int64)
    N=1   # first derivative
    f=zeros(n_coeffs)    # a constant term is zero for the first derivative
    @inbounds for i in 1:n_freqs
        f[i+1] = (Ω[i]^N) * cos(Ω[i] * t + N*π/2)
        f[i+1+n_freqs] = (Ω[i]^N) * sin(Ω[i] * t + N*π/2)
    end
    return f
end

function GSL_fourier_model_second_deriv(t::Float64, Ω::Vector{Float64}, n_freqs::Int64,  n_coeffs::Int64)
    N=2   # second derivative
    f=zeros(n_coeffs)    # a constant term is zero for the second derivative
    @inbounds for i in 1:n_freqs
        f[i+1] = (Ω[i]^N) * cos(Ω[i] * t + N*π/2)
        f[i+1+n_freqs] = (Ω[i]^N) * sin(Ω[i] * t + N*π/2)
    end
    return f
end

# fill predictor matrix X: N is number of harmonics, and Ω_fit is the fitting frequencies
function fill_predictor_matrix!(X::Ptr{gsl_matrix}, x::Vector{Float64}, n_p::Int64, n_freqs::Int64,  n_coeffs::Int64, Ω_fit::Vector{Float64})
    # construct the fit matrix X 
    @inbounds Threads.@threads for i=0:n_p-1
        Xij_zero = FourierFitGSL_Derivs.GSL_fourier_model_zero_deriv(x[i+1], Ω_fit, n_freqs, n_coeffs)
        Xij_first = FourierFitGSL_Derivs.GSL_fourier_model_first_deriv(x[i+1], Ω_fit, n_freqs, n_coeffs)
        Xij_second = FourierFitGSL_Derivs.GSL_fourier_model_second_deriv(x[i+1], Ω_fit, n_freqs, n_coeffs)
        # fill in row i of X for the zeroth deriv, and i + nPoints for first derivative etc
        for j=0:(2 * n_freqs)
            GSL.matrix_set(X, i, j, Xij_zero[j+1])
            GSL.matrix_set(X, i+n_p, j, Xij_first[j+1])
            GSL.matrix_set(X, i+2*n_p, j, Xij_second[j+1])
        end
    end
end

# call multilinear fit
function curve_fit!(y::Ptr{gsl_vector}, X::Ptr{gsl_matrix}, c::Ptr{gsl_vector}, cov::Ptr{gsl_matrix}, work::Ptr{gsl_multifit_linear_workspace}, chisq::Vector{Float64})
    GSL.multifit_linear(X, y, c, cov, chisq, work)
end

# master functions for carrying out fit with one, two or three fundamental frequencies
function GSL_fit_1!(xdata::Vector{Float64}, ydata::Vector{Float64}, ydata_1::Vector{Float64}, ydata_2::Vector{Float64}, n_p::Int64, nHarm::Int64, chisq::Vector{Float64},  Ω1::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their number
    Ω_fit = FourierFitGSL_Derivs.compute_fitting_frequenices_1(nHarm, Ω1)
    n_freqs = compute_num_fitting_freqs_1(nHarm)
    n_coeffs = 2 * n_freqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency

    # allocate memory and fill GSL vectors and matrices
    x, y, X, c, cov, work = FourierFitGSL_Derivs.allocate_memory(n_p, n_coeffs)
    FourierFitGSL_Derivs.fill_gsl_vectors!(x, y, xdata, ydata, ydata_1, ydata_2, n_p)
    FourierFitGSL_Derivs.fill_predictor_matrix!(X, xdata, n_p, n_freqs, n_coeffs, Ω_fit)

    # carry out fit and store best fit params
    # println("Carrying out fit")
    FourierFitGSL_Derivs.curve_fit!(y, X, c, cov, work, chisq)
    @views fit_params[:] = GSL.wrap_gsl_vector(c)

    # free memory
    FourierFitGSL_Derivs.free_memory!(x, y, X, c, cov, work)
    return Ω_fit
end

function GSL_fit_2!(xdata::Vector{Float64}, ydata::Vector{Float64}, ydata_1::Vector{Float64}, ydata_2::Vector{Float64}, n_p::Int64, nHarm::Int64, chisq::Vector{Float64},  Ω1::Float64, Ω2::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their number
    Ω_fit = FourierFitGSL_Derivs.compute_fitting_frequenices_2(nHarm, Ω1, Ω2)
    n_freqs = compute_num_fitting_freqs_2(nHarm)
    n_coeffs = 2 * n_freqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency

    # allocate memory and fill GSL vectors and matrices
    x, y, X, c, cov, work = FourierFitGSL_Derivs.allocate_memory(n_p, n_coeffs)
    FourierFitGSL_Derivs.fill_gsl_vectors!(x, y, xdata, ydata, ydata_1, ydata_2, n_p)
    FourierFitGSL_Derivs.fill_predictor_matrix!(X, xdata, n_p, n_freqs, n_coeffs, Ω_fit)

    # carry out fit and store best fit params
    # println("Carrying out fit")
    FourierFitGSL_Derivs.curve_fit!(y, X, c, cov, work, chisq)
    @views fit_params[:] = GSL.wrap_gsl_vector(c)

    # free memory
    FourierFitGSL_Derivs.free_memory!(x, y, X, c, cov, work)
    return Ω_fit
end

function GSL_fit_3!(xdata::Vector{Float64}, ydata::Vector{Float64}, ydata_1::Vector{Float64}, ydata_2::Vector{Float64}, n_p::Int64, nHarm::Int64, chisq::Vector{Float64},  Ω1::Float64, Ω2::Float64, Ω3::Float64, fit_params::Vector{Float64})
    # compute fitting frequncies and their number
    Ω_fit = FourierFitGSL_Derivs.compute_fitting_frequenices_3(nHarm, Ω1, Ω2, Ω3)
    n_freqs = compute_num_fitting_freqs_3(nHarm)
    n_coeffs = 2 * n_freqs + 1    # +1 to allocate memory for the constant term, and factor of 2 since we have sin and cos for each frequency

    # allocate memory and fill GSL vectors and matrices
    x, y, X, c, cov, work = FourierFitGSL_Derivs.allocate_memory(n_p, n_coeffs)
    FourierFitGSL_Derivs.fill_gsl_vectors!(x, y, xdata, ydata, ydata_1, ydata_2, n_p)
    FourierFitGSL_Derivs.fill_predictor_matrix!(X, xdata, n_p, n_freqs, n_coeffs, Ω_fit)

    # carry out fit and store best fit params
    # println("Carrying out fit")
    FourierFitGSL_Derivs.curve_fit!(y, X, c, cov, work, chisq)
    @views fit_params[:] = GSL.wrap_gsl_vector(c)

    # free memory
    FourierFitGSL_Derivs.free_memory!(x, y, X, c, cov, work)
    return Ω_fit
end

function GSL_fit_master!(xdata::Vector{Float64}, ydata::Vector{Float64}, ydata_1::Vector{Float64}, ydata_2::Vector{Float64}, n_p::Int64, nHarm::Int64, chisq::Vector{Float64},  Ω::Vector{Float64}, fit_params::Vector{Float64})
    freqs = Ω[Ω .< 1e9];
    num_freqs = length(freqs);
    if num_freqs==1
        GSL_fit_1!(xdata, ydata, ydata_1, ydata_2, n_p, nHarm, chisq, freqs..., fit_params)
    elseif num_freqs==2
        GSL_fit_2!(xdata, ydata, ydata_1, ydata_2, n_p, nHarm, chisq, freqs..., fit_params)
    elseif num_freqs==3
        GSL_fit_3!(xdata, ydata, ydata_1, ydata_2, n_p, nHarm, chisq, freqs..., fit_params)
    end
end

end