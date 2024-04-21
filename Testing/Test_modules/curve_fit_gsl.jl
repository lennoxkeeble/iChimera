#=

    In this module we write code to perform a multi-parameter linear least squares fit using the GSL library

=#
module CurveFitGSL
using GSL, DelimitedFiles

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

# fill predictor matrix X
function fill_predictor_matrix!(X::Ptr{gsl_matrix}, x::Vector{Float64}, model::Function, n_p::Int64, n_coeffs::Int64)
    # construct the fit matrix X 
    @inbounds Threads.@threads for i=0:n_p-1
        Xij = model(x[i+1])
        # fill in row i of X 
        for j=0:n_coeffs-1
            GSL.matrix_set(X, i, j, Xij[j+1])
        end
    end
end

# call multilinear fit
function curve_fit!(y::Ptr{gsl_vector}, X::Ptr{gsl_matrix}, c::Ptr{gsl_vector}, cov::Ptr{gsl_matrix}, work::Ptr{gsl_multifit_linear_workspace}, chisq::Vector{Float64})
    GSL.multifit_linear(X, y, c, cov, chisq, work)
end

# master function for carrying out fit
function GSL_fit!(xdata::Vector{Float64}, ydata::Vector{Float64}, model::Function, n_p::Int64, n_coeffs::Int64, chisq::Vector{Float64}, fit_params::Vector{Float64})
    x, y, X, c, cov, work = CurveFitGSL.allocate_memory(n_p, n_coeffs)
    CurveFitGSL.fill_gsl_vectors!(x, y, xdata, ydata, n_p)
    CurveFitGSL.fill_predictor_matrix!(X, xdata, model, n_p, n_coeffs)
    CurveFitGSL.curve_fit!(y, X, c, cov, work, chisq)
    @views fit_params[:] = GSL.wrap_gsl_vector(c)
    CurveFitGSL.free_memory!(x, y, X, c, cov, work)
end

end