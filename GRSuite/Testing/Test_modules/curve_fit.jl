module FourierFit
using LsqFit
using DelimitedFiles

# functional form to which we fit data
function curve_fit_functional(f::Vector{Float64}, tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, n_freqs::Int64)
    @inbounds Threads.@threads for i in eachindex(tdata)
        f[i]=0.
        @inbounds for j in eachindex(Ω)
            f[i] += params[j] * cos(Ω[j] * tdata[i]) + params[n_freqs + j] * sin(Ω[j] * tdata[i])
        end
    end
    return f
end

# # functional form to which we fit data
# function curve_fit_functional(f::Vector{Float64}, tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, n_freqs::Int64)
#     @inbounds Threads.@threads for i in eachindex(tdata)
#         f[i] = sum(@. @view(params[1:n_freqs]) * cos(Ω * tdata[i]) + @view(params[(n_freqs+1):(2*n_freqs)]) * sin(Ω * tdata[i]))
#     end
#     return f
# end

# # perform curve_fit with bounds
# function ls_fit(f::Vector{Float64}, tdata::Vector{Float64}, ydata::Vector{Float64}, Ω::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, p0::Vector{Float64}, n_freqs::Int64)
#     # construct curve_fit model
#     function model(t, p)
#         curve_fit_functional(f, t, Ω, p, n_freqs)
#     end

#     return curve_fit(model, tdata, ydata, p0, lower=lb, upper=ub, maxIter=5000, min_step_quality=1e-6, show_trace=true)
# end

# curve_fit without bounds
function ls_fit(f::Vector{Float64}, tdata::Vector{Float64}, ydata::Vector{Float64}, Ω::Vector{Float64}, p0::Vector{Float64}, n_freqs::Int64)
    # construct curve_fit model
    function model(t, p)
        curve_fit_functional(f, t, Ω, p, n_freqs)
    end
    
    return curve_fit(model, tdata, ydata, p0, maxIter=5000)
end

# compute Nth derivative
function curve_fit_functional_derivs(tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, N::Int64)
    # compute value of fourier expansion
    f=zeros(size(tdata, 1))
    n_freqs = size(Ω, 1)
    @inbounds for i in eachindex(tdata)
        for j in eachindex(Ω)  
            f[i] += (Ω[j]^N) * (params[j] * cos(Ω[j] * tdata[i] + N*π/2) + params[n_freqs + j] * sin(Ω[j] * tdata[i] + N*π/2))
        end
    end
    return f
end

# user function to carry out fit on data. returns: params, data
function fourier_fit(x::Vector{Float64}, y::Vector{Float64}, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, nHarm::Int64; p0=Float64[])
    f_container=zeros(size(x, 1))    # container for curve_fit_functional to store ydata

    # implement method used in Chimera code (e.g., the way of constructing the fitting frequencies)
    Ω=Float64[]
    # @inbounds for i_r in 0:nHarm
    #     @inbounds for i_θ in -i_r:nHarm
    #         @inbounds for i_ϕ in -(i_r+i_θ):nHarm
    #             if i_r==0 && i_θ==0 && i_ϕ==0
    #                 nothing
    #             else
    #                 append!(Ω, abs(i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ))
    #             end
    #         end
    #     end
    # end
    @inbounds for i_r in 0:nHarm
        @inbounds for i_θ in 0:(nHarm+i_r)
            @inbounds for i_ϕ in 0:(nHarm+i_r+i_θ)
                append!(Ω, i_r * Ωr + i_θ * Ωθ + i_ϕ * Ωϕ)
            end
        end
    end

    n_freqs = size(Ω, 1)

    if isequal(p0, Float64[])
        bound_fact = 100.0
        minVal = minimum(y); lbVal = minVal < 0 ? bound_fact * minVal : minVal / bound_fact
        maxVal = maximum(y); ubVal = maxVal < 0 ? maxVal / bound_fact : bound_fact * maxVal
        lb=lbVal * 50 * ones(2 * n_freqs)
        ub=ubVal * 50 * ones(2 * n_freqs)

        # initial guess
        append!(p0, 0.5 * (lb .+ ub))
    end

    # compute best fit parameters
    # if size(lb, 1)==0
    #     y_fit = ls_fit(f_container, x, y, Ω, p0, n_freqs)
    # else
    #     y_fit = ls_fit(f_container, x, y, Ω, lb, ub, p0, n_freqs)
    # end
    println("Carrying out fit")
    y_fit = ls_fit(f_container, x, y, Ω, p0, n_freqs)

    return Ω, y_fit, curve_fit_functional(f_container, x, Ω, coef(y_fit), n_freqs)
end
end