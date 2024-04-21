# # functional form to which we fit data
# function curve_fit_functional(f::Vector{Float64}, tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, n_freqs::Int64)
#     @inbounds Threads.@threads for i in eachindex(tdata)
#         f[i] = params[1]     #### constant term
#         @inbounds for j in 2:n_freqs
#             f[i] += params[j] * cos(Ω[j] * tdata[i]) + params[n_freqs-1 + j] * sin(Ω[j] * tdata[i])
#         end
#     end
#     return f
# end

# # compute Nth derivative
# function curve_fit_functional_derivs(tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, N::Int64)
#     # compute value of fourier expansion
#     f = N==0 ? params[1] * ones(size(tdata, 1)) : zeros(size(tdata, 1))
#     n_freqs = size(Ω, 1)
#     @inbounds for i in eachindex(tdata)
#         for j in 2:n_freqs    
#             f[i] += (Ω[j]^N) * (params[j] * cos(Ω[j] * tdata[i] + N*π/2) + params[n_freqs-1 + j] * sin(Ω[j] * tdata[i] + N*π/2))
#         end
#     end
#     return f
# end

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

# compute Nth derivative
function curve_fit_functional_derivs(tdata::Vector{Float64}, Ω::Vector{Float64}, params::Vector{Float64}, N::Int64)
    # compute value of fourier expansion
    f = zeros(size(tdata, 1))
    n_freqs = size(Ω, 1)
    @inbounds for i in eachindex(tdata)
        for j in eachindex(Ω)   
            f[i] += (Ω[j]^N) * (params[j] * cos(Ω[j] * tdata[i] + N*π/2) + params[n_freqs + j] * sin(Ω[j] * tdata[i] + N*π/2))
        end
    end
    return f
end


#### testing ####
f=zeros(5);
x=[1.2, 3.4, 5.6, 7.8, 9.9];
Ω = [0., 1., 2.]; n_freqs=size(Ω, 1)
true_func_0(x::Float64, Ω::Vector{Float64}, params::Vector{Float64}) = params[1] + params[2] * cos(Ω[2] * x) + params[3] * cos(Ω[3] * x) + params[4] * sin(Ω[2] * x) + params[5] * sin(Ω[3] * x)
true_func_1(x::Float64, Ω::Vector{Float64}, params::Vector{Float64}) = -Ω[2] * params[2] * sin(Ω[2] * x) - Ω[3] * params[3] * sin(Ω[3] * x) + Ω[2] * params[4] * cos(Ω[2] * x) + Ω[3] * params[5] * cos(Ω[3] * x)
true_func_2(x::Float64, Ω::Vector{Float64}, params::Vector{Float64}) = -(Ω[2]^2) * params[2] * cos(Ω[2] * x) - (Ω[3]^2) * params[3] * cos(Ω[3] * x) - (Ω[2]^2) * params[4] * sin(Ω[2] * x) - (Ω[3]^2) * params[5] * sin(Ω[3] * x)
params=[0., 1., 2., 3., 0.];

ytrue_0 = [true_func_0(x[i], Ω, params) for i in eachindex(x)]; y_ffit_0=curve_fit_functional(f, x, Ω, params, n_freqs)
y_ffit_00=curve_fit_functional_derivs(x, Ω, params, 0)
ytrue_1 = [true_func_1(x[i], Ω, params) for i in eachindex(x)]; y_ffit_1=curve_fit_functional_derivs(x, Ω, params, 1)
ytrue_2 = [true_func_2(x[i], Ω, params) for i in eachindex(x)]; y_ffit_2=curve_fit_functional_derivs(x, Ω, params, 2)

ytrue_0-y_ffit_0
ytrue_0-y_ffit_00
ytrue_1-y_ffit_1
ytrue_2-y_ffit_2