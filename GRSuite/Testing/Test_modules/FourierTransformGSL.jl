#=
    In this module we write code to compute FFTs using the GSL package
=#

module FourierTransformGSL
using GSL


# allocate memory for fitting method input
function allocate_memory(num_points::Int64)
    y = GSL.vector_alloc(2 * num_points)
    return y
end

# free memory used for fitting
function free_memory!(y::Ptr{gsl_vector})
    GSL.vector_free(y)
end

function set_real_part!(yGSL::Ptr{gsl_vector}, yJulia::Vector{Float64}, idx::Int64)
    GSL.vector_set(yGSL, 2 * (idx-1), real(yJulia[idx]))
end

function set_imaginary_part!(yGSL::Ptr{gsl_vector}, yJulia::Vector{Float64}, idx::Int64)
    GSL.vector_set(yGSL, 2 * (idx-1) + 1, imag(yJulia[idx]))
end

function extract_real_and_imaginary_parts!(yGSL::Ptr{gsl_vector}, real_y::Vector{Float64}, imag_y::Vector{Float64}, y::Vector{Float64}, num_points::Int64)
    @inbounds for i=1:num_points
        real_y[i] = yGSL[2 * (i-1)]
        imag_y[i] = yGSL[2 * (i-1) + 1]
        y[i] = real_y[i] * im * imag_y[i]
    end
end

# fill GSL vectors 'xdata' and 'ydata' for the fit
function fill_gsl_vector!(yGSL::Ptr{gsl_vector}, yJulia::Vector{Float64}, num_points::Int64)
    @inbounds Threads.@threads for i=1:num_points
        set_real_part!(yGSL, yJulia, i)
        set_imaginary_part!(yGSL, yJulia, i)
    end
end

function compute_complex_fft(yGSL::Ptr{gsl_vector}, num_points::Int64)
    GSL.gsl_fft_complex_radix2_forward(yGSL, 1, num_points);
end

function complex_fft(signal::Vector{Float64}, FFT::Vector{Float64}, FFT_real_part::Vector{Float64}, FFT_imaginary_part::Vector{Float64})
    total_num_points = length(signal)
    gsl_data = allocate_memory(total_num_points)
    fill_gsl_vector!(gsl_data, signal, total_num_points)
    compute_complex_fft(gsl_data, total_num_points)
    extract_real_and_imaginary_parts!(gsl_data, FFT_real_part, FFT_imaginary_part, FFT, total_num_points)
    free_memory!(gsl_data)
end

end