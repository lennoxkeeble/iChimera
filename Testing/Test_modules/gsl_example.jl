include("/home/lkeeble/GRSuite/Testing/Test_modules/curve_fit_gsl.jl");
using GSL, Distributions, Plots, .CurveFitGSL
n_p = 200  # number of data points to fit 
n_coeffs = 4 # number of fit coefficients 

# construct xdata
x_min=-5.0; x_max=5.0; Δx = (x_max-x_min)/(n_p-1);
xdata = x_min:Δx:x_max |> collect;

# construct model
model(x) = [cos(x), sin(x), cos(2.0x), sin(2.0x)]

# compute y-data
ydata = [sum(model(xx)) for xx in xdata]; y_noisy = @. ydata + rand(Uniform(0, 1));
chisq = [0.0]

# perform fit
fit_params = zeros(n_coeffs); fit_params_noisy=zeros(n_coeffs); # arrays for fitting parameters

CurveFitGSL.GSL_fit!(xdata, ydata, model, n_p, n_coeffs, chisq, fit_params)
CurveFitGSL.GSL_fit!(xdata, y_noisy, model, n_p, n_coeffs, chisq, fit_params_noisy)


# compute fitted functional form
y_fitted = [fit_params[1] * cos(xx) + fit_params[2] * sin(xx) + fit_params[3] * cos(2.0xx) +
    fit_params[4] * sin(2.0xx) for xx in xdata]

y_fitted_noisy = [fit_params_noisy[1] * cos(xx) + fit_params_noisy[2] * sin(xx) + fit_params_noisy[3] * cos(2.0xx) + 
    fit_params_noisy[4] * sin(2.0xx) for xx in xdata]

# plot fitted data
plot([xdata, xdata, xdata], [ydata, y_fitted, y_fitted_noisy], color=[:green :red :blue], label=["True" "Fitted w/o noise" "Fitted w noise"])
plot([xdata, xdata], [ydata, y_fitted_noisy], color=[:green :blue], label=["True" "Fitted w noise"])