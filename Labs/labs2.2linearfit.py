import numpy as np
import matplotlib.pyplot as plt

#Data from the poisson curve fit to the histograms
expectancy_values=np.array([2.092, 1.973, 1.833, 1.661])
standard_error_mean=np.array([0.007, 0.008, 0.010, 0.010])
#Make a logarithmic array for Beer - Lambert law I=I₀-e^(-μ x) => ln (I) = ln(I₀)- μ x
logarithmic_expectancy=np.log(expectancy_values)
#Thicknesses of the lead plate layer
thickness=np.array([0.49,0.975,1.4,1.83])
thickness_err=0.005
#Calculate errors for the logarithm of the expectancy values
errors=standard_error_mean/expectancy_values #Error propagation: δ_ln(I) = δI/I

#Create linear fit weighted by the uncertanty
coefficients, cov=np.polyfit(thickness, logarithmic_expectancy, 1, w=1/errors, cov=True)
linear_fit=np.poly1d(coefficients)

slope=coefficients[0]
#Calculate the error values for the slope and intercept using NumPy's Covariance Matrix 
#based on the weighted least squares method (AI recommended)
slope_err=np.sqrt(cov[0,0])
intercept=coefficients[1]
intercept_err=np.sqrt(cov[1,1])

#Smooth arrays for better looking plots
x_smooth=np.linspace(thickness.min()-0.1, thickness.max()+0.1, 500)
y_smooth=linear_fit(x_smooth)

#Create plot
fig, ax =plt.subplots(figsize=(8,6))
ax.plot(x_smooth, y_smooth, '-b', label=f'Linear fit ln(I)={intercept:.2f}±{intercept_err:.2f} {slope:.2f} ± {slope_err:.2f} x \n μ = {-slope:.3f} ± {slope_err:.3f} cm⁻¹')
ax.errorbar(thickness, logarithmic_expectancy, xerr=thickness_err, yerr=errors, fmt='o', color='blue', ecolor='gray', 
            elinewidth=1.5, capsize =4, capthick=1.5, markersize=8, markeredgecolor='white', zorder=3, label='Data ± δ')

#Set plot details and labels
ax.text(0.7,0.64,'ln(I)=ln(I₀)-μ x', color='blue', fontsize=10)
ax.set_xlabel('Lead layer thickness (cm)')
ax.set_ylabel('Natural Logarithm of Intensity (I)')
ax.grid(True, alpha=0.3)

plt.title('Radiation attenuation using Beer-Lambert law', fontsize=14, pad=15)
plt.legend()
plt.show()
