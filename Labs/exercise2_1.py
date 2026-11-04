import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data=np.loadtxt("/home/kivekia/Downloads/H2_data.txt", skiprows=1)

r=data[:, 0]
N=data[:, 1]

x=1/r**2   
y=N

def linear_model(x, A, B):
    return A*x+B

# Part (a)
params, cov=curve_fit(linear_model, x, y)
A_unw, B_unw=params

x_fit=np.linspace(min(x), max(x), 200)
y_fit_unw=linear_model(x_fit, A_unw, B_unw)

print("Part (a)")
print(f"A={A_unw:.1f}")
print(f"B={B_unw:.1f}")

plt.figure()
plt.plot(x, y, 'o', label="Data")
plt.plot(x_fit, y_fit_unw, '-', label=f"N = {A_unw:.1f}(1/r²) + {B_unw:.1f}")
plt.xlabel("1/r² (1/cm²)")
plt.ylabel("Photon count N")
plt.legend()
plt.grid()
plt.show()

# Part (b)
residuals=y-linear_model(x, A_unw, B_unw)
dof=len(y)-2
std_est = np.sqrt(np.sum(residuals**2)/dof)

print("\n Part (b)")
print(f"Estimated standard deviation={std_est:.1f}")

cov_scaled=cov*std_est**2

A_err=np.sqrt(cov_scaled[0, 0])
B_err=np.sqrt(cov_scaled[1, 1])

print(f"A={A_unw:.1f} ± {A_err:.1f}")
print(f"B={B_unw:.1f} ± {B_err:.1f}")

plt.figure()
plt.errorbar(x, y, yerr=std_est, fmt='o', label="Data with error bars")
plt.plot(x_fit, y_fit_unw, '-', label="Unweighted fit")
plt.xlabel("1/r² (1/cm²)")
plt.ylabel("Photon count N")
plt.legend()
plt.grid()
plt.show()

# Part (c)
std_poisson=np.sqrt(y)
params_w, cov_w=curve_fit(linear_model, x, y, sigma=std_poisson, absolute_sigma=True)
A_w, B_w=params_w
A_w_err=np.sqrt(cov_w[0, 0])
B_w_err=np.sqrt(cov_w[1, 1])

y_fit_w=linear_model(x_fit, A_w, B_w)

print("\n Part (c)")
print(f"A={A_w:.1f} ± {A_w_err:.1f}")
print(f"B={B_w:.1f} ± {B_w_err:.1f}")

plt.figure()
plt.errorbar(x, y, yerr=std_poisson, fmt='o', label="Data with Poisson errors")
plt.plot(x_fit, y_fit_w, '-', label="Weighted fit")
plt.xlabel("1/r² (1/cm²)")
plt.ylabel("Photon count N")
plt.legend()
plt.grid()
plt.show()
