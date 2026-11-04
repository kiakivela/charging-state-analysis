import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data=np.loadtxt("/home/kivekia/Downloads/exercise1_3.dat", skiprows=2)

T_data=data[:, 0]     
V_data=data[:, 1]     

# Part (a)

interp_T=interp1d(V_data, T_data, kind='quadratic', fill_value="extrapolate")

V_interp=np.linspace(min(V_data), max(V_data), 500)
T_interp=interp_T(V_interp)

plt.figure(figsize=(8, 5))
plt.plot(V_data, T_data, 'o', label="Calibration data")
plt.plot(V_interp, T_interp, '-', label="Interpolation curve")

plt.xlabel("Voltage (mV)", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Part (b)

V_measured=58.0     
V_uncertainty=1.0         

T_measured=interp_T(V_measured)
T_low=interp_T(V_measured-V_uncertainty)
T_high=interp_T(V_measured+V_uncertainty)

T_uncertainty=0.5*(T_high-T_low)

print("Measured voltage: ({:.1f} ± {:.1f}) mV".format(V_measured, V_uncertainty))
print("Estimated temperature: ({:.2f} ± {:.2f}) °C".format(T_measured, T_uncertainty))

