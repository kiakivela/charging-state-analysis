import numpy as np
import matplotlib.pyplot as plt

V0=1e-3     
T=1.0     

data=np.loadtxt("/home/kivekia/Downloads/exercise1_2.dat", skiprows=3)

t_data=data[:, 0]        
V_data=data[:, 1]*1e-3 

t_data_clean=np.delete(t_data, 22, axis=0)
V_data_clean=np.delete(V_data, 22, axis=0)

t_model=np.linspace(-3, 3, 500)

V1=2*V0*np.exp(-(t_model/(2*T))**2)
V2=V0*(1+np.cos(t_model/T))

plt.figure(figsize=(8, 5))

plt.plot(t_data_clean, V_data_clean, 'o', label='Experimental data')
plt.plot(t_model, V1, '-', linewidth=2, label=r'$V_1(t)=2V_0 e^{-(t/2T)^2}$')
plt.plot(t_model, V2, '-', linewidth=2, label=r'$V_2(t)=V_0(1+\cos(t/T))$')

plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Voltage (V)', fontsize=12)
plt.legend(loc="upper left", fontsize=9)
plt.grid(True)

plt.tight_layout()
plt.show()

