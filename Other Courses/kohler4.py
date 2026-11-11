import numpy as np
import matplotlib.pyplot as plt

T = 298.0
sigma = 72.8e-3
Mw = 0.01801528
rho_w = 1000.0
R = 8.314

M_s = np.array([0.13214, 0.05844]) 
rho_s = np.array([1770.0, 2160.0])

dry_radii = np.array([50e-9, 100e-9])

def kohler_S(r, r_dry, i,  salt_index):
    j = M_s[salt_index]     
    k = rho_s[salt_index]  
    
    n_s = ((r_dry**3 * np.pi * i) / 6 ) * (k / j)
    
    Raoult = (6 * n_s * Mw) / (np.pi * rho_w * r**3)
    
    Kelvin = (4 * sigma * Mw) / (rho_w * R * T * r)
    
    return 1 + Kelvin - Raoult

plt.figure(figsize=(8,6))

r_vals = np.linspace(20e-9, 5000e-9, 2000)

for r_dry in dry_radii:
    S_no_diss = kohler_S(r_vals, r_dry, i=3, salt_index=0)
    plt.plot(r_vals*1e9, S_no_diss, label=f"{int(r_dry*1e9)} nm (NH4)2SO4")
    
    S_full_diss = kohler_S(r_vals, r_dry, i=2, salt_index=1)
    plt.plot(r_vals*1e9, S_full_diss, "--", label=f"{int(r_dry*1e9)} nm NaCl")

plt.ylim(0.999, 1.005)
plt.xlabel("Droplet radius (nm)")
plt.ylabel("S")
plt.grid(True, linestyle=":")
plt.title("Köhler curves for NaCl and (NH4)2SO4 (full dissociation)")
plt.legend()
plt.tight_layout()
plt.show()
