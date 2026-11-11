import numpy as np
import matplotlib.pyplot as plt

T = 298.0
sigma = 72.8e-3
Mw = 0.01801528
rho_w = 1000.0
R = 8.314

# Your salts, but we choose ONE index per run instead of mixing everything
M_s = 0.05844
rho_s = 2160.0

dry_radii = np.array([50e-9, 100e-9])

def kohler_S(r, r_dry, i):
    
    # YOUR n_s formula
    n_s = ((r_dry**3 * np.pi * i) / 6) * (rho_s / M_s)
    
    # YOUR Raoult formula
    Raoult = (6 * n_s * Mw) / (np.pi * rho_w * r**3)
    
    # YOUR Kelvin term (not exponential)
    Kelvin = (4 * sigma * Mw) / (rho_w * R * T * r)
    
    # YOUR S formula
    return 1 + Kelvin - Raoult

plt.figure(figsize=(8,6))

# FIX 1: Much wider radius range + more points
r_vals = np.linspace(20e-9, 5000e-9, 2000)

for r_dry in dry_radii:
    S_no_diss = kohler_S(r_vals, r_dry, i=1)
    plt.plot(r_vals*1e9, S_no_diss, label=f"{int(r_dry*1e9)} nm no diss")
    
    S_full_diss = kohler_S(r_vals, r_dry, i=2)
    plt.plot(r_vals*1e9, S_full_diss, "--", label=f"{int(r_dry*1e9)} nm full diss")

plt.ylim(0.999, 1.005)
plt.xlabel("Droplet radius (nm)")
plt.ylabel("S")
plt.title("Köhler curves for NaCl (with no dissociation and full dissociation included)")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()
