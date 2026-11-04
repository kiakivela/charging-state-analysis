import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Constants
# -------------------------------
T = 298.0  # Temperature (K)
sigma = 72.8e-3  # Water surface tension (N/m)
Mw = 0.01801528  # Molar mass of water (kg/mol)
rho_w = 1000.0  # Density of water (kg/m^3)
R = 8.314  # Gas constant (J/mol/K)

# Salt properties for (NH4)2SO4
M_s = np.array([0.13214, 0.05844])   # molar mass (kg/mol)
rho_s = np.array([1770.0, 2160.0])  # density (kg/m^3)

# Dry particle radii (μm → m)
dry_radii = np.array([50e-9, 100e-9])


# -------------------------------
# Köhler Equation
# -------------------------------
def kohler_S(r, r_dry, i):
    # Solute mass and moles
    for j in M_s :
        for k in rho_s:
            n_s = ((r_dry**3 * np.pi) / i)* ( k / j)
            return n_s
  
    # Raoult term
    Raoult = (6 * n_s * Mw / (np.pi * rho_w * r**3))

    # Kelvin term
    Kelvin = (4 * sigma * Mw / (rho_w * R * T * r))

    return 1 + Kelvin - Raoult


# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(8, 6))

# Radius range (μm → m)
r_vals = np.linspace(10e-9, 800e-9, 1000)  # 0.01 μm → 10 μm

for r_dry in dry_radii:
    # No dissociation (i = 1)
    S_no_diss = kohler_S(r_vals, r_dry, i=1)
    plt.plot(r_vals * 1e9, S_no_diss, label=f"{r_dry*1e9:.02f} nm (no diss)", linestyle="-")

    # Complete dissociation (i = 3)
    S_full_diss = kohler_S(r_vals, r_dry, i=3)
    plt.plot(r_vals * 1e9, S_full_diss, linestyle="--", label=f"{r_dry*1e9:.02f} nm (full diss)")

plt.ylim(0.999, 1.005)

plt.xlabel("Droplet radius (nm)")
plt.ylabel("Saturation ratio S")
plt.title("(NH₄)₂SO₄ Köhler Curves at T = 298 K\nSolid = No dissociation, Dashed = Complete")
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()
