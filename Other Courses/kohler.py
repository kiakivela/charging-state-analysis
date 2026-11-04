import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Constants
# -------------------------------
T = 298.0  # Temperature (K)
sigma = 72.8e-3  # Water surface tension (N/m)
Mw = 0.01801528  # Molar mass of water (kg/mol)
rho_w = 1000.0  # Density of water (kg/m3)
R = 8.314  # Gas constant (J/mol/K)

# -------------------------------
# Salt properties
# -------------------------------
salts = {
    "NaCl": {"M": 0.058443, "rho": 2165.0, "i": 2},
    "(NH4)2SO4": {"M": 0.13214, "rho": 1770.0, "i": 3}
}

# Dry particle radii (50 and 100 nm)
dry_radii = [50e-9, 100e-9]

# -------------------------------
# Köhler equation
# -------------------------------
def kohler_S(r, r_dry, M_s, rho_s, i):
    # Solute mass and moles
    m_s = (4/3) * np.pi * r_dry**3 * rho_s
    n_s = m_s / M_s

    # Moles of water: mass / M_w
    m_w = (4/3) * np.pi * r**3 * rho_w
    n_w = m_w / Mw

    # Raoult's law term
    Raoult = n_w / (n_w + i * n_s)

    # Kelvin term
    Kelvin = np.exp(2 * sigma * Mw / (rho_w * R * T * r))

    return 1-Raoult + Kelvin

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(9, 6))
r_vals = np.linspace(20e-9, 2e-6, 2000)  # Radii from 20 nm → 2 µm

for salt_name, props in salts.items():
    for r_dry in dry_radii:
        S_vals = kohler_S(r_vals, r_dry, props["M"], props["rho"], props["i"])
        label = f"{salt_name}, {int(r_dry*1e9)} nm"
        plt.plot(r_vals * 1e9, S_vals, label=label)

plt.xlabel("Droplet radius (nm)")
plt.ylabel("Saturation ratio S")
plt.title("Köhler Curves for NaCl and (NH4)2SO4 at T = 298 K")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
