import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# PARAMETERS (feel free to adjust)
# -----------------------------------------------------------
Nx, Ny = 41, 41           # grid resolution (coarser -> more stable)
Visc = 0.1                # kinematic viscosity
dt = 0.001                # start small; code will check stability
MaxStep = 1000            # time steps
MaxIt = 200               # max SOR iterations
Beta = 1.4                # relaxation factor (1.3–1.6 stable)
MaxErr = 1e-4             # SOR convergence tolerance
U_top = 1.0               # lid velocity

# -----------------------------------------------------------
# GRID SETUP
# -----------------------------------------------------------
L = 1.0
h = L / (Nx - 1)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# Initialize fields
psi = np.zeros((Nx, Ny))   # stream function
omega = np.zeros((Nx, Ny)) # vorticity

# -----------------------------------------------------------
# STABILITY CHECK
# -----------------------------------------------------------
dt_diff = h**2 / (4 * Visc)
dt_adv = h / (abs(U_top) + 1e-8)
dt_stable = min(dt_diff, dt_adv)

if dt > dt_stable:
    print(f"⚠️ Reducing dt for stability: {dt:.2e} → {dt_stable:.2e}")
    dt = dt_stable

print(f"✅ Using dt = {dt:.4e}, h = {h:.4e}, ν = {Visc:.4e}")

# -----------------------------------------------------------
# TIME STEPPING
# -----------------------------------------------------------
plt.ion()
for step in range(MaxStep):
    # --- SOR iteration for streamfunction ψ ---
    for _ in range(MaxIt):
        psi_old = psi.copy()
        psi[1:-1, 1:-1] = (1 - Beta) * psi[1:-1, 1:-1] + 0.25 * Beta * (
            psi[2:, 1:-1] + psi[:-2, 1:-1] + psi[1:-1, 2:] + psi[1:-1, :-2]
            + h**2 * omega[1:-1, 1:-1]
        )
        Err = np.sum(np.abs(psi - psi_old))
        if Err < MaxErr:
            break

    # --- Boundary conditions for vorticity ω ---
    omega[1:-1, 0]   = 2 * (psi[1:-1, 1] - psi[1:-1, 0]) / h**2  # bottom
    omega[1:-1, -1]  = 2 * (psi[1:-1, -2] - psi[1:-1, -1]) / h**2 - 2 * U_top / h  # top (moving lid)
    omega[0, 1:-1]   = 2 * (psi[1, 1:-1] - psi[0, 1:-1]) / h**2   # left
    omega[-1, 1:-1]  = 2 * (psi[-2, 1:-1] - psi[-1, 1:-1]) / h**2 # right

    # --- Update vorticity (explicit scheme) ---
    omega_old = omega.copy()
    omega[1:-1, 1:-1] = omega_old[1:-1, 1:-1] + dt * (
        -0.25 * (
            (psi[1:-1, 2:] - psi[1:-1, :-2]) * (omega_old[2:, 1:-1] - omega_old[:-2, 1:-1])
            - (psi[2:, 1:-1] - psi[:-2, 1:-1]) * (omega_old[1:-1, 2:] - omega_old[1:-1, :-2])
        ) / h**2
        + Visc * (
            omega_old[2:, 1:-1] + omega_old[:-2, 1:-1]
            + omega_old[1:-1, 2:] + omega_old[1:-1, :-2]
            - 4 * omega_old[1:-1, 1:-1]
        ) / h**2
    )

    # --- Safety check ---
    if np.any(np.isnan(omega)) or np.any(np.isinf(omega)):
        print(f"💥 Instability detected at step {step}. Stop.")
        break

    # --- Plot every few steps ---
    if step % 50 == 0 or step == MaxStep - 1:
        plt.clf()

        # Vorticity
        if np.nanmax(omega) != np.nanmin(omega):
            plt.subplot(1, 2, 1)
            plt.contourf(X, Y, omega, levels=40, cmap='jet')
            plt.title(f"Vorticity ω (step={step})")
            plt.axis('square')

        # Stream function
        if np.nanmax(psi) != np.nanmin(psi):
            plt.subplot(1, 2, 2)
            plt.contourf(X, Y, psi, levels=40, cmap='jet')
            plt.title("Stream function ψ")
            plt.axis('square')

        plt.pause(0.05)

plt.ioff()
plt.show()

print("✅ Simulation complete.")
