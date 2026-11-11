import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Nx, Ny = 17, 17
MaxStep = 500
Visc = 0.1
dt = 0.0001
time = 0.0

MaxIt = 100        # Max SOR iterations
Beta = 1.1         # Over-relaxation factor
MaxErr = 0.001     # Convergence tolerance

# --- Initialize arrays ---
sf = np.zeros((Nx, Ny))    # Stream function ψ
vt = np.zeros((Nx, Ny))    # Vorticity ω
vto = np.zeros((Nx, Ny))   # Old vorticity
x = np.zeros((Nx, Ny))
y = np.zeros((Nx, Ny))

# Grid spacing
h = 1.0 / (Nx - 1)

# Coordinate grids
for i in range(Nx):
    for j in range(Ny):
        x[i, j] = h * i
        y[i, j] = h * j

# --- Time loop ---
for istep in range(MaxStep):

    # --- Solve Poisson equation for streamfunction (SOR) ---
    for it in range(MaxIt):
        sf_old = sf.copy()

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                sf[i, j] = (1 - Beta) * sf[i, j] + 0.25 * Beta * (
                    sf[i + 1, j] + sf[i - 1, j] +
                    sf[i, j + 1] + sf[i, j - 1] +
                    h * h * vt[i, j]
                )

        # Compute convergence error
        Err = np.sum(np.abs(sf - sf_old))
        if Err <= MaxErr:
            break

    # --- Boundary conditions for vorticity ---
    # Bottom wall (j=0)
    vt[1:-1, 0] = 2.0 * sf[1:-1, 1] / (h * h)

    # Top wall (j=Ny-1): lid moving with u = 1
    vt[1:-1, Ny - 1] = 2.0 * sf[1:-1, Ny - 2] / (h * h) - 2.0 / h

    # Left wall (i=0)
    vt[0, 1:-1] = 2.0 * sf[1, 1:-1] / (h * h)

    # Right wall (i=Nx-1)
    vt[Nx - 1, 1:-1] = 2.0 * sf[Nx - 2, 1:-1] / (h * h)

    # Store old vorticity
    vto = vt.copy()

    # --- Update interior vorticity using FTCS scheme ---
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            advect = 0.25 * (
                (sf[i, j + 1] - sf[i, j - 1]) * (vto[i + 1, j] - vto[i - 1, j]) -
                (sf[i + 1, j] - sf[i - 1, j]) * (vto[i, j + 1] - vto[i, j - 1])
            ) / (h * h)

            diffuse = Visc * (
                vto[i + 1, j] + vto[i - 1, j] + vto[i, j + 1] + vto[i, j - 1] -
                4.0 * vto[i, j]
            ) / (h * h)

            vt[i, j] = vto[i, j] + dt * (advect + diffuse)

    time += dt

    # --- Plot results ---
    if istep % 10 == 0 or istep == MaxStep - 1:
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.contour(x, y, vt, levels=40, cmap='jet')
        plt.title(f'Vorticity ω, t={time:.3f}')
        plt.axis('square')

        plt.subplot(1, 2, 2)
        plt.contour(x, y, sf, levels=40, cmap='jet')
        plt.title('Stream Function ψ')
        plt.axis('square')

        plt.pause(0.01)

plt.show()
