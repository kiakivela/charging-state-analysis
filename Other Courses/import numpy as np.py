import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters
# -----------------------------
Nx, Ny = 33, 33          # Grid points (including boundaries)
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
dt = 0.002
Visc = 0.1
rho = 1.0

MaxIt = 100
Beta = 1.5
MaxErr = 1e-3

# Lid velocity
u_top = 1.0
u_bottom = 0.0
v_left = 0.0
v_right = 0.0

# -----------------------------
# Initialize arrays
# -----------------------------
u = np.zeros((Nx+1, Ny+2))  # u on vertical faces
v = np.zeros((Nx+2, Ny+1))  # v on horizontal faces
p = np.zeros((Nx+2, Ny+2))  # pressure at cell centers
ut = np.zeros_like(u)
vt = np.zeros_like(v)
pold = np.zeros_like(p)

# Precompute coefficient matrix for pressure (SOR)
c = np.zeros((Nx+2, Ny+2)) + 1.0 / (2/dx**2 + 2/dy**2)
c[2:Nx, 2:Ny] = 1.0 / (2/dx**2 + 2/dy**2)

# Coordinate arrays for plotting
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

time = 0.0

# -----------------------------
# Plot setup
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title("Lid-Driven Cavity Flow")
ax.set_xlabel("x")
ax.set_ylabel("y")

quiv = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y))
cont = ax.contour(X, Y, np.zeros_like(X), levels=10, colors='k')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

# -----------------------------
# Update function for animation
# -----------------------------
def update(frame):
    global u, v, p, ut, vt, time

    # Run several time steps per frame
    for _ in range(5):

        # --- Boundary conditions ---
        u[:, 0] = -u[:, 1]             # bottom wall (no-slip)
        u[:, -1] = 2*u_top - u[:, -2]  # top wall (moving lid)
        v[0, :] = -v[1, :]             # left wall
        v[-1, :] = -v[-2, :]           # right wall

        # --- Predictor step for u and v ---
        for i in range(1, Nx):
            for j in range(1, Ny+1):
                ut[i, j] = u[i, j] + dt * (
                    -0.25 * (
                        ((u[i+1, j] + u[i, j])**2 - (u[i, j] + u[i-1, j])**2) / dx +
                        ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) -
                         (u[i, j] + u[i, j-1]) * (v[i+1, j-1] + v[i, j-1])) / dy
                    )
                    + Visc * (
                        (u[i+1, j] + u[i-1, j] - 2*u[i, j]) / dx**2 +
                        (u[i, j+1] + u[i, j-1] - 2*u[i, j]) / dy**2
                    )
                )

        for i in range(1, Nx+1):
            for j in range(1, Ny):
                vt[i, j] = v[i, j] + dt * (
                    -0.25 * (
                        ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) -
                         (u[i-1, j+1] + u[i-1, j]) * (v[i, j] + v[i-1, j])) / dx +
                        ((v[i, j+1] + v[i, j])**2 - (v[i, j] + v[i, j-1])**2) / dy
                    )
                    + Visc * (
                        (v[i+1, j] + v[i-1, j] - 2*v[i, j]) / dx**2 +
                        (v[i, j+1] + v[i, j-1] - 2*v[i, j]) / dy**2
                    )
                )

        # --- Solve pressure Poisson with SOR ---
        for _ in range(MaxIt):
            pold = p.copy()
            for i in range(1, Nx+1):
                for j in range(1, Ny+1):
                    p[i, j] = (1 - Beta) * p[i, j] + Beta * c[i, j] * (
                        (p[i+1, j] + p[i-1, j]) / dx**2 +
                        (p[i, j+1] + p[i, j-1]) / dy**2 -
                        (rho / dt) * ((ut[i, j] - ut[i-1, j]) / dx + (vt[i, j] - vt[i, j-1]) / dy)
                    )

            if np.sum(np.abs(p - pold)) < MaxErr:
                break

        # --- Corrector step ---
        for i in range(1, Nx):
            for j in range(1, Ny+1):
                u[i, j] = ut[i, j] - dt/dx * (p[i+1, j] - p[i, j])
        for i in range(1, Nx+1):
            for j in range(1, Ny):
                v[i, j] = vt[i, j] - dt/dy * (p[i, j+1] - p[i, j])

        time += dt

    # --- Update plot data ---
    uu = 0.5 * (u[1:-1, 1:-1] + u[1:-1, 2:])
    vv = 0.5 * (v[1:-1, 1:-1] + v[2:, 1:-1])

    quiv.set_UVC(uu.T, vv.T)  # transpose for correct orientation

    # Remove old contours and redraw
    for cobj in cont.collections:
        cobj.remove()
    cont = ax.contour(X, Y, p[1:Nx+1, 1:Ny+1].T, levels=20, colors='k')

    ax.set_title(f"Lid-Driven Cavity Flow  t = {time:.3f}")
    return quiv, cont

# -----------------------------
# Run animation
# -----------------------------
anim = FuncAnimation(fig, update, frames=300, interval=50, blit=False)
plt.show()
