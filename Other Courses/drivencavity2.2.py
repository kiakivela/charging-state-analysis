import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters ---
Nx, Ny = 33, 33        # grid points
Lx, Ly = 1.0, 1.0
Visc = 0.1             # kinematic viscosity
rho = 1.0
dt = 0.0001
MaxStep = 2000
MaxIt = 200
Beta = 1.5
MaxErr = 1e-3

# Boundary velocities
u_top = 1.0   # lid velocity
u_bottom = 0.0
v_left = 0.0
v_right = 0.0

# --- Derived parameters ---
dx = Lx / Nx
dy = Ly / Ny

# --- Initialize fields ---
u = np.zeros((Nx+1, Ny+2))   # u velocities (faces in x)
v = np.zeros((Nx+2, Ny+1))   # v velocities (faces in y)
p = np.zeros((Nx+2, Ny+2))   # pressure (centers)

ut = np.zeros_like(u)
vt = np.zeros_like(v)
p_old = np.zeros_like(p)

# Precompute SOR coefficient
c = np.ones_like(p) / (2/dx**2 + 2/dy**2)

# --- Simulation loop ---
for step in range(MaxStep):
    # Boundary conditions
    u[:, 0] = -u[:, 1]                   # bottom wall (no-slip)
    u[:, -1] = 2*u_top - u[:, -2]        # top lid
    v[0, :] = -v[1, :]                   # left wall
    v[-1, :] = -v[-2, :]                 # right wall

    # --- Predictor step for u ---
    for i in range(1, Nx):
        for j in range(1, Ny+1):
            du2dx = ((u[i+1,j] + u[i,j])**2 - (u[i,j] + u[i-1,j])**2) / (4*dx)
            duvdy = ((u[i,j+1] + u[i,j])*(v[i+1,j] + v[i,j]) -
                     (u[i,j] + u[i,j-1])*(v[i+1,j-1] + v[i,j-1])) / (4*dy)
            visc = Visc * ((u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx**2 +
                           (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy**2)
            ut[i,j] = u[i,j] + dt * (-du2dx - duvdy + visc)

    # --- Predictor step for v ---
    for i in range(1, Nx+1):
        for j in range(1, Ny):
            duvdx = ((u[i,j+1] + u[i,j])*(v[i+1,j] + v[i,j]) -
                     (u[i-1,j+1] + u[i-1,j])*(v[i,j] + v[i-1,j])) / (4*dx)
            dv2dy = ((v[i,j+1] + v[i,j])**2 - (v[i,j] + v[i,j-1])**2) / (4*dy)
            visc = Visc * ((v[i+1,j] - 2*v[i,j] + v[i-1,j]) / dx**2 +
                           (v[i,j+1] - 2*v[i,j] + v[i,j-1]) / dy**2)
            vt[i,j] = v[i,j] + dt * (-duvdx - dv2dy + visc)

    # --- Solve for pressure using SOR ---
    for it in range(MaxIt):
        p_old[:] = p
        for i in range(1, Nx+1):
            for j in range(1, Ny+1):
                rhs = (rho/dt) * ((ut[i,j] - ut[i-1,j]) / dx +
                                  (vt[i,j] - vt[i,j-1]) / dy)
                p[i,j] = (1 - Beta) * p[i,j] + Beta * c[i,j] * (
                    (p[i+1,j] + p[i-1,j]) / dx**2 +
                    (p[i,j+1] + p[i,j-1]) / dy**2 - rhs)
        if np.sum(np.abs(p - p_old)) < MaxErr:
            break

    # --- Correct velocities ---
    for i in range(1, Nx):
        for j in range(1, Ny+1):
            u[i,j] = ut[i,j] - (dt/dx) * (p[i+1,j] - p[i,j]) / rho
    for i in range(1, Nx+1):
        for j in range(1, Ny):
            v[i,j] = vt[i,j] - (dt/dy) * (p[i,j+1] - p[i,j]) / rho

    # --- Plot every 100 steps ---
    if step % 100 == 0 or step == MaxStep - 1:
        uu = 0.5 * (u[1:, 1:-1] + u[:-1, 1:-1])
        vv = 0.5 * (v[1:-1, 1:] + v[1:-1, :-1])
        pp = p[1:-1, 1:-1]

        x = np.linspace(dx/2, Lx - dx/2, Nx)
        y = np.linspace(dy/2, Ly - dy/2, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        plt.figure(figsize=(5,5))
        plt.contour(X, Y, pp, 15, colors='k', linewidths=0.7)
        plt.quiver(X, Y, uu, vv, scale=8)
        plt.title(f"Lid-Driven Cavity Flow (step={step})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("square")
        plt.tight_layout()
        plt.pause(0.01)
        #plt.close()

plt.show(block=True)

/home/kivekia/Python/drivencavity2.2.py