import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters ---
Nx, Ny = 32, 32        # grid points
Lx, Ly = 1.0, 1.0
Visc = 0.1             # kinematic viscosity
rho = 1.0              # density
dt = 0.002             # time step
MaxStep = 500          # total time steps
MaxIt = 100            # SOR iterations per step
Beta = 1.5             # over-relaxation parameter
MaxErr = 1e-3          # SOR stopping criterion

# --- Boundary velocities ---
u_top = 1.0   # lid velocity
u_bottom = 0.0
v_left = 0.0
v_right = 0.0

# --- Grid spacing ---
dx = Lx / Nx
dy = Ly / Ny

# --- Initialize fields ---
u = np.zeros((Nx+1, Ny+2))   # u velocity (faces in x)
v = np.zeros((Nx+2, Ny+1))   # v velocity (faces in y)
p = np.zeros((Nx+2, Ny+2))   # pressure (cell centers)

ut = np.zeros_like(u)
vt = np.zeros_like(v)
p_old = np.zeros_like(p)

# Precompute SOR coefficient matrix for pressure
c = np.ones_like(p) / (2/dx**2 + 2/dy**2)
c[2:Nx, 2:Ny] = 1.0 / (2/dx**2 + 2/dy**2)

# --- Simulation loop ---
for step in range(MaxStep):
    # Apply boundary conditions for u and v
    u[:, 0] = -u[:, 1]              # bottom wall (no-slip)
    u[:, -1] = 2*u_top - u[:, -2]   # top wall (moving lid)
    v[0, :] = -v[1, :]              # left wall
    v[-1, :] = -v[-2, :]            # right wall

    # --- Predictor step: temporary velocities ---
    # Compute ut (temporary u)
    for i in range(1, Nx):
        for j in range(1, Ny+1):
            du2dx = ((u[i+1,j] + u[i,j])**2 - (u[i,j] + u[i-1,j])**2) / (4*dx)
            duvdy = ((u[i,j+1] + u[i,j])*(v[i+1,j] + v[i,j]) -
                     (u[i,j] + u[i,j-1])*(v[i+1,j-1] + v[i,j-1])) / (4*dy)
            visc = Visc * ((u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx**2 +
                           (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy**2)
            ut[i,j] = u[i,j] + dt * (-du2dx - duvdy + visc)

    # Compute vt (temporary v)
    for i in range(1, Nx+1):
        for j in range(1, Ny):
            duvdx = ((u[i,j+1] + u[i,j])*(v[i+1,j] + v[i,j]) -
                     (u[i-1,j+1] + u[i-1,j])*(v[i,j] + v[i-1,j])) / (4*dx)
            dv2dy = ((v[i,j+1] + v[i,j])**2 - (v[i,j] + v[i,j-1])**2) / (4*dy)
            visc = Visc * ((v[i+1,j] - 2*v[i,j] + v[i-1,j]) / dx**2 +
                           (v[i,j+1] - 2*v[i,j] + v[i,j-1]) / dy**2)
            vt[i,j] = v[i,j] + dt * (-duvdx - dv2dy + visc)

    # --- Solve Poisson equation for pressure (SOR) ---
    for it in range(MaxIt):
        p_old[:] = p
        for i in range(1, Nx+1):
            for j in range(1, Ny+1):
                rhs = (rho/dt) * ((ut[i,j] - ut[i-1,j]) / dx +
                                  (vt[i,j] - vt[i,j-1]) / dy)
                p[i,j] = (1 - Beta) * p[i,j] + Beta * c[i,j] * (
                    ((p[i+1,j] + p[i-1,j]) / dx**2) +
                    ((p[i,j+1] + p[i,j-1]) / dy**2) - rhs)
        err = np.sum(np.abs(p - p_old))
        if err <= MaxErr:
            break

    # --- Correct velocity fields ---
    for i in range(1, Nx):
        for j in range(1, Ny+1):
            u[i,j] = ut[i,j] - (dt/dx) * (p[i+1,j] - p[i,j]) / rho
    for i in range(1, Nx+1):
        for j in range(1, Ny):
            v[i,j] = vt[i,j] - (dt/dy) * (p[i,j+1] - p[i,j]) / rho

    # --- Visualization every few steps ---
    if step % 20 == 0:
        # Interpolate u, v to pressure (cell) centers
        uu = 0.5 * (u[1:, 1:-1] + u[:-1, 1:-1])   # average in x
        vv = 0.5 * (v[1:-1, 1:] + v[1:-1, :-1])   # average in y

        # Define mesh for cell centers
        x = np.linspace(dx/2, Lx - dx/2, Nx)
        y = np.linspace(dy/2, Ly - dy/2, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Check shapes (for debugging, optional)
        # print("Shapes:", uu.shape, vv.shape, X.shape)

        plt.clf()
        plt.quiver(X, Y, uu, vv, scale=5)
        plt.title(f"Lid-driven cavity (step={step})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.pause(0.01)

plt.show()
