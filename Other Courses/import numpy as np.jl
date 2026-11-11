import numpy as np
import matplotlib.pyplot as plt
from time import time

# ---------------------------
# Parameters (user can change)
# ---------------------------
Nx = 41               # number of pressure cells in x
Ny = 41               # number of pressure cells in y
Lx = 1.0              # domain length (x)
Ly = 1.0              # domain length (y)
Re = 100.0            # Reynolds number
nu = 1.0 / Re         # kinematic viscosity
U_lid = 1.0           # lid velocity (top boundary)
dt = 0.001            # time step
nt = 5000             # number of time steps
plot_interval = 200   # how often to plot

# SOR parameters for Poisson solve
omega_sor = 1.6
poisson_tol = 1e-6
poisson_maxit = 5000

# ---------------------------
# Grid (MAC staggered)
# u velocities at vertical faces: shape (nx+1, ny)
# v velocities at horizontal faces: shape (nx, ny+1)
# p at cell centers: shape (nx, ny)
# ---------------------------
dx = Lx / Nx
dy = Ly / Ny

# create arrays
u = np.zeros((Nx + 1, Ny))       # u[i,j] located at x = i*dx, y = (j+0.5)*dy
v = np.zeros((Nx, Ny + 1))       # v[i,j] located at x = (i+0.5)*dx, y = j*dy
p = np.zeros((Nx, Ny))           # pressure at cell centers

# helper coordinates (cell centers for pressure)
xc = (np.arange(Nx) + 0.5) * dx
yc = (np.arange(Ny) + 0.5) * dy
Xc, Yc = np.meshgrid(xc, yc, indexing='ij')

# Apply initial lid BC (top boundary for u faces)
# top boundary: u[:, Ny-1] is interior face row near top; but for MAC we set u at top face index Ny-1 corresponds to face between last cell and top boundary? 
# Simpler approach: we'll apply boundary conditions explicitly each time step.

def apply_bcs(u, v):
    """Apply velocity boundary conditions for lid-driven cavity."""
    # u: shape (Nx+1, Ny)
    # v: shape (Nx, Ny+1)

    # Left wall (x=0): u[0,:] = 0
    u[0, :] = 0.0
    # Right wall (x=Lx): u[Nx,:] = 0
    u[-1, :] = 0.0

    # Bottom wall (y=0): v[:,0] = 0
    v[:, 0] = 0.0
    # Top wall (y=Ly): v[:,Ny] = 0
    v[:, -1] = 0.0

    # No-penetration for u at top and bottom faces:
    # bottom faces (between cell 0 and boundary): u[:,0] corresponds to face at y = 0.5*dy; no-slip requires u interior near walls adjust via ghost/BC in update step.
    # Enforce no-slip on horizontal walls by setting the tangential velocities on the boundary faces:
    # Bottom wall (y=0): u[:,0] = 0
    u[:, 0] = 0.0
    # Top wall (y=Ly): u[:, -1] set to lid velocity
    u[:, -1] = U_lid

    # Left and right vertical v-wall tangential velocities zero:
    v[0, :] = 0.0
    v[-1, :] = 0.0

    return u, v

# Poisson solver (SOR) for pressure correction phi (right-hand side = divergence / dt)
def solve_poisson(p, rhs, dx, dy, omega=1.6, tol=1e-6, maxit=5000):
    """
    Solve Poisson: Laplacian(p) = rhs
    Using SOR iteration on p-grid (cell centers).
    p and rhs shapes: (Nx, Ny)
    """
    nx, ny = p.shape
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)

    for it in range(maxit):
        p_old = p.copy()
        # update internal cells
        p[1:-1, 1:-1] = (1 - omega) * p[1:-1, 1:-1] + omega * (
            (dy2 * (p[2:, 1:-1] + p[:-2, 1:-1]) + dx2 * (p[1:-1, 2:] + p[1:-1, :-2]) - dx2 * dy2 * rhs[1:-1, 1:-1])
            / denom
        )
        # Boundary conditions for p: Neumann (dp/dn = 0) on cavity walls (commonly used)
        # dp/dx = 0 at left and right => p[0,:] = p[1,:], p[-1,:] = p[-2,:]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        # dp/dy = 0 at bottom and top => p[:,0] = p[:,1], p[:,-1] = p[:,-2]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        # convergence check (L2 norm)
        res = np.linalg.norm(p - p_old, ord=2)
        if res < tol:
            # print("Poisson converged in", it, "iters")
            break
    return p

# Utility: compute divergence at cell centers (using face velocities)
def divergence(u, v, dx, dy):
    """
    Compute divergence (du/dx + dv/dy) at cell centers from MAC velocities.
    u shape: (Nx+1, Ny), v shape: (Nx, Ny+1)
    returns div shape (Nx,Ny)
    """
    du_dx = (u[1:, :] - u[:-1, :]) / dx      # shape (Nx, Ny)
    dv_dy = (v[:, 1:] - v[:, :-1]) / dy     # shape (Nx, Ny)
    return du_dx + dv_dy

# Main time-stepping (explicit predictor - projection)
start_time = time()

# Pre-apply BCs
u, v = apply_bcs(u, v)

for n in range(nt):
    # Predictor step: compute tentative velocity u_star, v_star using explicit convective + diffusive terms
    u_star = u.copy()
    v_star = v.copy()

    # Compute convective terms using upwind (first-order) for stability
    # For u (faces at i=1..Nx-1, j=0..Ny-1)
    for i in range(1, Nx):
        for j in range(1, Ny-1):
            # velocities at surrounding points for convective fluxes
            # get u at face, and approximate u_x and u_y using neighboring face/cell values
            # u at (i,j): face center
            # approximate du/dx using neighboring u faces:
            ue = u[i+1, j] if i+1 <= Nx else 0.0
            uw = u[i-1, j]
            un = 0.25 * (u[i, j+1] + u[i-1, j+1]) if j+1 < Ny else 0.0
            us = 0.25 * (u[i, j-1] + u[i-1, j-1]) if j-1 >= 0 else 0.0

            # convective derivative approximations (upwind)
            # compute approximate velocities at cell-centered nodes to get v at u-face
            v_at_u = 0.25 * (v[i-1, j] + v[i, j] + v[i-1, j+1] + v[i, j+1])

            # du/dx (central)
            du_dx = (ue - uw) / (2 * dx)
            # du/dy (central)
            du_dy = (un - us) / (2 * dy)

            # convective term
            conv = u[i, j] * du_dx + v_at_u * du_dy

            # diffusion (second-order)
            u_e = u[i+1, j] if i+1 <= Nx else u[i, j]
            u_w = u[i-1, j]
            u_n = u[i, j+1] if j+1 < Ny else u[i, j]
            u_s = u[i, j-1] if j-1 >= 0 else u[i, j]
            diff = nu * ((u_e - 2*u[i, j] + u_w) / dx**2 + (u_n - 2*u[i, j] + u_s) / dy**2)

            u_star[i, j] = u[i, j] + dt * (-conv + diff)

    # For v (faces at i=0..Nx-1, j=1..Ny)
    for i in range(1, Nx-1):
        for j in range(1, Ny):
            # approximate values
            vn = v[i, j+1] if j+1 <= Ny else v[i, j]
            vs = v[i, j-1]
            ve = 0.25 * (v[i+1, j] + v[i+1, j-1]) if i+1 < Nx else 0.0
            vw = 0.25 * (v[i-1, j] + v[i-1, j-1]) if i-1 >= 0 else 0.0

            # approximate u at v-face
            u_at_v = 0.25 * (u[i, j-1] + u[i+1, j-1] + u[i, j] + u[i+1, j])

            dv_dx = (ve - vw) / (2 * dx)
            dv_dy = (vn - vs) / (2 * dy)

            conv = u_at_v * dv_dx + v[i, j] * dv_dy

            # diffusion
            v_e = v[i+1, j] if i+1 < Nx else v[i, j]
            v_w = v[i-1, j]
            v_n = v[i, j+1] if j+1 <= Ny else v[i, j]
            v_s = v[i, j-1]
            diff = nu * ((v_e - 2*v[i, j] + v_w) / dx**2 + (v_n - 2*v[i, j] + v_s) / dy**2)

            v_star[i, j] = v[i, j] + dt * (-conv + diff)

    # apply BCs to star velocities
    u_star, v_star = apply_bcs(u_star, v_star)

    # Compute RHS for Poisson: divergence(u_star)/dt
    div_star = divergence(u_star, v_star, dx, dy)
    rhs = div_star / dt

    # Solve Poisson for pressure p such that Lap(p) = rhs
    p = solve_poisson(p, rhs, dx, dy, omega=omega_sor, tol=poisson_tol, maxit=poisson_maxit)

    # Correct velocities: u_new = u_star - dt * dp/dx ; v_new = v_star - dt * dp/dy
    u_new = u_star.copy()
    v_new = v_star.copy()

    # dp/dx at u face centers (i=1..Nx-1)
    dpdx = (p[1:, :] - p[:-1, :]) / dx   # shape (Nx-1, Ny)
    u_new[1:-1, :] = u_star[1:-1, :] - dt * dpdx

    # dp/dy at v face centers (j=1..Ny-1)
    dpdy = (p[:, 1:] - p[:, :-1]) / dy   # shape (Nx, Ny-1)
    v_new[:, 1:-1] = v_star[:, 1:-1] - dt * dpdy

    # apply BCs
    u_new, v_new = apply_bcs(u_new, v_new)

    # update arrays
    u, v = u_new, v_new

    # optional diagnostic: max divergence
    if n % 100 == 0:
        max_div = np.abs(divergence(u, v, dx, dy)).max()
        print(f"step {n}, time {n*dt:.4f}, max_div = {max_div:.3e}")

    # plotting
    if n % plot_interval == 0 or n == nt - 1:
        plt.clf()
        # compute center velocities for plotting streamlines (u_c,v_c)
        u_c = 0.5 * (u[:-1, :] + u[1:, :])    # shape (Nx,Ny)
        v_c = 0.5 * (v[:, :-1] + v[:, 1:])    # shape (Nx,Ny)

        speed = np.sqrt(u_c**2 + v_c**2)
        plt.subplot(1,2,1)
        plt.contourf(Xc, Yc, speed.T, levels=50, cmap='jet')  # note: transpose for correct orientation
        plt.title(f"|V| at t={n*dt:.3f}")
        plt.colorbar()
        plt.axis('equal')

        plt.subplot(1,2,2)
        plt.streamplot(Xc.T, Yc.T, u_c.T, v_c.T, density=1.5, color='k')
        plt.title("Streamlines")
        plt.pause(0.01)

end_time = time()
print("Finished. Elapsed s:", end_time - start_time)
plt.show()
