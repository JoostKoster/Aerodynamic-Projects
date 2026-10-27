# AirfoilVortexSheet
# Numerical vortex panel method for a NACA 2412 airfoil
#
# Follows J.D. Anderson, "Fundamentals of Aerodynamics" (6th ed.),
# Sec. 4.10 "Lifting flows over arbitrary bodies: the vortex panel numerical
# method" (pp. 369-374). Equation numbers below are Anderson's.
#
# Outline of the method:
#   1. The body surface is wrapped with a vortex sheet gamma(s) (Fig. 4.15). There is
#      no closed-form solution for gamma(s), so the sheet is approximated by n straight
#      panels (Fig. 3.40) with a CONSTANT strength gamma_j per unit length on each
#      panel. gamma_1 ... gamma_n are the unknowns. Because gamma is constant per
#      panel this is the "first-order" method (p. 373).
#   2. Potential induced at a point P by panel j (4.72):
#          dphi_j = -(1/2pi) * integral_j( theta_pj * gamma_j ds_j )
#      with theta_pj = atan((y - y_j)/(x - x_j))   (4.73),
#      summed over all panels (4.74) and evaluated at control point i (4.75).
#   3. Boundary condition at the control point (= midpoint) of every panel: the
#      normal velocity is zero,  V_inf,n + V_n = 0   (4.78), where
#          V_inf,n = V_inf cos(beta_i)               (3.148)
#          V_n     = d(phi)/d(n_i)                   (4.76)
#      Combining these gives the crux of the method (4.79) / (4.80):
#          V_inf cos(beta_i) - sum_j (gamma_j / 2pi) * J_ij = 0
#          J_ij = integral_j( d(theta_ij)/d(n_i) ds_j )   (depends only on geometry)
#      -> n linear equations (one per panel) for the n unknowns gamma_j.
#   4. Unlike the source panel method (Sec. 3.17), a lifting body also needs the
#      Kutta condition, gamma(TE) = 0. Numerically, for the two panels i and i-1
#      that meet at the trailing edge (Fig. 4.32):   gamma_i = -gamma_(i-1)   (4.81)
#      This is an extra (n+1)-th equation -> the system is overdetermined.
#   5. Surface velocity and loads: the flow inside the body is at rest (Fig. 4.33),
#      so the velocity just outside the sheet equals gamma (from (4.8), gamma = u1 - u2
#      with u2 = 0); the pressure follows from Bernoulli. The circulation is
#          Gamma = sum_j gamma_j * s_j               (4.82)
#      and the lift per unit span is
#          L' = rho_inf * V_inf * Gamma              (4.83)
#
# Conventions used in this code:
#   - panels numbered clockwise, starting at the trailing edge on the lower surface
#   - positive gamma = clockwise circulation (the same sign convention as
#     Anderson's -1/(2pi) * integral(theta * gamma ds) potential in (4.72))
#   - n_i = unit normal of panel i pointing OUT of the body

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ============================================================
# GEOMETRY: NACA 2412
# ============================================================
M = 0.02       # Maximum camber
P = 4          # Maximum camber position (% chord)
p = P / 10     # Maximum camber position (fraction of chord)
t = 0.12       # Maximum thickness (fraction of chord)
c = 1.0        # Chord length

# Cosine-spaced x points (N + 1 points per surface -> 2N panels in total).
# Cosine spacing puts many small panels near the leading and trailing edge and
# fewer, larger panels in the middle of the airfoil, which Anderson (p. 373)
# calls usually advantageous for this method.
N = 100
x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, N + 1)))

# Mean camber line and derivative
y_c = np.zeros_like(x)
dyc_dx = np.zeros_like(x)

front = x < p
y_c[front] = (M / p**2) * (2 * p * x[front] - x[front]**2)
dyc_dx[front] = (2 * M / p**2) * (p - x[front])

back = ~front
y_c[back] = (M / (1 - p)**2) * (1 - 2 * p + 2 * p * x[back] - x[back]**2)
dyc_dx[back] = (2 * M / (1 - p)**2) * (p - x[back])

# Thickness distribution.
# Trailing edge is CLOSED (y_t(1) = 0). Anderson's method uses a closed contour in
# which boundary point n+1 coincides with point 1, so that the two panels at
# the trailing edge (panels i and i-1 in Fig. 4.32) touch at the TE point.
y_t = 5 * t * (
    0.2969 * np.sqrt(x)
    - 0.1260 * x
    - 0.3516 * x**2
    + 0.2843 * x**3
    - 0.1036 * x**4
)

theta = np.arctan(dyc_dx)

x_upper = c * (x - y_t * np.sin(theta))
y_upper = c * (y_c + y_t * np.cos(theta))
x_lower = c * (x + y_t * np.sin(theta))
y_lower = c * (y_c - y_t * np.cos(theta))

# Boundary points, CLOCKWISE (as in Anderson):
# lower trailing edge -> leading edge -> upper trailing edge
xb = np.concatenate((x_lower[::-1], x_upper[1:]))
yb = np.concatenate((y_lower[::-1], y_upper[1:]))

# Close the contour exactly (point n+1 = point 1)
xb[-1], yb[-1] = xb[0], yb[0]

np.savez(
    "NACA2412_geometry.npz",
    x=xb, y=yb,
    x_upper=x_upper, y_upper=y_upper,
    x_lower=x_lower, y_lower=y_lower
)

# ============================================================
# FREESTREAM
# ============================================================
V_inf = 10.0                # Freestream velocity [m/s]
alpha_deg = 5.0             # Angle of attack [degrees]
alpha = np.deg2rad(alpha_deg)

U_inf = V_inf * np.cos(alpha)
W_inf = V_inf * np.sin(alpha)

# ============================================================
# PANEL GEOMETRY  (Fig. 3.40: straight panels between boundary points)
# ============================================================
n = len(xb) - 1                      # number of panels 
X1, Y1 = xb[:-1], yb[:-1]            # panel start points (boundary points j)
X2, Y2 = xb[1:], yb[1:]              # panel end points   (boundary points j+1)

S = np.hypot(X2 - X1, Y2 - Y1)       # panel lengths s_j
if np.any(S < 1e-12):
    raise ValueError("Zero-length panel detected.")

phi = np.arctan2(Y2 - Y1, X2 - X1)   # panel angles

xc = 0.5 * (X1 + X2)                 # control points = panel midpoints (x_i, y_i),
yc = 0.5 * (Y1 + Y2)                 # where the boundary condition (4.78) is applied

tx, ty = np.cos(phi), np.sin(phi)    # unit tangents (clockwise direction)
nx, ny = -np.sin(phi), np.cos(phi)   # unit normals n_i, pointing OUT of the body
                                     # (outward for a clockwise contour)


# ============================================================
# Velocity induced by a constant-strength vortex panel
# ============================================================
def vortex_panel_velocity(xp, yp, j):
    """
    Velocity (u, v) at the points (xp, yp), induced by panel j carrying a
    UNIT vortex strength (positive = clockwise, as in Anderson).
    xp, yp may be scalars or arrays. Not valid for points on panel j itself
    (the self-influence is handled separately in the matrix).

    This is the velocity obtained from the potential of panel j, Eq. (4.72),
    dphi_j = -(1/2pi) * integral( theta_pj * gamma_j ds_j ),
    integrated in closed form along the straight panel. The same function gives
    both the normal-velocity influence coefficients at the control points
    (Eqs. 4.77 / 4.80) and the velocity at the grid points of the flow field.
    """
    # Panel-local coordinates (origin at panel start, x_l along the panel)
    dxp = xp - X1[j]
    dyp = yp - Y1[j]
    xl = dxp * np.cos(phi[j]) + dyp * np.sin(phi[j])
    yl = -dxp * np.sin(phi[j]) + dyp * np.cos(phi[j])

    # Angles and distances to the two panel end points
    # (theta of Eq. 4.73, measured from each end point of the panel)
    th1 = np.arctan2(yl, xl)
    th2 = np.arctan2(yl, xl - S[j])
    r1_sq = xl**2 + yl**2
    r2_sq = (xl - S[j])**2 + yl**2

    # Closed-form integral of the clockwise point-vortex velocity
    # along the panel (gamma constant over the panel)
    u_l = (th2 - th1) / (2.0 * np.pi)
    v_l = np.log(r2_sq / r1_sq) / (4.0 * np.pi)

    # Back to global coordinates
    u = u_l * np.cos(phi[j]) - v_l * np.sin(phi[j])
    v = u_l * np.sin(phi[j]) + v_l * np.cos(phi[j])
    return u, v


# ============================================================
# ASSEMBLE AND SOLVE THE SYSTEM
# Flow tangency at every control point, Anderson Eqs. (4.78)-(4.80):
#     V_inf,n + V_n = 0  ->  sum_j gamma_j * A_ij = -V_inf . n_i
# ============================================================
U = np.zeros((n, n))
V = np.zeros((n, n))
for j in range(n):
    U[:, j], V[:, j] = vortex_panel_velocity(xc, yc, j)

# A_ij = normal velocity at control point i due to a unit-strength vortex on panel j.
# This is the term -(1/2pi) * J_ij of Eqs. (4.77) / (4.80), with
# J_ij = integral_j( d(theta_ij)/d(n_i) ds_j ).
A = U * nx[:, None] + V * ny[:, None]

# A vortex panel induces ZERO normal velocity at its own midpoint
# (the jump across a vortex sheet is tangential), so the diagonal is zero.
np.fill_diagonal(A, 0.0)

# Right-hand side: cancel the freestream normal velocity,
# V_inf,n = V_inf cos(beta_i) = V_inf . n_i   (Eq. 3.148)
rhs = -(U_inf * nx + W_inf * ny)

# Kutta condition, Eq. (4.81): gamma_i = -gamma_(i-1) for the two panels that
# touch at the trailing edge (Fig. 4.32), i.e. gamma(TE) = 0 in the limit.
# With this numbering these are the first panel (lower surface) and the last
# panel (upper surface):  gamma_1 + gamma_m = 0.
# Together with the n equations above this gives n + 1 equations for n unknowns,
# i.e. an OVERDETERMINED system (p. 372).
kutta_row = np.zeros(n)
kutta_row[0] = 1.0
kutta_row[-1] = 1.0


A_full = np.vstack((A, kutta_row))
rhs_full = np.append(rhs, 0.0)
Uu, sv, Vt = np.linalg.svd(A_full, full_matrices=False)
keep = sv > 1e-3 * sv[0]           # truncate the near-null directions
gamma = Vt[keep].T @ ((Uu[:, keep].T @ rhs_full) / sv[keep])

# ============================================================
# AIRFOIL RESULTS
# ============================================================
# gamma_j is also the local surface velocity on panel j: the velocity inside the
# body is zero (Fig. 4.33), so gamma = u1 - u2 = u1 (Eq. 4.8) is the velocity just
# outside the vortex sheet, tangent to the surface (p. 373).
Gamma = np.sum(gamma * S)                 # total circulation, Eq. (4.82): sum(gamma_j * s_j)
Cl = 2.0 * Gamma / (V_inf * c)            # from L' = rho V_inf Gamma (Eq. 4.83), Cl = L' / (0.5 rho V_inf^2 c)
Cp = 1.0 - (gamma / V_inf)**2             # surface pressure coefficient from Bernoulli, V_surface = gamma

print(f"Panels: {n}")
print(f"Circulation Gamma = {Gamma:.4f} m^2/s")
print(f"Cl = {Cl:.4f}")
print(f"Kutta check: gamma_1 + gamma_m = {gamma[0] + gamma[-1]:.2e}")

# ============================================================
# VELOCITY FIELD ON A CARTESIAN GRID
# Freestream + the velocity induced by all vortex panels (superposition)
# ============================================================
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.6, 0.6
nx_grid, ny_grid = 180, 120

Xg, Yg = np.meshgrid(
    np.linspace(x_min, x_max, nx_grid),
    np.linspace(y_min, y_max, ny_grid)
)

Ug = U_inf * np.ones_like(Xg)
Vg = W_inf * np.ones_like(Yg)

# Vectorised over all grid points; loop over panels only
for j in range(n):
    u, v = vortex_panel_velocity(Xg, Yg, j)
    Ug += gamma[j] * u
    Vg += gamma[j] * v

# Mask points inside the airfoil (the velocity there should be ~0, Fig. 4.33)
airfoil_path = Path(np.column_stack((xb, yb)))
inside = airfoil_path.contains_points(
    np.column_stack((Xg.ravel(), Yg.ravel()))
).reshape(Xg.shape)

Ug[inside] = np.nan
Vg[inside] = np.nan

speed = np.sqrt(Ug**2 + Vg**2)

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))

levels = np.linspace(0, 2.0 * V_inf, 41)
contour = ax.contourf(Xg, Yg, speed, levels=levels, cmap="viridis", extend="max")
fig.colorbar(contour, ax=ax, label="Velocity magnitude [m/s]")

ax.streamplot(
    Xg, Yg, Ug, Vg,
    color="white", density=1.4, linewidth=0.7, arrowsize=0.8
)

ax.fill(xb, yb, color="white", zorder=5)
ax.plot(xb, yb, "k-", linewidth=1.5, zorder=6)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"NACA 2412: vortex panel method, alpha = {alpha_deg} deg, Cl = {Cl:.3f}")
ax.set_aspect("equal")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
fig.savefig("NACA2412_flowfield.png", dpi=150)
plt.show()
