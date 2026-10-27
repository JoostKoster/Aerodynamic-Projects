# Numerical lifting-line method (Anderson, Fundamentals of Aerodynamics, sec 5.4)
# rectangular wing with a non-linear lift curve of the airfoil

import numpy as np
import matplotlib.pyplot as plt

# ---------- input ----------
b = 10.0            # span [m]
AR = 8.0            # aspect ratio
c = b / AR          # chord [m], constant (rectangular wing)
S = b * c           # wing area [m^2]
V = 50.0            # freestream velocity [m/s]

k = 40              # number of intervals along the span, must be even (Simpson)
D = 0.05            # damping factor (book: about 0.05)
tol = 1e-4          # 0.01 percent
max_iter = 5000

# lift curve of the airfoil: alpha_eff [deg] vs cl
# NOTE: rough numbers for a cambered 12% airfoil, replace with real airfoil data
alpha_table = np.array([-30, -20, -16, -14, -12, -8, -4, 0, 4, 8, 12, 14, 15, 16, 18, 20, 30])
cl_table = np.array([-0.95, -0.95, -1.02, -1.10, -1.00, -0.60, -0.18, 0.15, 0.58,
                     1.00, 1.38, 1.50, 1.55, 1.48, 1.25, 1.10, 1.00])

# step 1: spanwise stations (k+1 stations from tip to tip)
y = np.linspace(-b / 2, b / 2, k + 1)
dy = y[1] - y[0]
chord = c * np.ones(k + 1)


def simpson(f, dy):
    # simpson's rule, weights 1 4 2 4 ... 4 1
    return dy / 3 * (f[0] + 4 * np.sum(f[1:-1:2]) + 2 * np.sum(f[2:-1:2]) + f[-1])


def solve(alpha_deg):
    alpha = np.deg2rad(alpha_deg)

    # step 2: assume elliptic circulation distribution
    # (for an elliptic wing alpha_eff = alpha / (1 + 2/AR), used for a first guess of gamma0)
    cl0 = np.interp(alpha_deg / (1 + 2 / AR), alpha_table, cl_table)
    gamma0 = 0.5 * V * c * cl0
    gamma = gamma0 * np.sqrt(1 - (2 * y / b) ** 2)

    n_ok = 0
    for it in range(max_iter):

        # step 3: induced angle of attack at every station, eq 5.76
        dgdy = np.gradient(gamma, dy, edge_order=2)
        alpha_i = np.zeros(k + 1)
        for n in range(k + 1):
            f = np.zeros(k + 1)
            for j in range(k + 1):
                if j != n:
                    f[j] = dgdy[j] / (y[n] - y[j])
            # singularity at j = n: take the average of the two neighbours
            if n == 0:
                f[n] = f[1]
            elif n == k:
                f[n] = f[k - 1]
            else:
                f[n] = 0.5 * (f[n - 1] + f[n + 1])
            alpha_i[n] = simpson(f, dy) / (4 * np.pi * V)

        # step 4: effective angle of attack
        alpha_eff = alpha - alpha_i

        # step 5: cl from the lift curve of the airfoil
        cl = np.interp(np.rad2deg(alpha_eff), alpha_table, cl_table)

        # step 6: new circulation (Kutta-Joukowski)
        gamma_new = 0.5 * V * chord * cl
        gamma_new[0] = 0.0      # gamma is zero at the wing tips
        gamma_new[-1] = 0.0

        # step 7: damped update
        error = np.max(np.abs(gamma_new - gamma)) / np.max(np.abs(gamma_new))
        gamma = gamma + D * (gamma_new - gamma)

        # step 8: converged when the error is small for 5 iterations in a row
        if error < tol:
            n_ok += 1
        else:
            n_ok = 0
        if n_ok == 5:
            break

    if n_ok < 5:
        print("warning: not converged for alpha =", alpha_deg)

    # gamma = 0 at the tips so cl is 0 there as well (only matters for the plot)
    cl[0] = 0.0
    cl[-1] = 0.0

    # step 9: lift and induced drag coefficient (eq 5.26 and 5.30)
    CL = 2 / (V * S) * simpson(gamma, dy)
    CDi = 2 / (V * S) * simpson(gamma * alpha_i, dy)
    return CL, CDi, gamma, cl, alpha_i, it + 1


# ---------- one angle of attack ----------
alpha_deg = 6.0
CL, CDi, gamma, cl, alpha_i, iterations = solve(alpha_deg)
e = CL ** 2 / (np.pi * AR * CDi)

print("alpha =", alpha_deg, "deg")
print("iterations:", iterations)
print("CL  =", round(CL, 4))
print("CDi =", round(CDi, 5))
print("e   =", round(e, 3))

# ---------- sweep over alpha (CL-alpha curve) ----------
alphas = np.arange(-2, 17, 1.0)
CL_list = []
for a in alphas:
    CL_list.append(solve(a)[0])

# ---------- plots ----------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(y / (b / 2), gamma / np.max(gamma))
plt.xlabel("y / (b/2)")
plt.ylabel("Gamma / Gamma_0")
plt.title("circulation, alpha = " + str(alpha_deg))
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(y / (b / 2), cl)
plt.xlabel("y / (b/2)")
plt.ylabel("section cl")
plt.title("section lift coefficient")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(alphas, CL_list, "o-")
plt.xlabel("alpha [deg]")
plt.ylabel("CL")
plt.title("CL - alpha, AR = " + str(AR))
plt.grid()

plt.tight_layout()
plt.show()
