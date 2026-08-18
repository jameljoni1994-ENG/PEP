"""
PEP Solver - OGM with Proper Auxiliary Point Tracking
=====================================================
OGM (Kim & Fessler 2016):
  y_0 = x_0
  x_1 = y_0 - (1/(L*theta_1)) * g(y_0)
  y_k = x_k + ((theta_k - 1)/theta_{k+1}) * (x_k - x_{k-1})
  x_{k+1} = y_k - (1/(L*theta_{k+1})) * g(y_k)

Key difference from GD: gradients are at y_k, not x_k.
We track f(y_k) and apply interpolation between y_i and y_j.
f(x_N) is connected via smoothness: f(x_N) <= f(y_{N-1}) + <g(y_{N-1}), x_N - y_{N-1}> + (L/2)||x_N - y_{N-1}||^2
"""
import numpy as np
from scipy.optimize import minimize
import time
import sys
import json


def ogm_theta(N):
    """theta_0 = 1, theta_{k+1} = (1 + sqrt(1 + 4*theta_k^2)) / 2"""
    theta = [1.0]
    for k in range(N):
        theta.append((1 + np.sqrt(1 + 4 * theta[-1]**2)) / 2)
    return theta


def ogm_bound(N):
    """Theoretical OGM bound: L*R^2 / (2 * theta_N^2)"""
    theta = ogm_theta(N)
    return 1.0 / (2.0 * theta[N]**2)


def drori_bound(N):
    return 1.0 / (4 * N + 2)


def solve_pep_ogm_v2(N, d, L=1.0, R=1.0, num_restarts=30, verbose=False):
    """
    Proper OGM PEP with auxiliary point tracking.

    Variables:
      vs0:   v_s[0]              (d)
      vg:    v_g[0..N-1]         (N*d)  - gradients at y_0..y_{N-1}
      fy:    f(y_0)..f(y_{N-1})  (N)    - function values at y-points

    Method constraint eliminates v_s[1..N] and v_y[0..N-1] from vs0 and vg.
    f(x_N) is eliminated via smoothness bound from y_{N-1}.
    Objective = f_y[N-1] + L<v_g[N-1], v_s[N] - v_y[N-1]> + (L/2)||v_s[N] - v_y[N-1]||^2
    """
    thetas = ogm_theta(N)

    # Precompute pair indices for interpolation between y-points
    ii, jj = np.meshgrid(range(N), range(N), indexing='ij')
    mask = ~np.eye(N, dtype=bool)
    pi = ii[mask]
    pj = jj[mask]

    def unpack(theta):
        vs0 = theta[:d]
        vg = theta[d:d + N * d].reshape(N, d)
        fy = theta[d + N * d:]

        # Compute y-positions and x-positions from method constraint
        vy = np.zeros((N, d))  # y_0, ..., y_{N-1}
        vs = np.zeros((N + 1, d))  # x_0, ..., x_N

        vs[0] = vs0
        vy[0] = vs[0]  # y_0 = x_0
        vs[1] = vy[0] - (1.0 / thetas[1]) * vg[0]

        for k in range(1, N):
            alpha_k = (thetas[k] - 1) / thetas[k + 1]
            vy[k] = vs[k] + alpha_k * (vs[k] - vs[k - 1])
            vs[k + 1] = vy[k] - (1.0 / thetas[k + 1]) * vg[k]

        return vs, vy, vg, fy

    def compute_objective(vs, vy, vg, fy):
        """f(x_N) = f(y_{N-1}) + L<g(y_{N-1}), x_N - y_{N-1}> + (L/2)||x_N - y_{N-1}||^2"""
        diff = vs[N] - vy[N - 1]
        fN = fy[N - 1] + L * np.dot(vg[N - 1], diff) + 0.5 * L * np.sum(diff**2)
        return fN

    def make_obj(mu):
        def penalty_obj(theta):
            vs, vy, vg, fy = unpack(theta)
            obj = -compute_objective(vs, vy, vg, fy)
            pen = 0.0

            # IC
            g_ic = R**2 - np.sum(vs[0]**2)
            if g_ic < 0:
                pen += g_ic**2

            # OPT at y-points
            vg_norm2 = np.sum(vg**2, axis=1)
            vy_dot_vg = np.sum(vg * vy, axis=1)

            lo_viol = np.maximum(0.5 * L * vg_norm2 - fy, 0.0)
            pen += np.sum(lo_viol**2)

            up_viol = np.maximum(fy - L * vy_dot_vg + 0.5 * L * vg_norm2, 0.0)
            pen += np.sum(up_viol**2)

            # INT between y-points
            diff_y = vy[pi] - vy[pj]
            dot_vg = np.sum(vg[pj] * diff_y, axis=1)
            dv_norm2 = np.sum((vg[pi] - vg[pj])**2, axis=1)
            g_int = fy[pi] - fy[pj] - L * dot_vg - 0.5 * L * dv_norm2
            viol_int = np.maximum(-g_int, 0.0)
            pen += np.sum(viol_int**2)

            return obj + mu * pen
        return penalty_obj

    best_val = -np.inf
    t0 = time.time()

    for k in range(num_restarts):
        alpha = np.random.uniform(0.3, 0.95) * R
        direction = np.random.randn(d)
        direction /= max(np.linalg.norm(direction), 1e-10)
        vs0_init = alpha * direction

        # Initialize vg to be near-optimal for the quadratic
        vg_init = np.zeros((N, d))
        for i in range(N):
            vg_init[i] = vs0_init * (0.5 + 0.3 * np.random.randn()) + np.random.randn(d) * 0.1

        # Compute vs from method constraint
        vs_init = np.zeros((N + 1, d))
        vy_init = np.zeros((N, d))
        vs_init[0] = vs0_init
        vy_init[0] = vs_init[0]
        vs_init[1] = vy_init[0] - (1.0 / thetas[1]) * vg_init[0]
        for kstep in range(1, N):
            alpha_k = (thetas[kstep] - 1) / thetas[kstep + 1]
            vy_init[kstep] = vs_init[kstep] + alpha_k * (vs_init[kstep] - vs_init[kstep - 1])
            vs_init[kstep + 1] = vy_init[kstep] - (1.0 / thetas[kstep + 1]) * vg_init[kstep]

        # Initialize fy within OPT bounds
        fy_init = np.zeros(N)
        for i in range(N):
            lo = 0.5 * L * np.sum(vg_init[i]**2)
            up = L * np.dot(vg_init[i], vy_init[i]) - lo
            if up > lo + 1e-8:
                fy_init[i] = 0.5 * (lo + up)
            else:
                fy_init[i] = lo + 1e-4

        theta0 = np.concatenate([vs0_init.ravel(), vg_init.ravel(), fy_init.ravel()])

        theta = theta0.copy()
        for mu in [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]:
            res = minimize(make_obj(mu), theta, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            theta = res.x

        # Compute final objective
        vs, vy, vg, fy = unpack(theta)
        fN = compute_objective(vs, vy, vg, fy)

        if fN > best_val:
            best_val = fN
            if verbose:
                sys.stdout.write(f"  [{k+1}] f_N = {best_val:.8f}\n")
                sys.stdout.flush()

    t1 = time.time()
    if verbose:
        sys.stdout.write(f"  Final: f_N = {best_val:.8f}, time = {t1-t0:.2f}s\n")
        sys.stdout.flush()
    return best_val, {'value': best_val, 'time': t1 - t0}


if __name__ == "__main__":
    sys.stdout.write("OGM PEP Solver v2 - Proper Auxiliary Point Tracking\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()

    tests = [
        (1, 2, "N=1,d=2 (OGM_ref=0.1910)"),
        (2, 3, "N=2,d=3 (OGM_ref=0.1039)"),
        (3, 5, "N=3,d=5 (OGM_ref=0.0661)"),
        (4, 6, "N=4,d=6 (OGM_ref=0.0461)"),
        (5, 7, "N=5,d=7 (OGM_ref=0.0340)"),
    ]

    for N, d, desc in tests:
        dr = drori_bound(N)
        og = ogm_bound(N)
        sys.stdout.write(f"\n{desc}:\n")
        sys.stdout.write(f"  Drori={dr:.6f}  OGM_ref={og:.6f}\n")
        sys.stdout.flush()
        val, info = solve_pep_ogm_v2(N=N, d=d, num_restarts=30, verbose=True)
        sys.stdout.write(f"  ratio_to_OGM = {val/og:.6f}  ratio_to_Drori = {val/dr:.6f}\n")
        sys.stdout.flush()
