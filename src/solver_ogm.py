"""
PEP Solver - OGM (Optimal Gradient Method)
===========================================
Kim & Fessler (2016):
  y_0 = x_0
  x_1 = y_0 - (1/(L*theta_1)) * g(y_0)
  y_k = x_k + ((theta_k - 1)/theta_{k+1}) * (x_k - x_{k-1})
  x_{k+1} = y_k - (1/(L*theta_{k+1})) * g(y_k)

Same BM framework as GD solver, different method constraint.
"""
import numpy as np
from scipy.optimize import minimize
import time
import sys
import json
import os


def ogm_theta(N):
    """theta_0 = 1, theta_{k+1} = (1 + sqrt(1 + 4*theta_k^2)) / 2"""
    theta = [1.0]
    for k in range(N):
        theta.append((1 + np.sqrt(1 + 4 * theta[-1]**2)) / 2)
    return theta


def ogm_bound(N):
    """Theoretical OGM bound: 1 / (2 * theta_N^2)"""
    theta = ogm_theta(N)
    return 1.0 / (2.0 * theta[N]**2)


def drori_bound(N):
    return 1.0 / (4 * N + 2)


def solve_pep_ogm(N, d, L=1.0, R=1.0, num_restarts=20, verbose=False):
    Np1 = N + 1
    thetas = ogm_theta(N)

    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]
    pj = jj[mask]

    def unpack(theta):
        vs0 = theta[:d]
        vg = theta[d:d + Np1 * d].reshape(Np1, d)
        fv = theta[d + Np1 * d:]
        vs = np.zeros((Np1, d))
        vs[0] = vs0
        vs[1] = vs[0] - (1.0 / thetas[1]) * vg[0]
        for k in range(1, N):
            alpha_k = (thetas[k] - 1) / thetas[k + 1]
            beta_k = 1.0 / thetas[k + 1]
            yk = vs[k] + alpha_k * (vs[k] - vs[k - 1])
            vs[k + 1] = yk - beta_k * vg[k]
        return vs, vg, fv

    def make_obj(mu):
        def penalty_obj(theta):
            vs, vg, fv = unpack(theta)
            obj = -fv[N]
            pen = 0.0

            g_ic = R**2 - np.sum(vs[0]**2)
            if g_ic < 0:
                pen += g_ic**2

            vy_norm2 = np.sum(vg**2, axis=1)
            lo_viol = np.maximum(0.5 * L * vy_norm2 - fv, 0.0)
            pen += np.sum(lo_viol**2)

            dot_vs_vg = np.sum(vg * vs, axis=1)
            up_viol = np.maximum(fv - L * dot_vs_vg + 0.5 * L * vy_norm2, 0.0)
            pen += np.sum(up_viol**2)

            diff_s = vs[pi] - vs[pj]
            dot_vg = np.sum(vg[pj] * diff_s, axis=1)
            dv_norm2 = np.sum((vg[pi] - vg[pj])**2, axis=1)
            g_int = fv[pi] - fv[pj] - L * dot_vg - 0.5 * L * dv_norm2
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

        vg_init = np.zeros((Np1, d))
        vs_init = np.zeros((Np1, d))
        vs_init[0] = vs0_init
        vg_init[0] = vs0_init + np.random.randn(d) * 0.3
        vs_init[1] = vs_init[0] - (1.0 / thetas[1]) * vg_init[0]
        for kstep in range(1, N):
            alpha_k = (thetas[kstep] - 1) / thetas[kstep + 1]
            beta_k = 1.0 / thetas[kstep + 1]
            yk = vs_init[kstep] + alpha_k * (vs_init[kstep] - vs_init[kstep - 1])
            vs_init[kstep + 1] = yk - np.random.randn(d) * 0.05
            vg_init[kstep] = np.random.randn(d) * 0.1

        fv_init = np.zeros(Np1)
        for i in range(Np1):
            lo = 0.5 * L * np.sum(vg_init[i]**2)
            up = L * np.dot(vg_init[i], vs_init[i]) - lo
            if up > lo + 1e-8:
                fv_init[i] = 0.5 * (lo + up)
            else:
                fv_init[i] = lo + 1e-4

        theta0 = np.concatenate([vs0_init.ravel(), vg_init.ravel(), fv_init.ravel()])

        theta = theta0.copy()
        for mu in [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]:
            res = minimize(make_obj(mu), theta, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            theta = res.x

        fv_res = theta[d + Np1 * d:]
        fN = fv_res[N]
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
    sys.stdout.write("OGM PEP Solver - Validation\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()

    tests = [
        (1, 2), (2, 3), (3, 5), (4, 6), (5, 7),
    ]
    for N, d in tests:
        dr = drori_bound(N)
        og = ogm_bound(N)
        sys.stdout.write(f"\nN={N}, d={d}:  Drori={dr:.6f}  OGM_ref={og:.6f}\n")
        sys.stdout.flush()
        val, info = solve_pep_ogm(N=N, d=d, num_restarts=20, verbose=True)
        sys.stdout.write(f"  ratio_to_Drori = {val/dr:.6f}\n")
        sys.stdout.flush()
