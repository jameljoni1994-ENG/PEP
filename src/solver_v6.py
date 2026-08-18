"""
PEP Solver v6 — Corrected interpolation constraint + Smooth barrier
===================================================================
Key fix: interpolation constraint is f_i - f_j >= ..., NOT f_j - f_i.
"""
import numpy as np
from scipy.optimize import minimize
import time
import sys


def solve_pep(N, d, L=1.0, R=1.0, num_restarts=50, verbose=False):
    Np1 = N + 1
    dv = Np1 * d
    
    # Precompute pair indices (i, j) for interpolation: ALL ordered pairs i != j
    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]  # "from" index
    pj = jj[mask]  # "to" index
    n_pairs = len(pi)
    
    def check_feasibility(theta):
        """Check all constraints, return (feasible, violations)."""
        vs = theta[:dv].reshape(Np1, d)
        vy = theta[dv:2*dv].reshape(Np1, d)
        fv = theta[2*dv:]
        
        violations = []
        
        # IC: R^2 - ||vs[0]||^2 >= 0
        violations.append(('IC', R**2 - np.sum(vs[0]**2)))
        
        # OPT: fv[i] - (L/2)||vy[i]||^2 >= 0
        vy_norm2 = np.sum(vy**2, axis=1)
        for i in range(Np1):
            violations.append(('OPT', fv[i] - 0.5*L*vy_norm2[i]))
        
        # INT: fv[i] - fv[j] - L<v_y_j, v_s_i - v_s_j> - (L/2)||v_y_i - v_y_j||^2 >= 0
        for k in range(n_pairs):
            i, j = pi[k], pj[k]
            dot = np.dot(vy[j], vs[i] - vs[j])
            dv_norm = np.sum((vy[i] - vy[j])**2)
            violations.append(('INT', fv[i] - fv[j] - L*dot - 0.5*L*dv_norm))
        
        return violations
    
    def penalty_obj(theta):
        vs = theta[:dv].reshape(Np1, d)
        vy = theta[dv:2*dv].reshape(Np1, d)
        fv = theta[2*dv:]
        
        obj = -fv[N]
        
        # IC
        g_ic = R**2 - np.sum(vs[0]**2)
        if g_ic < 0:
            obj += 1000.0 * g_ic**2
        
        # OPT (vectorized)
        vy_norm2 = np.sum(vy**2, axis=1)
        g_opt = fv - 0.5*L*vy_norm2
        viol_opt = np.minimum(g_opt, 0)
        if np.any(viol_opt != 0):
            obj += 1000.0 * np.sum(viol_opt**2)
        
        # INT (vectorized)
        # fv[i] - fv[j] - L<v_y_j, v_s_i - v_s_j> - (L/2)||v_y_i - v_y_j||^2
        diff_s = vs[pi] - vs[pj]               # (n_pairs, d)
        diff_y = vy[pi] - vy[pj]               # (n_pairs, d)
        dot_vy_js = np.sum(vy[pj] * diff_s, axis=1)  # (n_pairs,)
        dv_norm2 = np.sum(diff_y**2, axis=1)          # (n_pairs,)
        g_int = fv[pi] - fv[pj] - L*dot_vy_js - 0.5*L*dv_norm2
        viol_int = np.minimum(g_int, 0)
        if np.any(viol_int != 0):
            obj += 1000.0 * np.sum(viol_int**2)
        
        return obj
    
    best_val = -np.inf
    best_theta = None
    t0 = time.time()
    
    for k in range(num_restarts):
        alpha = np.random.uniform(0.15, 0.85) * R
        direction = np.random.randn(d)
        norm_d = np.linalg.norm(direction)
        direction = direction / max(norm_d, 1e-10)
        
        vs0 = np.zeros((Np1, d))
        vs0[0] = alpha * direction
        for i in range(1, Np1):
            # Simulate gradient descent steps toward x_*
            vs0[i] = vs0[i-1] * 0.5 + np.random.randn(d)*0.001
        
        vy0 = vs0.copy() + np.random.randn(Np1, d)*0.001
        fv0 = np.maximum(0.5*L*np.sum(vs0**2, axis=1), 1e-4)
        
        theta0 = np.concatenate([vs0.ravel(), vy0.ravel(), fv0.ravel()])
        
        # Verify feasibility of initial point
        viols = check_feasibility(theta0)
        min_viol = min(v for _, v in viols)
        if min_viol < -0.1:
            # Scale down
            scale = max(0.05, 0.1 / max(-min_viol, 1e-10))
            vs0 *= scale
            vy0 *= scale
            fv0 = np.maximum(0.5*L*np.sum(vs0**2, axis=1), 1e-6)
            theta0 = np.concatenate([vs0.ravel(), vy0.ravel(), fv0.ravel()])
        
        # Solve with increasing penalty
        theta = theta0.copy()
        for mu in [10, 100, 1000]:
            res = minimize(penalty_obj, theta, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
            theta = res.x
        
        fv_res = theta[2*dv:]
        fN = fv_res[N]
        
        if fN > best_val:
            best_val = fN
            best_theta = theta.copy()
            if verbose:
                sys.stdout.write(f"  [{k+1}] f_N = {best_val:.8f}\n")
                sys.stdout.flush()
    
    t1 = time.time()
    if verbose:
        sys.stdout.write(f"  Final: f_N = {best_val:.8f}, time = {t1-t0:.2f}s\n")
        sys.stdout.flush()
    
    return best_val, {'value': best_val, 'time': t1-t0}


def reference_bounds(N):
    drori = 1.0 / (4*N + 2)
    th = 1.0
    for t in range(1, N):
        th = (1 + np.sqrt(4*th**2 + 1)) / 2
    theta_N = (1 + np.sqrt(8*th**2 + 1)) / 2
    ogm = 1.0 / (2 * theta_N**2)
    return {'drori': drori, 'ogm': ogm}


if __name__ == "__main__":
    sys.stdout.write("PEP Solver v6 (Corrected INT constraint)\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()
    
    tests = [
        (1, 1, "d=1,N=1 (expected 0)"),
        (1, 2, "d=2,N=1 (expected 0)"),
        (2, 3, "d=3,N=2"),
        (3, 5, "d=5,N=3"),
        (3, 10, "d=10,N=3 (large d)"),
    ]
    
    for N, d, desc in tests:
        sys.stdout.write(f"\n{desc}:\n")
        sys.stdout.flush()
        val, info = solve_pep(N=N, d=d, num_restarts=25, verbose=True)
        refs = reference_bounds(N)
        sys.stdout.write(f"  Drori: {refs['drori']:.8f}, OGM: {refs['ogm']:.8f}\n")
        sys.stdout.flush()
