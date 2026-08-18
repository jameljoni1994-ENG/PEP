"""
PEP Solver v5 — Smooth Log-Barrier + Gradient
Uses log-barrier for smooth constraint handling + scipy minimize.
"""
import numpy as np
from scipy.optimize import minimize
import time
import sys

def solve_pep(N, d, L=1.0, R=1.0, num_restarts=50, verbose=False):
    Np1 = N + 1
    dv = Np1 * d
    
    # Precompute pair indices
    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]
    pj = jj[mask]
    
    def barrier_obj(theta, mu=1.0):
        vs = theta[:dv].reshape(Np1, d)
        vy = theta[dv:2*dv].reshape(Np1, d)
        fv = theta[2*dv:]
        
        obj = -fv[N]
        
        # Barrier terms: -mu * log(g_i) for each constraint g_i >= 0
        # IC: g = R^2 - ||vs[0]||^2
        g_ic = R**2 - np.sum(vs[0]**2)
        if g_ic <= 0:
            return 1e20
        obj -= mu * np.log(g_ic)
        
        # OPT: g_i = fv[i] - (L/2)||vy[i]||^2
        vy_norm2 = np.sum(vy**2, axis=1)
        g_opt = fv - 0.5*L*vy_norm2
        if np.any(g_opt <= 0):
            return 1e20
        obj -= mu * np.sum(np.log(g_opt))
        
        # INT: g = fv[pj] - fv[pi] - L*<vy[pj], vs[pi]-vs[pj]> - (L/2)||vy[pi]-vy[pj]||^2
        dv_vec = vy[pi] - vy[pj]
        dot_vy = np.sum(vy[pj] * (vs[pi] - vs[pj]), axis=1)
        dv_norm2 = np.sum(dv_vec**2, axis=1)
        g_int = fv[pj] - fv[pi] - L*dot_vy - 0.5*L*dv_norm2
        if np.any(g_int <= 0):
            return 1e20
        obj -= mu * np.sum(np.log(g_int))
        
        return obj
    
    best_val = -np.inf
    t0 = time.time()
    
    for k in range(num_restarts):
        alpha = np.random.uniform(0.15, 0.85) * R
        direction = np.random.randn(d)
        norm_d = np.linalg.norm(direction)
        if norm_d < 1e-10:
            direction = np.ones(d) / np.sqrt(d)
        else:
            direction /= norm_d
        
        vs0 = np.zeros((Np1, d))
        vs0[0] = alpha * direction
        for i in range(1, Np1):
            vs0[i] = vs0[0] * (1.0 - 0.15*i/Np1)
        
        vy0 = vs0.copy()
        fv0 = np.maximum(0.5*L*np.sum(vs0**2, axis=1), 1e-4)
        
        theta0 = np.concatenate([vs0.ravel(), vy0.ravel(), fv0.ravel()])
        
        # Central path: increase mu
        theta = theta0.copy()
        feasible = True
        for mu in [0.01, 0.1, 1.0, 10.0, 100.0]:
            res = minimize(lambda t: barrier_obj(t, mu), theta, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun > 1e19:
                feasible = False
                break
            theta = res.x
        
        if not feasible:
            continue
        
        fv_res = theta[2*dv:]
        fN = fv_res[N]
        
        if fN > best_val:
            best_val = fN
            if verbose:
                sys.stdout.write(f"  Restart {k+1}: f_N = {best_val:.8f}\n")
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
    sys.stdout.write("PEP Solver v5 (Log-Barrier)\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()
    
    tests = [
        (1, 1, 0.0, "d=1,N=1"),
        (1, 2, 0.0, "d=2,N=1"),
        (2, 3, None, "d=3,N=2"),
        (3, 5, None, "d=5,N=3"),
    ]
    
    for N, d, expected, desc in tests:
        sys.stdout.write(f"\n{desc}:\n")
        sys.stdout.flush()
        val, info = solve_pep(N=N, d=d, num_restarts=20, verbose=True)
        refs = reference_bounds(N)
        sys.stdout.write(f"  Drori: {refs['drori']:.8f}, OGM: {refs['ogm']:.8f}\n")
        if expected is not None:
            sys.stdout.write(f"  Expected: {expected}, PASS: {abs(val - expected) < 0.01}\n")
        sys.stdout.flush()
