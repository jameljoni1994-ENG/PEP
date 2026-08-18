"""
PEP Solver v8 — Method constraint via ELIMINATION (exact)
=========================================================
v_s[i+1] = v_s[i] - h*L*v_y[i]  =>  v_s[i] = v_s[0] - h*L * sum(v_y[0:i])

Free variables: v_s[0] (d), v_y[0..N] ((N+1)*d), f[0..N] (N+1), h (1)
"""
import numpy as np
from scipy.optimize import minimize
import time
import sys


def reference_bounds(N):
    drori = 1.0 / (4*N + 2)
    th = 1.0
    for t in range(1, N):
        th = (1 + np.sqrt(4*th**2 + 1)) / 2
    theta_N = (1 + np.sqrt(8*th**2 + 1)) / 2
    ogm = 1.0 / (2 * theta_N**2)
    return {'drori': drori, 'ogm': ogm}


def solve_pep(N, d, L=1.0, R=1.0, num_restarts=50, verbose=False):
    Np1 = N + 1
    
    # Pair indices for interpolation
    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]
    pj = jj[mask]
    
    def unpack(theta):
        """Unpack: h_raw, v_s0(d), v_y(Np1*d), f(Np1)"""
        h_raw = theta[0]
        h = 0.05 + 1.95 / (1.0 + np.exp(-h_raw))  # (0.05, 2.0)
        vs0 = theta[1:1+d]
        vy = theta[1+d:1+d+Np1*d].reshape(Np1, d)
        fv = theta[1+d+Np1*d:]
        
        # Compute all v_s from method constraint
        vs = np.zeros((Np1, d))
        vs[0] = vs0
        for i in range(N):
            vs[i+1] = vs[i] - h * L * vy[i]
        
        return h, vs, vy, fv
    
    def penalty_obj(theta):
        h, vs, vy, fv = unpack(theta)
        
        obj = -fv[N]
        penalty = 0.0
        
        # IC: R^2 - ||vs[0]||^2 >= 0
        g_ic = R**2 - np.sum(vs[0]**2)
        if g_ic < 0:
            penalty += g_ic**2
        
        # OPT (vectorized)
        vy_norm2 = np.sum(vy**2, axis=1)
        g_opt = fv - 0.5*L*vy_norm2
        viol_opt = np.minimum(g_opt, 0.0)
        penalty += np.sum(viol_opt**2)
        
        # INT (vectorized)
        diff_s = vs[pi] - vs[pj]
        diff_y = vy[pi] - vy[pj]
        dot_vy = np.sum(vy[pj] * diff_s, axis=1)
        dv_norm2 = np.sum(diff_y**2, axis=1)
        g_int = fv[pi] - fv[pj] - L*dot_vy - 0.5*L*dv_norm2
        viol_int = np.minimum(g_int, 0.0)
        penalty += np.sum(viol_int**2)
        
        return obj + 1e6 * penalty
    
    best_val = -np.inf
    best_theta = None
    t0 = time.time()
    
    for k in range(num_restarts):
        alpha = np.random.uniform(0.2, 0.9) * R
        direction = np.random.randn(d)
        direction /= max(np.linalg.norm(direction), 1e-10)
        
        # Initialize: quadratic f(x) = (L/2)||x||^2 with h=1/L
        h_init = 1.0 / L
        vs0_init = alpha * direction
        vy_init = np.zeros((Np1, d))
        vs_init = np.zeros((Np1, d))
        vs_init[0] = vs0_init
        vy_init[0] = vs0_init.copy()
        for i in range(N):
            vs_init[i+1] = vs_init[i] - h_init * L * vy_init[i]
            vy_init[i+1] = vs_init[i+1].copy()
        fv_init = np.maximum(0.5*L*np.sum(vs_init**2, axis=1), 1e-8)
        
        h_raw = np.log(h_init - 0.05) - np.log(2.0 - h_init)
        theta0 = np.concatenate([[h_raw], vs0_init.ravel(), vy_init.ravel(), fv_init.ravel()])
        
        theta = theta0.copy()
        for mu_exp in [1, 2, 3, 4, 5, 6]:
            mu = 10.0 ** mu_exp
            res = minimize(lambda t: -fv[N] + mu * np.maximum(0, -(R**2 - np.sum((unpack(t)[1])[0]**2)))**2
                          if False else penalty_obj(t), theta, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            theta = res.x
        
        h_res, vs_res, vy_res, fv_res = unpack(theta)
        fN = fv_res[N]
        
        if fN > best_val:
            best_val = fN
            best_theta = theta.copy()
            if verbose:
                sys.stdout.write(f"  [{k+1}] f_N = {best_val:.8f}, h = {h_res:.4f}\n")
                sys.stdout.flush()
    
    t1 = time.time()
    if verbose:
        sys.stdout.write(f"  Final: f_N = {best_val:.8f}, time = {t1-t0:.2f}s\n")
        sys.stdout.flush()
    
    return best_val, {'value': best_val, 'time': t1-t0}


if __name__ == "__main__":
    sys.stdout.write("PEP Solver v8 (Eliminated method constraint)\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()
    
    tests = [
        (1, 1, "d=1,N=1"),
        (1, 2, "d=2,N=1"),
        (2, 3, "d=3,N=2"),
        (3, 5, "d=5,N=3"),
        (5, 6, "d=6,N=5"),
    ]
    
    for N, d, desc in tests:
        sys.stdout.write(f"\n{desc}:\n")
        sys.stdout.flush()
        val, info = solve_pep(N=N, d=d, num_restarts=30, verbose=True)
        refs = reference_bounds(N)
        sys.stdout.write(f"  Drori: {refs['drori']:.8f}, OGM: {refs['ogm']:.8f}\n")
        sys.stdout.flush()
