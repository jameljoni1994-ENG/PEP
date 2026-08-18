"""
PEP Solver v7 — WITH method constraint
========================================
Gradient descent: v_s[i+1] = v_s[i] - h * L * v_y[i]
Step size h is also optimized (or fixed to 1/L for validation).
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


def solve_pep(N, d, L=1.0, R=1.0, num_restarts=50, verbose=False, optimize_h=True):
    Np1 = N + 1
    dv = Np1 * d
    
    # Pair indices for interpolation
    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]
    pj = jj[mask]
    
    def penalty_obj(theta):
        h_raw = theta[0]
        h = 0.01 + 2.0 / (1.0 + np.exp(-h_raw))  # map to (0.01, 2.01)
        vs = theta[1:1+dv].reshape(Np1, d)
        vy = theta[1+dv:1+2*dv].reshape(Np1, d)
        fv = theta[1+2*dv:]
        
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
        
        # INT (vectorized): fv[i] - fv[j] - L<v_y_j, v_s_i - v_s_j> - (L/2)||v_y_i - v_y_j||^2
        diff_s = vs[pi] - vs[pj]
        diff_y = vy[pi] - vy[pj]
        dot_vy = np.sum(vy[pj] * diff_s, axis=1)
        dv_norm2 = np.sum(diff_y**2, axis=1)
        g_int = fv[pi] - fv[pj] - L*dot_vy - 0.5*L*dv_norm2
        viol_int = np.minimum(g_int, 0.0)
        penalty += np.sum(viol_int**2)
        
        # METHOD: v_s[i+1] = v_s[i] - h*L*v_y[i] for i=0,...,N-1
        # => residual = vs[i+1] - vs[i] + h*L*vy[i]
        for i in range(N):
            residual = vs[i+1] - vs[i] + h*L*vy[i]
            penalty += np.sum(residual**2)
        
        return obj + 5000.0 * penalty
    
    best_val = -np.inf
    best_theta = None
    t0 = time.time()
    
    for k in range(num_restarts):
        alpha = np.random.uniform(0.2, 0.9) * R
        direction = np.random.randn(d)
        direction /= max(np.linalg.norm(direction), 1e-10)
        
        # Initialize on quadratic f(x) = (L/2)||x||^2 with gradient descent
        h_init = 1.0 / L
        vs0 = np.zeros((Np1, d))
        vs0[0] = alpha * direction
        vy0 = np.zeros((Np1, d))
        vy0[0] = vs0[0].copy()  # g_0 = L*x_0, so v_y_0 = x_0
        for i in range(N):
            vs0[i+1] = vs0[i] - h_init * L * vy0[i]
            vy0[i+1] = vs0[i+1].copy()  # v_y = v_s for quadratic
        fv0 = np.maximum(0.5*L*np.sum(vs0**2, axis=1), 1e-8)
        
        h_raw = np.log(h_init - 0.01) - np.log(2.01 - h_init)  # inverse sigmoid
        theta0 = np.concatenate([[h_raw], vs0.ravel(), vy0.ravel(), fv0.ravel()])
        
        theta = theta0.copy()
        for mu in [10, 100, 1000, 5000]:
            res = minimize(penalty_obj, theta, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
            theta = res.x
        
        fv_res = theta[1+2*dv:]
        fN = fv_res[N]
        
        if fN > best_val:
            best_val = fN
            best_theta = theta.copy()
            if verbose:
                h_opt = 0.01 + 2.0 / (1.0 + np.exp(-theta[0]))
                sys.stdout.write(f"  [{k+1}] f_N = {best_val:.8f}, h = {h_opt:.4f}\n")
                sys.stdout.flush()
    
    t1 = time.time()
    if verbose:
        sys.stdout.write(f"  Final: f_N = {best_val:.8f}, time = {t1-t0:.2f}s\n")
        sys.stdout.flush()
    
    return best_val, {'value': best_val, 'time': t1-t0}


if __name__ == "__main__":
    sys.stdout.write("PEP Solver v7 (with method constraint)\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()
    
    # Test cases
    tests = [
        (1, 1, "d=1,N=1 (gradient descent)"),
        (1, 2, "d=2,N=1"),
        (2, 3, "d=3,N=2"),
        (3, 5, "d=5,N=3"),
        (5, 6, "d=6,N=5"),
    ]
    
    for N, d, desc in tests:
        sys.stdout.write(f"\n{desc}:\n")
        sys.stdout.flush()
        val, info = solve_pep(N=N, d=d, num_restarts=25, verbose=True)
        refs = reference_bounds(N)
        sys.stdout.write(f"  Drori: {refs['drori']:.8f}, OGM: {refs['ogm']:.8f}\n")
        sys.stdout.flush()
