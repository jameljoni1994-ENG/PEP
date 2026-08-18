"""
PEP Solver v4 — Fully Vectorized Penalty Method
All constraint evaluation is vectorized (no Python loops in hot path).
"""
import numpy as np
from scipy.optimize import minimize
import time


def solve_pep(N, d, L=1.0, R=1.0, num_restarts=50, verbose=False):
    Np1 = N + 1
    dv = Np1 * d
    total = 2 * dv + Np1
    
    # Precompute pair indices as arrays
    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]  # shape: n_pairs
    pj = jj[mask]  # shape: n_pairs
    n_pairs = len(pi)
    
    def penalty_obj(theta):
        vs = theta[:dv].reshape(Np1, d)   # (Np1, d)
        vy = theta[dv:2*dv].reshape(Np1, d)
        fv = theta[2*dv:]                  # (Np1,)
        
        obj = -fv[N]
        
        # IC: R^2 - ||vs[0]||^2
        ic = R**2 - np.sum(vs[0]**2)
        if ic < 0:
            obj += 1000.0 * ic**2
        
        # OPT: fv[i] - (L/2)||vy[i]||^2  (vectorized)
        vy_norm2 = np.sum(vy**2, axis=1)  # (Np1,)
        opt_viol = (L/2) * vy_norm2 - fv  # violation: positive means violated
        mask_opt = opt_viol > 0
        if np.any(mask_opt):
            obj += 1000.0 * np.sum(opt_viol[mask_opt]**2)
        
        # INT: vectorized over all pairs
        # fv[pj] - fv[pi] - L * <vy[pj], vs[pi] - vs[pj]> - (L/2)||vy[pi] - vy[pj]||^2
        dv_vec = vy[pi] - vy[pj]                    # (n_pairs, d)
        dot_vy = np.sum(vy[pj] * (vs[pi] - vs[pj]), axis=1)  # (n_pairs,)
        dv_norm2 = np.sum(dv_vec**2, axis=1)         # (n_pairs,)
        int_vals = fv[pj] - fv[pi] - L*dot_vy - 0.5*L*dv_norm2  # (n_pairs,)
        int_viol = -int_vals  # positive means violated
        mask_int = int_viol > 0
        if np.any(mask_int):
            obj += 1000.0 * np.sum(int_viol[mask_int]**2)
        
        return obj
    
    best_val = -np.inf
    best_theta = None
    t0 = time.time()
    
    for k in range(num_restarts):
        alpha = np.random.uniform(0.15, 0.85) * R
        direction = np.random.randn(d)
        direction /= max(np.linalg.norm(direction), 1e-10)
        
        vs0 = np.zeros((Np1, d))
        vs0[0] = alpha * direction
        for i in range(1, Np1):
            vs0[i] = vs0[0] * (1.0 - 0.2*i/Np1) + np.random.randn(d)*0.01*R
        
        vy0 = vs0.copy() + np.random.randn(Np1, d)*0.005
        fv0 = np.maximum(0.5*L*np.sum(vs0**2, axis=1), 1e-6)
        
        theta0 = np.concatenate([vs0.ravel(), vy0.ravel(), fv0.ravel()])
        
        theta = theta0.copy()
        for mu in [10, 100, 1000]:
            res = minimize(penalty_obj, theta, method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12})
            theta = res.x
        
        fv_res = theta[2*dv:]
        fN = fv_res[N]
        
        if fN > best_val:
            best_val = fN
            best_theta = theta.copy()
    
    t1 = time.time()
    
    if verbose:
        print(f"  f_N = {best_val:.8f}, time = {t1-t0:.2f}s, restarts = {num_restarts}")
    
    return best_val, {'value': best_val, 'time': t1-t0, 'N': N, 'd': d}


def reference_bounds(N):
    drori = 1.0 / (4*N + 2)
    theta = 1.0
    for t in range(1, N):
        theta = (1 + np.sqrt(4*theta**2 + 1)) / 2
    theta_N = (1 + np.sqrt(8*theta**2 + 1)) / 2
    ogm = 1.0 / (2 * theta_N**2)
    return {'drori': drori, 'ogm': ogm}


if __name__ == "__main__":
    print("PEP Solver v4 (Vectorized Penalty)")
    print("=" * 60)
    
    tests = [
        (1, 1, 0.0, "d=1,N=1: Newton step"),
        (1, 2, 0.0, "d=2,N=1: should be 0"),
        (2, 3, None, "d=3,N=2"),
        (3, 5, None, "d=5,N=3"),
        (3, 10, None, "d=10,N=3 (large-d)"),
    ]
    
    for N, d, expected, desc in tests:
        print(f"\n{desc}:")
        val, info = solve_pep(N=N, d=d, num_restarts=30, verbose=True)
        refs = reference_bounds(N)
        print(f"  Drori: {refs['drori']:.8f}, OGM: {refs['ogm']:.8f}")
        if expected is not None:
            print(f"  Expected: {expected}, Got: {val:.8f}, PASS: {abs(val - expected) < 0.01}")
