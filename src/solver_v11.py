"""
PEP Solver v11 — Fixed h=1/L, validated against analytical
===========================================================
Fix h=1/L (gradient descent). Both OPT bounds.
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


def solve_pep(N, d, L=1.0, R=1.0, num_restarts=50, verbose=False, h_step=None):
    """Solve PEP. If h_step is None, optimizes over h. If given, fixes h."""
    Np1 = N + 1
    
    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]
    pj = jj[mask]
    
    def unpack(theta):
        if h_step is None:
            h = 0.05 + 1.95 / (1.0 + np.exp(-theta[0]))
            offset = 1
        else:
            h = h_step
            offset = 0
        vs0 = theta[offset:offset+d]
        vy = theta[offset+d:offset+d+Np1*d].reshape(Np1, d)
        fv = theta[offset+d+Np1*d:]
        vs = np.zeros((Np1, d))
        vs[0] = vs0
        for i in range(N):
            vs[i+1] = vs[i] - h * L * vy[i]
        return h, vs, vy, fv
    
    def make_obj(mu):
        def penalty_obj(theta):
            h, vs, vy, fv = unpack(theta)
            obj = -fv[N]
            pen = 0.0
            g_ic = R**2 - np.sum(vs[0]**2)
            if g_ic < 0:
                pen += g_ic**2
            vy_norm2 = np.sum(vy**2, axis=1)
            lo_viol = np.maximum(0.5*L*vy_norm2 - fv, 0.0)
            pen += np.sum(lo_viol**2)
            dot_vs_vy = np.sum(vy * vs, axis=1)
            up_viol = np.maximum(fv - L*dot_vs_vy + 0.5*L*vy_norm2, 0.0)
            pen += np.sum(up_viol**2)
            diff_s = vs[pi] - vs[pj]
            dot_vy = np.sum(vy[pj] * diff_s, axis=1)
            dv_norm2 = np.sum((vy[pi] - vy[pj])**2, axis=1)
            g_int = fv[pi] - fv[pj] - L*dot_vy - 0.5*L*dv_norm2
            viol_int = np.maximum(-g_int, 0.0)
            pen += np.sum(viol_int**2)
            return obj + mu * pen
        return penalty_obj
    
    best_val = -np.inf
    best_theta = None
    t0 = time.time()
    
    for k in range(num_restarts):
        alpha = np.random.uniform(0.3, 0.95) * R
        direction = np.random.randn(d)
        direction /= max(np.linalg.norm(direction), 1e-10)
        
        h_init = h_step if h_step else np.random.uniform(0.3, 1.5) / L
        vs0_init = alpha * direction
        vy_init = np.zeros((Np1, d))
        vs_init = np.zeros((Np1, d))
        vs_init[0] = vs0_init
        vy_init[0] = vs0_init + np.random.randn(d)*0.3
        for i in range(N):
            vs_init[i+1] = vs_init[i] - h_init * L * vy_init[i]
            vy_init[i+1] = np.random.randn(d)*0.1
        
        fv_init = np.zeros(Np1)
        for i in range(Np1):
            lo = 0.5*L*np.sum(vy_init[i]**2)
            up = L*np.dot(vy_init[i], vs_init[i]) - lo
            if up > lo + 1e-8:
                fv_init[i] = 0.5*(lo + up)
            else:
                fv_init[i] = lo + 1e-4
        
        if h_step is None:
            h_raw = np.log(max(h_init - 0.05, 0.01)) - np.log(max(2.0 - h_init, 0.01))
            theta0 = np.concatenate([[h_raw], vs0_init.ravel(), vy_init.ravel(), fv_init.ravel()])
        else:
            theta0 = np.concatenate([vs0_init.ravel(), vy_init.ravel(), fv_init.ravel()])
        
        theta = theta0.copy()
        for mu in [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]:
            res = minimize(make_obj(mu), theta, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            theta = res.x
        
        if h_step is None:
            fv_res = theta[1+d+Np1*d:]
        else:
            fv_res = theta[d+Np1*d:]
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


if __name__ == "__main__":
    sys.stdout.write("PEP Solver v11 (fixed h=1/L)\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()
    
    # First: analytical verification for d=1, N=1
    sys.stdout.write("\nAnalytical grid search (d=1, N=1, h=1/L):\n")
    best = 0
    for a in np.linspace(-0.5, 2.0, 1000):
        for b in np.linspace(-2.0, 2.0, 1000):
            lo0 = a**2/2; lo1 = b**2/2
            up0 = a - a**2/2; up1 = b*(1-a) - b**2/2
            c01 = 2*a*b + (a-b)**2/2
            c10 = -2*a**2 + (a-b)**2/2
            f1_max = min(up1, up0 - c01)
            f1_min = max(lo1, lo0 + c10)
            if f1_max >= f1_min + 1e-10 and f1_max > best:
                best = f1_max
    sys.stdout.write(f"  Grid max f1 = {best:.8f}\n\n")
    sys.stdout.flush()
    
    # Now: check what the SDP bound actually is for d=1, N=1
    tests = [
        (1, 1, "d=1,N=1"),
        (1, 2, "d=2,N=1"),
        (2, 3, "d=3,N=2"),
        (3, 5, "d=5,N=3"),
        (3, 10, "d=10,N=3"),
    ]
    
    for N, d, desc in tests:
        sys.stdout.write(f"\n{desc}:\n")
        sys.stdout.flush()
        val, info = solve_pep(N=N, d=d, num_restarts=30, verbose=True, h_step=1.0)
        refs = reference_bounds(N)
        sys.stdout.write(f"  Drori: {refs['drori']:.8f}, OGM: {refs['ogm']:.8f}\n")
        sys.stdout.flush()
