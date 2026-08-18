"""
PEP Solver v9 — Both OPT bounds + eliminated method constraint
================================================================
Correct formulation with ALL constraints from SDP-PEP:
- Lower OPT: f_i >= (L/2)||v_y_i||^2
- Upper OPT: f_i <= L<v_y_i, v_s_i> - (L/2)||v_y_i||^2
- INT: f_i - f_j - L<v_y_j, v_s_i-v_s_j> - (L/2)||v_y_i-v_y_j||^2 >= 0
- IC: ||v_s_0||^2 <= R^2
- METHOD: v_s[i+1] = v_s[i] - h*L*v_y[i] (eliminated)
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
    
    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]
    pj = jj[mask]
    
    def unpack(theta):
        h = 0.05 + 1.95 / (1.0 + np.exp(-theta[0]))
        vs0 = theta[1:1+d]
        vy = theta[1+d:1+d+Np1*d].reshape(Np1, d)
        fv = theta[1+d+Np1*d:]
        
        vs = np.zeros((Np1, d))
        vs[0] = vs0
        for i in range(N):
            vs[i+1] = vs[i] - h * L * vy[i]
        
        return h, vs, vy, fv
    
    def penalty_obj(theta):
        h, vs, vy, fv = unpack(theta)
        obj = -fv[N]
        pen = 0.0
        
        # IC
        g_ic = R**2 - np.sum(vs[0]**2)
        if g_ic < 0:
            pen += g_ic**2
        
        # Lower OPT: f_i >= (L/2)||v_y_i||^2
        vy_norm2 = np.sum(vy**2, axis=1)
        lo_opt = 0.5*L*vy_norm2 - fv  # violation: positive = violated
        viol = np.maximum(lo_opt, 0.0)
        pen += np.sum(viol**2)
        
        # Upper OPT: f_i <= L<v_y_i, v_s_i> - (L/2)||v_y_i||^2
        dot_vs_vy = np.sum(vy * vs, axis=1)  # (Np1,)
        up_opt = fv - L*dot_vs_vy + 0.5*L*vy_norm2  # violation: positive = violated
        viol_up = np.maximum(up_opt, 0.0)
        pen += np.sum(viol_up**2)
        
        # INT (vectorized)
        diff_s = vs[pi] - vs[pj]
        diff_y = vy[pi] - vy[pj]
        dot_vy = np.sum(vy[pj] * diff_s, axis=1)
        dv_norm2 = np.sum(diff_y**2, axis=1)
        g_int = fv[pi] - fv[pj] - L*dot_vy - 0.5*L*dv_norm2
        viol_int = np.maximum(-g_int, 0.0)
        pen += np.sum(viol_int**2)
        
        return obj + 1e6 * pen
    
    best_val = -np.inf
    best_theta = None
    t0 = time.time()
    
    for k in range(num_restarts):
        alpha = np.random.uniform(0.2, 0.9) * R
        direction = np.random.randn(d)
        direction /= max(np.linalg.norm(direction), 1e-10)
        
        h_init = 1.0 / L
        vs0_init = alpha * direction
        vy_init = np.zeros((Np1, d))
        vs_init = np.zeros((Np1, d))
        vs_init[0] = vs0_init
        vy_init[0] = vs0_init.copy()
        for i in range(N):
            vs_init[i+1] = vs_init[i] - h_init * L * vy_init[i]
            vy_init[i+1] = vs_init[i+1].copy()
        
        # f values within [lower_opt, upper_opt]
        fv_init = np.zeros(Np1)
        for i in range(Np1):
            lo = 0.5*L*np.sum(vy_init[i]**2)
            up = L*np.dot(vy_init[i], vs_init[i]) - lo
            fv_init[i] = 0.5*(lo + up) if up > lo else lo + 1e-6
        
        h_raw = np.log(h_init - 0.05) - np.log(2.0 - h_init)
        theta0 = np.concatenate([[h_raw], vs0_init.ravel(), vy_init.ravel(), fv_init.ravel()])
        
        theta = theta0.copy()
        for mu_exp in [1, 2, 3, 4, 5, 6]:
            res = minimize(penalty_obj, theta, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            theta = res.x
        
        fv_res = theta[1+d+Np1*d:]
        fN = fv_res[N]
        
        if fN > best_val:
            best_val = fN
            best_theta = theta.copy()
            if verbose:
                h_res = 0.05 + 1.95 / (1.0 + np.exp(-theta[0]))
                sys.stdout.write(f"  [{k+1}] f_N = {best_val:.8f}, h = {h_res:.4f}\n")
                sys.stdout.flush()
    
    t1 = time.time()
    if verbose:
        sys.stdout.write(f"  Final: f_N = {best_val:.8f}, time = {t1-t0:.2f}s\n")
        sys.stdout.flush()
    
    return best_val, {'value': best_val, 'time': t1-t0}


if __name__ == "__main__":
    sys.stdout.write("PEP Solver v9 (Both OPT bounds)\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()
    
    tests = [
        (1, 1, "d=1,N=1 (expect 0.1667)"),
        (1, 2, "d=2,N=1 (expect 0.1667)"),
        (2, 3, "d=3,N=2 (expect 0.1000)"),
        (3, 5, "d=5,N=3 (expect 0.0714)"),
        (5, 6, "d=6,N=5 (expect 0.0417)"),
    ]
    
    for N, d, desc in tests:
        sys.stdout.write(f"\n{desc}:\n")
        sys.stdout.flush()
        val, info = solve_pep(N=N, d=d, num_restarts=30, verbose=True)
        refs = reference_bounds(N)
        sys.stdout.write(f"  Drori: {refs['drori']:.8f}, OGM: {refs['ogm']:.8f}, gap: {abs(val-refs['drori']):.8f}\n")
        sys.stdout.flush()
