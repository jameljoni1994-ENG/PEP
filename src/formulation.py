"""
PEP Solver — Final Formulation (v12)
=====================================
Burer-Monteiro factorization of rank-constrained SDP-PEP.
Fixed step size h=1/L (gradient descent).

Constraints:
  IC:   ||v_s_0||^2 <= R^2
  OPT_L: f_i >= (L/2)||v_y_i||^2           (lower, from j=*)
  OPT_U: f_i <= L<v_y_i, v_s_i> - (L/2)||v_y_i||^2  (upper, from i=*)
  INT:  f_i - f_j - L<v_y_j, v_s_i-v_s_j> - (L/2)||v_y_i-v_y_j||^2 >= 0
  METH: v_s[i+1] = v_s[i] - h*L*v_y[i]    (eliminated)
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
    """
    Solve the rank-constrained PEP via Burer-Monteiro factorization.
    
    Parameters:
        N: number of iterations (N+1 total iterates including x_0)
        d: rank parameter (dimension of v_s, v_y)
        L: smoothness constant
        R: initial radius ||x_0 - x_*|| <= R
        num_restarts: number of random restarts
        verbose: print progress
        h_step: fixed step size (default: 1/L)
    
    Returns:
        (f_N, info_dict)
    """
    if h_step is None:
        h_step = 1.0 / L
    
    Np1 = N + 1
    
    # Precompute pair indices for interpolation
    ii, jj = np.meshgrid(range(Np1), range(Np1), indexing='ij')
    mask = ~np.eye(Np1, dtype=bool)
    pi = ii[mask]
    pj = jj[mask]
    
    def unpack(theta):
        vs0 = theta[:d]
        vy = theta[d:d+Np1*d].reshape(Np1, d)
        fv = theta[d+Np1*d:]
        vs = np.zeros((Np1, d))
        vs[0] = vs0
        for i in range(N):
            vs[i+1] = vs[i] - h_step * L * vy[i]
        return vs, vy, fv
    
    def make_obj(mu):
        def penalty_obj(theta):
            vs, vy, fv = unpack(theta)
            obj = -fv[N]
            pen = 0.0
            
            # IC
            g_ic = R**2 - np.sum(vs[0]**2)
            if g_ic < 0:
                pen += g_ic**2
            
            # Lower OPT
            vy_norm2 = np.sum(vy**2, axis=1)
            lo_viol = np.maximum(0.5*L*vy_norm2 - fv, 0.0)
            pen += np.sum(lo_viol**2)
            
            # Upper OPT
            dot_vs_vy = np.sum(vy * vs, axis=1)
            up_viol = np.maximum(fv - L*dot_vs_vy + 0.5*L*vy_norm2, 0.0)
            pen += np.sum(up_viol**2)
            
            # INT (vectorized)
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
        
        vs0_init = alpha * direction
        vy_init = np.zeros((Np1, d))
        vs_init = np.zeros((Np1, d))
        vs_init[0] = vs0_init
        vy_init[0] = vs0_init + np.random.randn(d)*0.3
        for i in range(N):
            vs_init[i+1] = vs_init[i] - h_step * L * vy_init[i]
            vy_init[i+1] = np.random.randn(d)*0.1
        
        fv_init = np.zeros(Np1)
        for i in range(Np1):
            lo = 0.5*L*np.sum(vy_init[i]**2)
            up = L*np.dot(vy_init[i], vs_init[i]) - lo
            if up > lo + 1e-8:
                fv_init[i] = 0.5*(lo + up)
            else:
                fv_init[i] = lo + 1e-4
        
        theta0 = np.concatenate([vs0_init.ravel(), vy_init.ravel(), fv_init.ravel()])
        
        theta = theta0.copy()
        for mu in [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]:
            res = minimize(make_obj(mu), theta, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            theta = res.x
        
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
    
    return best_val, {'value': best_val, 'time': t1-t0, 'N': N, 'd': d, 'L': L, 'R': R}
