"""
APAN vs GD vs OGM on worst-case quadratics (fixed fstar, R^2 normalization).
"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\gidooo\gidooo\src')

import numpy as np, json, time, os
from apan.algorithms import run_apan, run_bfgs, run_lbfgs, estimate_L
from apan.problems import Quadratic


def drori_bound(N):
    return 1.0 / (4 * N + 2)

def ogm_theoretical(N):
    th = 1.0
    for t in range(1, N):
        th = (1 + np.sqrt(4*th**2 + 1)) / 2
    theta_N = (1 + np.sqrt(8*th**2 + 1)) / 2
    return 1.0 / (2 * theta_N**2)


def run_gd_pure(prob, x0, N, h=1.0):
    x = x0.copy().astype(float)
    g = prob.grad(x)
    hist_g = [float(np.linalg.norm(g))]
    for _ in range(N):
        x = x - h * g
        g = prob.grad(x)
        hist_g.append(float(np.linalg.norm(g)))
    return float(prob.f(x)), hist_g


def run_ogm_pure(prob, x0, N, L=1.0):
    thetas = [1.0]
    for _ in range(N):
        thetas.append((1 + np.sqrt(1 + 4 * thetas[-1]**2)) / 2)
    x_prev = x0.copy().astype(float)
    g = prob.grad(x_prev)
    hist_g = [float(np.linalg.norm(g))]
    x_curr = x_prev - (1.0 / (L * thetas[1])) * g
    hist_g.append(float(np.linalg.norm(prob.grad(x_curr))))
    for k in range(1, N):
        y = x_curr + ((thetas[k] - 1) / thetas[k+1]) * (x_curr - x_prev)
        g_y = prob.grad(y)
        x_prev = x_curr
        x_curr = y - (1.0 / (L * thetas[k+1])) * g_y
        hist_g.append(float(np.linalg.norm(prob.grad(x_curr))))
    return float(prob.f(x_curr)), hist_g


def run_one(prob, x0, N, kappa, part_label):
    d = prob.n
    R2 = float(np.sum((x0 - prob.xstar)**2))
    L_true = float(kappa)
    h = 1.0 / L_true

    gd_f, gd_g = run_gd_pure(prob, x0, N, h=h)
    ogm_f, ogm_g = run_ogm_pure(prob, x0, N, L=L_true)

    L_est = estimate_L(prob, x0)
    cfg = {'sigma_high': L_est * 0.1, 'relative_thresh': True, 'tol': 1e-12, 'max_iter': 200}
    t0 = time.perf_counter()
    res_a = run_apan(prob, x0, cfg)
    t_a = time.perf_counter() - t0
    res_b = run_bfgs(prob, x0, {'tol': 1e-12, 'max_iter': 200})
    res_l = run_lbfgs(prob, x0, {'tol': 1e-12, 'max_iter': 200, 'mem': 20})

    dr = drori_bound(N)
    og = ogm_theoretical(N)
    apan_g = [h['gnorm'] for h in res_a.history]

    return {
        'part': part_label, 'd': d, 'N': N, 'kappa': kappa, 'R2': R2,
        'drori_ref': dr, 'ogm_ref': og,
        'gd_f': gd_f, 'gd_norm': gd_f / (L_true * R2), 'gd_ratio_drori': (gd_f / (L_true * R2)) / dr,
        'ogm_f': ogm_f, 'ogm_norm': ogm_f / (L_true * R2), 'ogm_ratio_ref': (ogm_f / (L_true * R2)) / og,
        'apan_f': res_a.final['f'], 'apan_iters': res_a.n_iter,
        'apan_rate': res_a.final.get('rate'), 'apan_rho': res_a.final.get('rho_ones', 0),
        'bfgs_f': res_b.final['f'], 'bfgs_iters': res_b.n_iter,
        'lbfgs_f': res_l.final['f'], 'lbfgs_iters': res_l.n_iter,
        'time_apan': t_a,
        'hist_gd': gd_g, 'hist_ogm': ogm_g, 'hist_apan': apan_g,
        'hist_bfgs': [h['gnorm'] for h in res_b.history],
        'hist_lbfgs': [h['gnorm'] for h in res_l.history],
    }


def main():
    print("=" * 78)
    print("  APAN vs GD vs OGM — Worst-Case Quadratics")
    print("=" * 78)

    all_results = []
    all_hist = {}

    # Part A: kappa sweep
    print("\n--- A: kappa sweep (N=5, d=7) ---")
    for kappa in [1, 10, 100, 1000, 10000]:
        prob = Quadratic(n=7, cond=kappa, seed=42)
        x0 = prob.init(dist=1.0, seed=0)
        r = run_one(prob, x0, 5, kappa, 'A')
        all_results.append(r)
        all_hist[f'A_k{kappa}'] = {k: r[k] for k in ['hist_gd','hist_ogm','hist_apan','hist_bfgs','hist_lbfgs']}
        print(f"  kappa={kappa:>5d}: GD={r['gd_norm']:.4f} ({r['gd_ratio_drori']:.1f}x Drori)  "
              f"OGM={r['ogm_norm']:.4f} ({r['ogm_ratio_ref']:.1f}x OGM_ref)  "
              f"APAN={r['apan_f']:.2e}({r['apan_iters']}it)  "
              f"BFGS={r['bfgs_f']:.2e}({r['bfgs_iters']}it)  "
              f"L-BFGS={r['lbfgs_f']:.2e}({r['lbfgs_iters']}it)")

    # Part B: dim sweep
    print("\n--- B: dimension sweep (kappa=100, N=d-2) ---")
    for d in [3, 5, 10, 20, 50]:
        N = d - 2
        prob = Quadratic(n=d, cond=100, seed=42)
        x0 = prob.init(dist=1.0, seed=0)
        r = run_one(prob, x0, N, 100, 'B')
        all_results.append(r)
        all_hist[f'B_d{d}'] = {k: r[k] for k in ['hist_gd','hist_ogm','hist_apan','hist_bfgs','hist_lbfgs']}
        print(f"  d={d:>3d} N={N:>2d}: GD={r['gd_norm']:.4f} ({r['gd_ratio_drori']:.1f}x)  "
              f"OGM={r['ogm_norm']:.4f} ({r['ogm_ratio_ref']:.1f}x)  "
              f"APAN={r['apan_f']:.2e}({r['apan_iters']}it)  "
              f"BFGS={r['bfgs_f']:.2e}({r['bfgs_iters']}it)")

    # Part C: N sweep
    print("\n--- C: N sweep (kappa=100, d=N+2) ---")
    for N in [1, 2, 3, 4, 5, 6]:
        d = N + 2
        prob = Quadratic(n=d, cond=100, seed=42)
        x0 = prob.init(dist=1.0, seed=0)
        r = run_one(prob, x0, N, 100, 'C')
        all_results.append(r)
        all_hist[f'C_N{N}'] = {k: r[k] for k in ['hist_gd','hist_ogm','hist_apan','hist_bfgs','hist_lbfgs']}
        print(f"  N={N} d={d}: GD={r['gd_norm']:.4f} ({r['gd_ratio_drori']:.1f}x Drori)  "
              f"OGM={r['ogm_norm']:.4f} ({r['ogm_ratio_ref']:.1f}x OGM_ref)  "
              f"APAN={r['apan_f']:.2e}({r['apan_iters']}it)")

    # Save
    out_dir = r'C:\Users\Windows.11\Desktop\pep\results'
    with open(os.path.join(out_dir, 'apan_comparison.json'), 'w') as fp:
        json.dump(all_results, fp, indent=2)
    with open(os.path.join(out_dir, 'apan_histories.json'), 'w') as fp:
        json.dump(all_hist, fp, indent=2)
    print(f"\nSaved to {out_dir}/apan_comparison.json")


if __name__ == '__main__':
    main()
