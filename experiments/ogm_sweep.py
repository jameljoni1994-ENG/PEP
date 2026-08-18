"""
OGM heatmap sweep: d x N for OGM method constraint
"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
import json
import time
import os
from src.solver_ogm import solve_pep_ogm, ogm_bound, drori_bound

results = {}
N_max = 5

for N in range(1, N_max + 1):
    max_d = N + 1
    results[str(N)] = {}
    for d in range(1, max_d + 1):
        t0 = time.time()
        val, info = solve_pep_ogm(N=N, d=d, L=1.0, R=1.0, num_restarts=15, verbose=False)
        t1 = time.time()
        results[str(N)][str(d)] = {
            'f_N': val,
            'drori': drori_bound(N),
            'ogm_ref': ogm_bound(N),
            'time': t1 - t0,
        }
        sys.stdout.write(f"N={N}, d={d}: f_N={val:.8f}  (drori={drori_bound(N):.6f}, ogm_ref={ogm_bound(N):.6f})\n")
        sys.stdout.flush()

outpath = r'C:\Users\Windows.11\Desktop\pep\results\ogm_phase1_data.json'
with open(outpath, 'w') as f:
    json.dump({'results': results}, f, indent=2)
sys.stdout.write(f"\nSaved to {outpath}\n")
sys.stdout.flush()
