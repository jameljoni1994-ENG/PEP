"""
Phase 1: d x N Heatmap Sweep (incremental save)
"""
import sys, json, time, os
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
import numpy as np
from src.formulation import solve_pep, reference_bounds

OUT = r'C:\Users\Windows.11\Desktop\pep\results\phase1_data.json'

sys.stdout.write("PHASE 1: d x N HEATMAP SWEEP (incremental)\n")
sys.stdout.write("=" * 60 + "\n")
sys.stdout.flush()

N_max = 6
L, R = 1.0, 1.0
num_restarts = 15

results = {}
t_total = time.time()

for N in range(1, N_max + 1):
    results[N] = {}
    refs = reference_bounds(N)
    sys.stdout.write(f"\nN={N}  Drori={refs['drori']:.6f}  OGM={refs['ogm']:.6f}\n")
    sys.stdout.flush()
    
    for d in range(1, N + 2):
        t0 = time.time()
        val, info = solve_pep(N=N, d=d, L=L, R=R, num_restarts=num_restarts)
        dt = time.time() - t0
        results[N][d] = {'f_N': val, 'drori': refs['drori'], 'ogm': refs['ogm'], 'time': dt}
        
        ratio = val / refs['drori'] if refs['drori'] > 0 else 0
        sys.stdout.write(f"  d={d}: f_N={val:.8f}  ratio={ratio:.4f}  [{dt:.1f}s]\n")
        sys.stdout.flush()
        
        # Save after each computation
        output = {
            'results': {str(k): {str(d): v for d, v in v.items()} for k, v in results.items()},
            'meta': {'N_max': N_max, 'L': L, 'R': R, 'num_restarts': num_restarts,
                     'method': 'gradient_descent_h=1/L'}
        }
        with open(OUT, 'w') as f:
            json.dump(output, f, indent=2)

total_time = time.time() - t_total

# Summary
sys.stdout.write("\n" + "=" * 70 + "\n")
sys.stdout.write(f"{'N':>3} | " + " | ".join([f"d={d}" for d in range(1, N_max+2)]) + " | Drori\n")
sys.stdout.write("-" * 70 + "\n")
for N in range(1, N_max + 1):
    refs = reference_bounds(N)
    row = f"{N:3d} | "
    for d in range(1, N + 2):
        row += f"{results[N][d]['f_N']:.6f} | "
    row += f"{refs['drori']:.6f}"
    sys.stdout.write(row + "\n")

sys.stdout.write(f"\nTotal: {total_time:.0f}s\n")
sys.stdout.flush()
