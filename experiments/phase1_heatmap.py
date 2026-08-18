"""
Phase 1: d × N Heatmap Sweep
==============================
Core experiment: sweep d ∈ {1,...,N+1} for N ∈ {1,...,6}.
For each (N, d), solve the rank-constrained PEP and record f_N.
Also compute the Drori SDP bound (d → ∞) for comparison.
"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
import numpy as np
import json
import time
from src.formulation import solve_pep, reference_bounds

sys.stdout.write("PHASE 1: d × N HEATMAP SWEEP\n")
sys.stdout.write("=" * 60 + "\n")
sys.stdout.flush()

N_max = 6
L, R = 1.0, 1.0
num_restarts = 20

results = {}
results_meta = {
    'N_max': N_max, 'L': L, 'R': R,
    'num_restarts': num_restarts,
    'method': 'gradient_descent_h=1/L',
    'description': 'Rank-constrained PEP via Burer-Monteiro factorization'
}

t_total = time.time()

for N in range(1, N_max + 1):
    results[N] = {}
    refs = reference_bounds(N)
    sys.stdout.write(f"\nN = {N}  (Drori={refs['drori']:.6f}, OGM={refs['ogm']:.6f})\n")
    sys.stdout.flush()
    
    for d in range(1, N + 2):
        t0 = time.time()
        val, info = solve_pep(N=N, d=d, L=L, R=R, num_restarts=num_restarts)
        dt = time.time() - t0
        
        results[N][d] = {
            'f_N': val,
            'drori': refs['drori'],
            'ogm': refs['ogm'],
            'time': dt
        }
        
        ratio = val / refs['drori'] if refs['drori'] > 0 else 0
        sys.stdout.write(f"  d={d:2d}: f_N = {val:.8f}  (ratio to Drori: {ratio:.4f})  [{dt:.1f}s]\n")
        sys.stdout.flush()

total_time = time.time() - t_total

# Print summary table
sys.stdout.write("\n" + "=" * 80 + "\n")
sys.stdout.write("SUMMARY TABLE\n")
sys.stdout.write("=" * 80 + "\n")

header = f"{'N':>3} | " + " | ".join([f"d={d}" for d in range(1, N_max + 2)]) + " | Drori"
sys.stdout.write(header + "\n")
sys.stdout.write("-" * len(header) + "\n")

for N in range(1, N_max + 1):
    refs = reference_bounds(N)
    row = f"{N:3d} | "
    vals = []
    for d in range(1, N + 2):
        v = results[N][d]['f_N']
        vals.append(v)
        row += f"{v:.6f} | "
    row += f"{refs['drori']:.6f}"
    sys.stdout.write(row + "\n")
sys.stdout.flush()

# Save results
output = {
    'results': results,
    'meta': results_meta,
    'total_time': total_time
}

# Convert keys to strings for JSON
output_json = {
    'results': {str(k): {str(d): v for d, v in v.items()} for k, v in results.items()},
    'meta': results_meta,
    'total_time': total_time
}

with open(r'C:\Users\Windows.11\Desktop\pep\results\phase1_data.json', 'w') as f:
    json.dump(output_json, f, indent=2)

sys.stdout.write(f"\nTotal time: {total_time:.1f}s\n")
sys.stdout.write(f"Results saved to results/phase1_data.json\n")
sys.stdout.flush()
