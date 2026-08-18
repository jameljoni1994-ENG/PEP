"""
Main experiment runner for all phases.
Executes: sanity checks, solver benchmarking, heatmap generation,
duality gap analysis, and publication figure generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import time
from src.formulation import solve_pep, reference_bounds, build_scipy_constraints, pack_theta, unpack_theta
from src.visualization import (plot_heatmap, plot_performance_ratio, 
                               plot_convergence_curves, plot_solver_comparison,
                               plot_duality_gap, generate_latex_table)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
DATA_DIR = os.path.join(RESULTS_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def phase0_sanity_checks():
    """Phase 0: Validate formulation on known cases."""
    print("="*70)
    print("PHASE 0: SANITY CHECKS")
    print("="*70)
    
    results = {}
    
    # Test 1: d=1, N=1 => f_N = 0 (perfect Newton step)
    print("\n[Test 1] d=1, N=1 => expected f_N = 0")
    val, info = solve_pep(N=1, d=1, num_restarts=30, verbose=True)
    results['d1_N1'] = {'value': val, 'expected': 0.0, 'pass': abs(val) < 1e-3}
    print(f"  PASS: {results['d1_N1']['pass']}")
    
    # Test 2: d=2, N=1 => f_N = 0
    print("\n[Test 2] d=2, N=1 => expected f_N = 0")
    val, info = solve_pep(N=1, d=2, num_restarts=30, verbose=True)
    results['d2_N1'] = {'value': val, 'expected': 0.0, 'pass': abs(val) < 1e-3}
    print(f"  PASS: {results['d2_N1']['pass']}")
    
    # Test 3: d=10, N=3 => should approach Drori bound 1/14 ≈ 0.0714
    print("\n[Test 3] d=10, N=3 => expected ≈ Drori = 1/14 ≈ 0.07143")
    val, info = solve_pep(N=3, d=10, num_restarts=30, verbose=True)
    refs = reference_bounds(3)
    print(f"  Drori: {refs['drori']:.8f}")
    print(f"  OGM:   {refs['ogm']:.8f}")
    print(f"  Got:   {val:.8f}")
    results['d10_N3'] = {'value': val, 'drori': refs['drori'], 
                          'pass': abs(val - refs['drori']) < 0.02}
    print(f"  PASS: {results['d10_N3']['pass']}")
    
    # Test 4: d=20, N=5 => should approach Drori bound 1/22 ≈ 0.04545
    print("\n[Test 4] d=20, N=5 => expected ≈ Drori = 1/22 ≈ 0.04545")
    val, info = solve_pep(N=5, d=20, num_restarts=30, verbose=True)
    refs = reference_bounds(5)
    print(f"  Drori: {refs['drori']:.8f}")
    print(f"  OGM:   {refs['ogm']:.8f}")
    print(f"  Got:   {val:.8f}")
    results['d20_N5'] = {'value': val, 'drori': refs['drori'],
                          'pass': abs(val - refs['drori']) < 0.02}
    print(f"  PASS: {results['d20_N5']['pass']}")
    
    # Save results
    with open(os.path.join(DATA_DIR, 'phase0_sanity.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    all_pass = all(r['pass'] for r in results.values())
    print(f"PHASE 0 OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("="*70)
    
    return results


def phase1_solver_benchmark():
    """Phase 1: Compare solver methods."""
    print("\n" + "="*70)
    print("PHASE 1: SOLVER BENCHMARKING")
    print("="*70)
    
    results = {}
    
    for method in ['trust-constr', 'SLSQP']:
        print(f"\n--- Method: {method} ---")
        results[method] = []
        
        for N in [1, 2, 3, 4]:
            d = min(N + 1, 10)  # Use large d to approximate known bounds
            t_start = time.time()
            val, info = solve_pep(N=N, d=d, num_restarts=20, method=method, verbose=False)
            t_end = time.time()
            
            results[method].append({
                'N': N, 'd': d, 'value': val,
                'time': t_end - t_start,
                'restarts': info['successful_restarts']
            })
            print(f"  N={N}, d={d}: val={val:.6f}, time={t_end-t_start:.2f}s")
    
    with open(os.path.join(DATA_DIR, 'phase1_benchmark.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_solver_comparison(results)
    return results


def phase2_heatmap():
    """Phase 2: Generate full heatmap of epsilon*(N,d)."""
    print("\n" + "="*70)
    print("PHASE 2: HEATMAP GENERATION")
    print("="*70)
    
    N_values = list(range(1, 7))
    max_d_values = {N: N + 1 for N in N_values}
    
    # Collect all d values needed
    all_d = set()
    for N in N_values:
        for d in range(1, max_d_values[N] + 1):
            all_d.add(d)
    all_d = sorted(all_d)
    
    # Results matrix: rows = N, cols = d
    results_matrix = np.full((len(N_values), len(all_d)), np.nan)
    
    for i, N in enumerate(N_values):
        for d in range(1, max_d_values[N] + 1):
            j = all_d.index(d)
            print(f"  Solving N={N}, d={d}...", end=' ', flush=True)
            
            val, info = solve_pep(N=N, d=d, num_restarts=50, verbose=False)
            results_matrix[i, j] = val
            
            ref = reference_bounds(N)
            drori = ref['drori']
            ratio = val / drori if drori > 0 else np.nan
            print(f"eps={val:.6f}, ratio_to_drori={ratio:.3f}")
    
    # Save results
    np.save(os.path.join(DATA_DIR, 'heatmap_matrix.npy'), results_matrix)
    with open(os.path.join(DATA_DIR, 'heatmap_values.json'), 'w') as f:
        json.dump({
            'N_values': N_values,
            'd_values': all_d,
            'matrix': results_matrix.tolist()
        }, f, indent=2)
    
    # Generate figures
    plot_heatmap(results_matrix, N_values, all_d)
    
    # Performance ratio
    ref_values = np.array([reference_bounds(N)['drori'] for N in N_values])
    plot_performance_ratio(results_matrix, N_values, all_d, ref_values)
    
    # Convergence curves
    results_dict = {}
    for i, N in enumerate(N_values):
        results_dict[N] = {}
        for j, d in enumerate(all_d):
            if not np.isnan(results_matrix[i, j]):
                results_dict[N][d] = results_matrix[i, j]
    
    plot_convergence_curves(N_values, results_dict, 
                           {N: reference_bounds(N) for N in N_values})
    
    # Generate LaTeX table
    generate_latex_table(results_matrix, N_values, all_d)
    
    return results_matrix, N_values, all_d


def phase3_comparison():
    """Phase 3: Compare with known bounds."""
    print("\n" + "="*70)
    print("PHASE 3: COMPARISON WITH KNOWN BOUNDS")
    print("="*70)
    
    N_values = list(range(1, 7))
    
    print(f"\n{'N':>3} | {'d':>3} | {'eps*(d)':>10} | {'Drori':>10} | {'OGM':>10} | {'Ratio':>8}")
    print("-" * 70)
    
    for N in N_values:
        refs = reference_bounds(N)
        for d in range(1, N + 2):
            val, _ = solve_pep(N=N, d=d, num_restarts=30, verbose=False)
            ratio = val / refs['drori'] if refs['drori'] > 0 else np.nan
            print(f"{N:3d} | {d:3d} | {val:10.6f} | {refs['drori']:10.6f} | {refs['ogm']:10.6f} | {ratio:8.3f}")


def phase5_duality_gap():
    """Phase 5: Compute duality gap via SDP relaxation."""
    print("\n" + "="*70)
    print("PHASE 5: DUALITY GAP ANALYSIS")
    print("="*70)
    
    # Upper bound: solve SDP relaxation (no rank constraint)
    # This is equivalent to the standard PEP (d >= N+2)
    
    N_values = [1, 2, 3, 4, 5]
    lower_bounds = {}
    upper_bounds = {}
    
    for N in N_values:
        lower_bounds[N] = {}
        upper_bounds[N] = {}
        
        for d in range(1, N + 2):
            # Lower bound: our QCQP solution
            val, _ = solve_pep(N=N, d=d, num_restarts=30, verbose=False)
            lower_bounds[N][d] = val
            
            # Upper bound: solve with d = N+2 (no rank constraint effect)
            if d < N + 2:
                upper_val, _ = solve_pep(N=N, d=N+2, num_restarts=20, verbose=False)
                upper_bounds[N][d] = upper_val
            else:
                upper_bounds[N][d] = val
            
            gap = (upper_bounds[N][d] - lower_bounds[N][d]) / max(upper_bounds[N][d], 1e-10) * 100
            print(f"  N={N}, d={d}: lower={val:.6f}, upper={upper_bounds[N][d]:.6f}, gap={gap:.2f}%")
    
    plot_duality_gap(N_values, lower_bounds, upper_bounds)
    
    return lower_bounds, upper_bounds


def run_all_phases():
    """Run all experimental phases."""
    print("PEP Rank-Constrained Solver - Full Experiment Suite")
    print("="*70)
    
    t_start = time.time()
    
    # Phase 0: Sanity checks
    sanity_results = phase0_sanity_checks()
    
    # Phase 1: Solver benchmarking
    benchmark_results = phase1_solver_benchmark()
    
    # Phase 2: Heatmap generation (main experiment)
    heatmap_results, N_vals, d_vals = phase2_heatmap()
    
    # Phase 3: Comparison with known bounds
    phase3_comparison()
    
    # Phase 5: Duality gap analysis
    duality_results = phase5_duality_gap()
    
    t_end = time.time()
    
    print("\n" + "="*70)
    print(f"ALL PHASES COMPLETED in {t_end - t_start:.1f} seconds")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    run_all_phases()
