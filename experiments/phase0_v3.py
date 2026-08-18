"""Phase 0: Sanity checks — v3 (Fast Penalty-Based)"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
from src.formulation import solve_pep, reference_bounds

print("=" * 60)
print("PHASE 0: SANITY CHECKS (v3 - Penalty Method)")
print("=" * 60)

# Test 1: d=1, N=1
print("\n[Test 1] d=1, N=1 => expected f_N = 0")
val, info = solve_pep(N=1, d=1, num_restarts=30, verbose=True)
print("  PASS:", abs(val) < 1e-2)

# Test 2: d=2, N=1
print("\n[Test 2] d=2, N=1 => expected f_N = 0")
val2, info2 = solve_pep(N=1, d=2, num_restarts=30, verbose=True)
print("  PASS:", abs(val2) < 1e-2)

# Test 3: d=5, N=2
print("\n[Test 3] d=5, N=2")
val3, info3 = solve_pep(N=2, d=5, num_restarts=30, verbose=True)
refs2 = reference_bounds(2)
print("  Drori:", round(refs2['drori'], 6))
print("  OGM:", round(refs2['ogm'], 6))

# Test 4: d=10, N=3
print("\n[Test 4] d=10, N=3")
val4, info4 = solve_pep(N=3, d=10, num_restarts=30, verbose=True)
refs3 = reference_bounds(3)
print("  Drori:", round(refs3['drori'], 6))
print("  OGM:", round(refs3['ogm'], 6))

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
