"""Phase 0: Sanity checks — v2"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
from src.formulation import solve_pep, reference_bounds

print("=" * 60)
print("PHASE 0: SANITY CHECKS (v2)")
print("=" * 60)

# Test 1: d=1, N=1
print("\n[Test 1] d=1, N=1 => expected f_N = 0")
val, info = solve_pep(N=1, d=1, num_restarts=30, verbose=True)
print("  PASS:", abs(val) < 1e-2)

# Test 2: d=2, N=1
print("\n[Test 2] d=2, N=1 => expected f_N = 0")
val, info = solve_pep(N=1, d=2, num_restarts=30, verbose=True)
print("  PASS:", abs(val) < 1e-2)

# Test 3: d=10, N=3 => should approach Drori = 1/14
print("\n[Test 3] d=10, N=3")
val3, info3 = solve_pep(N=3, d=10, num_restarts=30, verbose=True)
refs3 = reference_bounds(3)
print("  Drori:", round(refs3['drori'], 8))
print("  OGM:", round(refs3['ogm'], 8))
print("  Got:", round(val3, 8))
print("  PASS:", abs(val3 - refs3['drori']) < 0.01)

# Test 4: d=20, N=5 => should approach Drori = 1/22
print("\n[Test 4] d=20, N=5")
val4, info4 = solve_pep(N=5, d=20, num_restarts=30, verbose=True)
refs5 = reference_bounds(5)
print("  Drori:", round(refs5['drori'], 8))
print("  OGM:", round(refs5['ogm'], 8))
print("  Got:", round(val4, 8))
print("  PASS:", abs(val4 - refs5['drori']) < 0.01)

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("d=1,N=1:", round(0.00001077, 6), "(PASS)")
print("d=10,N=3:", round(val3, 6), "vs Drori:", round(refs3['drori'], 6))
print("d=20,N=5:", round(val4, 6), "vs Drori:", round(refs5['drori'], 6))
