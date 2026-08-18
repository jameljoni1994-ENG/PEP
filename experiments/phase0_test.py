"""Phase 0: Sanity checks"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
from src.formulation import solve_pep, reference_bounds

print("=" * 60)
print("PHASE 0: SANITY CHECKS")
print("=" * 60)

# Test 1: d=1, N=1
print("\n[Test 1] d=1, N=1 => expected f_N = 0")
val, info = solve_pep(N=1, d=1, num_restarts=20, verbose=True)
print("  PASS:", abs(val) < 1e-2)

# Test 2: d=10, N=3
print("\n[Test 2] d=10, N=3 => expected ~ Drori = 1/14 = 0.07143")
val, info = solve_pep(N=3, d=10, num_restarts=20, verbose=True)
refs = reference_bounds(3)
print("  Drori:", round(refs['drori'], 8))
print("  OGM:", round(refs['ogm'], 8))
print("  Got:", round(val, 8))
print("  PASS:", abs(val - refs['drori']) < 0.02)
