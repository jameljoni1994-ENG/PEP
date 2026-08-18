"""Phase 0: Sanity checks with corrected solver"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
from src.formulation import solve_pep, reference_bounds

print("=" * 60)
print("PHASE 0: SANITY CHECKS (v12 — corrected formulation)")
print("=" * 60)

all_pass = True

# Test 1: d=1, N=1 => Drori = 1/6
print("\n[Test 1] d=1, N=1 => expect ~0.1667 (Drori 1/6)")
val, info = solve_pep(N=1, d=1, num_restarts=20, verbose=True)
refs = reference_bounds(1)
pass1 = abs(val - refs['drori']) < 0.005
print(f"  PASS: {pass1} (gap={abs(val-refs['drori']):.6f})")
all_pass = all_pass and pass1

# Test 2: d=2, N=1 => same Drori bound (rank not restrictive yet)
print("\n[Test 2] d=2, N=1 => expect ~0.1667")
val2, info2 = solve_pep(N=1, d=2, num_restarts=20, verbose=True)
pass2 = abs(val2 - refs['drori']) < 0.005
print(f"  PASS: {pass2} (gap={abs(val2-refs['drori']):.6f})")
all_pass = all_pass and pass2

# Test 3: d=5, N=3 => Drori = 1/14
print("\n[Test 3] d=5, N=3 => expect ~0.0714 (Drori 1/14)")
val3, info3 = solve_pep(N=3, d=5, num_restarts=20, verbose=True)
refs3 = reference_bounds(3)
pass3 = abs(val3 - refs3['drori']) < 0.005
print(f"  PASS: {pass3} (gap={abs(val3-refs3['drori']):.6f})")
all_pass = all_pass and pass3

# Test 4: d=10, N=3 => Drori = 1/14 (d >= N+1, rank not restrictive)
print("\n[Test 4] d=10, N=3 => expect ~0.0714 (Drori 1/14)")
val4, info4 = solve_pep(N=3, d=10, num_restarts=20, verbose=True)
pass4 = abs(val4 - refs3['drori']) < 0.005
print(f"  PASS: {pass4} (gap={abs(val4-refs3['drori']):.6f})")
all_pass = all_pass and pass4

# Test 5: d=2, N=1 with d < N+1 should give SAME bound
# (d=2 >= N+1=2, so rank constraint not active)
print("\n[Test 5] d=1, N=1 => expect 0.1667 (rank d=1, N+1=2, rank IS active)")
val5, info5 = solve_pep(N=1, d=1, num_restarts=20, verbose=True)
pass5 = abs(val5 - refs['drori']) < 0.005
print(f"  PASS: {pass5}")
all_pass = all_pass and pass5

print("\n" + "=" * 60)
if all_pass:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗")
print("=" * 60)
