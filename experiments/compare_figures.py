"""
Generate comparison figures: APAN vs GD vs OGM convergence + gap bars.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

FIG_DIR = r'C:\Users\Windows.11\Desktop\pep\results\figures'
os.makedirs(FIG_DIR, exist_ok=True)

with open(r'C:\Users\Windows.11\Desktop\pep\results\apan_comparison.json') as f:
    results = json.load(f)

with open(r'C:\Users\Windows.11\Desktop\pep\results\apan_histories.json') as f:
    histories = json.load(f)

# ── Figure A1: Convergence curves (gradient norm) for kappa sweep ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
kappas = [1, 100, 10000]
for idx, kappa in enumerate(kappas):
    ax = axes[idx]
    key = f'A_k{kappa}'
    h = histories[key]
    methods = [
        ('GD', h['hist_gd'], '#d62728', '-', 'o', 4),
        ('OGM', h['hist_ogm'], '#1f77b4', '-', 's', 4),
        ('APAN', h['hist_apan'], '#2ca02c', '-', 'D', 6),
        ('BFGS', h['hist_bfgs'], '#9467bd', '--', '^', 4),
    ]
    for name, gn, color, ls, mk, ms in methods:
        if len(gn) > 0:
            iters = list(range(len(gn)))
            gn_arr = np.array(gn)
            mask = gn_arr > 1e-16
            if mask.any():
                ax.plot(np.array(iters)[mask], gn_arr[mask], color=color, ls=ls,
                        marker=mk, markersize=ms, linewidth=1.5, label=name)
    ax.set_xlabel('Iteration $k$', fontsize=10)
    ax.set_ylabel(r'$\|g_k\|$', fontsize=10)
    ax.set_title(f'$\\kappa={kappa}$', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, max(len(h['hist_gd']), len(h['hist_ogm']), len(h['hist_apan']), len(h['hist_bfgs'])) + 0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig7_apan_convergence_kappa.pdf'), dpi=150)
plt.savefig(os.path.join(FIG_DIR, 'fig7_apan_convergence_kappa.png'), dpi=150)
plt.close()
print("Saved fig7_apan_convergence_kappa")

# ── Figure A2: Convergence curves for dimension sweep ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
dims = [5, 20, 50]
for idx, d in enumerate(dims):
    ax = axes[idx]
    key = f'B_d{d}'
    h = histories[key]
    methods = [
        ('GD', h['hist_gd'], '#d62728', '-', 'o', 4),
        ('OGM', h['hist_ogm'], '#1f77b4', '-', 's', 4),
        ('APAN', h['hist_apan'], '#2ca02c', '-', 'D', 6),
        ('BFGS', h['hist_bfgs'], '#9467bd', '--', '^', 4),
    ]
    for name, gn, color, ls, mk, ms in methods:
        if len(gn) > 0:
            iters = list(range(len(gn)))
            gn_arr = np.array(gn)
            mask = gn_arr > 1e-16
            if mask.any():
                ax.plot(np.array(iters)[mask], gn_arr[mask], color=color, ls=ls,
                        marker=mk, markersize=ms, linewidth=1.5, label=name)
    N_val = d - 2
    ax.set_xlabel('Iteration $k$', fontsize=10)
    ax.set_ylabel(r'$\|g_k\|$', fontsize=10)
    ax.set_title(f'$d={d},\\ N={N_val}$', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig8_apan_convergence_dim.pdf'), dpi=150)
plt.savefig(os.path.join(FIG_DIR, 'fig8_apan_convergence_dim.png'), dpi=150)
plt.close()
print("Saved fig8_apan_convergence_dim")

# ── Figure A3: Normalized gap bar chart (kappa sweep) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
part_a = [r for r in results if r['part'] == 'A']
kappas_a = [r['kappa'] for r in part_a]
gd_norms = [r['gd_norm'] for r in part_a]
ogm_norms = [r['ogm_norm'] for r in part_a]
drori_refs = [r['drori_ref'] for r in part_a]
ogm_refs = [r['ogm_ref'] for r in part_a]

x = np.arange(len(kappas_a))
w = 0.22
ax1.bar(x - w, gd_norms, w, label='GD', color='#d62728', alpha=0.8)
ax1.bar(x, ogm_norms, w, label='OGM', color='#1f77b4', alpha=0.8)
ax1.axhline(y=drori_refs[0], color='gray', ls='--', label=f'Drori bound ($N=5$)', alpha=0.7)
ax1.axhline(y=ogm_refs[0], color='gray', ls=':', label=f'OGM ref bound', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{k}' for k in kappas_a])
ax1.set_xlabel('Condition number $\\kappa$', fontsize=10)
ax1.set_ylabel(r'Normalized gap $(f_N - f_*) / (L \cdot R^2)$', fontsize=10)
ax1.set_title('(a) Gap vs. conditioning ($d=7, N=5$)', fontsize=11)
ax1.set_yscale('log')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# ── Part B: dimension sweep bars ──
part_b = [r for r in results if r['part'] == 'B']
dims_b = [r['d'] for r in part_b]
gd_norms_b = [r['gd_norm'] for r in part_b]
ogm_norms_b = [r['ogm_norm'] for r in part_b]
drori_b = [r['drori_ref'] for r in part_b]
ogm_b = [r['ogm_ref'] for r in part_b]

x2 = np.arange(len(dims_b))
ax2.bar(x2 - w, gd_norms_b, w, label='GD', color='#d62728', alpha=0.8)
ax2.bar(x2, ogm_norms_b, w, label='OGM', color='#1f77b4', alpha=0.8)
for i, (d, dr, og) in enumerate(zip(dims_b, drori_b, ogm_b)):
    ax2.plot([i-0.3, i+0.1], [dr, dr], 'k--', alpha=0.5, linewidth=0.8)
    ax2.plot([i-0.1, i+0.2], [og, og], 'k:', alpha=0.5, linewidth=0.8)
ax2.set_xticks(x2)
ax2.set_xticklabels([f'{d}' for d in dims_b])
ax2.set_xlabel('Dimension $d$  ($N=d-2, \\kappa=100$)', fontsize=10)
ax2.set_ylabel(r'Normalized gap $(f_N - f_*) / (L \cdot R^2)$', fontsize=10)
ax2.set_title('(b) Gap vs. dimension', fontsize=11)
ax2.set_yscale('log')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig9_apan_gap_comparison.pdf'), dpi=150)
plt.savefig(os.path.join(FIG_DIR, 'fig9_apan_gap_comparison.png'), dpi=150)
plt.close()
print("Saved fig9_apan_gap_comparison")

# ── Figure A4: Iteration count comparison (APAN vs BFGS vs L-BFGS) ──
fig, ax = plt.subplots(figsize=(7, 4))
part_c = sorted([r for r in results if r['part'] == 'C'], key=lambda r: r['N'])
Ns = [r['N'] for r in part_c]
apan_iters = [r['apan_iters'] for r in part_c]
bfgs_iters = [r['bfgs_iters'] for r in part_c]

ax.bar(np.array(Ns) - 0.15, apan_iters, 0.3, label='APAN', color='#2ca02c', alpha=0.8)
ax.bar(np.array(Ns) + 0.15, bfgs_iters, 0.3, label='BFGS', color='#9467bd', alpha=0.8)
ax.set_xlabel('Iterations $N$  ($d=N+2, \\kappa=100$)', fontsize=11)
ax.set_ylabel('Iterations to convergence', fontsize=11)
ax.set_title('Second-order method iteration count', fontsize=12)
ax.set_xticks(Ns)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig10_apan_iteration_count.pdf'), dpi=150)
plt.savefig(os.path.join(FIG_DIR, 'fig10_apan_iteration_count.png'), dpi=150)
plt.close()
print("Saved fig10_apan_iteration_count")

print("\nAll figures saved.")
