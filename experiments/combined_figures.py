"""
Combined figures: GD vs OGM PEP bounds, rank sensitivity analysis
"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

figdir = r'C:\Users\Windows.11\Desktop\pep\results\figures'

with open(r'C:\Users\Windows.11\Desktop\pep\results\phase1_data.json') as f:
    gd_data = json.load(f)['results']
with open(r'C:\Users\Windows.11\Desktop\pep\results\ogm_phase1_data.json') as f:
    ogm_data = json.load(f)['results']

N_max = 5
max_d = N_max + 2

# --- Figure 4: GD f_N vs d for each N ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
colors = plt.cm.tab10(np.linspace(0, 0.5, N_max))

for idx, N in enumerate(range(1, N_max + 1)):
    ax = axes[idx // 3, idx % 3]
    ds = list(range(1, N + 2))
    vals = [gd_data[str(N)][str(d)]['f_N'] for d in ds]
    drori = gd_data[str(N)]['1']['drori']
    ogm_ref = gd_data[str(N)]['1']['ogm']

    ax.plot(ds, vals, 'o-', color=colors[idx], linewidth=2, markersize=6, label=f'Rank-d solver')
    ax.axhline(y=drori, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Drori SDP = {drori:.6f}')
    ax.axhline(y=ogm_ref, color='blue', linestyle=':', alpha=0.5, linewidth=1, label=f'OGM ref = {ogm_ref:.6f}')
    ax.set_xlabel('Rank d', fontsize=10)
    ax.set_ylabel('$f_N - f_*$', fontsize=10)
    ax.set_title(f'N={N}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0.5, N + 1.5)

axes[0, 2].legend(fontsize=7, loc='upper right')
plt.suptitle('Rank-Constrained PEP Bound vs Rank d (Gradient Descent, h=1/L)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{figdir}/fig4_gd_fd_vs_d_allN.pdf', dpi=150, bbox_inches='tight')
plt.savefig(f'{figdir}/fig4_gd_fd_vs_d_allN.png', dpi=150, bbox_inches='tight')
plt.close()
sys.stdout.write("Saved fig4_gd_fd_vs_d_allN\n")

# --- Figure 5: Ratio heatmap (GD) ---
fig, ax = plt.subplots(figsize=(8, 4))
ratio_matrix = np.full((N_max, max_d), np.nan)
for i, N in enumerate(range(1, N_max + 1)):
    for j, d in enumerate(range(1, max_d + 1)):
        if str(d) in gd_data[str(N)] and 'f_N' in gd_data[str(N)][str(d)]:
            ratio_matrix[i, j] = gd_data[str(N)][str(d)]['f_N'] / gd_data[str(N)][str(d)]['drori']

cmap = LinearSegmentedColormap.from_list('r', ['#2166ac', '#f7f7f7', '#b2182b'])
im = ax.imshow(ratio_matrix, cmap=cmap, aspect='auto', origin='lower', vmin=0.999, vmax=1.001)
ax.set_xticks(range(max_d))
ax.set_xticklabels([str(d) for d in range(1, max_d + 1)])
ax.set_yticks(range(N_max))
ax.set_yticklabels([f'N={N}' for N in range(1, N_max + 1)])
ax.set_xlabel('Rank parameter d', fontsize=11)
ax.set_ylabel('Iterations N', fontsize=11)
ax.set_title('Ratio $f_N^{\\mathrm{rank\\text{-}}d} / f_N^{\\mathrm{SDP}}$ (Gradient Descent)', fontsize=12)

for i in range(N_max):
    for j in range(max_d):
        if not np.isnan(ratio_matrix[i, j]):
            ax.text(j, i, f'{ratio_matrix[i,j]:.5f}', ha='center', va='center', fontsize=8,
                    color='black' if 0.9999 < ratio_matrix[i,j] < 1.0001 else 'red')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Ratio', fontsize=10)
plt.tight_layout()
plt.savefig(f'{figdir}/fig5_ratio_heatmap.pdf', dpi=150)
plt.savefig(f'{figdir}/fig5_ratio_heatmap.png', dpi=150)
plt.close()
sys.stdout.write("Saved fig5_ratio_heatmap\n")

# --- Figure 6: Convergence curves (GD bound vs Drori vs OGM ref) ---
fig, ax = plt.subplots(figsize=(8, 5))
Ns = list(range(1, N_max + 1))
gd_bounds = [gd_data[str(N)]['1']['f_N'] for N in Ns]
drori_bounds = [gd_data[str(N)]['1']['drori'] for N in Ns]
ogm_refs = [gd_data[str(N)]['1']['ogm'] for N in Ns]

ax.plot(Ns, gd_bounds, 'ko-', linewidth=2, markersize=7, label='Rank-$d$ solver ($d=1$)')
ax.plot(Ns, drori_bounds, 'r--s', linewidth=2, markersize=6, label='Drori SDP bound')
ax.plot(Ns, ogm_refs, 'b:^', linewidth=2, markersize=6, label='OGM reference')
ax.set_xlabel('Number of iterations N', fontsize=12)
ax.set_ylabel('$f_N - f_*$', fontsize=12)
ax.set_title('Worst-Case Bound vs Iteration Count', fontsize=13)
ax.legend(fontsize=11)
ax.set_xticks(Ns)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{figdir}/fig6_convergence.pdf', dpi=150)
plt.savefig(f'{figdir}/fig6_convergence.png', dpi=150)
plt.close()
sys.stdout.write("Saved fig6_convergence\n")

# --- LaTeX Table (combined) ---
sys.stdout.write("\n--- Combined LaTeX Table ---\n")
sys.stdout.write("\\begin{tabular}{c|cccccc|cc}\n")
sys.stdout.write("\\toprule\n")
sys.stdout.write("N & d=1 & d=2 & d=3 & d=4 & d=5 & d=6 & Drori & OGM$_{ref}$ \\\\\n")
sys.stdout.write("\\midrule\n")
for N in range(1, N_max + 1):
    row = f"{N}"
    for d in range(1, N_max + 1):
        if str(d) in gd_data[str(N)] and 'f_N' in gd_data[str(N)][str(d)]:
            row += f" & {gd_data[str(N)][str(d)]['f_N']:.6f}"
        else:
            row += " & ---"
    row += f" & {gd_data[str(N)]['1']['drori']:.6f} & {gd_data[str(N)]['1']['ogm']:.6f} \\\\"
    sys.stdout.write(row + "\n")
sys.stdout.write("\\bottomrule\n")
sys.stdout.write("\\end{tabular}\n")
sys.stdout.flush()
