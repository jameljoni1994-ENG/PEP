"""
Phase 1b: Generate heatmap + performance ratio figures
"""
import sys
sys.path.insert(0, r'C:\Users\Windows.11\Desktop\pep')
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

with open(r'C:\Users\Windows.11\Desktop\pep\results\phase1_data.json', 'r') as f:
    data = json.load(f)

N_max = 6
max_d = N_max + 2  # d goes up to N+2 for the largest N
ds = list(range(1, max_d + 1))
Ns = list(range(1, N_max + 1))

# Build matrices: rows = N, cols = d
heatmap = np.full((N_max, max_d), np.nan)
ratio_matrix = np.full((N_max, max_d), np.nan)
drori_row = np.zeros(N_max)

for i, N in enumerate(Ns):
    refs_str = data['results'][str(N)]
    drori_val = data['results'][str(N)]['1']['drori']
    drori_row[i] = drori_val
    for j, d in enumerate(ds):
        if str(d) in refs_str:
            heatmap[i, j] = refs_str[str(d)]['f_N']
            ratio_matrix[i, j] = refs_str[str(d)]['f_N'] / drori_val

# Figure 1: Absolute f_N heatmap
fig, ax = plt.subplots(figsize=(10, 5))
cmap = LinearSegmentedColormap.from_list('pep', ['#f0f0f0', '#2166ac', '#053061'])
im = ax.imshow(heatmap, cmap=cmap, aspect='auto', origin='lower')
ax.set_xticks(range(max_d))
ax.set_xticklabels([str(d) for d in ds])
ax.set_yticks(range(N_max))
ax.set_yticklabels([f'N={N}' for N in Ns])
ax.set_xlabel('Rank parameter d', fontsize=12)
ax.set_ylabel('Iterations N', fontsize=12)
ax.set_title('Rank-Constrained PEP Bound $f_N - f_*$\n(Gradient Descent, $h=1/L$)', fontsize=13)

for i in range(N_max):
    for j in range(max_d):
        if not np.isnan(heatmap[i, j]):
            color = 'white' if heatmap[i, j] > 0.06 else 'black'
            ax.text(j, i, f'{heatmap[i,j]:.4f}', ha='center', va='center', fontsize=8, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('$f_N - f_*$', fontsize=11)
plt.tight_layout()
plt.savefig(r'C:\Users\Windows.11\Desktop\pep\results\figures\fig1_heatmap_absolute.pdf', dpi=150)
plt.savefig(r'C:\Users\Windows.11\Desktop\pep\results\figures\fig1_heatmap_absolute.png', dpi=150)
plt.close()
sys.stdout.write("Saved fig1_heatmap_absolute.pdf/png\n")
sys.stdout.flush()

# Figure 2: Ratio to Drori heatmap
fig, ax = plt.subplots(figsize=(10, 5))
cmap2 = LinearSegmentedColormap.from_list('ratio', ['#053061', '#67a9cf', '#f7f7f7'])
im2 = ax.imshow(ratio_matrix, cmap=cmap2, aspect='auto', origin='lower', vmin=0.98, vmax=1.02)
ax.set_xticks(range(max_d))
ax.set_xticklabels([str(d) for d in ds])
ax.set_yticks(range(N_max))
ax.set_yticklabels([f'N={N}' for N in Ns])
ax.set_xlabel('Rank parameter d', fontsize=12)
ax.set_ylabel('Iterations N', fontsize=12)
ax.set_title('Ratio: Rank-Constrained Bound / Drori SDP Bound\n(Gradient Descent, $h=1/L$)', fontsize=13)

for i in range(N_max):
    for j in range(max_d):
        if not np.isnan(ratio_matrix[i, j]):
            ax.text(j, i, f'{ratio_matrix[i,j]:.4f}', ha='center', va='center', fontsize=8)

cbar2 = plt.colorbar(im2, ax=ax, shrink=0.8)
cbar2.set_label('Ratio', fontsize=11)
plt.tight_layout()
plt.savefig(r'C:\Users\Windows.11\Desktop\pep\results\figures\fig2_ratio_to_drori.pdf', dpi=150)
plt.savefig(r'C:\Users\Windows.11\Desktop\pep\results\figures\fig2_ratio_to_drori.png', dpi=150)
plt.close()
sys.stdout.write("Saved fig2_ratio_to_drori.pdf/png\n")
sys.stdout.flush()

# Figure 3: Convergence curve — f_N vs d for each N
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, N_max))

for i, N in enumerate(Ns):
    ds_for_N = list(range(1, N + 2))
    vals = [data['results'][str(N)][str(d)]['f_N'] for d in ds_for_N]
    drori = data['results'][str(N)]['1']['drori']
    ax.plot(ds_for_N, vals, 'o-', color=colors[i], label=f'N={N}', markersize=5)
    ax.axhline(y=drori, color=colors[i], linestyle='--', alpha=0.4, linewidth=1)

ax.set_xlabel('Rank parameter d', fontsize=12)
ax.set_ylabel('$f_N - f_*$', fontsize=12)
ax.set_title('Rank-Constrained PEP Bound vs d\n(Gradient Descent, $h=1/L$, $R=1$, $L=1$)', fontsize=13)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0.5, N_max + 1.5)
plt.tight_layout()
plt.savefig(r'C:\Users\Windows.11\Desktop\pep\results\figures\fig3_fd_vs_d.pdf', dpi=150)
plt.savefig(r'C:\Users\Windows.11\Desktop\pep\results\figures\fig3_fd_vs_d.png', dpi=150)
plt.close()
sys.stdout.write("Saved fig3_fd_vs_d.pdf/png\n")
sys.stdout.flush()

# Print LaTeX table
sys.stdout.write("\nLaTeX Table:\n")
sys.stdout.write("\\begin{tabular}{c|" + "c"*N_max + "|c}\n")
sys.stdout.write(f"{'N':>3}" + "".join([f" & d={d}" for d in range(1, N_max+1)]) + " & Drori \\\\\n")
sys.stdout.write("\\hline\n")
for N in Ns:
    row = f"{N}"
    for d in range(1, N_max + 1):
        if str(d) in data['results'][str(N)]:
            row += f" & {data['results'][str(N)][str(d)]['f_N']:.6f}"
        else:
            row += " & ---"
    row += f" & {data['results'][str(N)]['1']['drori']:.6f} \\\\"
    sys.stdout.write(row + "\n")
sys.stdout.write("\\end{tabular}\n")
sys.stdout.flush()
