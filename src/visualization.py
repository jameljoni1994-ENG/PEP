"""
Visualization module for PEP experiments.
Generates publication-quality figures with LaTeX labels.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os


# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_heatmap(results_matrix: np.ndarray, N_values: list, d_values: list,
                 title: str = r'$\epsilon^*(N, d)$ Heatmap',
                 filename: str = 'heatmap_epsilon.png',
                 xlabel: str = r'Dimension $d$',
                 ylabel: str = r'Iterations $N$'):
    """
    Plot a heatmap of epsilon*(N,d) values.
    
    results_matrix: (len(N_values), len(d_values)) array of epsilon values
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    im = ax.imshow(results_matrix, aspect='auto', cmap='YlOrRd',
                   origin='lower', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\epsilon^*(N, d)$', fontsize=14)
    
    # Set tick labels
    ax.set_xticks(range(len(d_values)))
    ax.set_xticklabels(d_values)
    ax.set_yticks(range(len(N_values)))
    ax.set_yticklabels(N_values)
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    
    # Add text annotations
    for i in range(len(N_values)):
        for j in range(len(d_values)):
            val = results_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.15 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       fontsize=9, color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_performance_ratio(results_matrix: np.ndarray, N_values: list, d_values: list,
                           reference_values: np.ndarray,
                           filename: str = 'performance_ratio.png'):
    """
    Plot ratio epsilon*(N,d) / epsilon*(N,inf) showing performance loss.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Compute ratio
    ratio_matrix = results_matrix / reference_values[:, np.newaxis]
    ratio_matrix = np.clip(ratio_matrix, 0, 1)
    
    im = ax.imshow(ratio_matrix, aspect='auto', cmap='RdYlGn',
                   origin='lower', vmin=0, vmax=1, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\epsilon^*(N,d) / \epsilon^*(N,\infty)$', fontsize=14)
    
    ax.set_xticks(range(len(d_values)))
    ax.set_xticklabels(d_values)
    ax.set_yticks(range(len(N_values)))
    ax.set_yticklabels(N_values)
    
    ax.set_xlabel(r'Dimension $d$', fontsize=14)
    ax.set_ylabel(r'Iterations $N$', fontsize=14)
    ax.set_title('Performance Ratio: Low-d vs Large-d', fontsize=16)
    
    for i in range(len(N_values)):
        for j in range(len(d_values)):
            val = ratio_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=9, color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_convergence_curves(N_values: list, results_dict: dict,
                            reference_bounds: dict,
                            filename: str = 'convergence_curves.png'):
    """
    Plot epsilon*(d) vs d for each N, compared with reference bounds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(N_values)))
    
    for idx, N in enumerate(N_values):
        d_vals = sorted(results_dict[N].keys())
        eps_vals = [results_dict[N][d] for d in d_vals]
        ax.plot(d_vals, eps_vals, 'o-', color=colors[idx], 
                label=f'$N={N}$', markersize=5)
    
    # Add reference lines
    for N in N_values:
        if N in reference_bounds:
            refs = reference_bounds[N]
            ax.axhline(y=refs['drori'], color='red', linestyle='--', alpha=0.3,
                       label=f'Drori (N={N})' if N == N_values[0] else '')
            ax.axhline(y=refs['ogm'], color='blue', linestyle=':', alpha=0.3,
                       label=f'OGM (N={N})' if N == N_values[0] else '')
    
    ax.set_xlabel(r'Dimension $d$', fontsize=14)
    ax.set_ylabel(r'$\epsilon^*(N, d)$', fontsize=14)
    ax.set_title('Worst-Case Performance vs Dimension', fontsize=16)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_solver_comparison(timing_results: dict, filename: str = 'solver_comparison.png'):
    """
    Plot solver timing comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    solvers = list(timing_results.keys())
    N_values = sorted(set([r['N'] for results in timing_results.values() for r in results]))
    
    # Left: timing
    for solver in solvers:
        times = [r['time'] for r in timing_results[solver]]
        Ns = [r['N'] for r in timing_results[solver]]
        ax1.plot(Ns, times, 'o-', label=solver)
    
    ax1.set_xlabel(r'$N$', fontsize=14)
    ax1.set_ylabel('Time (seconds)', fontsize=14)
    ax1.set_title('Solver Timing', fontsize=16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: objective values
    for solver in solvers:
        vals = [r['value'] for r in timing_results[solver]]
        Ns = [r['N'] for r in timing_results[solver]]
        ax2.plot(Ns, vals, 'o-', label=solver)
    
    ax2.set_xlabel(r'$N$', fontsize=14)
    ax2.set_ylabel(r'$\epsilon^*$', fontsize=14)
    ax2.set_title('Objective Values', fontsize=16)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def plot_duality_gap(N_values: list, lower_bounds: dict, upper_bounds: dict,
                     filename: str = 'duality_gap.png'):
    """
    Plot duality gap analysis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for N in N_values:
        if N in lower_bounds and N in upper_bounds:
            ds = sorted(lower_bounds[N].keys())
            lbs = [lower_bounds[N][d] for d in ds]
            ubs = [upper_bounds[N][d] for d in ds]
            ax1.fill_between(ds, lbs, ubs, alpha=0.3, label=f'$N={N}$')
            ax1.plot(ds, lbs, 'o-', markersize=4)
            ax1.plot(ds, ubs, 's--', markersize=4)
    
    ax1.set_xlabel(r'Dimension $d$', fontsize=14)
    ax1.set_ylabel(r'$\epsilon^*$', fontsize=14)
    ax1.set_title('Lower and Upper Bounds', fontsize=16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gap percentage
    for N in N_values:
        if N in lower_bounds and N in upper_bounds:
            ds = sorted(lower_bounds[N].keys())
            gaps = [(upper_bounds[N][d] - lower_bounds[N][d]) / max(upper_bounds[N][d], 1e-10) * 100
                    for d in ds]
            ax2.plot(ds, gaps, 'o-', label=f'$N={N}$')
    
    ax2.set_xlabel(r'Dimension $d$', fontsize=14)
    ax2.set_ylabel('Gap (%)', fontsize=14)
    ax2.set_title('Duality Gap (%)', fontsize=16)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved: {filename}")


def generate_latex_table(results_matrix: np.ndarray, N_values: list, d_values: list,
                        filename: str = 'results_table.tex'):
    """Generate LaTeX table of results."""
    header = r"""
\begin{table}[htbp]
\centering
\caption{Worst-case performance $\epsilon^*(N,d)$ for rank-constrained PEP.}
\label{tab:results}
\small
\begin{tabular}{c|""" + 'c' * len(d_values) + r"""}
\hline
\textbf{$N$} & """ + ' & '.join([f'$d={d}$' for d in d_values]) + r""" \\
\hline
"""
    rows = []
    for i, N in enumerate(N_values):
        row_vals = []
        for j, d in enumerate(d_values):
            val = results_matrix[i, j]
            if np.isnan(val):
                row_vals.append('---')
            else:
                row_vals.append(f'{val:.6f}')
        rows.append(f'{N} & ' + ' & '.join(row_vals) + r' \\')
    
    footer = r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(os.path.dirname(RESULTS_DIR), 'tables', filename), 'w') as f:
        f.write(header)
        f.write('\n'.join(rows))
        f.write('\n')
        f.write(footer)
    
    print(f"Saved LaTeX table: {filename}")


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print(f"Figures will be saved to: {RESULTS_DIR}")
