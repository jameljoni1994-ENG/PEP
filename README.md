# Rank-Constrained PEP via Burer--Monteiro Factorization

Rank-constrained Performance Estimation Problem (PEP) solver using Burer--Monteiro factorization for computing dimension-dependent worst-case bounds of first-order optimization methods.

## Overview

The standard SDP-PEP relaxes the rank constraint on the Gram matrix `G`, potentially introducing conservatism. This project reformulates the rank-constrained PEP as a nonconvex QCQP via the Burer--Monteiro factorization `G = VV^T` with `d` columns, yielding a low-dimensional problem solvable by off-the-shelf local optimizers.

### Key Results

- **Gradient Descent**: SDP relaxation is tight -- the rank constraint is inactive across all tested dimensions (`1 <= N <= 6`, `1 <= d <= N+1`). The BM solver recovers Drori--Teboulle bounds to machine precision (`< 1e-5`).
- **OGM**: PEP bounds consistent with SDP relaxation; auxiliary-point tracking introduces additional complexity for methods with non-standard gradient evaluations.
- **APAN (Second-Order)**: Breaks all first-order PEP bounds in 2 iterations via Hessian information, confirming the fundamental distinction between first- and second-order methods.

## Project Structure

```
pep/
  src/
    formulation.py          # GD solver (v12) -- core BM-PEP formulation
    solver_ogm.py           # OGM PEP solver (SDP-PEP framework)
    solver_ogm_v2.py        # OGM with auxiliary points + smoothness
    solver_ogm_v3.py        # OGM v3 (confirmed = v1)
    visualization.py        # Plotting utilities
  experiments/
    phase1_sweep.py         # GD sweep: N=1..6, d=1..N+1
    ogm_sweep.py            # OGM sweep
    compare_apan.py         # APAN vs GD vs OGM comparison
    compare_figures.py      # APAN comparison figures (fig7--fig10)
    phase1_figures.py       # GD figures (fig1--fig3)
    combined_figures.py     # GD+OGM combined figures (fig4--fig6)
    run_all.py              # Run all experiments
  results/
    phase1_data.json        # GD sweep results
    ogm_phase1_data.json    # OGM sweep results
    apan_comparison.json    # APAN comparison results
    apan_histories.json     # APAN convergence histories
    figures/                # All figures (PDF + PNG)
  paper/
    main.tex                # English paper (pdflatex)
    main.pdf
    main_ar.tex             # Arabic paper (xelatex + polyglossia)
    main_ar.pdf
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run GD sweep
```bash
python experiments/phase1_sweep.py
```

### Run OGM sweep
```bash
python experiments/ogm_sweep.py
```

### Run APAN comparison
Requires the APAN library (from the [gidooo](https://github.com/jameljoni1994-ENG/gidooo) project):
```bash
python experiments/compare_apan.py
```

### Generate figures
```bash
python experiments/phase1_figures.py      # fig1--fig3
python experiments/combined_figures.py     # fig4--fig6
python experiments/compare_figures.py      # fig7--fig10
```

### Compile papers
```bash
cd paper
pdflatex main.tex         # English
xelatex main_ar.tex       # Arabic
```

## Dependencies

- **numpy** >= 1.24
- **scipy** >= 1.10
- **matplotlib** >= 3.7
- **MiKTeX** or **TeX Live** (for paper compilation)
- **APAN library** (optional, for second-order method comparison)

## References

1. O. Drori and M. Teboulle, *Performing oracle experiments for first-order methods*, Mathematical Programming, 2014.
2. A. Taylor, J. Hendrickx, and F. Glineur, *Smooth strongly convex functions and their impact on lower bounds for first-order methods*, Optimization Methods and Software, 2017.
3. A. Taylor, J. Hendrickx, and F. Glineur, *Constructing exact nonlinear worst-case performance estimates*, SIAM Journal on Optimization, 2019.
4. D. Kim and J. Fessler, *Optimizing the memory efficiency of the gradient method*, NeurIPS, 2016.
5. S. Burer and R. Monteiro, *A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization*, Mathematical Programming, 2003.

## Authors

- Ghidaa Najib Khudur
- Jamil Ibrahim Jouni
