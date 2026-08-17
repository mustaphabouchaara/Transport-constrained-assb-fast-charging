# Transport-constrained fast charging of all-solid-state batteries

Reproducibility package for the manuscript **“Transport-Constrained
Minimum-Time Fast Charging of All-Solid-State Batteries: Mathematical
Analysis and Structure-Exploiting Computation.”**

The package contains the Python code, parameter files, notebooks, generated
PDF figures, and CSV tables needed to reproduce the manuscript's numerical
results. Figure and table numbers below match the manuscript.

## Requirements

- Python 3.13 (reference environment: Python 3.13.14 on Windows)
- Packages pinned in `requirements.txt`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux and macOS users can activate the environment with
`source .venv/bin/activate`.

## Reproduce the manuscript

```powershell
python paper_figures/generate_figure_03.py
python paper_figures/generate_figure_04.py
python paper_figures/generate_figure_05.py
python paper_figures/generate_figure_06.py
python paper_figures/generate_figure_07.py
python paper_tables/generate_tables.py all
```

Table 3 includes direct particle-swarm optimization and nonlinear
verification; on the reference workstation it takes several minutes.

## Manuscript-to-code map

| Result | Generator | Output |
|---|---|---|
| Figure 3: temporal B-spline basis | `paper_figures/generate_figure_03.py` | `paper_figures/figure_03_temporal_bspline_basis.pdf` |
| Figure 4: local temporal refinement | `paper_figures/generate_figure_04.py` | `paper_figures/figure_04_temporal_refinement.pdf` |
| Figure 5: constant-D validation | `paper_figures/generate_figure_05.py` | `paper_figures/figure_05_linear_validation.pdf` |
| Figure 6: DNN robustness | `paper_figures/generate_figure_06.py` | `paper_figures/figure_06_dnn_diffusivity_robustness.pdf` |
| Figure 7: nonlinear correction | `paper_figures/generate_figure_07.py` | `paper_figures/figure_07_nonlinear_maximal_feasible.pdf` |
| Table 2: convergence | `paper_tables/generate_tables.py 2` | `paper_tables/table_02_convergence.csv` |
| Table 3: policy comparison | `paper_tables/generate_tables.py 3` | `paper_tables/table_03_policy_comparison.csv` |

Figures 1 and 2 are conceptual manuscript schematics and are not numerical
outputs of this package.

## Repository structure

- `iga_core/`: B-spline, quadrature, assembly, and electrochemistry utilities.
- `paper_figures/`: figure generators, settings, and reference PDFs.
- `paper_tables/`: table generator and reference CSV outputs.
- `notebooks/`: optional interactive entry points.

All command-line generators are headless. Relative output paths are resolved
from the repository root.

## Parameter regimes

The manuscript intentionally uses distinct numerical studies:

- Figures 4–5 and Table 2: constant `D = 1.76e-15 m^2/s`, `H_max = 0.1`.
- Figures 6–7: `D_ref = D_DNN(c=1) = 7e-15 m^2/s`; the constraint is recorded
  in the matching JSON settings.
- Table 3: constant `D = D_DNN(c=1) = 7e-15 m^2/s` and
  `H_con = 0.85 H_peak` for the full-current constant-D reference.

Do not compare raw times across these studies without accounting for their
different diffusivity and constraint settings.

## License and citation

The code is released under BSD-3-Clause; see `LICENSE`. Citation metadata are
provided in `CITATION.cff`. Replace its repository placeholder before creating
the archival release, then add the Zenodo DOI after publication of the deposit.
