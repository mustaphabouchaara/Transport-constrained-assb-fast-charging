# Manuscript figures

Figure numbers and filenames match the manuscript.

```powershell
python paper_figures/generate_figure_03.py
python paper_figures/generate_figure_04.py
python paper_figures/generate_figure_05.py
python paper_figures/generate_figure_06.py
python paper_figures/generate_figure_07.py
```

- Figure 3: representative quadratic temporal B-spline basis.
- Figure 4: local temporal-grid refinement comparison.
- Figure 5: constant-diffusivity maximal-feasible validation.
- Figure 6: robustness of a transferred policy under DNN diffusivity.
- Figure 7: nonlinear maximal-feasible admissibility correction.

Figures 4, 6, and 7 have editable JSON settings beside their generators.
Figure 5 uses the canonical interactive notebook in `../notebooks/`; its
command-line generator executes only the cells needed for the PDF.
