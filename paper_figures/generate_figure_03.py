#!/usr/bin/env python3
"""Reproduce manuscript Figure 3: quadratic temporal B-spline basis."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iga_core import make_knots, plots_splines  # noqa: E402


def main() -> None:
    """Write the representative temporal-basis plot used in the manuscript."""
    degree = 2
    elements = 16
    sample_points = 1000
    breakpoints = np.linspace(0.0, 1.0, elements + 1)
    knots = make_knots(breakpoints, degree, periodic=False)
    time, basis = plots_splines(knots, degree, sample_points)

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure, axis = plt.subplots(figsize=(7.1, 3.6))
    for index in range(basis.shape[1]):
        axis.plot(time, basis[:, index], linewidth=1.2)
    axis.scatter(
        breakpoints,
        np.zeros_like(breakpoints),
        color="black",
        s=14,
        zorder=3,
        label="Breakpoints",
    )
    axis.set_xlabel("Nondimensional time")
    axis.set_ylabel(r"Temporal B-spline basis $N_i(t)$")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(False)
    axis.legend(frameon=False, loc="upper right")
    figure.tight_layout()

    output = PROJECT_ROOT / "paper_figures" / "figure_03_temporal_bspline_basis.pdf"
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
