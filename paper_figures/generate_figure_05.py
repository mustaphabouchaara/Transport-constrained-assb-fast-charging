#!/usr/bin/env python3
"""Generate Figure 5 from its canonical linear-validation notebook.

The notebook remains the single source for the numerical workflow.  This
command-line entry point executes only the cells required to assemble the
model, compute the policy, and write Figure 5; exploratory tables are skipped.
It uses a headless Matplotlib backend and therefore works without opening any
plot windows.

Run from any directory with::

    python paper_figures/generate_figure_05.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "figure_05_linear_validation.ipynb"
OUTPUT_PATH = PROJECT_ROOT / "paper_figures" / "figure_05_linear_validation.pdf"

# These are the notebook's model, solve, reconstruction, and final plotting
# cells.  The final table cell is intentionally excluded.
REQUIRED_CODE_CELLS = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15)


def load_notebook_code() -> list[str]:
    """Load and validate the code cells used by the Figure 5 workflow."""
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Figure 5 notebook was not found: {NOTEBOOK_PATH}")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if not cells or max(REQUIRED_CODE_CELLS) >= len(cells):
        raise RuntimeError("The Figure 5 notebook does not have the expected cell layout.")

    code: list[str] = []
    for index in REQUIRED_CODE_CELLS:
        cell = cells[index]
        if cell.get("cell_type") != "code":
            raise RuntimeError(f"Expected notebook cell {index} to be a code cell.")
        code.append("".join(cell.get("source", [])))

    final_plot = code[-1]
    if "np.asarray(avg_list)" not in final_plot or r"$c_{\mathrm{avg}}(t)$" not in final_plot:
        raise RuntimeError(
            "The notebook's Figure 5 panel (c) must plot c_avg(t) before this generator runs."
        )
    return code


def main() -> None:
    """Execute the canonical notebook workflow and write the Figure 5 PDF."""
    mpl.use("Agg")
    code_cells = load_notebook_code()
    namespace: dict[str, object] = {"__name__": "__figure_05__"}
    previous_directory = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        for cell_index, source in zip(REQUIRED_CODE_CELLS, code_cells):
            exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{cell_index}", "exec"), namespace)
            if cell_index == 0:
                namespace["plt"].show = lambda *args, **kwargs: None  # type: ignore[union-attr]
            # Notebook development cells create diagnostic figures.  Close
            # them immediately; cell 15 writes the one publication figure.
            if cell_index != 15 and "plt" in namespace:
                namespace["plt"].close("all")  # type: ignore[union-attr]
    finally:
        os.chdir(previous_directory)

    if not OUTPUT_PATH.is_file():
        raise RuntimeError(f"Figure 5 was not written to {OUTPUT_PATH}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
