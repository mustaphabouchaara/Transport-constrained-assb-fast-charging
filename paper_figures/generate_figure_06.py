#!/usr/bin/env python3
r"""Reproduce Figure 6: DNN diffusivity robustness.

The figure quantifies the robustness of a transport-constrained charging
protocol when the diffusivity changes from the constant reference value to a
concentration-dependent DNN interpolation.  The health functional is evaluated
directly by isogeometric quadrature,

.. math::

   H(c) = \bar c-c_s, \qquad
   \bar c=\int_0^1 c(x)\,\mathrm{d}x, \qquad c_s=c(0).

For every interpolation parameter :math:`\beta`, the diffusivity is

.. math::

   D_\beta(c)=(1-\beta)D_{\rm ref}+\beta D_{\rm DNN}(c).

The protocol is generated once for :math:`\beta=0` by selecting the largest
admissible current at every step, then replayed unchanged for every
:math:`\beta`.  Panel (b) reports the resulting constraint margin

.. math::

   \Delta(\beta)=\max_{0\leq t\leq T}\left[H_\beta(t)-H_{\rm con}\right].

This is a clean, cache-free reproducer.  It deliberately does not read legacy
notebooks or ``.npz`` files; its settings are stored alongside the final PDF.

The configuration expresses the dimensionless control as ``j(t)``. Its
physical current is ``capacity_Ah * j(t)`` in amperes before Faraday-law scaling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl

# Reproduction must run identically on headless CI/archival machines and on a
# workstation.  A PDF is the only graphical artifact written by this script.
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as sparse_linalg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iga_core import (  # noqa: E402
    L2_projection,
    Mass_Matrix,
    Stiffness_Matrix,
    basis_ders_on_quad_grid,
    elements_spans,
    gauss_legendre,
    make_knots,
    quadrature_grid,
)


@dataclass(frozen=True)
class SpatialModel:
    r"""IGA spatial operators and exact quadrature representation of :math:`\bar c`."""

    mass: np.ndarray
    stiffness: np.ndarray
    connectivity: np.ndarray
    values: np.ndarray
    gradients: np.ndarray
    weights: np.ndarray
    average_weights: np.ndarray
    initial_state: np.ndarray


@dataclass
class TrajectorySummary:
    """Compact diagnostics for one replayed diffusivity trajectory."""

    beta: float
    delta: float
    maximum_health: float
    time_of_maximum_health_s: float
    target_crossing_time_s: float | None


@dataclass(frozen=True)
class TimeGrid:
    """Time steps that preserve the configured step and the exact horizon.

    Every full step has the configured nondimensional width.  If the requested
    physical horizon is not an integer multiple of that width, one smaller
    final step closes the interval exactly.  This is the same convention used
    by the canonical Figure 6 reproducer.
    """

    step_sizes_nd: np.ndarray
    step_end_times_s: np.ndarray
    full_step_nd: float
    time_scale_s: float
    full_step_count: int
    final_partial_step_nd: float | None

    @property
    def n_steps(self) -> int:
        """Return the total number of full and (if needed) partial steps."""
        return int(self.step_sizes_nd.size)

    @property
    def full_step_s(self) -> float:
        """Return the configured full-step width in physical seconds."""
        return float(self.full_step_nd * self.time_scale_s)

    @property
    def final_partial_step_s(self) -> float | None:
        """Return the optional final partial width in physical seconds."""
        return None if self.final_partial_step_nd is None else float(self.final_partial_step_nd * self.time_scale_s)


def parse_arguments() -> argparse.Namespace:
    """Parse optional configuration and non-interactive plotting arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "figure_06_settings.json",
        help="Path to the Figure 6 JSON configuration.",
    )
    return parser.parse_args()


def load_configuration(path: Path) -> dict[str, Any]:
    """Load and minimally validate the JSON configuration file."""
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "figure",
        "physical_parameters",
        "dnn_diffusivity",
        "spatial_discretization",
        "time_discretization",
        "operating_condition",
        "robustness_assessment",
        "output",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing required sections: {missing}")
    return config


def build_time_grid(horizon_s: float, time_scale_s: float, full_step_nd: float) -> TimeGrid:
    r"""Build a 65 s-compatible time grid without rescaling the configured step.

    Let :math:`T_0=L_p^2/D_{\rm ref}` and :math:`T=T_{\rm phys}/T_0`.
    The grid contains ``floor(T/dt)`` full steps of exactly ``dt`` and, when
    required, one residual step :math:`T-N\,dt`.  This avoids silently
    changing the numerical scheme merely to fit a prescribed final time.
    """
    if horizon_s <= 0.0 or time_scale_s <= 0.0 or full_step_nd <= 0.0:
        raise ValueError("The horizon, time scale, and full time step must be positive.")

    horizon_nd = horizon_s / time_scale_s
    full_step_count = int(np.floor(horizon_nd / full_step_nd + 1.0e-12))
    residual_nd = horizon_nd - full_step_count * full_step_nd
    residual_tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(horizon_nd))
    if residual_nd < residual_tolerance:
        residual_nd = 0.0
    if residual_nd < 0.0:
        raise RuntimeError("Time-grid construction produced a negative residual step.")

    step_sizes = np.full(full_step_count, full_step_nd, dtype=float)
    final_partial_step_nd: float | None = None
    if residual_nd > 0.0:
        step_sizes = np.append(step_sizes, residual_nd)
        final_partial_step_nd = float(residual_nd)
    if step_sizes.size == 0:
        raise RuntimeError("The requested horizon is shorter than numerical round-off.")

    step_end_times_s = np.cumsum(step_sizes) * time_scale_s
    # Make the archival record and terminal numerical time exactly agree with
    # the configured physical horizon, while preserving every full step.
    step_end_times_s[-1] = horizon_s
    if not np.isclose(step_end_times_s[-1], horizon_s, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Time grid does not terminate at the configured physical horizon.")
    return TimeGrid(
        step_sizes_nd=step_sizes,
        step_end_times_s=step_end_times_s,
        full_step_nd=float(full_step_nd),
        time_scale_s=float(time_scale_s),
        full_step_count=full_step_count,
        final_partial_step_nd=final_partial_step_nd,
    )


def make_spatial_model(config: dict[str, Any]) -> SpatialModel:
    r"""Assemble the cubic IGA operators and exact coefficient-space average.

    The average is represented by a vector ``w`` such that
    :math:`\bar c=w^\mathsf{T}c`.  This avoids the sampled-grid ``mean`` or
    trapezoidal approximations that can alter the constraint value.
    """
    physical = config["physical_parameters"]
    spatial = config["spatial_discretization"]
    elements = int(spatial["elements"])
    degree = int(spatial["spline_degree"])
    gauss_points = int(spatial["gauss_points_per_element"])
    if elements < 1 or degree < 1:
        raise ValueError("The spatial element count and spline degree must be positive.")

    grid = np.linspace(0.0, 1.0, elements + 1)
    knots = make_knots(grid, degree, periodic=False)
    n_basis = len(knots) - degree - 1
    spans = np.asarray(elements_spans(knots, degree), dtype=int)
    # ``iga_core.gauss_legendre(q)`` returns q+1 points.  The reference
    # cubic implementation used ``gauss_legendre(p)`` and hence four points.
    quadrature_points, quadrature_weights = gauss_legendre(gauss_points - 1)
    points, weights = quadrature_grid(grid, quadrature_points, quadrature_weights)
    basis = basis_ders_on_quad_grid(knots, degree, points, nders=1, normalize=False)

    mass = np.zeros((n_basis, n_basis))
    stiffness = np.zeros_like(mass)
    Mass_Matrix(elements, degree, spans, basis, weights, points, mass)
    Stiffness_Matrix(elements, degree, spans, basis, weights, points, stiffness)

    connectivity = spans[:, None] - degree + np.arange(degree + 1)[None, :]
    values = basis[:, :, 0, :]
    gradients = basis[:, :, 1, :]
    average_weights = np.zeros(n_basis)
    local_average_weights = np.einsum("eaq,eq->ea", values, weights)
    np.add.at(average_weights, connectivity.ravel(), local_average_weights.ravel())

    c_initial = float(physical["initial_normalized_concentration"])
    initial_state = L2_projection(knots, degree, lambda x: c_initial)
    return SpatialModel(
        mass=mass,
        stiffness=stiffness,
        connectivity=connectivity,
        values=values,
        gradients=gradients,
        weights=weights,
        average_weights=average_weights,
        initial_state=initial_state,
    )


def dnn_diffusivity(concentration: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Evaluate the piecewise-linear DNN diffusivity with endpoint clamping."""
    return np.interp(concentration, nodes[:, 0], nodes[:, 1])


def health(state: np.ndarray, average_weights: np.ndarray) -> float:
    r"""Return :math:`H=\bar c-c_s` from IGA coefficients and exact quadrature."""
    return float(average_weights @ state - state[0])


def average_concentration(state: np.ndarray, average_weights: np.ndarray) -> float:
    r"""Return the normalized volume average :math:`\bar c` by IGA quadrature."""
    return float(average_weights @ state)


def diffusion_vector(
    state: np.ndarray,
    beta: float,
    model: SpatialModel,
    reference_diffusivity: float,
    dnn_nodes: np.ndarray,
) -> np.ndarray:
    r"""Assemble the semi-implicit diffusion vector for :math:`D_\beta(c)`.

    For ``beta=0`` this is exactly ``K c``.  For other values, diffusivity is
    evaluated at the previous state at quadrature points, matching the
    semi-implicit nonlinear update used in the original study.
    """
    if beta == 0.0:
        return model.stiffness @ state

    local_coefficients = state[model.connectivity]
    concentration_q = np.einsum("eaq,ea->eq", model.values, local_coefficients)
    gradient_q = np.einsum("eaq,ea->eq", model.gradients, local_coefficients)
    d_dnn_q = dnn_diffusivity(concentration_q, dnn_nodes)
    diffusivity_q = (1.0 - beta) * reference_diffusivity + beta * d_dnn_q
    local_vector = np.einsum(
        "eq,eq,eaq->ea",
        model.weights,
        diffusivity_q * gradient_q / reference_diffusivity,
        model.gradients,
    )
    assembled = np.zeros_like(state)
    np.add.at(assembled, model.connectivity.ravel(), local_vector.ravel())
    return assembled


def advance_one_step(
    state: np.ndarray,
    current_nd: float,
    beta: float,
    model: SpatialModel,
    mass_lu: Any,
    dt_nd: float,
    surface_vector: np.ndarray,
    reference_diffusivity: float,
    dnn_nodes: np.ndarray,
) -> np.ndarray:
    """Advance one semi-implicit IGA step under a non-negative charge current."""
    diffusion = diffusion_vector(state, beta, model, reference_diffusivity, dnn_nodes)
    rhs = model.mass @ state - dt_nd * (diffusion + current_nd * surface_vector)
    return mass_lu.solve(rhs)


def run_full_current_reference(
    model: SpatialModel,
    mass_lu: Any,
    time_grid: TimeGrid,
    current_nd: float,
    surface_vector: np.ndarray,
    reference_diffusivity: float,
    dnn_nodes: np.ndarray,
) -> tuple[float, float]:
    r"""Return :math:`H_{\mathrm{peak}}` and its time for beta=0 full-current charging."""
    state = model.initial_state.copy()
    peak = health(state, model.average_weights)
    peak_time_s = 0.0
    for dt_step_nd, time_s in zip(time_grid.step_sizes_nd, time_grid.step_end_times_s):
        state = advance_one_step(
            state, current_nd, 0.0, model, mass_lu, float(dt_step_nd), surface_vector,
            reference_diffusivity, dnn_nodes,
        )
        value = health(state, model.average_weights)
        if value > peak:
            peak = value
            peak_time_s = float(time_s)
    return peak, peak_time_s


def construct_reference_policy(
    model: SpatialModel,
    mass_lu: Any,
    time_grid: TimeGrid,
    maximum_current_nd: float,
    h_constraint: float,
    target_average_concentration: float,
    surface_vector: np.ndarray,
    reference_diffusivity: float,
    dnn_nodes: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Construct a fixed beta=0 maximal-feasible protocol.

    Fixed-state dynamics are affine in the current,
    :math:`c^{n+1}=c_0^{n+1}+j^n r`.  The procedure selects the largest
    current in :math:`[0,j_{\max}]` with :math:`H(c^{n+1})\leq H_{\rm con}`.
    At the final active step it solves the equally affine average constraint to
    stop exactly at the configured target average concentration.
    """
    state = model.initial_state.copy()
    currents = np.zeros(time_grid.n_steps)
    first_constraint_time_s: float | None = None
    target_time_s: float | None = None
    active_step_count = 0
    maximum_health = health(state, model.average_weights)

    for index, (dt_step_nd, end_time_s) in enumerate(zip(time_grid.step_sizes_nd, time_grid.step_end_times_s)):
        # The response to the boundary current is affine for a fixed state,
        # but it depends on the step width.  Recompute it for the optional
        # final partial step instead of reusing the full-step response.
        current_response = mass_lu.solve(-float(dt_step_nd) * surface_vector)
        health_gain = health(current_response, model.average_weights)
        average_gain = average_concentration(current_response, model.average_weights)
        if health_gain <= 0.0 or average_gain >= 0.0:
            raise RuntimeError(
                "Unexpected current response: charging must increase H and reduce the average concentration."
            )
        zero_current_state = advance_one_step(
            state, 0.0, 0.0, model, mass_lu, float(dt_step_nd), surface_vector,
            reference_diffusivity, dnn_nodes,
        )
        h_zero = health(zero_current_state, model.average_weights)
        if h_zero > h_constraint + 1.0e-10:
            raise RuntimeError(
                f"Zero-current transport violation at {end_time_s:.6f} s: "
                f"{h_zero:.6e} > {h_constraint:.6e}."
            )

        allowable_current = float(np.clip((h_constraint - h_zero) / health_gain, 0.0, maximum_current_nd))
        next_state = zero_current_state + allowable_current * current_response
        next_average = average_concentration(next_state, model.average_weights)

        if next_average <= target_average_concentration:
            target_current = (target_average_concentration - average_concentration(zero_current_state, model.average_weights)) / average_gain
            if target_current < -1.0e-12 or target_current > allowable_current + 1.0e-10:
                raise RuntimeError("The target average is not reachable within the final feasible time step.")
            allowable_current = float(np.clip(target_current, 0.0, allowable_current))
            next_state = zero_current_state + allowable_current * current_response
            target_time_s = float(end_time_s)
            currents[index] = allowable_current
            maximum_health = max(maximum_health, health(next_state, model.average_weights))
            state = next_state
            active_step_count = index + 1
            break

        currents[index] = allowable_current
        active_step_count = index + 1
        state = next_state
        current_health = health(state, model.average_weights)
        maximum_health = max(maximum_health, current_health)
        if allowable_current < maximum_current_nd - 1.0e-12 and first_constraint_time_s is None:
            first_constraint_time_s = float(end_time_s)
    else:
        raise RuntimeError(
            f"The beta=0 reference protocol did not reach c_avg={target_average_concentration} "
            f"within the configured horizon."
        )

    if active_step_count == 0:
        raise RuntimeError("The reference policy did not contain an active time step.")
    if first_constraint_time_s is None:
        first_constraint_time_s = target_time_s
    target_average_error = average_concentration(state, model.average_weights) - target_average_concentration
    return currents, {
        "first_constraint_time_s": first_constraint_time_s,
        "target_time_s": target_time_s,
        "active_step_count": active_step_count,
        "target_average_error": target_average_error,
        "maximum_health": maximum_health,
        "maximum_constraint_residual": maximum_health - h_constraint,
        "active_minimum_current_nd": float(np.min(currents[:active_step_count])),
    }


def replay_fixed_policy(
    beta: float,
    currents: np.ndarray,
    model: SpatialModel,
    mass_lu: Any,
    time_grid: TimeGrid,
    h_constraint: float,
    target_average_concentration: float,
    surface_vector: np.ndarray,
    reference_diffusivity: float,
    dnn_nodes: np.ndarray,
) -> TrajectorySummary:
    """Replay one common current protocol and measure its transport margin."""
    state = model.initial_state.copy()
    maximum_health = health(state, model.average_weights)
    maximum_time_s = 0.0
    target_crossing_time_s: float | None = None
    for current_nd, dt_step_nd, end_time_s in zip(
        currents, time_grid.step_sizes_nd, time_grid.step_end_times_s
    ):
        state = advance_one_step(
            state, float(current_nd), beta, model, mass_lu, float(dt_step_nd), surface_vector,
            reference_diffusivity, dnn_nodes,
        )
        current_health = health(state, model.average_weights)
        if current_health > maximum_health:
            maximum_health = current_health
            maximum_time_s = float(end_time_s)
        if target_crossing_time_s is None and average_concentration(state, model.average_weights) <= target_average_concentration + 1.0e-10:
            target_crossing_time_s = float(end_time_s)
    return TrajectorySummary(
        beta=float(beta),
        delta=float(maximum_health - h_constraint),
        maximum_health=float(maximum_health),
        time_of_maximum_health_s=maximum_time_s,
        target_crossing_time_s=target_crossing_time_s,
    )


def configure_matplotlib() -> None:
    """Apply publication-safe PDF typography without requiring a LaTeX install."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def plot_figure(
    config: dict[str, Any],
    nodes: np.ndarray,
    summaries: list[TrajectorySummary],
    h_max: float,
    h_constraint: float,
    output_path: Path,
) -> None:
    """Create the two-panel PDF with diffusivity family and robustness margin."""
    configure_matplotlib()
    diffusion = config["dnn_diffusivity"]
    tolerance = float(config["robustness_assessment"]["violation_tolerance"])
    reference_diffusivity = float(diffusion["reference_diffusivity_m2_s"])
    displayed_betas = [float(value) for value in diffusion["display_beta_values"]]
    c_values = np.linspace(nodes[0, 0], nodes[-1, 0], 600)
    dnn_values = dnn_diffusivity(c_values, nodes)
    beta_values = np.asarray([summary.beta for summary in summaries])
    delta_values = np.asarray([summary.delta for summary in summaries])
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(displayed_betas)))

    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), constrained_layout=True)
    diffusivity_axis, margin_axis = axes

    for beta, color in zip(displayed_betas, colors):
        d_beta = (1.0 - beta) * reference_diffusivity + beta * dnn_values
        diffusivity_axis.plot(c_values, d_beta, color=color, label=rf"$\beta={beta:g}$")
    diffusivity_axis.plot(nodes[:, 0], nodes[:, 1], "o", color="black", markersize=3.4, label="DNN nodes")
    diffusivity_axis.axhline(reference_diffusivity, color="black", linestyle="--", linewidth=1.1, label=r"$D_{\mathrm{ref}}$")
    diffusivity_axis.set(
        yscale="log",
        xlabel=r"Normalized concentration $c$",
        ylabel=r"$D_{\beta}(c)$ [m$^2$ s$^{-1}$]",
        title="(a)",
    )
    diffusivity_axis.title.set_fontweight("bold")
    diffusivity_axis.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.8)

    # Preserve a visible strict-feasibility band even when beta=0 is exactly
    # on the active constraint to round-off accuracy.
    lower_limit = min(float(np.min(delta_values)) * 1.15, -0.12 * tolerance)
    upper_limit = max(tolerance * 1.28, float(np.max(delta_values)) * 1.12)
    if upper_limit <= 0.0:
        upper_limit = tolerance * 1.28
    margin_axis.axhspan(0.0, tolerance, facecolor=(0.90, 0.97, 0.90), edgecolor="none", zorder=0)
    if upper_limit > tolerance:
        margin_axis.axhspan(tolerance, upper_limit, facecolor=(1.00, 0.94, 0.86), edgecolor="none", zorder=0)
    margin_axis.plot(beta_values, delta_values, color="0.35", linewidth=1.1, zorder=2)

    numerical_zero = 1.0e-12
    strictly_feasible = delta_values <= numerical_zero
    within_tolerance = (delta_values > numerical_zero) & (delta_values <= tolerance)
    violating = delta_values > tolerance
    margin_axis.scatter(beta_values[strictly_feasible], delta_values[strictly_feasible], color="black", marker="o", s=21, label="strictly feasible", zorder=3)
    margin_axis.scatter(beta_values[within_tolerance], delta_values[within_tolerance], color="green", marker="s", s=21, label="within tolerance", zorder=3)
    margin_axis.scatter(beta_values[violating], delta_values[violating], color="red", marker="^", s=25, label="violating", zorder=3)
    margin_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1, label=r"$\Delta=0$")
    margin_axis.axhline(tolerance, color="0.45", linestyle=":", linewidth=1.2, label=rf"$\varepsilon_{{\mathrm{{tol}}}}={tolerance:.3g}$")
    margin_axis.set(
        ylim=(lower_limit, upper_limit),
        xlabel=r"Interpolation parameter $\beta$",
        ylabel=r"$\Delta(\beta)=\max_t[H_\beta(t)-H_{\mathrm{con}}]$",
        title="(b)",
    )
    margin_axis.title.set_fontweight("bold")
    margin_axis.legend(frameon=False, loc="best")

    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def resolve_output_path(path_string: str) -> Path:
    """Resolve configuration paths relative to the repository root."""
    candidate = Path(path_string)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
def main() -> None:
    """Run the cache-free Figure 6 workflow and write its PDF."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_configuration(config_path)
    physical = config["physical_parameters"]
    diffusion = config["dnn_diffusivity"]
    temporal = config["time_discretization"]
    operating = config["operating_condition"]
    robustness = config["robustness_assessment"]

    model = make_spatial_model(config)
    nodes = np.asarray(diffusion["nodes"], dtype=float)
    if nodes.shape[1] != 2 or not np.all(np.diff(nodes[:, 0]) > 0.0):
        raise ValueError("DNN diffusivity nodes must be an ascending two-column array.")
    reference_diffusivity = float(diffusion["reference_diffusivity_m2_s"])
    if not np.isclose(nodes[-1, 1], reference_diffusivity, rtol=0.0, atol=1.0e-30):
        raise ValueError("D_ref must equal the DNN diffusivity at c=1.")

    t_scale_s = float(physical["electrode_thickness_m"]) ** 2 / reference_diffusivity
    requested_dt_nd = float(temporal["time_step_nondimensional"])
    horizon_s = float(temporal["horizon_s"])
    time_grid = build_time_grid(horizon_s, t_scale_s, requested_dt_nd)
    maximum_input = float(operating["maximum_charging_input"])
    if not bool(robustness["common_fixed_policy_for_all_beta"]):
        raise ValueError(
            "This Figure 6 reproducer is defined for one fixed beta=0 policy "
            "replayed across beta; set common_fixed_policy_for_all_beta to true."
        )
    if not np.isclose(float(robustness["reference_beta"]), 0.0, atol=0.0):
        raise ValueError("The canonical fixed policy must be constructed at beta=0.")
    if robustness["policy"] != "one_step_maximal_feasible":
        raise ValueError("Unsupported Figure 6 reference-policy construction.")
    current_scale = (
        float(physical["capacity_Ah"]) * float(physical["electrode_thickness_m"])
        / (float(physical["faraday_constant_C_mol"]) * float(physical["electrode_area_m2"])
           * reference_diffusivity * float(physical["maximum_concentration_mol_m3"]))
    )
    maximum_current_nd = maximum_input * current_scale
    target_average = float(operating["target_average_concentration"])
    constraint_fraction = float(operating["constraint_fraction_of_hmax"])
    if not 0.0 < constraint_fraction < 1.0:
        raise ValueError("constraint_fraction_of_hmax must lie strictly between zero and one.")
    beta_values = [float(value) for value in diffusion["beta_values"]]
    if not beta_values or not np.isclose(float(robustness["reference_beta"]), 0.0):
        raise ValueError("This Figure 6 reproducer requires beta=0 as the reference policy.")
    if 0.0 not in beta_values:
        raise ValueError("beta_values must include beta=0 for the reference case.")

    surface_vector = np.zeros(model.mass.shape[0])
    surface_vector[0] = 1.0
    mass_lu = sparse_linalg.splu(csc_matrix(model.mass))

    h_max, h_max_time_s = run_full_current_reference(
        model, mass_lu, time_grid, maximum_current_nd, surface_vector,
        reference_diffusivity, nodes,
    )
    h_constraint = constraint_fraction * h_max
    currents, _policy_summary = construct_reference_policy(
        model, mass_lu, time_grid, maximum_current_nd, h_constraint,
        target_average, surface_vector, reference_diffusivity, nodes,
    )
    summaries = [
        replay_fixed_policy(
            beta, currents, model, mass_lu, time_grid, h_constraint, target_average,
            surface_vector, reference_diffusivity, nodes,
        )
        for beta in beta_values
    ]

    figure_path = resolve_output_path(config["output"]["figure_pdf"])
    plot_figure(config, nodes, summaries, h_max, h_constraint, figure_path)

    print(f"Saved Figure 6 PDF: {figure_path}")
    partial_text = "none" if time_grid.final_partial_step_s is None else f"{time_grid.final_partial_step_s:.9g} s"
    print(
        f"H_peak={h_max:.9g}; H_con={h_constraint:.9g}; "
        f"full steps={time_grid.full_step_count}; total steps={time_grid.n_steps}; "
        f"dt={time_grid.full_step_s:.9g} s; final partial step={partial_text}"
    )
    for summary in summaries:
        print(
            f"beta={summary.beta:.2f}: Delta={summary.delta:.9g}, "
            f"max H={summary.maximum_health:.9g}, "
            f"t_max={summary.time_of_maximum_health_s:.6f} s"
        )
if __name__ == "__main__":
    main()
