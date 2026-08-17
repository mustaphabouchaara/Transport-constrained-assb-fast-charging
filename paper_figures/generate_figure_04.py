#!/usr/bin/env python3
"""Reproduce Figure 4: temporal-control-grid refinement for linear diffusion.

The script is deliberately independent of notebook state and generated ``.npz``
files.  It assembles the one-dimensional isogeometric Galerkin operators,
constructs the coarse and switch-refined temporal B-spline control spaces, and
computes both policies from the same linear response model.

The health functional is evaluated directly from IGA coefficients,

    H(t) = c_avg(t) - c_s(t),

where ``c_avg`` is the quadrature-consistent spatial average and ``c_s`` is the
concentration at the flux boundary. The dimensionless control is denoted by
``j(t)``; its physical current is ``capacity_Ah * j(t)`` in amperes. No sampled concentration profile is used
to evaluate the health functional.

Run from the repository root, for example::

    python paper_figures/generate_figure_04.py

An alternative JSON configuration can be passed with ``--config``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

# The canonical workflow writes a vector PDF and must also run on headless
# continuous-integration and archival machines.
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "figure_04_settings.json"


class ConfigurationError(ValueError):
    """Raised when a Figure 4 configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class SpatialOperators:
    """IGA matrices and coefficient vectors needed by the diffusion solver."""

    mass: np.ndarray
    stiffness: np.ndarray
    average: np.ndarray
    surface: np.ndarray

    @property
    def health(self) -> np.ndarray:
        """Return the coefficient vector for ``c_avg - c_s``."""

        return self.average - self.surface


@dataclass(frozen=True)
class PolicyResult:
    """A temporal B-spline policy and its collocation diagnostics."""

    name: str
    control_breakpoints: np.ndarray
    control_knots: np.ndarray
    coefficients: np.ndarray
    current: np.ndarray
    health: np.ndarray
    switch_index: int
    switch_time_s: float
    pre_switch_basis: tuple[int, ...]
    switching_basis: tuple[int, ...]
    post_switch_basis: tuple[int, ...]
    linear_system_rank: int
    linear_system_condition_number: float
    linear_system_raw_condition_number: float
    boundary_residual_inf: float
    phase_one_residual_inf: float


def load_config(path: Path) -> dict[str, Any]:
    """Read and validate a JSON configuration file."""

    if not path.is_file():
        raise FileNotFoundError(f"Figure 4 configuration was not found: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigurationError("The Figure 4 configuration must be a JSON object.")
    validate_config(config)
    return config


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required mapping section with a concise error on failure."""

    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section '{name}' must be an object.")
    return value


def _positive(section: dict[str, Any], key: str) -> float:
    """Read one finite, strictly positive numeric configuration value."""

    try:
        value = float(section[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Missing or invalid positive value '{key}'.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ConfigurationError(f"'{key}' must be finite and strictly positive.")
    return value


def _positive_integer(section: dict[str, Any], key: str) -> int:
    """Read one strictly positive integer configuration value."""

    value = _positive(section, key)
    if not value.is_integer():
        raise ConfigurationError(f"'{key}' must be an integer.")
    return int(value)


def validate_config(config: dict[str, Any]) -> None:
    """Validate physical, discretization, control, and output settings."""

    model = _section(config, "model")
    spatial = _section(config, "spatial_discretization")
    time = _section(config, "time_discretization")
    control = _section(config, "control")
    output = _section(config, "output")

    for key in (
        "capacity_Ah",
        "electrode_thickness_m",
        "electrode_area_m2",
        "maximum_concentration_mol_m3",
        "initial_normalized_concentration",
        "faraday_constant_C_mol",
        "reference_diffusivity_m2_s",
    ):
        _positive(model, key)

    if not np.isclose(_positive(spatial, "domain_length_normalized"), 1.0):
        raise ConfigurationError(
            "This normalized Figure 4 formulation requires domain_length_normalized = 1."
        )
    degree = _positive_integer(spatial, "spline_degree")
    elements = _positive_integer(spatial, "elements")
    quadrature = _positive_integer(spatial, "gauss_points_per_element")
    if quadrature < degree + 1:
        raise ConfigurationError(
            "gauss_points_per_element must be at least spline_degree + 1."
        )
    if elements < 1:
        raise ConfigurationError("At least one spatial element is required.")

    _positive(time, "duration_s")
    _positive(time, "time_step_nondimensional")
    if not isinstance(time.get("include_final_partial_step"), bool):
        raise ConfigurationError("include_final_partial_step must be true or false.")

    _positive(control, "maximum_charging_input")
    _positive(control, "health_upper_bound")
    temporal_degree = _positive_integer(control, "temporal_spline_degree")
    if temporal_degree < 1:
        raise ConfigurationError("temporal_spline_degree must be at least one.")
    if _positive_integer(control, "coarse_control_elements") < 1:
        raise ConfigurationError("At least one coarse control element is required.")
    _positive(control, "phase_one_weight")
    for key in (
        "refined_left_breakpoint_count",
        "refined_switch_breakpoint_count",
        "refined_right_breakpoint_count",
    ):
        if _positive_integer(control, key) < 2:
            raise ConfigurationError(f"'{key}' must be at least two.")
    _positive(control, "refined_window_fraction_of_coarse_element")

    for key in ("figure_pdf",):
        value = output.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"output.{key} must be a non-empty path string.")


def make_open_clamped_knots(breakpoints: np.ndarray, degree: int) -> np.ndarray:
    """Create the open clamped B-spline knot vector used by the notebooks."""

    breakpoints = np.asarray(breakpoints, dtype=float)
    if breakpoints.ndim != 1 or breakpoints.size < 2:
        raise ValueError("At least two one-dimensional breakpoints are required.")
    if not np.all(np.isfinite(breakpoints)) or not np.all(np.diff(breakpoints) > 0.0):
        raise ValueError("Breakpoints must be finite and strictly increasing.")
    if degree < 1:
        raise ValueError("The spline degree must be positive.")
    return np.concatenate(
        (
            np.full(degree, breakpoints[0]),
            breakpoints,
            np.full(degree, breakpoints[-1]),
        )
    )


def find_span(knots: np.ndarray, degree: int, coordinate: float) -> int:
    """Return the active B-spline span at ``coordinate`` on a clamped knot vector."""

    low = degree
    high = len(knots) - 1 - degree
    if coordinate <= knots[low]:
        return low
    if coordinate >= knots[high]:
        return high - 1
    span = (low + high) // 2
    while coordinate < knots[span] or coordinate >= knots[span + 1]:
        if coordinate < knots[span]:
            high = span
        else:
            low = span
        span = (low + high) // 2
    return span


def basis_funs(knots: np.ndarray, degree: int, coordinate: float, span: int) -> np.ndarray:
    """Evaluate the nonzero B-spline basis functions at one coordinate."""

    left = np.empty(degree, dtype=float)
    right = np.empty(degree, dtype=float)
    values = np.empty(degree + 1, dtype=float)
    values[0] = 1.0
    for order in range(degree):
        left[order] = coordinate - knots[span - order]
        right[order] = knots[span + 1 + order] - coordinate
        saved = 0.0
        for basis_index in range(order + 1):
            denominator = right[basis_index] + left[order - basis_index]
            temporary = values[basis_index] / denominator
            values[basis_index] = saved + right[basis_index] * temporary
            saved = left[order - basis_index] * temporary
        values[order + 1] = saved
    return values


def basis_funs_first_derivative(
    knots: np.ndarray, degree: int, coordinate: float, span: int
) -> np.ndarray:
    """Evaluate first derivatives of the nonzero B-spline basis functions."""

    if degree < 1:
        return np.zeros(1, dtype=float)
    lower_degree_values = basis_funs(knots, degree - 1, coordinate, span)
    derivatives = np.empty(degree + 1, dtype=float)
    saved = degree * lower_degree_values[0] / (
        knots[span + 1] - knots[span + 1 - degree]
    )
    derivatives[0] = -saved
    for basis_index in range(1, degree):
        previous = saved
        saved = degree * lower_degree_values[basis_index] / (
            knots[span + basis_index + 1] - knots[span + basis_index + 1 - degree]
        )
        derivatives[basis_index] = previous - saved
    derivatives[degree] = saved
    return derivatives


def basis_matrix(knots: np.ndarray, degree: int, coordinates: np.ndarray) -> np.ndarray:
    """Evaluate every temporal B-spline basis function on a time grid."""

    coordinates = np.asarray(coordinates, dtype=float)
    n_basis = len(knots) - degree - 1
    matrix = np.zeros((coordinates.size, n_basis), dtype=float)
    for row, coordinate in enumerate(coordinates):
        span = find_span(knots, degree, float(coordinate))
        matrix[row, span - degree : span + 1] = basis_funs(
            knots, degree, float(coordinate), span
        )
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=2.0e-13, rtol=0.0):
        raise RuntimeError("Temporal B-spline basis does not form a partition of unity.")
    return matrix


def assemble_spatial_operators(
    elements: int, degree: int, gauss_points: int, domain_length: float
) -> SpatialOperators:
    """Assemble mass, diffusion stiffness, average, and surface IGA operators.

    The spatial variable is normalized to ``[0, 1]``.  The average vector is
    built with the same element quadrature as the Galerkin matrices, so the
    reported health is coefficient-level and quadrature-consistent.
    """

    breakpoints = np.linspace(0.0, domain_length, elements + 1)
    knots = make_open_clamped_knots(breakpoints, degree)
    n_basis = len(knots) - degree - 1
    mass = np.zeros((n_basis, n_basis), dtype=float)
    stiffness = np.zeros_like(mass)
    average = np.zeros(n_basis, dtype=float)
    abscissas, quadrature_weights = np.polynomial.legendre.leggauss(gauss_points)

    for element, (left, right) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        for abscissa, reference_weight in zip(abscissas, quadrature_weights):
            coordinate = midpoint + half_width * abscissa
            weight = half_width * reference_weight
            span = find_span(knots, degree, float(coordinate))
            indices = np.arange(span - degree, span + 1)
            values = basis_funs(knots, degree, float(coordinate), span)
            derivatives = basis_funs_first_derivative(knots, degree, float(coordinate), span)
            mass[np.ix_(indices, indices)] += weight * np.outer(values, values)
            stiffness[np.ix_(indices, indices)] += weight * np.outer(derivatives, derivatives)
            average[indices] += weight * values / domain_length

    span_left = find_span(knots, degree, float(breakpoints[0]))
    surface = np.zeros(n_basis, dtype=float)
    surface[span_left - degree : span_left + 1] = basis_funs(
        knots, degree, float(breakpoints[0]), span_left
    )

    if not np.allclose(mass, mass.T, atol=2.0e-14, rtol=0.0):
        raise RuntimeError("Assembled mass matrix is not symmetric.")
    if not np.allclose(stiffness, stiffness.T, atol=2.0e-12, rtol=0.0):
        raise RuntimeError("Assembled stiffness matrix is not symmetric.")
    constant_coefficients = np.ones(n_basis)
    if not np.isclose(average @ constant_coefficients, 1.0, atol=2.0e-13, rtol=0.0):
        raise RuntimeError("IGA average operator does not preserve a constant field.")
    if not np.isclose((average - surface) @ constant_coefficients, 0.0, atol=2.0e-13):
        raise RuntimeError("Health functional must vanish for a spatially constant field.")

    return SpatialOperators(mass=mass, stiffness=stiffness, average=average, surface=surface)


def make_time_grid(time_config: dict[str, Any], reference_time_s: float) -> np.ndarray:
    """Build an exact-duration nondimensional grid with an optional final partial step."""

    duration_nd = _positive(time_config, "duration_s") / reference_time_s
    nominal_step = _positive(time_config, "time_step_nondimensional")
    include_partial = bool(time_config["include_final_partial_step"])
    full_steps = int(np.floor(duration_nd / nominal_step + 1.0e-13))
    time = nominal_step * np.arange(full_steps + 1, dtype=float)
    remaining = duration_nd - time[-1]

    if remaining > 2.0e-13:
        if not include_partial:
            raise ConfigurationError(
                "duration_s is not divisible by time_step_nondimensional; "
                "set include_final_partial_step to true."
            )
        time = np.append(time, duration_nd)
    elif remaining < -2.0e-13:
        raise RuntimeError("The nondimensional time grid overshot the requested duration.")
    else:
        time[-1] = duration_nd

    if time.size < 2 or not np.all(np.diff(time) > 0.0):
        raise RuntimeError("The time grid must contain strictly increasing nodes.")
    return time


def _factorizations(
    mass: np.ndarray, stiffness: np.ndarray, time: np.ndarray
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Cache implicit-Euler factorizations for the distinct step lengths."""

    factors: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for step in np.diff(time):
        key = round(float(step), 15)
        if key not in factors:
            factors[key] = lu_factor(mass + step * stiffness)
    return factors


def simulate_current(
    current: np.ndarray,
    time: np.ndarray,
    operators: SpatialOperators,
    initial_concentration: float,
    current_scale: float,
) -> np.ndarray:
    """Integrate the linear diffusion problem for a nodal charging-input history."""

    if current.shape != time.shape:
        raise ValueError("The current history and time grid must have identical shape.")
    factors = _factorizations(operators.mass, operators.stiffness, time)
    concentration = np.full(operators.mass.shape[0], initial_concentration, dtype=float)
    health = np.empty(time.size, dtype=float)
    health[0] = operators.health @ concentration

    for index, step in enumerate(np.diff(time), start=1):
        factor = factors[round(float(step), 15)]
        right_hand_side = (
            operators.mass @ concentration
            - step * current_scale * current[index] * operators.surface
        )
        concentration = lu_solve(factor, right_hand_side)
        health[index] = operators.health @ concentration

    if not np.all(np.isfinite(health)):
        raise RuntimeError("The linear diffusion integration produced non-finite health values.")
    return health


def build_health_response_matrix(
    temporal_basis: np.ndarray,
    time: np.ndarray,
    operators: SpatialOperators,
    current_scale: float,
) -> np.ndarray:
    """Build health responses to all temporal B-spline control basis functions.

    Linearity permits all basis responses to be propagated together as columns
    of one coefficient matrix.  This is exactly equivalent to solving one PDE
    per basis function and avoids dependence on precomputed response files.
    """

    if temporal_basis.shape[0] != time.size:
        raise ValueError("Temporal basis rows must equal the number of time nodes.")
    factors = _factorizations(operators.mass, operators.stiffness, time)
    n_control_basis = temporal_basis.shape[1]
    coefficients = np.zeros((operators.mass.shape[0], n_control_basis), dtype=float)
    response = np.zeros((time.size, n_control_basis), dtype=float)

    for index, step in enumerate(np.diff(time), start=1):
        factor = factors[round(float(step), 15)]
        right_hand_side = operators.mass @ coefficients - step * current_scale * np.outer(
            operators.surface, temporal_basis[index]
        )
        coefficients = lu_solve(factor, right_hand_side)
        response[index] = operators.health @ coefficients

    if not np.all(np.isfinite(response)):
        raise RuntimeError("A temporal-basis response contained non-finite values.")
    return response


def first_constraint_crossing(health: np.ndarray, limit: float) -> int:
    """Return the first discrete index at which a health limit is reached."""

    crossings = np.flatnonzero(health >= limit)
    if crossings.size == 0:
        raise RuntimeError(
            f"The full-current trajectory does not reach H_max={limit:.8g}; "
            f"its maximum is {health.max():.8g}."
        )
    return int(crossings[0])


def refined_control_breakpoints(
    duration_nd: float, switch_time_nd: float, control: dict[str, Any]
) -> np.ndarray:
    """Build the switch-focused temporal breakpoints used in the refined policy."""

    coarse_elements = _positive_integer(control, "coarse_control_elements")
    coarse_width = duration_nd / coarse_elements
    window_fraction = _positive(control, "refined_window_fraction_of_coarse_element")
    right_edge = min(duration_nd, switch_time_nd + window_fraction * coarse_width)
    if not right_edge > switch_time_nd:
        raise RuntimeError("The refined temporal window has zero width.")

    left = np.linspace(
        0.0,
        switch_time_nd,
        _positive_integer(control, "refined_left_breakpoint_count"),
        endpoint=False,
    )
    switching_window = np.linspace(
        switch_time_nd,
        right_edge,
        _positive_integer(control, "refined_switch_breakpoint_count"),
        endpoint=False,
    )
    right = np.linspace(
        right_edge,
        duration_nd,
        _positive_integer(control, "refined_right_breakpoint_count"),
    )
    breakpoints = np.unique(np.concatenate((left, switching_window, right)))
    if not np.isclose(breakpoints[0], 0.0) or not np.isclose(breakpoints[-1], duration_nd):
        raise RuntimeError("The refined control grid does not span the time horizon.")
    if not np.all(np.diff(breakpoints) > 0.0):
        raise RuntimeError("The refined control grid contains repeated breakpoints.")
    return breakpoints


def solve_policy(
    name: str,
    control_breakpoints: np.ndarray,
    time: np.ndarray,
    reference_time_s: float,
    operators: SpatialOperators,
    current_scale: float,
    control: dict[str, Any],
    switch_index: int,
) -> PolicyResult:
    """Fit the maximal-feasible temporal B-spline policy on one control grid.

    The pre-switch coefficients are fixed at the current upper bound.  The
    remaining coefficients solve the same weighted collocation system as the
    reference notebooks: health is fitted to ``H_max`` after the observed
    switch, while the initial full-current arc receives a large weight.
    """

    degree = _positive_integer(control, "temporal_spline_degree")
    maximum_input = _positive(control, "maximum_charging_input")
    health_upper_bound = _positive(control, "health_upper_bound")
    phase_one_weight = _positive(control, "phase_one_weight")
    knots = make_open_clamped_knots(control_breakpoints, degree)
    temporal_basis = basis_matrix(knots, degree, time)
    response = build_health_response_matrix(temporal_basis, time, operators, current_scale)
    n_control_basis = temporal_basis.shape[1]
    switch_time_nd = time[switch_index]
    tolerance = 50.0 * np.finfo(float).eps * max(1.0, time[-1])

    pre_switch: list[int] = []
    switching: list[int] = []
    post_switch: list[int] = []
    for basis_index in range(n_control_basis):
        support_start = knots[basis_index]
        support_end = knots[basis_index + degree + 1]
        if support_end <= switch_time_nd + tolerance:
            pre_switch.append(basis_index)
        elif support_start >= switch_time_nd - tolerance:
            post_switch.append(basis_index)
        else:
            switching.append(basis_index)

    unknown = switching + post_switch
    if not unknown:
        raise RuntimeError("No temporal control coefficients are available after switching.")
    coefficients = np.zeros(n_control_basis, dtype=float)
    coefficients[pre_switch] = maximum_input
    boundary_indices = np.arange(switch_index, time.size)
    phase_one_indices = np.arange(0, switch_index + 1)

    known_health = np.zeros(boundary_indices.size, dtype=float)
    known_current = np.zeros(phase_one_indices.size, dtype=float)
    if pre_switch:
        known_health = response[np.ix_(boundary_indices, pre_switch)] @ coefficients[pre_switch]
        known_current = temporal_basis[np.ix_(phase_one_indices, pre_switch)] @ coefficients[pre_switch]

    health_matrix = response[np.ix_(boundary_indices, unknown)]
    health_rhs = health_upper_bound - known_health
    phase_one_matrix = temporal_basis[np.ix_(phase_one_indices, unknown)]
    phase_one_rhs = maximum_input - known_current
    system_matrix = np.vstack((health_matrix, phase_one_weight * phase_one_matrix))
    system_rhs = np.concatenate((health_rhs, phase_one_weight * phase_one_rhs))
    # Column equilibration leaves the weighted least-squares objective
    # unchanged after coefficients are unscaled, while avoiding artificial
    # scale differences between temporal control columns.
    column_scales = np.linalg.norm(system_matrix, axis=0)
    if np.any(column_scales <= np.finfo(float).tiny):
        raise RuntimeError(f"{name} policy has an unobservable temporal-control coefficient.")
    scaled_matrix = system_matrix / column_scales
    scaled_coefficients, _, rank, singular_values = np.linalg.lstsq(
        scaled_matrix, system_rhs, rcond=None
    )
    unknown_coefficients = scaled_coefficients / column_scales
    coefficients[unknown] = unknown_coefficients

    current = temporal_basis @ coefficients
    health = response @ coefficients
    current_tolerance = 2.0e-9 * max(1.0, maximum_input)
    if current.min() < -current_tolerance or current.max() > maximum_input + current_tolerance:
        raise RuntimeError(
            f"{name} policy violates the current bounds: "
            f"[{current.min():.8g}, {current.max():.8g}] versus [0, {maximum_input:.8g}]."
        )
    if not np.all(np.isfinite(health)):
        raise RuntimeError(f"{name} policy produced non-finite health values.")

    boundary_residual = health[boundary_indices] - health_upper_bound
    phase_one_residual = current[phase_one_indices] - maximum_input
    condition_number = math.inf
    if singular_values.size and singular_values[-1] > 0.0:
        condition_number = float(singular_values[0] / singular_values[-1])
    raw_singular_values = np.linalg.svd(system_matrix, compute_uv=False)
    raw_condition_number = math.inf
    if raw_singular_values.size and raw_singular_values[-1] > 0.0:
        raw_condition_number = float(raw_singular_values[0] / raw_singular_values[-1])
    return PolicyResult(
        name=name,
        control_breakpoints=control_breakpoints,
        control_knots=knots,
        coefficients=coefficients,
        current=current,
        health=health,
        switch_index=switch_index,
        switch_time_s=float(switch_time_nd * reference_time_s),
        pre_switch_basis=tuple(pre_switch),
        switching_basis=tuple(switching),
        post_switch_basis=tuple(post_switch),
        linear_system_rank=int(rank),
        linear_system_condition_number=condition_number,
        linear_system_raw_condition_number=raw_condition_number,
        boundary_residual_inf=float(np.linalg.norm(boundary_residual, ord=np.inf)),
        phase_one_residual_inf=float(np.linalg.norm(phase_one_residual, ord=np.inf)),
    )


def plot_figure(
    time_s: np.ndarray,
    unrefined: PolicyResult,
    refined: PolicyResult,
    health_upper_bound: float,
    output_path: Path,
) -> None:
    """Export the two-panel temporal-refinement comparison as a PDF only."""

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
    colors = {"unrefined": "#1f77b4", "refined": "#ff7f0e"}

    axis = axes[0]
    axis.plot(time_s, unrefined.current, color=colors["unrefined"], linewidth=1.8, label="Unrefined")
    axis.plot(
        time_s,
        refined.current,
        color=colors["refined"],
        linestyle="--",
        linewidth=1.8,
        label="Refined",
    )
    axis.set_ylabel(r"Charging input $j(t)$")
    axis.text(0.02, 0.95, "(a)", transform=axis.transAxes, fontweight="bold", va="top")
    axis.legend(frameon=False, loc="best")
    axis.grid(False)

    axis = axes[1]
    axis.plot(time_s, unrefined.health, color=colors["unrefined"], linewidth=1.8, label="Unrefined")
    axis.plot(
        time_s,
        refined.health,
        color=colors["refined"],
        linestyle="--",
        linewidth=1.8,
        label="Refined",
    )
    axis.axhline(
        health_upper_bound,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=r"$H_{\max}$",
    )
    axis.set_xlabel("Time [s]")
    axis.set_ylabel(r"$H(t)=c_{\mathrm{avg}}(t)-c_s(t)$")
    axis.text(0.02, 0.95, "(b)", transform=axis.transAxes, fontweight="bold", va="top")
    axis.legend(frameon=False, loc="best")
    axis.grid(False)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def policy_summary(policy: PolicyResult, health_upper_bound: float) -> dict[str, Any]:
    """Return compact numerical diagnostics for one policy."""

    return {
        "control_basis_count": int(policy.coefficients.size),
        "control_breakpoint_count": int(policy.control_breakpoints.size),
        "switch_time_s": policy.switch_time_s,
        "current_min": float(policy.current.min()),
        "current_max": float(policy.current.max()),
        "health_min": float(policy.health.min()),
        "health_max": float(policy.health.max()),
        "maximum_health_violation": float(policy.health.max() - health_upper_bound),
        "boundary_collocation_residual_inf": policy.boundary_residual_inf,
        "phase_one_collocation_residual_inf": policy.phase_one_residual_inf,
        "linear_system_rank": policy.linear_system_rank,
        "linear_system_condition_number": policy.linear_system_condition_number,
        "linear_system_raw_condition_number": policy.linear_system_raw_condition_number,
        "pre_switch_basis_indices": list(policy.pre_switch_basis),
        "switching_basis_indices": list(policy.switching_basis),
        "post_switch_basis_indices": list(policy.post_switch_basis),
    }


def reproduce(config: dict[str, Any]) -> dict[str, Any]:
    """Compute both temporal-grid policies and write the Figure 4 PDF."""

    model = _section(config, "model")
    spatial = _section(config, "spatial_discretization")
    time_config = _section(config, "time_discretization")
    control = _section(config, "control")
    output = _section(config, "output")

    reference_time_s = _positive(model, "electrode_thickness_m") ** 2 / _positive(
        model, "reference_diffusivity_m2_s"
    )
    time_nd = make_time_grid(time_config, reference_time_s)
    time_s = time_nd * reference_time_s
    operators = assemble_spatial_operators(
        _positive_integer(spatial, "elements"),
        _positive_integer(spatial, "spline_degree"),
        _positive_integer(spatial, "gauss_points_per_element"),
        _positive(spatial, "domain_length_normalized"),
    )
    current_scale = (
        _positive(model, "capacity_Ah") * _positive(model, "electrode_thickness_m")
    ) / (
        _positive(model, "faraday_constant_C_mol")
        * _positive(model, "electrode_area_m2")
        * _positive(model, "reference_diffusivity_m2_s")
        * _positive(model, "maximum_concentration_mol_m3")
    )
    maximum_input = _positive(control, "maximum_charging_input")
    health_upper_bound = _positive(control, "health_upper_bound")
    full_current = np.full(time_nd.size, maximum_input, dtype=float)
    full_current_health = simulate_current(
        full_current,
        time_nd,
        operators,
        _positive(model, "initial_normalized_concentration"),
        current_scale,
    )
    switch_index = first_constraint_crossing(full_current_health, health_upper_bound)

    duration_nd = float(time_nd[-1])
    coarse_breakpoints = np.linspace(
        0.0,
        duration_nd,
        _positive_integer(control, "coarse_control_elements") + 1,
    )
    refined_breakpoints = refined_control_breakpoints(
        duration_nd, float(time_nd[switch_index]), control
    )
    unrefined = solve_policy(
        "Unrefined",
        coarse_breakpoints,
        time_nd,
        reference_time_s,
        operators,
        current_scale,
        control,
        switch_index,
    )
    refined = solve_policy(
        "Refined",
        refined_breakpoints,
        time_nd,
        reference_time_s,
        operators,
        current_scale,
        control,
        switch_index,
    )

    figure_path = PROJECT_ROOT / output["figure_pdf"]
    plot_figure(time_s, unrefined, refined, health_upper_bound, figure_path)
    return {
        "figure_pdf": output["figure_pdf"],
        "first_crossing_time_s": float(time_s[switch_index]),
        "unrefined": policy_summary(unrefined, health_upper_bound),
        "refined": policy_summary(refined, health_upper_bound),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse the optional configuration path without notebook-specific state."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the Figure 4 JSON configuration (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standalone Figure 4 reproduction and print concise diagnostics."""

    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_config(config_path)
    summary = reproduce(config)
    print(f"Wrote {summary['figure_pdf']}")
    print(
        "Full-current first crossing: "
        f"{summary['first_crossing_time_s']:.6f} s"
    )
    for name in ("unrefined", "refined"):
        result = summary[name]
        print(
            f"{name.capitalize()}: max(H-H_max)={result['maximum_health_violation']:.6e}, "
            f"boundary residual={result['boundary_collocation_residual_inf']:.6e}"
        )


if __name__ == "__main__":
    main()
