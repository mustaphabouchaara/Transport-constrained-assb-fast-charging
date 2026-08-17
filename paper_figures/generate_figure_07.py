"""Reproduce Figure 7: nonlinear maximal-feasible charging.

The normalized concentration field c(x,t) is represented in the IGA spline
basis on x in [0, 1].  The health metric is evaluated using the same Gaussian
quadrature as the PDE discretization,

    H(t) = c_bar(t) - c_s(t),
    c_bar(t) = integral_0^1 c(x,t) dx,
    c_s(t) = c(0,t).

For each beta, the controller takes the greatest one-step current in
[0, j_max] that makes H(t_{n+1}) <= H_con.  Thus it is a pointwise
maximal-feasible feedback law for the semi-implicit update.  It does not
claim that every nonlinear case stays on the boundary after first contact:
when full current becomes feasible again, H can fall below H_con.

The configured input is a C-rate; the physical current used in the
Faraday-law boundary-flux scale is ``capacity_Ah * C_rate`` in amperes.

Run from the repository root:
    python paper_figures/generate_figure_07.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

# The canonical workflow writes a vector PDF and must also run on headless
# continuous-integration and archival machines.
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as sparse_linalg

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from iga_core import (  # noqa: E402
    L2_projection,
    Mass_Matrix,
    Stiffness_Matrix,
    basis_ders_on_quad_grid,
    elements_spans,
    gauss_legendre,
    int_approx,
    make_knots,
    quadrature_grid,
)

CONFIG_PATH = Path(__file__).resolve().parent / "figure_07_settings.json"


@dataclass(frozen=True)
class Model:
    """Immutable numerical model assembled from the Figure 7 configuration."""

    capacity: float
    thickness: float
    area: float
    maximum_concentration: float
    faraday: float
    c_initial: float
    d_ref: float
    j_max_c_rate: float
    target_average: float
    dt_nd: float
    horizon_s: float
    elements: int
    degree: int
    spans: np.ndarray
    basis: np.ndarray
    weights: np.ndarray
    mass: np.ndarray
    stiffness: np.ndarray
    initial: np.ndarray
    t0: float
    dt_s: float
    step_sizes_nd: np.ndarray
    times_s: np.ndarray
    j_max_nd: float
    surface_vector: np.ndarray
    mass_solver: object
    average_weights: np.ndarray
    linear_zero_step: np.ndarray
    regular_current_response: np.ndarray

    def average(self, coefficients: np.ndarray) -> float:
        """Return c_bar using coefficient-level IGA quadrature."""
        return float(self.average_weights @ coefficients)

    def health(self, coefficients: np.ndarray) -> float:
        """Return H=c_bar-c_s with c_s represented by the first coefficient."""
        return self.average(coefficients) - float(coefficients[0])


def load_config(path: Path) -> dict:
    """Load and validate a self-contained Figure 7 configuration."""
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "physical_parameters", "discretization", "operating_conditions",
        "dnn_diffusivity", "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing Figure 7 configuration sections: {sorted(missing)}")
    physical = config["physical_parameters"]
    discretization = config["discretization"]
    operating = config["operating_conditions"]
    output = config["output"]
    for section, keys in (
        (physical, ("capacity_Ah", "electrode_thickness_m", "electrode_area_m2",
                    "maximum_concentration_mol_m3", "faraday_constant_C_mol",
                    "initial_normalized_concentration")),
        (discretization, ("elements", "spline_degree", "gauss_points_per_element",
                          "nonlinear_time_step_nondimensional", "horizon_s")),
        (operating, ("reference_diffusivity_m2_s", "maximum_charging_C_rate",
                     "target_average_concentration",
                     "constraint_fraction_of_linear_full_current_peak", "beta_values")),
        (output, ("figure_pdf",)),
    ):
        absent = set(keys).difference(section)
        if absent:
            raise ValueError(f"Missing configuration keys: {sorted(absent)}")
    nodes = np.asarray(config["dnn_diffusivity"].get("nodes"), dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 2:
        raise ValueError("DNN diffusivity nodes must be a two-column array with at least two rows.")
    if not np.all(np.diff(nodes[:, 0]) > 0.0) or np.any(nodes[:, 1] <= 0.0):
        raise ValueError("DNN diffusivity nodes require increasing concentrations and positive values.")
    if not np.isclose(nodes[-1, 0], 1.0, atol=1.0e-12):
        raise ValueError("The final DNN node must be located at c=1.")
    if not np.isclose(nodes[-1, 1], operating["reference_diffusivity_m2_s"], atol=1.0e-30):
        raise ValueError("D_ref must equal the DNN diffusivity at c=1.")
    if int(discretization["gauss_points_per_element"]) < int(discretization["spline_degree"]) + 1:
        raise ValueError("At least degree+1 Gauss points per element are required.")
    return config


def build_model(config: dict) -> Model:
    """Assemble the reference cubic IGA spatial discretization and exact final time."""
    physical = config["physical_parameters"]
    discretization = config["discretization"]
    operating = config["operating_conditions"]

    elements = int(discretization["elements"])
    degree = int(discretization["spline_degree"])
    grid = np.linspace(0.0, 1.0, elements + 1)
    knots = make_knots(grid, degree, False)
    spans = elements_spans(knots, degree)
    gauss_points = int(discretization["gauss_points_per_element"])
    # iga_core.gauss_legendre(order) returns order+1 points.
    quadrature_points, quadrature_weights = gauss_legendre(gauss_points - 1)
    points, weights = quadrature_grid(grid, quadrature_points, quadrature_weights)
    basis = basis_ders_on_quad_grid(knots, degree, points, 1, normalize=False)
    n_basis = len(knots) - degree - 1
    mass = np.zeros((n_basis, n_basis))
    stiffness = np.zeros_like(mass)
    Mass_Matrix(elements, degree, spans, basis, weights, points, mass)
    Stiffness_Matrix(elements, degree, spans, basis, weights, points, stiffness)

    d_ref = float(operating["reference_diffusivity_m2_s"])
    thickness = float(physical["electrode_thickness_m"])
    t0 = thickness**2 / d_ref
    dt_nd = float(discretization["nonlinear_time_step_nondimensional"])
    dt_s = dt_nd * t0
    horizon_s = float(discretization["horizon_s"])
    full_steps = int(np.floor(horizon_s / dt_s + 1.0e-12))
    remainder_s = horizon_s - full_steps * dt_s
    step_sizes_nd = np.full(full_steps, dt_nd)
    if remainder_s > 1.0e-12:
        step_sizes_nd = np.append(step_sizes_nd, remainder_s / t0)
    times_s = np.concatenate(([0.0], np.cumsum(step_sizes_nd) * t0))
    if not np.isclose(times_s[-1], horizon_s, atol=1.0e-12):
        raise RuntimeError("Time grid does not terminate exactly at the configured horizon.")

    initial_concentration = float(physical["initial_normalized_concentration"])
    initial = L2_projection(knots, degree, lambda _: initial_concentration)
    surface_vector = np.zeros(n_basis)
    surface_vector[0] = 1.0
    average_weights = np.array([
        int_approx(elements, degree, spans, basis, weights, np.eye(n_basis)[i])
        for i in range(n_basis)
    ])
    if not np.isclose(float(average_weights.sum()), 1.0, atol=1.0e-12):
        raise RuntimeError("IGA average weights must integrate the constant field exactly.")

    capacity = float(physical["capacity_Ah"])
    area = float(physical["electrode_area_m2"])
    maximum_concentration = float(physical["maximum_concentration_mol_m3"])
    faraday = float(physical["faraday_constant_C_mol"])
    j_max = float(operating["maximum_charging_C_rate"])
    j_max_nd = j_max * capacity * thickness / (faraday * area * d_ref * maximum_concentration)
    mass_solver = sparse_linalg.splu(csc_matrix(mass))
    linear_zero_step = mass_solver.solve(mass - dt_nd * stiffness)
    regular_current_response = mass_solver.solve(-dt_nd * surface_vector)
    return Model(
        capacity=capacity, thickness=thickness, area=area,
        maximum_concentration=maximum_concentration, faraday=faraday,
        c_initial=initial_concentration, d_ref=d_ref, j_max_c_rate=j_max,
        target_average=float(operating["target_average_concentration"]),
        dt_nd=dt_nd, horizon_s=horizon_s, elements=elements, degree=degree,
        spans=spans, basis=basis, weights=weights, mass=mass, stiffness=stiffness,
        initial=initial, t0=t0, dt_s=dt_s, step_sizes_nd=step_sizes_nd,
        times_s=times_s, j_max_nd=j_max_nd, surface_vector=surface_vector,
        mass_solver=mass_solver,
        average_weights=average_weights,
        linear_zero_step=linear_zero_step,
        regular_current_response=regular_current_response,
    )


def d_dnn(concentration: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Evaluate the configured piecewise-linear DNN diffusivity law."""
    return np.interp(np.asarray(concentration, dtype=float), nodes[:, 0], nodes[:, 1])


def diffusion_vector(
    model: Model, beta: float, coefficients: np.ndarray, dnn_nodes: np.ndarray
) -> np.ndarray:
    """Assemble K(c) in M c' + K(c) + j e_s = 0."""
    if beta == 0.0:
        return model.stiffness @ coefficients
    vector = np.zeros_like(coefficients)
    values = model.basis[:, :, 0, :]
    derivatives = model.basis[:, :, 1, :]
    for element, span in enumerate(model.spans):
        indices = slice(span - model.degree, span + 1)
        local = coefficients[indices]
        concentration_q = values[element].T @ local
        gradient_q = derivatives[element].T @ local
        diffusivity_q = (1.0 - beta) * model.d_ref + beta * d_dnn(concentration_q, dnn_nodes)
        vector[indices] += derivatives[element] @ (
            model.weights[element] * diffusivity_q / model.d_ref * gradient_q
        )
    return vector


def zero_current_update(
    model: Model, beta: float, coefficients: np.ndarray, dt_step: float, dnn_nodes: np.ndarray
) -> np.ndarray:
    """Advance one semi-implicit step with zero applied surface current."""
    if beta == 0.0 and np.isclose(dt_step, model.dt_nd, rtol=0.0, atol=1.0e-16):
        return model.linear_zero_step @ coefficients
    return model.mass_solver.solve(
        model.mass @ coefficients - dt_step * diffusion_vector(model, beta, coefficients, dnn_nodes)
    )


def current_response(model: Model, dt_step: float) -> np.ndarray:
    """Return the affine coefficient response to one unit of nondimensional current."""
    if np.isclose(dt_step, model.dt_nd, rtol=0.0, atol=1.0e-16):
        return model.regular_current_response
    return model.mass_solver.solve(-dt_step * model.surface_vector)


def full_current_peak(model: Model, dnn_nodes: np.ndarray) -> tuple[float, float]:
    """Return H_max and its time for beta=0 under full current over the full horizon."""
    coefficients = model.initial.copy()
    peak = model.health(coefficients)
    peak_time = 0.0
    for index, dt_step in enumerate(model.step_sizes_nd, start=1):
        zero = zero_current_update(model, 0.0, coefficients, dt_step, dnn_nodes)
        coefficients = zero + model.j_max_nd * current_response(model, dt_step)
        value = model.health(coefficients)
        if value > peak:
            peak, peak_time = value, float(model.times_s[index])
    return float(peak), peak_time


def interpolate_target(
    model: Model,
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Linearly locate the average-concentration target within an implicit step."""
    average_before = model.average(before)
    average_after = model.average(after)
    fraction = (average_before - model.target_average) / (average_before - average_after)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction, before + fraction * (after - before)


def solve_maximal_feasible(
    model: Model, beta: float, h_con: float, dnn_nodes: np.ndarray
) -> dict:
    """Integrate the full horizon with a maximal-feasible charging phase.

    Charging stops at the exact target event.  The remaining configured
    horizon is propagated at zero current, which cleanly separates charge
    completion from post-charge relaxation in the published trajectory.
    """
    coefficients = model.initial.copy()
    times = [0.0]
    health_values = [model.health(coefficients)]
    currents = [0.0]
    averages = [model.average(coefficients)]
    entry_time = None
    boundary_exit_time = None
    target_time = None
    charged = False
    maximum_violation = health_values[0] - h_con
    active_steps = 0

    for index, dt_step in enumerate(model.step_sizes_nd, start=1):
        t_previous = model.times_s[index - 1]
        t_current = model.times_s[index]
        if charged:
            coefficients = zero_current_update(model, beta, coefficients, dt_step, dnn_nodes)
            h_next = model.health(coefficients)
            j_next = 0.0
        else:
            zero = zero_current_update(model, beta, coefficients, dt_step, dnn_nodes)
            response = current_response(model, dt_step)
            h_zero = model.health(zero)
            gain = model.health(response)
            if gain <= 0.0:
                raise RuntimeError("The one-step health gain must be positive.")
            if h_zero > h_con + 1.0e-10:
                raise RuntimeError(
                    f"beta={beta:g}: zero current violates H_con at {t_current:.9f} s."
                )
            j_next = float(np.clip((h_con - h_zero) / gain, 0.0, model.j_max_nd))
            next_coefficients = zero + j_next * response
            h_next = model.health(next_coefficients)
            if h_next > h_con + 1.0e-9:
                raise RuntimeError("Internal maximal-feasibility check failed.")

            if j_next < model.j_max_nd - 1.0e-12:
                active_steps += 1
                if entry_time is None:
                    entry_time = float(t_current)
            elif entry_time is not None and boundary_exit_time is None:
                boundary_exit_time = float(t_current)

            if model.average(next_coefficients) <= model.target_average:
                fraction, target_coefficients = interpolate_target(model, coefficients, next_coefficients)
                target_time = float(t_previous + fraction * (t_current - t_previous))
                times.append(target_time)
                health_values.append(model.health(target_coefficients))
                currents.append(0.0)
                averages.append(model.average(target_coefficients))
                remaining_step = (1.0 - fraction) * dt_step
                coefficients = (
                    zero_current_update(model, beta, target_coefficients, remaining_step, dnn_nodes)
                    if remaining_step > 0.0 else target_coefficients
                )
                charged = True
                # The state at t_current is the zero-current relaxation of
                # the interpolated target state, not the unconstrained end
                # state that would have resulted from charging over the full
                # original step. Record that physically consistent state.
                h_next = model.health(coefficients)
                j_next = 0.0
            else:
                coefficients = next_coefficients

        maximum_violation = max(maximum_violation, h_next - h_con)
        times.append(float(t_current))
        health_values.append(float(h_next))
        currents.append(float(j_next))
        averages.append(model.average(coefficients))

    if target_time is None:
        raise RuntimeError(
            f"beta={beta:g}: target c_bar={model.target_average:g} was not reached within "
            f"{model.horizon_s:g} s."
        )
    return {
        "beta": beta,
        "time_s": np.asarray(times),
        "health": np.asarray(health_values),
        "current_nd": np.asarray(currents),
        "average": np.asarray(averages),
        "entry_time_s": entry_time,
        "boundary_exit_time_s": boundary_exit_time,
        "target_time_s": target_time,
        "maximum_violation": float(maximum_violation),
        "active_steps": int(active_steps),
    }


def configure_plot_style() -> None:
    """Apply PDF-safe, compact publication settings."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })


def plot_results(model: Model, results: dict[float, dict], h_max: float, h_con: float, output: Path) -> None:
    """Write the Figure 7 vector PDF with only the active H_con reference line."""
    configure_plot_style()
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(results)))
    fig, (axis_h, axis_j) = plt.subplots(
        1, 2, figsize=(7.16, 2.55), constrained_layout=True
    )
    scale = model.faraday * model.area * model.d_ref * model.maximum_concentration / (
        model.capacity * model.thickness
    )
    for (beta, result), color in zip(results.items(), colors):
        time = result["time_s"]
        health = result["health"]
        current = result["current_nd"] * scale
        axis_h.plot(time, health, color=color, linewidth=2.1, label=rf"$\beta={beta:g}$")
        axis_j.step(time, current, where="post", color=color, linewidth=2.1)
        entry = result["entry_time_s"]
        if entry is not None:
            axis_h.plot(entry, np.interp(entry, time, health), "o", ms=4,
                        mfc="white", mec=color, zorder=5)
            axis_j.plot(entry, np.interp(entry, time, current), "o", ms=4,
                        mfc="white", mec=color, zorder=5)
        target = result["target_time_s"]
        axis_h.plot(target, np.interp(target, time, health), "o", ms=3, color=color)
        axis_j.plot(target, 0.0, "o", ms=3, color=color)

    axis_h.axhline(h_con, color="black", linestyle="--", linewidth=1.1, label=r"$H_{\rm con}$")
    axis_j.axhline(model.j_max_c_rate, color="black", linestyle="--", linewidth=1.1)
    axis_h.set(xlabel="Time [s]", ylabel=r"$H(t)=\bar c(t)-c_s(t)$")
    axis_j.set(xlabel="Time [s]", ylabel=r"Charging C-rate $C(t)$")
    axis_h.set_title("(a)", loc="left", fontweight="bold")
    axis_j.set_title("(b)", loc="left", fontweight="bold")
    axis_h.legend(frameon=False, loc="lower left")
    axis_h.text(0.98, 0.07, rf"$\bar c: 1\!\to\!{model.target_average:.1f}$",
                transform=axis_h.transAxes, ha="right", va="bottom", fontsize=7)
    axis_j.text(0.98, 0.07, rf"$C_{{\max}}={model.j_max_c_rate:g}$",
                transform=axis_j.transAxes, ha="right", va="bottom", fontsize=7)
    for axis in (axis_h, axis_j):
        axis.set_xlim(0.0, model.horizon_s)
        axis.tick_params(direction="in", top=True, right=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def reproduce(config_path: Path = CONFIG_PATH) -> dict[str, object]:
    """Generate Figure 7 from one settings file and return run diagnostics.

    This public function is used by the canonical Figure 7 notebook as well as
    by the command-line entry point. Keeping the numerical implementation here
    prevents the notebook and script from drifting apart.
    """

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    model = build_model(config)
    dnn_nodes = np.asarray(config["dnn_diffusivity"]["nodes"], dtype=float)
    fraction = float(config["operating_conditions"]["constraint_fraction_of_linear_full_current_peak"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("The H_con fraction must lie strictly between zero and one.")
    h_max, h_max_time = full_current_peak(model, dnn_nodes)
    h_con = fraction * h_max
    beta_values = tuple(float(value) for value in config["operating_conditions"]["beta_values"])
    results = {
        beta: solve_maximal_feasible(model, beta, h_con, dnn_nodes)
        for beta in beta_values
    }
    for beta, result in results.items():
        if result["maximum_violation"] > 1.0e-8:
            raise RuntimeError(f"beta={beta:g}: constraint violation remains.")
        if result["entry_time_s"] is None:
            raise RuntimeError(f"beta={beta:g}: the constraint never becomes active.")
        if not np.isclose(result["target_time_s"], result["target_time_s"]):
            raise RuntimeError(f"beta={beta:g}: target event is invalid.")

    output = REPOSITORY_ROOT / config["output"]["figure_pdf"]
    plot_results(model, results, h_max, h_con, output)
    return {
        "output": output,
        "h_max": h_max,
        "h_max_time_s": h_max_time,
        "h_con": h_con,
        "constraint_fraction": fraction,
        "results": results,
    }


def main() -> None:
    """Run the configured Figure 7 simulation, validations, and PDF export."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help="Path to a Figure 7 JSON configuration (default: canonical config).",
    )
    arguments = parser.parse_args()
    summary = reproduce(arguments.config)
    print(f"H_max = {summary['h_max']:.12g} at {summary['h_max_time_s']:.9g} s")
    print(
        f"H_con = {summary['h_con']:.12g} "
        f"({summary['constraint_fraction']:.6g} H_max)"
    )
    results = summary["results"]
    for beta, result in results.items():
        print(
            f"beta={beta:g}: entry={result['entry_time_s']:.9g} s, "
            f"target={result['target_time_s']:.9g} s, "
            f"boundary_exit={result['boundary_exit_time_s']}, "
            f"max(H-H_con)={result['maximum_violation']:.3e}"
        )
    print(f"Wrote {summary['output']}")


if __name__ == "__main__":
    main()
