"""Generate manuscript Tables 2 and 3 from their numerical algorithms.

Table 2 evaluates spatial/temporal convergence for the constant-diffusivity
maximal-feasible solution.  Table 3 compares charging policies on a common
verification grid.  The algorithms remain separate because they use different
discretizations and objectives; this module only provides one entry point for
their reproducible execution.

Examples
--------
python paper_tables/generate_tables.py 2
python paper_tables/generate_tables.py 3
python paper_tables/generate_tables.py all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import linalg as sla
from scipy.sparse.linalg import splu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "paper_tables"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iga_core import (  # noqa: E402
    L2_projection,
    Mass_Matrix,
    Stiffness_Matrix,
    basis_ders_on_quad_grid,
    elements_spans,
    gauss_legendre,
    int_approx,
    make_knots,
    plots_splines,
    quadrature_grid,
    varying_diff_coeff,
)


# ---------------------------------------------------------------------------
# Table 2: convergence of the constant-diffusivity maximal-feasible solution.
# ---------------------------------------------------------------------------

T2_CAPACITY = 1e-5
T2_LP = 3.2e-7
T2_SA = 1e-4
T2_A_MAX = 2.33e4
T2_C0 = 1.0
T2_FARADAY = 96485.0
T2_D_REF = 1.76e-15
T2_J_MAX = 60.0
T2_H_MAX = 0.1
T2_T_PHYS = 60.0
T2_PX = 3
T2_PT = 2
T2_DT_BASE = 5e-3


def _table2_assemble_space(nx: int):
    grid = np.linspace(0.0, 1.0, nx + 1)
    knots = make_knots(grid, T2_PX, False)
    nbasis = len(knots) - T2_PX - 1
    spans = elements_spans(knots, T2_PX)
    u_quad, w_quad = gauss_legendre(T2_PX)
    points, weights = quadrature_grid(grid, u_quad, w_quad)
    basis = basis_ders_on_quad_grid(knots, T2_PX, points, 1, normalize=False)
    mass = np.zeros((nbasis, nbasis))
    stiffness = np.zeros((nbasis, nbasis))
    Mass_Matrix(nx, T2_PX, spans, basis, weights, points, mass)
    Stiffness_Matrix(nx, T2_PX, spans, basis, weights, points, stiffness)
    initial = L2_projection(knots, T2_PX, lambda x: T2_C0)
    return knots, spans, basis, weights, mass, stiffness, initial


def _table2_health(c, nx: int, spans, basis, weights) -> float:
    return int_approx(nx, T2_PX, spans, basis, weights, c) - c[0]


def _table2_refined_time_grid(T: float, ts: float, nt_control: int) -> np.ndarray:
    """Construct the original 1:2:1 switching-centred control grid."""
    intervals = nt_control - 1
    n_left = intervals // 4
    n_refined = intervals // 2
    n_right = intervals - n_left - n_refined
    coarse_h = T / max(n_left + n_right, 1)
    t_right = min(T, ts + 0.5 * coarse_h)
    left = np.linspace(0.0, ts, n_left + 1)
    refined = np.linspace(ts, t_right, n_refined + 1)[1:]
    right = np.linspace(t_right, T, n_right + 1)[1:]
    grid = np.concatenate((left, refined, right))
    assert len(grid) == nt_control
    return grid


def solve_level(nx: int, nt_control: int) -> dict[str, float | int]:
    """Solve one Table-2 spatial/control-grid refinement level.

    This public function is retained for the historical notebook that reports
    the individual refinement levels.
    """
    knots, spans, basis, weights, mass, stiffness, initial = _table2_assemble_space(nx)
    nbasis = len(initial)
    T0 = T2_LP**2 / T2_D_REF
    T = T2_T_PHYS / T0
    nt_pde = int(T / T2_DT_BASE)
    t_pde = np.linspace(0.0, T, nt_pde)
    dt = t_pde[1] - t_pde[0]
    e0 = np.zeros(nbasis)
    e0[0] = 1.0
    j_scale = T2_CAPACITY * T2_LP / (T2_FARADAY * T2_SA * T2_D_REF * T2_A_MAX)

    lu = sla.splu(csr_matrix(mass + dt * stiffness).tocsc())
    c = initial.copy()
    h_jmax = np.zeros(nt_pde)
    for n in range(1, nt_pde):
        c = lu.solve(mass @ c - dt * (T2_J_MAX * j_scale) * e0)
        h_jmax[n] = _table2_health(c, nx, spans, basis, weights)

    crossings = np.flatnonzero(h_jmax >= T2_H_MAX)
    if not crossings.size:
        raise RuntimeError(f"Level nx={nx}: H_max is never reached")
    idx_s = int(crossings[0])
    ts = float(t_pde[idx_s])

    control_grid = _table2_refined_time_grid(T, ts, nt_control)
    knots_t = make_knots(control_grid, T2_PT, periodic=False)
    _, phi = plots_splines(knots_t, T2_PT, nt_pde)
    n_params = phi.shape[1]

    h_basis = np.zeros((n_params, nt_pde))
    for k in range(n_params):
        c = np.zeros(nbasis)
        for n in range(1, nt_pde):
            c = lu.solve(mass @ c - dt * (phi[n, k] * j_scale) * e0)
            h_basis[k, n] = _table2_health(c, nx, spans, basis, weights)

    k_pre, k_mixed, k_post = [], [], []
    for k in range(n_params):
        left = knots_t[k]
        right = knots_t[k + T2_PT + 1]
        if right <= ts:
            k_pre.append(k)
        elif left >= ts:
            k_post.append(k)
        else:
            k_mixed.append(k)
    unknown = k_mixed + k_post

    alpha = np.zeros(n_params)
    alpha[k_pre] = T2_J_MAX
    idx_active = np.arange(idx_s, nt_pde - 1)
    idx_pre = np.arange(0, idx_s + 1)
    a_h = h_basis[unknown][:, idx_active].T
    b_h = T2_H_MAX - h_basis[k_pre][:, idx_active].T @ alpha[k_pre]
    a_j = phi[idx_pre][:, unknown]
    b_j = T2_J_MAX - phi[idx_pre][:, k_pre] @ (T2_J_MAX * np.ones(len(k_pre)))
    weight_phase1 = 1e4
    a_stack = np.vstack((a_h, weight_phase1 * a_j))
    b_stack = np.concatenate((b_h, weight_phase1 * b_j))
    alpha[unknown], *_ = np.linalg.lstsq(a_stack, b_stack, rcond=None)

    h_rec = h_basis.T @ alpha
    r_h = h_rec[idx_active] - T2_H_MAX
    r_inf = float(np.max(np.abs(r_h)))
    r_rms = float(np.sqrt(np.mean(r_h**2)))

    n_verify = 10 * (len(idx_active) - 1) + 1
    t_verify = np.linspace(0.0, T, n_verify)
    dt_verify = t_verify[1] - t_verify[0]
    _, phi_verify = plots_splines(knots_t, T2_PT, n_verify)
    current_verify = phi_verify @ alpha
    lu_verify = sla.splu(csr_matrix(mass + dt_verify * stiffness).tocsc())
    c = initial.copy()
    h_verify = np.zeros(n_verify)
    for n in range(1, n_verify):
        c = lu_verify.solve(
            mass @ c - dt_verify * (current_verify[n] * j_scale) * e0
        )
        h_verify[n] = _table2_health(c, nx, spans, basis, weights)
    epsilon_h = float(np.max(np.maximum(h_verify - T2_H_MAX, 0.0)))

    return {
        "nx": nx,
        "Nt": nt_control,
        "ts_nd": ts,
        "tf_nd": float(t_pde[-2]),
        "ts_s": ts * T0,
        "tf_s": float(t_pde[-2] * T0),
        "rH_inf": r_inf,
        "rH_rms": r_rms,
        "epsilon_H": epsilon_h,
        "verify_points": n_verify,
    }


def compute_table2() -> pd.DataFrame:
    """Return the three-level convergence table used in manuscript Table 2."""
    levels = (("Coarse", 32, 25), ("Intermediate", 64, 49), ("Fine", 128, 97))
    rows = []
    for level, nx_level, nt_level in levels:
        result = solve_level(nx_level, nt_level)
        rows.append(
            {
                "Level": level,
                "n_x": nx_level,
                "N_t": nt_level,
                "t_s [s]": result["ts_s"],
                "t_f [s]": result["tf_s"],
                "||r_H||_inf": result["rH_inf"],
                "RMS(r_H)": result["rH_rms"],
                "epsilon_H": result["epsilon_H"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 3: policy comparison for constant and concentration-dependent DNN D.
# ---------------------------------------------------------------------------

T3_CAPACITY, T3_LP, T3_SA, T3_A_MAX = 1e-5, 3.2e-7, 1e-4, 2.33e4
# Use the DNN law's value at the initial concentration c=1 as the constant-D
# reference, so the linear and nonlinear comparisons start from the same
# diffusivity: D_DNN(1) = 7e-15 m^2/s.
T3_FARADAY, T3_D_REF = 96485.0, 7.0e-15
T3_C_INITIAL, T3_C_TARGET = 1.0, 0.6
T3_J_MAX, T3_HORIZON = 60.0, 120.0
# On this verification grid, full-current constant-D charging gives
# H_peak=0.04066925452335535.  Use 85% of that reproducible reference peak so
# the transport constraint is active and the policy comparison is informative.
T3_H_FRACTION_OF_LINEAR_FULL_CURRENT_PEAK = 0.85
T3_LINEAR_FULL_CURRENT_PEAK = 0.04066925452335535
T3_H_CON = T3_H_FRACTION_OF_LINEAR_FULL_CURRENT_PEAK * T3_LINEAR_FULL_CURRENT_PEAK
T3_NX, T3_PX = 4, 3
T3_DT_ND = 5e-5
T3_T0 = T3_LP**2 / T3_D_REF
T3_DT_S = T3_DT_ND * T3_T0
T3_N_STEPS = int(np.ceil(T3_HORIZON / T3_DT_S))
T3_J_SCALE = T3_CAPACITY * T3_LP / (T3_FARADAY * T3_SA * T3_D_REF * T3_A_MAX)
T3_Q_RATE = T3_CAPACITY / (T3_FARADAY * T3_SA * T3_LP * T3_A_MAX)
T3_Q_REQUIRED = T3_C_INITIAL - T3_C_TARGET
T3_TOL = 1e-6


def _table3_model():
    grid = np.linspace(0.0, 1.0, T3_NX + 1)
    knots = make_knots(grid, T3_PX, False)
    spans = elements_spans(knots, T3_PX)
    uq, wq = gauss_legendre(T3_PX)
    points, weights = quadrature_grid(grid, uq, wq)
    basis = basis_ders_on_quad_grid(knots, T3_PX, points, 1, normalize=False)
    nbasis = len(knots) - T3_PX - 1
    mass = np.zeros((nbasis, nbasis))
    stiffness = np.zeros_like(mass)
    Mass_Matrix(T3_NX, T3_PX, spans, basis, weights, points, mass)
    Stiffness_Matrix(T3_NX, T3_PX, spans, basis, weights, points, stiffness)
    initial = L2_projection(knots, T3_PX, lambda x: T3_C_INITIAL)
    return spans, basis, weights, mass, stiffness, initial


T3_SPANS, T3_BASIS, T3_WEIGHTS, T3_MASS, T3_STIFFNESS, T3_INITIAL = _table3_model()
T3_E0 = np.zeros(len(T3_INITIAL))
T3_E0[0] = 1.0
T3_LU_LINEAR = splu(csc_matrix(T3_MASS + T3_DT_ND * T3_STIFFNESS))
T3_LU_MASS = splu(csc_matrix(T3_MASS))


def _table3_health(c) -> float:
    return int_approx(T3_NX, T3_PX, T3_SPANS, T3_BASIS, T3_WEIGHTS, c) - c[0]


def _table3_cutoff_current(current):
    current = np.clip(np.asarray(current, float), 0.0, T3_J_MAX).copy()
    delivered = 0.0
    cutoff = None
    for n, value in enumerate(current):
        dq = value * T3_Q_RATE * T3_DT_S
        if delivered + dq >= T3_Q_REQUIRED and value > 0.0:
            fraction = (T3_Q_REQUIRED - delivered) / dq
            cutoff = (n + fraction) * T3_DT_S
            current[n] *= fraction
            current[n + 1 :] = 0.0
            delivered = T3_Q_REQUIRED
            break
        delivered += dq
    return current, cutoff, delivered


def _table3_simulate(current, nonlinear: bool = False):
    current, cutoff, delivered = _table3_cutoff_current(current)
    c = T3_INITIAL.copy()
    h = np.zeros(len(current) + 1)
    active_indices = np.flatnonzero(current > 0.0)
    active_count = int(active_indices[-1] + 1) if active_indices.size else 0
    for n, value in enumerate(current[:active_count], start=1):
        if nonlinear:
            kvec = np.zeros(len(c))
            varying_diff_coeff(
                "DNN", T3_D_REF, T3_NX, T3_PX, T3_SPANS, T3_BASIS, T3_WEIGHTS, c, kvec
            )
            c = T3_LU_MASS.solve(
                T3_MASS @ c - T3_DT_ND * (value * T3_J_SCALE * T3_E0 + kvec)
            )
        else:
            c = T3_LU_LINEAR.solve(
                T3_MASS @ c - T3_DT_ND * value * T3_J_SCALE * T3_E0
            )
        h[n] = _table3_health(c)
    violation = float(np.max(h - T3_H_CON))
    return {
        "current": current,
        "H": h,
        "tf": cutoff,
        "violation": violation,
        "delivered": delivered,
    }


def _table3_best_constant_current():
    lo, hi = 0.0, T3_J_MAX
    # Twenty-two bisection iterations resolve the normalized input to better
    # than 1.5e-5, well below the verification tolerance and faster than the
    # formerly excessive 28 iterations.
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        trial = _table3_simulate(np.full(T3_N_STEPS, mid))
        if trial["violation"] <= T3_TOL:
            lo = mid
        else:
            hi = mid
    return np.full(T3_N_STEPS, lo)


def _table3_greedy_linear_policy():
    c = T3_INITIAL.copy()
    policy = np.zeros(T3_N_STEPS)
    delivered = 0.0
    for n in range(T3_N_STEPS):
        remaining = T3_Q_REQUIRED - delivered
        if remaining <= 1e-12:
            break
        upper = min(T3_J_MAX, remaining / (T3_Q_RATE * T3_DT_S))
        c0 = T3_LU_LINEAR.solve(T3_MASS @ c)
        c1 = T3_LU_LINEAR.solve(T3_MASS @ c - T3_DT_ND * upper * T3_J_SCALE * T3_E0)
        h0, h1 = _table3_health(c0), _table3_health(c1)
        value = upper
        if h1 > T3_H_CON and h1 > h0:
            value = upper * np.clip((T3_H_CON - h0) / (h1 - h0), 0.0, 1.0)
        c = T3_LU_LINEAR.solve(T3_MASS @ c - T3_DT_ND * value * T3_J_SCALE * T3_E0)
        policy[n] = value
        delivered += value * T3_Q_RATE * T3_DT_S
    return policy


def _table3_direct_pso_policy(
    seed: int = 7,
    temporal_elements: int = 16,
    temporal_degree: int = 2,
    particles: int = 36,
    iterations: int = 70,
):
    """Optimize a smooth current through its temporal B-spline coefficients.

    Each particle is one complete coefficient vector ``alpha`` and defines
    ``j(t) = Phi(t) @ alpha``.  No maximal-feasible seed, assumed switching
    structure, piecewise-constant time blocks, feasibility rescaling, or
    fallback policy is used.
    """
    rng = np.random.default_rng(seed)
    temporal_grid = np.linspace(0.0, 1.0, temporal_elements + 1)
    temporal_knots = make_knots(temporal_grid, temporal_degree, periodic=False)
    _, phi = plots_splines(temporal_knots, temporal_degree, T3_N_STEPS)
    n_coefficients = phi.shape[1]

    # Propagate all B-spline unit responses in one matrix-valued time march.
    states = np.repeat(T3_INITIAL[:, None], n_coefficients, axis=1)
    response = np.zeros((T3_N_STEPS, n_coefficients))
    health_vector = np.asarray([
        _table3_health(np.eye(len(T3_INITIAL))[:, i])
        for i in range(len(T3_INITIAL))
    ])
    for n in range(T3_N_STEPS):
        states = T3_LU_LINEAR.solve(
            T3_MASS @ states
            - T3_DT_ND * T3_J_SCALE * T3_E0[:, None] * phi[n][None, :]
        )
        response[n] = health_vector @ states

    def objective(x):
        current = x @ phi.T
        h = x @ response.T
        cumulative_charge = np.cumsum(current, axis=1) * T3_Q_RATE * T3_DT_S
        reaches_target = cumulative_charge[:, -1] >= T3_Q_REQUIRED
        crossing = np.argmax(cumulative_charge >= T3_Q_REQUIRED, axis=1)
        crossing = np.where(reaches_target, crossing, T3_N_STEPS - 1)
        previous_index = np.maximum(crossing - 1, 0)
        previous_charge = np.where(
            crossing > 0,
            cumulative_charge[np.arange(len(x)), previous_index],
            0.0,
        )
        step_charge = current[np.arange(len(x)), crossing] * T3_Q_RATE * T3_DT_S
        fraction = np.clip(
            np.divide(
                T3_Q_REQUIRED - previous_charge,
                step_charge,
                out=np.ones_like(step_charge),
                where=step_charge > 0.0,
            ),
            0.0,
            1.0,
        )
        final_time = (crossing + fraction) * T3_DT_S
        before_cutoff = np.arange(T3_N_STEPS)[None, :] <= crossing[:, None]
        peak_health = np.max(np.where(before_cutoff, h, -np.inf), axis=1)
        violation = np.maximum(peak_health - T3_H_CON, 0.0)
        shortfall = np.maximum(T3_Q_REQUIRED - cumulative_charge[:, -1], 0.0)
        return final_time + 1e9 * violation**2 + 1e9 * shortfall**2

    # Every particle starts independently from a random coefficient vector.
    x = rng.uniform(0.0, T3_J_MAX, (particles, n_coefficients))
    v = np.zeros_like(x)
    pbest = x.copy()
    pscore = objective(x)
    gbest = pbest[np.argmin(pscore)].copy()
    gscore = np.min(pscore)
    for _ in range(iterations):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = 0.72 * v + 1.45 * r1 * (pbest - x) + 1.45 * r2 * (gbest - x)
        x = np.clip(x + v, 0.0, T3_J_MAX)
        score = objective(x)
        improve = score < pscore
        pbest[improve], pscore[improve] = x[improve], score[improve]
        if np.min(score) < gscore:
            gbest, gscore = x[np.argmin(score)].copy(), np.min(score)
    policy, _, _ = _table3_cutoff_current(phi @ gbest)
    return policy


def _table3_nonlinear_blockwise(block: int = 20):
    c = T3_INITIAL.copy()
    policy = np.zeros(T3_N_STEPS)
    delivered = 0.0
    held_value = T3_J_MAX
    for n in range(T3_N_STEPS):
        if delivered >= T3_Q_REQUIRED - 1e-12:
            break
        kvec = np.zeros(len(c))
        varying_diff_coeff(
            "DNN", T3_D_REF, T3_NX, T3_PX, T3_SPANS, T3_BASIS, T3_WEIGHTS, c, kvec
        )
        upper = min(T3_J_MAX, (T3_Q_REQUIRED - delivered) / (T3_Q_RATE * T3_DT_S))
        c_zero = T3_LU_MASS.solve(T3_MASS @ c - T3_DT_ND * kvec)
        c_upper = T3_LU_MASS.solve(
            T3_MASS @ c - T3_DT_ND * (upper * T3_J_SCALE * T3_E0 + kvec)
        )
        h_zero, h_upper = _table3_health(c_zero), _table3_health(c_upper)
        if n % block == 0 or _table3_health(c) > T3_H_CON - 5e-4:
            held_value = upper
            if h_upper > T3_H_CON and h_upper > h_zero:
                held_value = upper * np.clip(
                    (T3_H_CON - h_zero) / (h_upper - h_zero), 0.0, 1.0
                )
        value = min(held_value, upper)
        c = T3_LU_MASS.solve(
            T3_MASS @ c - T3_DT_ND * (value * T3_J_SCALE * T3_E0 + kvec)
        )
        policy[n] = value
        delivered += value * T3_Q_RATE * T3_DT_S
    return policy


def compute_table3() -> pd.DataFrame:
    """Return the charging-policy comparison reported in manuscript Table 3."""
    started = time.time()
    maximum = np.full(T3_N_STEPS, T3_J_MAX)
    best_constant = _table3_best_constant_current()
    pso = _table3_direct_pso_policy()
    offline_online = _table3_greedy_linear_policy()
    nonlinear_corrected = _table3_nonlinear_blockwise()
    cases = [
        ("Maximum charging rate", "constant D", maximum, False, "Low"),
        ("Best feasible constant rate", "constant D", best_constant, False, "Low"),
        ("Direct PSO", "constant D", pso, False, "High"),
        ("Offline-online maximal-feasible", "constant D", offline_online, False, "Low online"),
        ("Transferred constant-D policy", "DNN D(c)", offline_online, True, "Low"),
        ("Nonlinear blockwise correction", "DNN D(c)", nonlinear_corrected, True, "Medium"),
    ]
    rows = []
    for policy, model_name, current, nonlinear, effort in cases:
        result = _table3_simulate(current, nonlinear=nonlinear)
        rows.append(
            {
                "Policy": policy,
                "Model": model_name,
                "Feasible?": "Yes" if result["violation"] <= T3_TOL else "No",
                "t_f [s]": result["tf"],
                "max_t(H-H_con)": max(0.0, result["violation"]),
                "Effort": effort,
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["runtime_s"] = time.time() - started
    return table


def _write_table(table: pd.DataFrame, filename: str) -> Path:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIRECTORY / filename
    table.to_csv(output_path, index=False)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "table",
        nargs="?",
        choices=("2", "3", "all"),
        default="all",
        help="table to generate (default: all)",
    )
    args = parser.parse_args(argv)

    if args.table in ("2", "all"):
        table2 = compute_table2()
        output2 = _write_table(table2, "table_02_convergence.csv")
        print(table2.to_string(index=False))
        print(f"Saved {output2}")

    if args.table in ("3", "all"):
        table3 = compute_table3()
        output3 = _write_table(table3, "table_03_policy_comparison.csv")
        print(table3.to_string(index=False))
        print(f"Runtime: {table3.attrs['runtime_s']:.1f} s")
        print(f"Saved {output3}")


if __name__ == "__main__":
    main()
