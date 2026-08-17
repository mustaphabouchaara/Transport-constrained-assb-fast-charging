"""Nonlinear electrochemical assembly and post-processing utilities.

The routines implement the compact algebraic forms used by the battery-model
notebooks and retain their historical function names for compatibility.
"""

import numpy as np


def assemble_non_linear(nelements, degree, spans, basis, weights, cn1, rhs, nw_alpha, nw_beta, nw_gamma):
    """Assemble the Galerkin vector for ``alpha*c**2 + beta*c + gamma`` in place.

    The nonlinear reaction is evaluated at quadrature points from the IGA
    coefficient vector ``cn1`` and accumulated into ``rhs``.
    """
    for element in range(nelements):
        span = spans[element]
        indices = np.arange(span - degree, span + 1)
        values = basis[element, :, 0, :]
        concentration_q = cn1[indices] @ values
        reaction_q = (
            nw_alpha * concentration_q**2
            + nw_beta * concentration_q
            + nw_gamma
        )
        rhs[indices] += values @ (weights[element] * reaction_q)
    return rhs


def Non_linear_jacobian(nw_alpha, nw_beta, nelements, degree, spans, basis, weights, matrix, cn):
    """Assemble the Jacobian of the nonlinear Galerkin reaction vector in place."""
    for element in range(nelements):
        span = spans[element]
        indices = np.arange(span - degree, span + 1)
        values = basis[element, :, 0, :]
        concentration_q = cn[indices] @ values
        derivative_q = 2.0 * nw_alpha * concentration_q + nw_beta
        local = (values * (weights[element] * derivative_q)) @ values.T
        matrix[np.ix_(indices, indices)] += local
    return matrix


def c_avg_var(nelements, degree, spans, basis, weights, cn1, nw_alpha, nw_beta, nw_gamma):
    """Return the quadrature-weighted domain average of the nonlinear reaction."""
    integral = 0.0
    volume = 0.0
    for element in range(nelements):
        span = spans[element]
        indices = np.arange(span - degree, span + 1)
        values = basis[element, :, 0, :]
        concentration_q = cn1[indices] @ values
        reaction_q = (
            nw_alpha * concentration_q**2
            + nw_beta * concentration_q
            + nw_gamma
        )
        integral += float(weights[element] @ reaction_q)
        volume += float(np.sum(weights[element]))
    return integral / volume


def dc_dx(cn, dx):
    """Return a finite-difference gradient of a one-dimensional profile."""
    return np.gradient(cn, dx)


def dc_dx_all(Q_matrix, y):
    """Return spatial finite-difference gradients for every row of ``Q_matrix``."""
    return np.gradient(Q_matrix, y, axis=1)


def E_field_e(cn, dc_dx_profile, I, R, T, F, D_pos, D_neg):
    """Return the electric-field-like profile used by the transport post-processing.

    ``cn`` is retained in the established function signature although the
    algebraic expression depends on the supplied gradient and transport terms.
    """
    return (I * R * T) / (F * D_pos * D_neg) * dc_dx_profile


def E_e_field_matrix(Q_matrix, dc_dx_matrix, I, R, T, F, D_pos, D_neg):
    """Return the electric-field-like quantity for a matrix of concentration gradients.

    ``Q_matrix`` remains in the signature for compatibility with notebook code;
    the returned expression uses ``dc_dx_matrix`` and the transport parameters.
    """
    return (I * R * T) / (F * D_pos * D_neg) * dc_dx_matrix


def integrate_field_trapz(E_profile, y):
    """Integrate a one-dimensional field over coordinates ``y`` by the trapezoid rule."""
    return np.trapz(E_profile, y)


def eta_mt_profile_original(c_profile, E_profile, y, R, T, F):
    """Return the original profile-level mass-transport overpotential metric."""
    return np.trapz(c_profile * E_profile, y) / (R * T * F)


def eta_mt_time_series_original(ce_matrix, E_matrix, y, R, T, F):
    """Evaluate the original mass-transport metric for each time-series column."""
    return np.array([eta_mt_profile_original(ce_matrix[:, i], E_matrix[:, i], y, R, T, F) for i in range(ce_matrix.shape[1])])
