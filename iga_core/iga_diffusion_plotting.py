"""Diffusivity laws and plotting helpers for concentration-field simulations."""

import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(Q_matrix, xs, T_vals):
    """Display a concentration history as a space-time heatmap.

    ``Q_matrix`` follows the notebook convention of time along rows and space
    along columns; the transpose is plotted so space is vertical.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(Q_matrix.T, aspect='auto', origin='lower', extent=[T_vals[0], T_vals[-1], xs[0], xs[-1]], cmap='viridis')
    plt.colorbar(label=r'$c(x,t)$')
    plt.xlabel('Time $t$')
    plt.ylabel('Space $x$')
    plt.title('Evolution of $c(x,t)$ over time (heatmap)')
    plt.show()


def plot_surface(Q_matrix, xs, T_vals):
    """Display a concentration history as a three-dimensional Matplotlib surface."""
    X, Y = np.meshgrid(xs, T_vals, indexing='ij')
    Z = Q_matrix.T
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('Space $x$')
    ax.set_ylabel('Time $t$')
    ax.set_zlabel('$c(x,t)$')
    ax.set_title('3D Plot of $c(x,t)$ Over Time')
    plt.show()


def Diff_coef(c, text):
    """Return the piecewise-linear diffusivity for a named material data set.

    Only the retained ``'DNN'`` law is supported. Its tabulated values and
    concentration breakpoints reproduce the DNN model used in this project.
    """
    if text != 'DNN':
        raise ValueError("Only the DNN diffusivity law is retained.")

    c1, d1 = 0.5, 8.5e-14
    c2, d2 = 0.8, 2.5e-14
    c3, d3 = 0.95, 5e-16
    c4, d4 = 1, 7e-15

    if c <= c1:
        return d1
    elif c1 < c < c2:
        return d1 + (d2 - d1) * (c - c1) / (c2 - c1)
    elif c2 <= c < c3:
        return d2 + (d3 - d2) * (c - c2) / (c3 - c2)
    else:
        return d3 + (d4 - d3) * (c - c3) / (c4 - c3)


def varying_diff_coeff(data, D_ref, nelements, degree, spans, basis, weights, cn1, K):
    """Assemble the nonlinear diffusivity contribution into ``K`` in place.

    The concentration and its spatial derivative are evaluated at quadrature
    nodes from ``cn1``.  The assembled term is scaled by ``D_ref`` in the same
    nondimensional form used by the nonlinear diffusion solver.
    """
    ne1 = nelements
    p1 = degree
    spans_1 = spans
    basis_1 = basis
    weights_1 = weights
    k1 = weights.shape[1]
    coef_u = np.zeros(p1 + 1)
    C = np.zeros(k1)
    dC = np.zeros(k1)
    D = np.zeros(k1)
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        coef_u[:] = cn1[i_span_1 - p1:i_span_1 + 1]
        for g1 in range(0, k1):
            l = 0.0
            ll = 0.0
            for jl_1 in range(0, p1 + 1):
                bdx = basis_1[ie1, jl_1, 0, g1]
                l += bdx * coef_u[jl_1]
                bdx1 = basis_1[ie1, jl_1, 1, g1]
                ll += bdx1 * coef_u[jl_1]
            C[g1] = l
            D[g1] = Diff_coef(C[g1], data)
            dC[g1] = ll
        for il_1 in range(0, p1 + 1):
            i1 = i_span_1 - p1 + il_1
            v = 0.0
            for g1 in range(0, k1):
                bi_1 = basis_1[ie1, il_1, 1, g1]
                wvol = weights_1[ie1, g1]
                v += wvol * D[g1] * dC[g1] * bi_1
            K[i1] += v / D_ref
