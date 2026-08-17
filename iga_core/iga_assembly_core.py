"""Assembly, projection, quadrature, and visualization helpers for 1-D IGA.

The functions in this module retain the legacy public names used in the
notebooks.  Matrix and vector arguments are assembled in place unless a
function explicitly returns a newly computed array.
"""

import numpy as np
from scipy.sparse import csc_matrix, linalg as sla
import matplotlib.pyplot as plt

try:
    from pyccel.decorators import types
except ImportError:
    def types(*_signatures):
        """Provide a no-op replacement when optional ``pyccel`` is unavailable."""
        def decorator(function):
            """Return the undecorated Python function."""
            return function
        return decorator


@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def Stiffness_Matrix(nelements, degree, spans, basis, weights, points, matrix):
    """Assemble the one-dimensional diffusion stiffness matrix in place.

    ``basis[..., 1, :]`` is interpreted as the spatial derivative of the
    active B-splines at the supplied element quadrature nodes.  ``points`` is
    accepted for the established public API but is not required by this form.
    """
    k1 = weights.shape[1]
    ne = nelements
    p = degree
    for i_e in range(ne):
        i_span = spans[i_e]
        for i_1 in range(0, p + 1):
            i = i_span - p + i_1
            for j_1 in range(0, p + 1):
                j = i_span - p + j_1
                v = 0.0
                for i_k1 in range(0, k1):
                    b_xi = basis[i_e, i_1, 1, i_k1] * basis[i_e, j_1, 1, i_k1]
                    v += b_xi * weights[i_e, i_k1]
                matrix[i, j] += v


@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def Mass_Matrix(nelements, degree, spans, basis, weights, points, matrix):
    """Assemble the one-dimensional IGA mass matrix in place.

    ``basis[..., 0, :]`` supplies the active B-spline values at the element
    quadrature nodes.  ``points`` is retained for API compatibility.
    """
    k1 = weights.shape[1]
    ne = nelements
    p = degree
    for i_e in range(ne):
        i_span = spans[i_e]
        for i_1 in range(0, p + 1):
            i = i_span - p + i_1
            for j_1 in range(0, p + 1):
                j = i_span - p + j_1
                v = 0.0
                for i_k1 in range(0, k1):
                    b_xi = basis[i_e, i_1, 0, i_k1] * basis[i_e, j_1, 0, i_k1]
                    v += b_xi * weights[i_e, i_k1]
                matrix[i, j] += v


def L2_projection(knots, p, g):
    """Interpolate a scalar function into the B-spline coefficient space.

    The historical function name is preserved.  Its implementation constructs
    a collocation system at evenly spaced coordinates and solves it directly.
    """
    nbasis = len(knots) - p - 1
    X = np.linspace(knots[0], knots[-p], nbasis)
    matrix = np.zeros((nbasis, nbasis))
    G = np.zeros(nbasis)
    for i, ix in enumerate(X):
        i_span = find_span(knots, p, ix)
        values_xi = basis_funs(knots, p, ix, i_span)
        matrix[i, i_span - p: i_span + 1] = values_xi[:]
        G[i] = g(ix)
    lu = sla.splu(csc_matrix(matrix))
    gh = lu.solve(G)
    return gh


def point_on_bspline_curve(knots, P, x):
    """Evaluate a vector-valued B-spline curve with control points ``P`` at ``x``."""
    degree = len(knots) - len(P) - 1
    d = P.shape[-1]
    span = find_span(knots, degree, x)
    b = basis_funs(knots, degree, x, span)
    c = np.zeros(d)
    for k in range(0, degree + 1):
        c[:] += b[k] * P[span - degree + k, :]
    return c


def plot_field_1d(knots, degree, u, label, nx=101, color='b', xmin=None, xmax=None):
    """Plot a scalar B-spline field using the active Matplotlib axes.

    The function intentionally returns ``None`` to preserve notebook usage.
    """
    if xmin is None:
        xmin = knots[degree]
        xmax = knots[-degree - 1]
    xs = np.linspace(xmin, xmax, nx)
    P = np.zeros((len(u), 1))
    P[:, 0] = u[:]
    Q = np.zeros((nx, 1))
    for i, x in enumerate(xs):
        Q[i, :] = point_on_bspline_curve(knots, P, x)
    plt.plot(xs, Q[:, 0], '-' + color, linewidth=1, label=label)


def sol_plot(knots, degree, u, nx=101, xmin=None, xmax=None):
    """Sample a scalar B-spline field and return ``(values, coordinates)``.

    ``values`` has shape ``(nx, 1)`` for compatibility with existing notebook
    code that treats field samples as a column vector.
    """
    if xmin is None:
        xmin = knots[degree]
        xmax = knots[-degree - 1]
    xs = np.linspace(xmin, xmax, nx)
    P = np.zeros((len(u), 1))
    P[:, 0] = u[:]
    Q = np.zeros((nx, 1))
    for i, x in enumerate(xs):
        Q[i, :] = point_on_bspline_curve(knots, P, x)
    return Q, xs


def plots_splines(knots, degree, nx=101):
    """Sample every B-spline basis function on a uniformly spaced grid.

    Returns the coordinate vector and an array whose rows are coordinates and
    whose columns are global basis functions.
    """
    nbasis = len(knots) - degree - 1
    xs = np.linspace(knots[degree], knots[len(knots) - degree], nx)
    Q = np.zeros((nx, nbasis))
    for i, xq in enumerate(xs):
        low = degree
        high = len(knots) - 1 - degree
        if xq <= knots[low]:
            span = low
        elif xq >= knots[high]:
            span = high - 1
        else:
            span = (low + high) // 2
            while xq < knots[span] or xq >= knots[span + 1]:
                if xq < knots[span]:
                    high = span
                else:
                    low = span
                span = (low + high) // 2
        vals = basis_funs(knots, degree, xq, span)
        Q[i, span - degree:span + 1] = vals
    return xs, Q


def int_approx(nelements, degree, spans, basis, weights, U):
    """Integrate a scalar IGA field from its coefficients by element quadrature."""
    n = weights.shape[1]
    v = 0.0
    p = degree
    ne = nelements
    for ie1 in range(ne):
        i_span_1 = spans[ie1]
        for g1 in range(n):
            p_p_m = 0.0
            for i_1 in range(p + 1):
                i1 = i_span_1 - p + i_1
                bi_0 = basis[ie1, i_1, 0, g1]
                p_p_m += U[i1] * bi_0
            is1 = weights[ie1, g1]
            v += p_p_m * is1
    return v


def NIA(f, a, b, ordergl, tol=10e-14):
    """Approximate the integral of ``f`` over ``[a, b]`` by Gauss-Legendre quadrature.

    The Newton iteration and rule order match the legacy implementation.
    """
    m = ordergl + 1
    from math import cos, pi
    from numpy import zeros

    def legendre(t, m):
        """Return the Legendre polynomial of degree ``m`` and its derivative."""
        p0 = 1.0
        p1 = t
        for k in range(1, m):
            p = ((2.0 * k + 1.0) * t * p1 - k * p0) / (1.0 + k)
            p0 = p1
            p1 = p
        dp = m * (p0 - t * p1) / (1.0 - t**2)
        return p1, dp

    A = zeros(m)
    x = zeros(m)
    nRoots = (m + 1) // 2
    for i in range(nRoots):
        t = cos(pi * (i + 0.75) / (m + 0.5))
        for j in range(30):
            p, dp = legendre(t, m)
            dt = -p / dp
            t = t + dt
            if abs(dt) < tol:
                x[i] = t
                x[m - i - 1] = -t
                A[i] = 2.0 / (1.0 - t**2) / (dp**2)
                A[m - i - 1] = A[i]
                break
    val = 0.0
    for i in range(m):
        val += (a + b) / 2 * A[i] * f((a + b) / 2 * x[i] + (b - a) / 2)
    return val


def assemble_rhs(f, nelements, degree, spans, basis, weights, points, bounds, rhs):
    """Assemble the legacy diffusion-reaction right-hand-side vector in place.

    ``bounds`` stores IGA coefficients for the field whose spatial derivative
    is used in the second contribution.  The routine is retained for notebook
    compatibility and leaves ``rhs`` mutated in place.
    """
    p = degree
    k1 = weights.shape[1]
    for ie1 in range(nelements):
        i_span = spans[ie1]
        coef_u = bounds[i_span - p: i_span + 1]
        valuesdx = np.zeros(k1)
        for g1 in range(k1):
            l = 0.0
            for jl_1 in range(p + 1):
                bdx = basis[ie1, jl_1, 1, g1]
                l += bdx * coef_u[jl_1]
            valuesdx[g1] = l
        for il_1 in range(p + 1):
            i1 = i_span - p + il_1
            v = 0.0
            for g1 in range(k1):
                bi_0 = basis[ie1, il_1, 0, g1]
                bi_x = basis[ie1, il_1, 1, g1]
                x1 = points[ie1, g1]
                wvol = weights[ie1, g1]
                sx = valuesdx[g1]
                v += bi_0 * f(x1) * wvol - sx * bi_x * wvol
            rhs[i1] += v


from .iga_bspline import basis_funs, find_span
