# coding: utf-8
"""B-spline utilities used by the one-dimensional IGA discretizations.

The routines implement span lookup, local basis evaluation, quadrature-grid
construction, and collocation matrices for the battery transport models.  They
preserve the indexing conventions of the historical simulation notebooks.
"""

import numpy as np

__all__ = [
    'find_span',
    'basis_funs',
    'basis_funs_1st_der',
    'basis_funs_all_ders',
    'collocation_matrix',
    'breakpoints',
    'greville',
    'elements_spans',
    'make_knots',
    'quadrature_grid',
    'basis_ders_on_quad_grid',
]


def find_span(knots, degree, x):
    """Return the knot-span index containing ``x`` for a clamped B-spline basis.

    Coordinates at either endpoint are assigned to the first or last valid
    span, respectively, so that endpoint basis evaluation remains well-defined.
    """
    low = degree
    high = len(knots) - 1 - degree
    if x <= knots[low]:
        return low
    if x >= knots[high]:
        return high - 1
    span = (low + high) // 2
    while x < knots[span] or x >= knots[span + 1]:
        if x < knots[span]:
            high = span
        else:
            low = span
        span = (low + high) // 2
    return span


def basis_funs(knots, degree, x, span):
    """Evaluate the ``degree + 1`` B-splines nonzero at ``x`` in ``span``.

    The returned values correspond to global basis indices
    ``span - degree`` through ``span``.
    """
    left = np.empty(degree, dtype=float)
    right = np.empty(degree, dtype=float)
    values = np.empty(degree + 1, dtype=float)
    values[0] = 1.0
    for j in range(degree):
        left[j] = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved = 0.0
        for r in range(j + 1):
            temp = values[r] / (right[r] + left[j - r])
            values[r] = saved + right[r] * temp
            saved = left[j - r] * temp
        values[j + 1] = saved
    return values


def basis_funs_1st_der(knots, degree, x, span):
    """Evaluate first derivatives of the B-splines nonzero at ``x``.

    The ordering matches :func:`basis_funs` and therefore maps to global basis
    indices ``span - degree`` through ``span``.
    """
    values = basis_funs(knots, degree - 1, x, span)
    ders = np.empty(degree + 1, dtype=float)
    saved = degree * values[0] / (knots[span + 1] - knots[span + 1 - degree])
    ders[0] = -saved
    for j in range(1, degree):
        temp = saved
        saved = degree * values[j] / (knots[span + j + 1] - knots[span + j + 1 - degree])
        ders[j] = temp - saved
    ders[degree] = saved
    return ders


def basis_funs_all_ders(knots, degree, x, span, n):
    """Evaluate nonzero B-splines and derivatives through order ``n`` at ``x``.

    Rows of the returned array are derivative orders and columns follow the
    local basis-function ordering used by :func:`basis_funs`.
    """
    left = np.empty(degree)
    right = np.empty(degree)
    ndu = np.empty((degree + 1, degree + 1))
    a = np.empty((2, degree + 1))
    ders = np.zeros((n + 1, degree + 1))
    ne = min(n, degree)
    ndu[0, 0] = 1.0
    for j in range(degree):
        left[j] = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved = 0.0
        for r in range(j + 1):
            ndu[j + 1, r] = 1.0 / (right[r] + left[j - r])
            temp = ndu[r, j] * ndu[j + 1, r]
            ndu[r, j + 1] = saved + right[r] * temp
            saved = left[j - r] * temp
        ndu[j + 1, j + 1] = saved
    ders[0, :] = ndu[:, degree]
    for r in range(degree + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, ne + 1):
            d = 0.0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] * ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if (rk > -1) else -rk
            j2 = k - 1 if (r - 1 <= pk) else degree - r
            a[s2, j1:j2 + 1] = (a[s1, j1:j2 + 1] - a[s1, j1 - 1:j2]) * ndu[pk + 1, rk + j1:rk + j2 + 1]
            d += np.dot(a[s2, j1:j2 + 1], ndu[rk + j1:rk + j2 + 1, pk])
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] * ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            j = s1
            s1 = s2
            s2 = j
    r = degree
    for k in range(1, ne + 1):
        ders[k, :] = ders[k, :] * r
        r = r * (degree - k)
    return ders


def collocation_matrix(knots, degree, xgrid, periodic):
    """Construct the B-spline collocation matrix on ``xgrid``.

    For periodic bases, local basis contributions are wrapped into the reduced
    set of independent periodic coefficients.
    """
    nb = len(knots) - degree - 1
    if periodic:
        nb -= degree
    nx = len(xgrid)
    mat = np.zeros((nx, nb))
    if periodic:
        js = lambda span: [(span - degree + s) % nb for s in range(degree + 1)]
    else:
        js = lambda span: slice(span - degree, span + 1)
    for i, x in enumerate(xgrid):
        span = find_span(knots, degree, x)
        basis = basis_funs(knots, degree, x, span)
        mat[i, js(span)] = basis
    return mat


def breakpoints(knots, degree):
    """Return the unique element breakpoints associated with a knot vector."""
    return np.unique(knots[degree:-degree])


def greville(knots, degree, periodic):
    """Return Greville abscissae for an open or periodic B-spline basis."""
    T = knots
    p = degree
    s = 1 + p // 2 if periodic else 1
    n = len(T) - 2 * p - 1 if periodic else len(T) - p - 1
    xg = np.around([sum(T[i:i + p]) / p for i in range(s, s + n)], decimals=15)
    if periodic:
        a = T[p]
        b = T[-p]
        xg = np.around((xg - a) % (b - a) + a, decimals=15)
    return xg


def elements_spans(knots, degree):
    """Return the active knot span associated with every nonempty element."""
    breaks = breakpoints(knots, degree)
    nk = len(knots)
    ne = len(breaks) - 1
    spans = np.zeros(ne, dtype=int)
    ie = 0
    for ik in range(degree, nk - degree):
        if knots[ik] != knots[ik + 1]:
            spans[ie] = ik
            ie += 1
        if ie == ne:
            break
    return spans


def make_knots(breaks, degree, periodic):
    """Build an open-clamped or periodic knot vector from element breakpoints.

    Parameters follow the conventions used throughout the original IGA code:
    non-periodic vectors repeat each endpoint ``degree`` additional times,
    whereas periodic vectors extend the breakpoint sequence across the period.
    """
    assert isinstance(degree, int)
    assert isinstance(periodic, bool)
    assert len(breaks) > 1
    assert all(np.diff(breaks) > 0)
    assert degree > 0
    if periodic:
        assert len(breaks) > degree
    p = degree
    T = np.zeros(len(breaks) + 2 * p)
    T[p:-p] = breaks
    if periodic:
        period = breaks[-1] - breaks[0]
        T[0:p] = [xi - period for xi in breaks[-p - 1:-1]]
        T[-p:] = [xi + period for xi in breaks[1:p + 1]]
    else:
        T[0:p] = breaks[0]
        T[-p:] = breaks[-1]
    return T


def quadrature_grid(breaks, quad_rule_x, quad_rule_w):
    """Map a reference Gauss rule from ``[-1, 1]`` to every spatial element."""
    assert len(breaks) >= 2
    assert len(quad_rule_x) == len(quad_rule_w)
    assert min(quad_rule_x) >= -1
    assert max(quad_rule_x) <= 1
    quad_rule_x = np.asarray(quad_rule_x)
    quad_rule_w = np.asarray(quad_rule_w)
    ne = len(breaks) - 1
    nq = len(quad_rule_x)
    quad_x = np.zeros((ne, nq))
    quad_w = np.zeros((ne, nq))
    for ie, (a, b) in enumerate(zip(breaks[:-1], breaks[1:])):
        c0 = 0.5 * (a + b)
        c1 = 0.5 * (b - a)
        quad_x[ie, :] = c1 * quad_rule_x[:] + c0
        quad_w[ie, :] = c1 * quad_rule_w[:]
    return quad_x, quad_w


def basis_ders_on_quad_grid(knots, degree, quad_grid, nders, normalize=False):
    """Evaluate B-spline values and first derivatives at element quadrature nodes.

    ``normalize`` is retained for compatibility with earlier notebook calls;
    this implementation preserves the physical-coordinate derivatives used by
    the assembly routines.
    """
    ne, nq = quad_grid.shape
    basis = np.zeros((ne, degree + 1, nders + 1, nq))
    for ie in range(ne):
        for iq in range(nq):
            x = quad_grid[ie, iq]
            span = find_span(knots, degree, x)
            vals = basis_funs(knots, degree, x, span)
            basis[ie, :, 0, iq] = vals
            if nders >= 1:
                ders = basis_funs_1st_der(knots, degree, x, span)
                basis[ie, :, 1, iq] = ders
    return basis
