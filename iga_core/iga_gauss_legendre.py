"""Gauss-Legendre quadrature rules used by the IGA assembly routines."""

import numpy as np


def gauss_legendre(ordergl, tol=10e-14):
    """Return a Gauss-Legendre rule on ``[-1, 1]``.

    ``ordergl`` follows the legacy convention: the returned rule contains
    ``ordergl + 1`` abscissas and weights.  Roots are found by Newton iteration
    using the supplied convergence tolerance.
    """
    m = ordergl + 1
    from math import cos, pi

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

    A = np.zeros(m)
    x = np.zeros(m)
    nRoots = (m + 1) // 2
    for i in range(nRoots):
        t = cos(pi * (i + 0.75) / (m + 0.5))
        for _ in range(30):
            p, dp = legendre(t, m)
            dt = -p / dp
            t = t + dt
            if abs(dt) < tol:
                x[i] = t
                x[m - i - 1] = -t
                A[i] = 2.0 / (1.0 - t**2) / (dp**2)
                A[m - i - 1] = A[i]
                break
    return x, A
