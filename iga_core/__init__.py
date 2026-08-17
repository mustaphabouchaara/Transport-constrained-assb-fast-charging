"""Public compatibility namespace for the battery-model IGA utilities.

Importing :mod:`iga_core` exposes the spline, quadrature, assembly, diffusion,
and electrochemical helpers historically used by the project notebooks.
"""

from .iga_bspline import (
    find_span,
    basis_funs,
    basis_funs_1st_der,
    basis_funs_all_ders,
    collocation_matrix,
    breakpoints,
    greville,
    elements_spans,
    make_knots,
    quadrature_grid,
    basis_ders_on_quad_grid,
)
from .iga_gauss_legendre import gauss_legendre
from .iga_assembly_core import (
    Stiffness_Matrix,
    Mass_Matrix,
    assemble_rhs,
    int_approx,
    NIA,
    L2_projection,
    plots_splines,
    plot_field_1d,
    sol_plot,
)
from .iga_diffusion_plotting import Diff_coef, varying_diff_coeff, plot_heatmap, plot_surface
from .iga_electrochemistry import (
    assemble_non_linear,
    Non_linear_jacobian,
    c_avg_var,
    dc_dx,
    dc_dx_all,
    E_field_e,
    E_e_field_matrix,
    eta_mt_profile_original,
    eta_mt_time_series_original,
    integrate_field_trapz,
)
