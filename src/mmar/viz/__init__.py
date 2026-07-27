r"""Publication-quality figures for the multifractal formalism.

Two modules:

:mod:`mmar.viz.spectra`
    The results — :math:`f(\alpha)`, :math:`\zeta(q)`/:math:`\tau(q)`,
    :math:`h(q)`, the structure-function family, and estimator comparisons.

:mod:`mmar.viz.diagnostics`
    The evidence — log--log residuals, the selected scaling range, surrogate
    nulls, the pointwise Hölder field, cascade measures, and a six-panel
    dashboard.

Contract obeyed by every function here
--------------------------------------
* takes ``ax: Axes | None`` (or ``fig: Figure | None`` for the multi-panel
  figures) and creates one when it is ``None``;
* returns the :class:`~matplotlib.axes.Axes` it drew on, or the
  :class:`~matplotlib.figure.Figure` when the plot is inherently multi-panel;
* **never** calls ``plt.show()`` — display is the caller's decision, and calling
  it inside a library makes the function unusable in a script and untestable;
* never mutates global rcParams unless :func:`~mmar.viz.spectra.use_style` is
  called explicitly with ``permanent=True``.

matplotlib is an optional dependency. Importing this package is safe without
it: neither submodule imports matplotlib at module scope, and every entry point
imports it lazily and raises an :exc:`ImportError` naming the extra
(``pip install 'mmar[viz]'``) if it is absent.
"""

from __future__ import annotations

from mmar.viz.diagnostics import (
    multifractal_dashboard,
    plot_cascade,
    plot_fit_residuals,
    plot_pointwise_holder,
    plot_scaling_range_selection,
    plot_surrogate_null,
)
from mmar.viz.spectra import (
    CVD_PALETTE,
    HOUSE_STYLE,
    plot_comparison,
    plot_exponents,
    plot_hq,
    plot_spectrum,
    plot_structure_functions,
    use_style,
)

__all__ = [
    "CVD_PALETTE",
    "HOUSE_STYLE",
    "multifractal_dashboard",
    "plot_cascade",
    "plot_comparison",
    "plot_exponents",
    "plot_fit_residuals",
    "plot_hq",
    "plot_pointwise_holder",
    "plot_scaling_range_selection",
    "plot_spectrum",
    "plot_structure_functions",
    "plot_surrogate_null",
    "use_style",
]
