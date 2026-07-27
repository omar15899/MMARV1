r"""Publication figures for scaling exponents and singularity spectra.

Every function here is a *renderer*, never an estimator: it draws what it is
given and computes nothing that changes a reported number. The one exception is
the refitting of a straight line inside :func:`plot_structure_functions`, which
is done through :func:`mmar.core.regression.loglog_fit` so that the line drawn
is the same object the estimator would have fitted, not a second implementation.

Three rules the module enforces because breaking them is how multifractal
figures mislead:

1. **A scaling plot always shows its fitted window.** A log--log family of
   :math:`S(q,\delta)` without the selected scaling range shaded is a picture,
   not evidence: the reader cannot tell whether the slope was taken across a
   crossover. :func:`plot_structure_functions` shades the window.
2. **A** :math:`\zeta(q)` **plot always shows the monofractal line**
   :math:`q\,h(2)`. Multifractality *is* the departure of :math:`\zeta` from
   that straight line; a curve shown alone is unreadable, because the eye cannot
   judge curvature of order :math:`10^{-2}` against nothing.
3. **Nothing calls** ``plt.show()``. Display is the caller's decision. Showing
   inside a library function makes it unusable in a script, in a notebook that
   composes panels, and in a test.

matplotlib is an optional dependency (``pip install 'mmar[viz]'``); it is
imported lazily inside each entry point so that importing :mod:`mmar` never
requires it.

Colour
------
:data:`CVD_PALETTE` is the Okabe--Ito colour-blind-safe qualitative palette
(Okabe & Ito 2008), reordered so that every *adjacent* pair keeps a perceptual
separation of :math:`\Delta E \ge 9.6` under deuteranopia and
:math:`\ge 8.5` under tritanopia on a white surface. Series identity is never
carried by colour alone: every multi-series plot also emits a legend.

References
----------
Halsey, T., Jensen, M., Kadanoff, L., Procaccia, I., Shraiman, B. (1986).
Fractal measures and their singularities. *Phys. Rev. A* **33**, 1141.

Okabe, M. and Ito, K. (2008). Color universal design: how to make figures and
presentations that are friendly to colorblind people.

Tufte, E. (1983). *The Visual Display of Quantitative Information.*
Graphics Press. (The grid is recessive here for the reason given on p. 92.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from mmar.core.regression import loglog_fit
from mmar.core.types import (
    Convention,
    FitMethod,
    FloatArray,
    HolderSpectrum,
    ScalingExponents,
    ScalingRange,
    StructureFunctions,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only exists for the type checker
    from matplotlib.axes import Axes

__all__ = [
    "CVD_PALETTE",
    "HOUSE_STYLE",
    "plot_comparison",
    "plot_exponents",
    "plot_hq",
    "plot_spectrum",
    "plot_structure_functions",
    "use_style",
]

BoolArray = NDArray[np.bool_]

CVD_PALETTE: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
)
"""Okabe--Ito qualitative palette, ordered for maximal adjacent-pair CVD separation."""

_REFERENCE_COLOUR: Final = "#4D4D4D"
"""Neutral grey for a theoretical or null curve: it must never read as a series."""

_NULL_COLOUR: Final = "#8C8C8C"

HOUSE_STYLE: Final[Mapping[str, Any]] = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "errorbar.capsize": 2.0,
}
"""The house rcParams. Applied only when :func:`use_style` is called explicitly."""

_ANNOTATION_BOX: Final[dict[str, Any]] = {
    "boxstyle": "round,pad=0.35",
    "facecolor": "white",
    "edgecolor": "#BFBFBF",
    "linewidth": 0.6,
    "alpha": 0.9,
}

_INSTALL_HINT: Final = (
    "matplotlib is required by mmar.viz but is not installed. Install the optional "
    "extra:  pip install 'mmar[viz]'   (or 'mmar[full]' for every optional feature)."
)


# --------------------------------------------------------------------------- #
# Lazy backend access
# --------------------------------------------------------------------------- #


def _pyplot() -> ModuleType:
    """Import :mod:`matplotlib.pyplot`, naming the extra if it is absent.

    Deferred to call time on purpose: :mod:`mmar` must import, and every
    estimator must run, in an environment without matplotlib.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415 - optional dependency, probed lazily
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(_INSTALL_HINT) from exc
    return plt


def new_axes(*, figsize: tuple[float, float] = (6.0, 4.0)) -> Axes:
    """A fresh single-panel Axes on its own Figure."""
    _, ax = _pyplot().subplots(figsize=figsize)
    return cast("Axes", ax)


def _axes(ax: Axes | None, *, figsize: tuple[float, float] = (6.0, 4.0)) -> Axes:
    return new_axes(figsize=figsize) if ax is None else ax


def use_style(*, permanent: bool = False) -> AbstractContextManager[None]:
    r"""The house style: serif mathtext, recessive grid, CVD-safe colour cycle.

    Returns a context manager, so the style is scoped by default::

        with use_style():
            ax = plot_spectrum(spectrum)

    A library must not mutate global rcParams as a side effect of being
    imported — that silently restyles every unrelated figure in the caller's
    session. Pass ``permanent=True`` to opt into a process-wide change; the
    returned context manager is then a no-op.

    The palette was validated against a white (print) surface. On a dark
    surface the orange, purple and sky-blue steps leave the recommended
    lightness band and should be re-stepped before use.
    """
    plt = _pyplot()
    style = {**HOUSE_STYLE, "axes.prop_cycle": plt.cycler(color=list(CVD_PALETTE))}
    if permanent:
        import matplotlib as mpl  # noqa: PLC0415 - already proven importable by _pyplot

        # rcParams is typed with a Literal key union; the style mapping is validated
        # by matplotlib itself on assignment, so the widening is safe.
        mpl.rcParams.update(cast("Any", style))
        from contextlib import nullcontext  # noqa: PLC0415 - local to the branch

        return nullcontext()
    return cast("AbstractContextManager[None]", plt.rc_context(style))


def _colour(index: int) -> str:
    """Cycle position ``index`` of the palette, wrapping only past its length."""
    return CVD_PALETTE[index % len(CVD_PALETTE)]


def _finite_pair(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Drop points where either coordinate is non-finite, preserving order.

    :math:`f(\alpha)=-\infty` outside the support is the *exact* value of a
    Legendre--Fenchel transform, not missing data (see
    :func:`mmar.models.theory.cascade_falpha`); it simply cannot be drawn.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    return np.asarray(x[ok], dtype=np.float64), np.asarray(y[ok], dtype=np.float64)


def _sorted_by_x(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    order = np.argsort(x, kind="stable")
    return np.asarray(x[order], dtype=np.float64), np.asarray(y[order], dtype=np.float64)


def _reference_curve(
    reference: HolderSpectrum | tuple[FloatArray, FloatArray],
) -> tuple[FloatArray, FloatArray]:
    if isinstance(reference, HolderSpectrum):
        return reference.alpha, reference.f_alpha
    a, f = reference
    return np.asarray(a, dtype=np.float64), np.asarray(f, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Singularity spectrum
# --------------------------------------------------------------------------- #


def plot_spectrum(
    spectrum: HolderSpectrum,
    *,
    ax: Axes | None = None,
    reference: HolderSpectrum | tuple[FloatArray, FloatArray] | None = None,
    show_descriptors: bool = True,
    label: str | None = None,
    colour: str | None = None,
) -> Axes:
    r"""Draw the singularity spectrum :math:`\alpha \mapsto f(\alpha)`.

    The curve is the Legendre--Fenchel transform
    :math:`f(\alpha)=\inf_q\{q\alpha-\tau(q)\}` in the measure convention, so a
    space-filling support has :math:`\max_\alpha f(\alpha)=d`
    (:mod:`mmar.core.types`). Three things are drawn on top of it:

    * a shaded band :math:`f\pm\hat\sigma_f` when ``spectrum.f_stderr`` is
      populated. Note that the delta-method errors propagated by
      :func:`~mmar.core.legendre.spectrum_from_exponents` are a *lower* bound on
      the true uncertainty; a band from
      :mod:`mmar.inference.bootstrap` is the honest one;
    * a dashed reference curve (typically the closed-form spectrum from
      :mod:`mmar.models.theory`), drawn in neutral grey so it never reads as a
      competing estimate;
    * an annotation giving :math:`\alpha_0`, the width
      :math:`\Delta\alpha=\alpha_{\max}-\alpha_{\min}` and the asymmetry
      :math:`B=(\alpha_0-\alpha_{\min})/(\alpha_{\max}-\alpha_0)`, together with
      a vertical marker at :math:`\alpha_0`.

    Failure modes the figure makes visible rather than hiding: a spectrum whose
    apex sits away from :math:`f=d` has not been estimated over a wide enough
    :math:`q` window; a spectrum with a backward fold came from the parametric
    (tangent) construction on a non-concave :math:`\hat\zeta` and should be
    recomputed with ``method="fenchel"``.

    Parameters
    ----------
    spectrum
        The estimated spectrum.
    ax
        Target Axes; a new single-panel figure is created when ``None``.
    reference
        A :class:`~mmar.core.types.HolderSpectrum` or a raw
        ``(alpha, f)`` pair, drawn dashed.
    show_descriptors
        Draw the :math:`\alpha_0` marker and the descriptor box.

    Returns
    -------
    Axes
        The Axes drawn on. ``plt.show()`` is never called.
    """
    ax = _axes(ax)
    colour = colour or _colour(0)
    a, f = _sorted_by_x(*_finite_pair(spectrum.alpha, spectrum.f_alpha))
    if a.size == 0:
        raise ValueError(
            "the spectrum has no finite (alpha, f) pair to draw; f(alpha) = -inf "
            "everywhere means the q window resolved no Holder exponents at all"
        )

    if spectrum.f_stderr is not None:
        se = np.asarray(spectrum.f_stderr, dtype=np.float64)
        if se.shape == spectrum.f_alpha.shape:
            ok = np.isfinite(spectrum.alpha) & np.isfinite(spectrum.f_alpha) & np.isfinite(se)
            order = np.argsort(spectrum.alpha[ok], kind="stable")
            aa = spectrum.alpha[ok][order]
            ff = spectrum.f_alpha[ok][order]
            ss = se[ok][order]
            ax.fill_between(aa, ff - ss, ff + ss, color=colour, alpha=0.18, linewidth=0.0)

    ax.plot(a, f, color=colour, label=label or spectrum.method)

    if reference is not None:
        ra, rf = _sorted_by_x(*_finite_pair(*_reference_curve(reference)))
        ax.plot(ra, rf, color=_REFERENCE_COLOUR, linestyle="--", linewidth=1.2, label="reference")

    if show_descriptors:
        d = spectrum.descriptors()
        ax.axvline(d.alpha_0, color=colour, linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(
            0.02,
            0.98,
            "\n".join(
                (
                    rf"$\alpha_0$ = {d.alpha_0:.4g}",
                    rf"$\Delta\alpha$ = {d.width:.4g}",
                    rf"$B$ = {d.asymmetry:.4g}",
                    rf"$f_{{\max}}$ = {d.f_max:.4g}  ($d$ = {spectrum.d_support:g})",
                )
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=_ANNOTATION_BOX,
        )

    ax.set_xlabel(r"Hölder exponent $\alpha$")
    ax.set_ylabel(r"$f(\alpha)$")
    ax.set_title("singularity spectrum")
    if label or reference is not None:
        ax.legend(loc="upper right")
    return ax


def plot_comparison(
    spectra: Mapping[str, HolderSpectrum],
    ax: Axes | None = None,
    *,
    reference: HolderSpectrum | tuple[FloatArray, FloatArray] | None = None,
) -> Axes:
    r"""Overlay several estimators' spectra on one axis.

    Estimator disagreement is information, not noise to be averaged away: MFDFA,
    wavelet leaders and WTMM have different biases at the ends of the
    :math:`q` window, and a width :math:`\Delta\alpha` that varies by a factor
    of two across methods is evidence that the scaling assumption, not the
    arithmetic, is what is uncertain. The legend therefore carries each
    estimator's width, so the disagreement is quantified and not merely visible.

    Insertion order of the mapping is the drawing order and the colour order;
    pass a :class:`dict` if that matters.
    """
    ax = _axes(ax)
    if not spectra:
        raise ValueError("nothing to compare: the spectra mapping is empty")
    for i, (name, spec) in enumerate(spectra.items()):
        a, f = _sorted_by_x(*_finite_pair(spec.alpha, spec.f_alpha))
        d = spec.descriptors()
        ax.plot(a, f, color=_colour(i), label=rf"{name}  ($\Delta\alpha$ = {d.width:.3g})")
    if reference is not None:
        ra, rf = _sorted_by_x(*_finite_pair(*_reference_curve(reference)))
        ax.plot(ra, rf, color=_REFERENCE_COLOUR, linestyle="--", linewidth=1.2, label="reference")
    ax.set_xlabel(r"Hölder exponent $\alpha$")
    ax.set_ylabel(r"$f(\alpha)$")
    ax.set_title("estimator comparison")
    ax.legend(loc="lower center")
    return ax


# --------------------------------------------------------------------------- #
# Scaling exponents
# --------------------------------------------------------------------------- #


def plot_exponents(
    exponents: ScalingExponents,
    *,
    ax: Axes | None = None,
    convention: Convention = Convention.SIGNAL,
    reference: ScalingExponents | None = None,
    show_monofractal: bool = True,
    colour: str | None = None,
) -> Axes:
    r"""Draw :math:`\zeta(q)` (signal view) or :math:`\tau(q)` (measure view).

    The single most informative element of this figure is the straight line
    :math:`q\mapsto q\,h(2)`, drawn by default. For a monofractal
    :math:`\zeta(q)=qH` *exactly*, so :math:`\hat\zeta` lies on that line to
    within its error bars; **the vertical gap between the curve and the line is
    the multifractality**, and it is typically of order :math:`10^{-2}` — far
    too small to be judged from the curve alone, which always looks straight.

    Error bars are :math:`\pm\hat\sigma_{\zeta}(q)` from the log--log
    regressions. They are strongly correlated across :math:`q` (the same
    :math:`S(q,\delta)` surface feeds every fit), so overlapping bars do *not*
    imply that the difference between two exponents is insignificant; use the
    bootstrap in :mod:`mmar.inference` for that.

    Parameters
    ----------
    convention
        :attr:`~mmar.core.types.Convention.SIGNAL` plots :math:`\zeta`,
        :attr:`~mmar.core.types.Convention.MEASURE` plots
        :math:`\tau=\zeta-d`. The monofractal reference is offset by the same
        :math:`d`, so the comparison is convention-consistent.
    reference
        Closed-form exponents (e.g. :func:`mmar.models.theory.lognormal_exponents`),
        drawn dashed in neutral grey.
    """
    ax = _axes(ax)
    colour = colour or _colour(0)
    q = exponents.q
    y = exponents.exponent(convention)
    offset = 0.0 if convention is Convention.SIGNAL else -exponents.d_support
    symbol = r"\zeta" if convention is Convention.SIGNAL else r"\tau"

    yerr = None
    if exponents.zeta_stderr is not None:
        se = np.asarray(exponents.zeta_stderr, dtype=np.float64)
        # tau = zeta - d is a deterministic shift, so the standard error is shared.
        yerr = se if se.shape == q.shape else None

    ax.errorbar(
        q,
        y,
        yerr=yerr,
        color=colour,
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        elinewidth=0.9,
        label=exponents.estimator,
    )

    if show_monofractal:
        h2 = exponents.h2
        ax.plot(
            q,
            q * h2 + offset,
            color=_NULL_COLOUR,
            linestyle="-.",
            linewidth=1.0,
            label=rf"monofractal $q\,h(2)$, $h(2)$ = {h2:.4g}",
        )

    if reference is not None:
        ax.plot(
            reference.q,
            reference.exponent(convention),
            color=_REFERENCE_COLOUR,
            linestyle="--",
            linewidth=1.2,
            label=f"reference: {reference.estimator}",
        )

    ax.axhline(0.0, color="#D9D9D9", linewidth=0.8, zorder=0)
    ax.set_xlabel("$q$")
    ax.set_ylabel(rf"${symbol}(q)$")
    ax.set_title(f"scaling function, {convention.value} convention")
    ax.legend(loc="best")
    return ax


def plot_hq(
    exponents: ScalingExponents,
    ax: Axes | None = None,
    *,
    reference: ScalingExponents | None = None,
    colour: str | None = None,
) -> Axes:
    r"""Draw the generalised Hurst function :math:`h(q)=\zeta(q)/q`.

    Flat for a monofractal, by definition: :math:`\zeta(q)=qH \Rightarrow
    h\equiv H`. The horizontal reference is :math:`h(2)`, the classical Hurst
    exponent, so the plot reads as "how far is this process from the
    second-order Hurst description of itself".

    The point :math:`q=0` is a removable singularity of :math:`\zeta/q` and is
    left as NaN rather than filled by a limit
    (:math:`h(0^{\pm})=\zeta'(0)=\alpha_0`, which is a different estimator with
    a different variance); matplotlib renders it as a gap.

    Monotonicity is the diagnostic: :math:`h` must be non-increasing, because
    :math:`\zeta` is concave and :math:`\zeta(0)=0` make :math:`\zeta(q)/q` a
    chord slope from the origin. A fitted :math:`\hat h` that rises anywhere is
    evidence of a failed scaling range, not of an exotic process.
    """
    ax = _axes(ax)
    colour = colour or _colour(0)
    q = exponents.q
    h = exponents.hq

    yerr = None
    if exponents.zeta_stderr is not None:
        se = np.asarray(exponents.zeta_stderr, dtype=np.float64)
        if se.shape == q.shape:
            with np.errstate(divide="ignore", invalid="ignore"):
                yerr = np.where(q != 0.0, se / np.abs(q), np.nan)

    ax.errorbar(
        q,
        h,
        yerr=yerr,
        color=colour,
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        elinewidth=0.9,
        label=exponents.estimator,
    )
    h2 = exponents.h2
    ax.axhline(h2, color=_NULL_COLOUR, linestyle="-.", linewidth=1.0)
    ax.text(
        0.98,
        0.96,
        rf"$h(2)$ = {h2:.4g}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox=_ANNOTATION_BOX,
    )
    if reference is not None:
        ax.plot(
            reference.q,
            reference.hq,
            color=_REFERENCE_COLOUR,
            linestyle="--",
            linewidth=1.2,
            label=f"reference: {reference.estimator}",
        )
        ax.legend(loc="best")
    ax.set_xlabel("$q$")
    ax.set_ylabel("$h(q)$")
    ax.set_title("generalised Hurst function")
    return ax


# --------------------------------------------------------------------------- #
# Structure functions
# --------------------------------------------------------------------------- #


def _default_q_subset(q: FloatArray, n_show: int) -> FloatArray:
    """At most ``n_show`` moment orders, evenly spaced across the grid."""
    if q.size <= n_show:
        return q
    idx = np.unique(np.round(np.linspace(0, q.size - 1, n_show)).astype(int))
    return np.asarray(q[idx], dtype=np.float64)


def _range_mask(scales: FloatArray, scaling_range: ScalingRange | None) -> BoolArray:
    """Boolean selector for the fitted window.

    The explicit ``mask`` recorded by
    :func:`~mmar.core.regression.select_scaling_range` is authoritative when its
    length matches; otherwise the window is reconstructed from its endpoints,
    which is equivalent for a contiguous selection and is all the type promises.
    """
    if scaling_range is None:
        return np.ones(scales.size, dtype=np.bool_)
    if len(scaling_range.mask) == scales.size:
        return np.asarray(scaling_range.mask, dtype=np.bool_)
    inside = (scales >= scaling_range.s_min) & (scales <= scaling_range.s_max)
    return np.asarray(inside, dtype=np.bool_)


def plot_structure_functions(
    structure: StructureFunctions,
    *,
    ax: Axes | None = None,
    q_subset: Sequence[float] | FloatArray | None = None,
    scaling_range: ScalingRange | None = None,
    n_show: int = 5,
    show_fits: bool = True,
) -> Axes:
    r"""Log--log family :math:`\delta \mapsto S(q,\delta)`, with the fitted window shaded.

    .. math::
        S(q,\delta) \sim \delta^{\zeta(q)}
        \quad\Longleftrightarrow\quad
        \log S(q,\delta) = \zeta(q)\log\delta + \text{const}.

    The shaded band is the scale window the exponents were actually fitted over.
    Showing it is not decoration: a power law holds *on a window*, and the two
    commonest ways to manufacture a scaling exponent are to fit through a
    crossover and to include coarse scales at which the moment is an average of
    three windows. Neither is visible in an :math:`R^2`; both are obvious once
    the window is marked and the fit is extended across it.

    Parameters
    ----------
    structure
        The raw scaling surface. ``values[i, j]`` must be the moment
        :math:`S(q_i,\delta_j)` itself, not its logarithm.
    q_subset
        Moment orders to draw (nearest grid match). Defaults to ``n_show``
        orders evenly spaced across the grid — a full grid of 41 curves is
        unreadable and hides exactly the tail behaviour that matters.
    scaling_range
        The window selected by the estimator. When ``None`` the whole scale axis
        is the window and nothing is shaded, which is the honest rendering of
        "everything was fitted".
    show_fits
        Overlay the weighted log--log fit on each drawn curve. Weights are the
        window counts when ``structure.counts`` is present, since the variance
        of a log-moment estimated from :math:`n_j` windows scales as
        :math:`1/n_j`.

    Raises
    ------
    ValueError
        If a drawn row has fewer than three strictly positive, finite moments.
        Dropping such a row silently is how spurious power laws get published.
    """
    ax = _axes(ax, figsize=(6.0, 4.5))
    scales = np.asarray(structure.scales, dtype=np.float64)
    q_grid = np.asarray(structure.q, dtype=np.float64)
    wanted = (
        _default_q_subset(q_grid, n_show)
        if q_subset is None
        else np.asarray(q_subset, dtype=np.float64)
    )
    mask = _range_mask(scales, scaling_range)
    weights_all = (
        None if structure.counts is None else np.asarray(structure.counts, dtype=np.float64)
    )

    if scaling_range is not None:
        ax.axvspan(
            scaling_range.s_min,
            scaling_range.s_max,
            color="#000000",
            alpha=0.055,
            linewidth=0.0,
            zorder=0,
        )

    for i, q_val in enumerate(wanted):
        row_idx = int(np.argmin(np.abs(q_grid - q_val)))
        q_i = float(q_grid[row_idx])
        s = np.asarray(structure.values[row_idx], dtype=np.float64)
        good = np.isfinite(s) & (s > 0.0)
        if int(good.sum()) < 3:
            raise ValueError(
                f"row q = {q_i:g} has only {int(good.sum())} positive finite moments; "
                "a log-log fit is undefined there. Restrict the q grid or widen the scales."
            )
        col = _colour(i)
        ax.plot(
            scales[good],
            s[good],
            color=col,
            marker="o",
            markersize=3.0,
            linewidth=0.9,
            alpha=0.9,
            label=f"$q$ = {q_i:g}",
        )
        if not show_fits:
            continue
        fit_sel = good & mask
        if int(fit_sel.sum()) < 3:
            raise ValueError(
                f"row q = {q_i:g} keeps only {int(fit_sel.sum())} scales inside the "
                "selected scaling range; the window is too narrow to fit."
            )
        w = None if weights_all is None else weights_all[fit_sel]
        fit = loglog_fit(
            scales[fit_sel],
            s[fit_sel],
            method=FitMethod.OLS if w is None else FitMethod.WLS,
            weights=w,
        )
        xs = scales[fit_sel]
        ax.plot(xs, np.exp(fit.predict(xs)), color=col, linestyle="--", linewidth=1.3, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"scale $\delta$")
    ax.set_ylabel(r"$S(q,\delta)$")
    title = "structure functions"
    if scaling_range is not None:
        title += (
            f"  ·  fitted window [{scaling_range.s_min:g}, {scaling_range.s_max:g}]"
            f" ({scaling_range.decades:.2f} decades, {scaling_range.selection})"
        )
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    return ax
