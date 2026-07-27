r"""Diagnostic figures: the plots that decide whether a spectrum may be believed.

:mod:`mmar.viz.spectra` draws the *result*. This module draws the *evidence*:
the residual structure a high :math:`R^2` conceals, the scale window that was
selected and the one that was not, the null distribution a test statistic is
being judged against, and the time locations of the singular events that the
:math:`f(\alpha)` curve only counts.

The organising idea is that every scalar this library reports is a slope, a
tail probability, or a Legendre transform of one of those, and each has a
canonical picture in which its failure mode is obvious:

============================  ==================================================
Reported quantity             Picture in which it fails visibly
============================  ==================================================
:math:`\hat\zeta(q)`, R^2     residuals against :math:`\log\delta` (curvature)
scaling range                 the log--log family with the window shaded
:math:`p`-value               the null histogram with the observation marked
:math:`f(\alpha)` width       :math:`h(t)` under the series (is it one event?)
============================  ==================================================

matplotlib is optional (``pip install 'mmar[viz]'``) and imported lazily. No
function calls ``plt.show()``.

References
----------
Anscombe, F.J. (1973). Graphs in statistical analysis. *Am. Stat.* **27**, 17.
(The four datasets share a regression, an :math:`R^2` and a residual variance,
and differ entirely in their residual plots. That is this module's premise.)

Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., Farmer, J.D. (1992).
Testing for nonlinearity in time series: the method of surrogate data.
*Physica D* **58**, 77--94.

Peltier, R.F. and Lévy Véhel, J. (1995). Multifractional Brownian motion.
*INRIA Research Report* 2645.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from mmar.core.regression import loglog_fit
from mmar.core.types import (
    FitMethod,
    FloatArray,
    LogLogFit,
    MultifractalReport,
    PointwiseHolder,
    ScalingRange,
    StructureFunctions,
)
from mmar.core.validate import validate_measure
from mmar.viz.spectra import (
    _ANNOTATION_BOX,
    _NULL_COLOUR,
    _REFERENCE_COLOUR,
    CVD_PALETTE,
    _axes,
    _colour,
    _pyplot,
    _range_mask,
    new_axes,
    plot_exponents,
    plot_hq,
    plot_spectrum,
    plot_structure_functions,
    use_style,
)

if TYPE_CHECKING:  # pragma: no cover - types only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "multifractal_dashboard",
    "plot_cascade",
    "plot_fit_residuals",
    "plot_pointwise_holder",
    "plot_scaling_range_selection",
    "plot_surrogate_null",
    "use_style",
]

_SINGULAR_QUANTILE: Final = 0.05
"""Default fraction of the sample flagged as locally roughest by :func:`plot_pointwise_holder`."""


def _figure(figsize: tuple[float, float]) -> Figure:
    return cast("Figure", _pyplot().figure(figsize=figsize))


# --------------------------------------------------------------------------- #
# Regression diagnostics
# --------------------------------------------------------------------------- #


def plot_fit_residuals(fit: LogLogFit, ax: Axes | None = None) -> Axes:
    r"""Residuals of a log--log fit against :math:`\log\delta`.

    .. math::
        r_j \;=\; \log y_j - \big(\hat\beta\log\delta_j + \hat c\big).

    A power law is the hypothesis that :math:`\mathbb E[r_j]=0` *uniformly in
    scale*. The :math:`R^2` of a scaling fit is almost always above 0.99 —
    :math:`\log\delta` spans decades, so the fit explains an enormous total
    variance even when the relation is visibly curved — and it is therefore
    nearly useless as a check. The residual plot is the check: a smile or a
    frown is a crossover between two scaling regimes, a monotone drift is a
    mis-specified window, and a single outlier at the coarse end is the
    small-window-count artefact that weighting is meant to suppress.

    A least-squares quadratic in :math:`\log\delta` is overlaid when at least
    four scales are present. Its curvature coefficient is printed: compare it
    with the residual standard deviation, since a curvature that is small
    relative to the scatter is not evidence of a crossover.

    Raises
    ------
    ValueError
        If the fit carries no stored residuals (``LogLogFit`` built by hand
        without ``log_x``/``residuals``).
    """
    ax = _axes(ax, figsize=(6.0, 3.2))
    lx = np.asarray(fit.log_x, dtype=np.float64)
    r = np.asarray(fit.residuals, dtype=np.float64)
    if lx.size == 0 or r.size != lx.size:
        raise ValueError(
            "this LogLogFit carries no residuals to plot (log_x is empty or its length "
            "disagrees with residuals); refit through mmar.core.regression.loglog_fit, "
            "which always stores them"
        )

    ax.axhline(0.0, color=_NULL_COLOUR, linewidth=1.0)
    ax.plot(lx, r, color=_colour(0), marker="o", markersize=4.0, linestyle="none")

    note = [
        rf"slope = {fit.slope:.4g} $\pm$ {fit.slope_stderr:.2g}",
        rf"$R^2$ = {fit.r_squared:.5f},  $n$ = {fit.n_points}",
        rf"$s_r$ = {float(np.std(r, ddof=min(2, max(r.size - 1, 0)))):.3g}",
    ]
    if lx.size >= 4:
        coeffs = np.polyfit(lx, r, 2)
        grid = np.linspace(float(lx.min()), float(lx.max()), 200)
        ax.plot(
            grid,
            np.polyval(coeffs, grid),
            color=_REFERENCE_COLOUR,
            linestyle="--",
            linewidth=1.1,
        )
        note.append(rf"quadratic curvature = {2.0 * float(coeffs[0]):.3g}")
    else:
        # Keep the drawn-line count independent of n: an invisible placeholder
        # would be dishonest, so the note states why the trend is absent instead.
        note.append("quadratic trend needs 4+ scales")

    ax.text(
        0.02,
        0.04,
        "\n".join(note),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox=_ANNOTATION_BOX,
    )
    ax.set_xlabel(r"$\log \delta$")
    ax.set_ylabel("residual")
    ax.set_title(f"log--log residuals ({fit.method.value})")
    return ax


def plot_scaling_range_selection(
    structure: StructureFunctions,
    selection: ScalingRange,
    ax: Axes | None = None,
    *,
    q: float = 2.0,
) -> Axes:
    r"""One row of the scaling surface with the selected window shaded and fitted.

    Draws :math:`\log S(q,\delta)` against :math:`\log\delta` for a single
    moment order (:math:`q=2` by default, the order that defines the classical
    Hurst exponent), shades
    :math:`[\delta_{\min},\delta_{\max}]` as chosen by
    :func:`mmar.core.regression.select_scaling_range`, and draws the fitted line
    **solid inside the window and dashed outside it**. The extrapolation is the
    point of the figure: if the dashed continuation tracks the data the window
    was conservative, and if it peels away the excluded scales genuinely belong
    to another regime.

    ``selection.decades`` is annotated. Below roughly one decade a power-law
    claim is weak whatever the :math:`R^2` says, because two points and a slope
    always fit.
    """
    ax = _axes(ax, figsize=(6.0, 3.6))
    scales = np.asarray(structure.scales, dtype=np.float64)
    row_idx = int(np.argmin(np.abs(np.asarray(structure.q, dtype=np.float64) - q)))
    q_i = float(structure.q[row_idx])
    s = np.asarray(structure.values[row_idx], dtype=np.float64)
    good = np.isfinite(s) & (s > 0.0)
    inside = _range_mask(scales, selection) & good
    if int(inside.sum()) < 3:
        raise ValueError(
            f"the selected range keeps {int(inside.sum())} usable scales at q = {q_i:g}; "
            "at least three are needed to draw the fit that produced the exponent"
        )

    ax.axvspan(
        selection.s_min, selection.s_max, color="#000000", alpha=0.07, linewidth=0.0, zorder=0
    )
    ax.plot(
        scales[good],
        s[good],
        color=_colour(0),
        marker="o",
        markersize=4.0,
        linestyle="none",
        label=f"$S({q_i:g}, \\delta)$",
    )
    weights = (
        None if structure.counts is None else np.asarray(structure.counts, dtype=np.float64)[inside]
    )
    fit = loglog_fit(
        scales[inside],
        s[inside],
        method=FitMethod.OLS if weights is None else FitMethod.WLS,
        weights=weights,
        scaling_range=selection,
    )
    full = scales[good]
    ax.plot(
        full,
        np.exp(fit.predict(full)),
        color=_REFERENCE_COLOUR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="fit, extrapolated",
    )
    ax.plot(
        scales[inside],
        np.exp(fit.predict(scales[inside])),
        color=_REFERENCE_COLOUR,
        linestyle="-",
        linewidth=1.6,
        label=rf"fit, slope = {fit.slope:.4g}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"scale $\delta$")
    ax.set_ylabel(rf"$S({q_i:g},\delta)$")
    ax.set_title(
        f"scaling range: {selection.selection}, {selection.n_scales} scales, "
        f"{selection.decades:.2f} decades, $R^2$ = {fit.r_squared:.5f}"
    )
    ax.legend(loc="best")
    return ax


# --------------------------------------------------------------------------- #
# Surrogate / null distribution
# --------------------------------------------------------------------------- #

_NULL_KEYS: Final = (
    "null_distribution",
    "null",
    "null_statistics",
    "surrogate_statistics",
    "surrogates",
)
_STAT_KEYS: Final = ("statistic", "observed", "observed_statistic", "value")
_PVALUE_KEYS: Final = ("p_value", "pvalue", "p")
_NAME_KEYS: Final = ("name", "statistic_name", "test", "label")


def _field(obj: object, keys: Sequence[str]) -> Any:
    """First present attribute (or mapping key) among ``keys``; ``None`` if absent.

    Deliberately structural rather than nominal: the surrogate-test result type
    lives in :mod:`mmar.inference`, and a plotting module that imports it would
    couple rendering to inference. Anything exposing a null sample, an observed
    statistic and a :math:`p`-value can be drawn.
    """
    if isinstance(obj, Mapping):
        for k in keys:
            if k in obj:
                return obj[k]
        return None
    for k in keys:
        value = getattr(obj, k, None)
        if value is not None:
            return value
    return None


def plot_surrogate_null(test_result: object, ax: Axes | None = None, *, bins: int = 40) -> Axes:
    r"""The null distribution as a histogram, with the observed statistic marked.

    A surrogate test compares an observed statistic :math:`T(x)` against its
    distribution under a null generated by :math:`m` surrogates that preserve
    everything except the property under test (Theiler et al. 1992). The
    :math:`p`-value alone hides three things this figure shows: whether the null
    is unimodal, whether the observation sits in the bulk or beyond every
    surrogate, and how coarse the resolution is.

    That last point is annotated explicitly. With :math:`m` surrogates the
    smallest attainable one-sided :math:`p`-value is
    :math:`1/(m+1)` — the correct estimator being
    :math:`\hat p = (1+\#\{T_i \ge T\})/(m+1)` (Davison & Hinkley 1997) — so
    ``p = 0.001`` from 99 surrogates is not a small :math:`p`-value, it is a
    rounding of "off the end of the null".

    Parameters
    ----------
    test_result
        Any object or mapping exposing a null sample (``null_distribution``,
        ``null``, ``surrogates``, ...), an observed statistic (``statistic``,
        ``observed``, ...) and optionally a :math:`p`-value (``p_value``).

    Raises
    ------
    TypeError
        If the null sample or the observed statistic cannot be located. The
        attribute names searched are named in the message.
    """
    ax = _axes(ax, figsize=(5.5, 3.4))
    raw_null = _field(test_result, _NULL_KEYS)
    raw_stat = _field(test_result, _STAT_KEYS)
    if raw_null is None or raw_stat is None:
        raise TypeError(
            f"{type(test_result).__name__} does not expose a surrogate null: looked for a "
            f"sample in {list(_NULL_KEYS)} and an observed statistic in {list(_STAT_KEYS)}. "
            "Pass the test result object, or a mapping with those keys."
        )
    null = np.asarray(raw_null, dtype=np.float64).ravel()
    null = null[np.isfinite(null)]
    if null.size < 2:
        raise ValueError(f"the null sample has {null.size} finite values; nothing to histogram")
    observed = float(np.asarray(raw_stat, dtype=np.float64).ravel()[0])

    ax.hist(
        null,
        bins=min(bins, max(5, null.size // 5)),
        color=_NULL_COLOUR,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(observed, color=CVD_PALETTE[1], linewidth=1.8)

    raw_p = _field(test_result, _PVALUE_KEYS)
    m = int(null.size)
    # The plug-in estimator, used only when the caller supplied no p-value.
    p = float(raw_p) if raw_p is not None else float((1 + int(np.sum(null >= observed))) / (m + 1))
    name = _field(test_result, _NAME_KEYS)
    lines = [
        rf"observed $T$ = {observed:.4g}",
        rf"$p$ = {p:.4g}",
        rf"$m$ = {m} surrogates, min attainable $p$ = {1.0 / (m + 1):.3g}",
        f"null mean {float(np.mean(null)):.4g}, sd {float(np.std(null, ddof=1)):.3g}",
    ]
    # TestResult.name defaults to "": inserting it would open the box with a blank line.
    if isinstance(name, str) and name.strip():
        lines.insert(0, name)
    ax.text(
        0.98,
        0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox=_ANNOTATION_BOX,
    )
    ax.set_xlabel("statistic under the surrogate null")
    ax.set_ylabel("count")
    ax.set_title("surrogate null distribution")
    return ax


# --------------------------------------------------------------------------- #
# Pointwise regularity
# --------------------------------------------------------------------------- #


def plot_pointwise_holder(
    pointwise: PointwiseHolder,
    *,
    x: FloatArray | None = None,
    ax: Axes | None = None,
    threshold: float | None = None,
) -> Figure:
    r"""The series above, its pointwise Hölder exponent :math:`h(t)` below.

    :class:`~mmar.core.types.HolderSpectrum` is the *distribution* of exponents;
    :class:`~mmar.core.types.PointwiseHolder` is the *field*. This figure is the
    payoff of estimating the field: the singular events acquire dates. A width
    :math:`\Delta\alpha` produced by three crashes is a different object from
    the same width produced by regularity that varies smoothly all year, and
    :math:`f(\alpha)` cannot tell them apart because it has integrated time out.

    Points with :math:`h(t)` below ``threshold`` are highlighted on both panels.
    The default threshold is the empirical ``5%`` quantile of the finite
    exponents — a *descriptive* cut, not a test: there is no null distribution
    for a pointwise exponent here, and calling the flagged points "significant"
    would be unwarranted.

    Parameters
    ----------
    x
        The series itself, aligned with ``pointwise.t``. When ``None`` the top
        panel is omitted from the data but the panel is still created, so the
        figure geometry does not depend on the argument.
    ax
        Axes for the *series*; the :math:`h(t)` panel is appended below it with
        a shared time axis. A new two-panel figure is created when ``None``.

    Returns
    -------
    Figure
        The parent figure, because the plot is inherently two-panel.
    """
    if ax is None:
        ax = new_axes(figsize=(7.0, 4.4))
    # Imported after the axes exist so that a missing matplotlib surfaces as the
    # library's own ImportError, naming the extra, rather than as a bare failure here.
    from mpl_toolkits.axes_grid1 import (  # type: ignore[import-untyped]  # noqa: PLC0415
        make_axes_locatable,
    )

    fig = cast("Figure", ax.get_figure())
    ax_h = cast(
        "Axes",
        make_axes_locatable(ax).append_axes("bottom", size="80%", pad=0.12, sharex=ax),
    )

    t = np.asarray(pointwise.t, dtype=np.float64)
    h = np.asarray(pointwise.h, dtype=np.float64)
    finite = np.isfinite(h)
    if not finite.any():
        raise ValueError("every pointwise Hölder exponent is non-finite; nothing to draw")
    cut = float(np.quantile(h[finite], _SINGULAR_QUANTILE)) if threshold is None else threshold
    singular = finite & (h < cut)

    if x is not None:
        xa = np.asarray(x, dtype=np.float64)
        if xa.shape != t.shape:
            raise ValueError(f"x{xa.shape} must align with pointwise.t{t.shape}")
        ax.plot(t, xa, color="#333333", linewidth=0.8)
        if singular.any():
            ax.plot(
                t[singular],
                xa[singular],
                color=CVD_PALETTE[1],
                marker="o",
                markersize=3.0,
                linestyle="none",
            )
    ax.set_ylabel("$X(t)$")
    ax.set_title(f"pointwise regularity ({pointwise.method})")
    ax.tick_params(labelbottom=False)

    if pointwise.stderr is not None:
        se = np.asarray(pointwise.stderr, dtype=np.float64)
        if se.shape == h.shape:
            ax_h.fill_between(t, h - se, h + se, color=_colour(0), alpha=0.16, linewidth=0.0)
    ax_h.plot(t, h, color=_colour(0), linewidth=0.9)
    ax_h.axhline(cut, color=_NULL_COLOUR, linestyle="--", linewidth=1.0)
    ax_h.plot(
        t[singular],
        h[singular],
        color=CVD_PALETTE[1],
        marker="o",
        markersize=3.0,
        linestyle="none",
        label=f"$h < {cut:.3g}$  ({int(singular.sum())} points)",
    )
    ax_h.set_xlabel("$t$")
    ax_h.set_ylabel("$h(t)$")
    ax_h.legend(loc="lower right")
    return fig


# --------------------------------------------------------------------------- #
# Cascade measure
# --------------------------------------------------------------------------- #


def plot_cascade(
    measure: FloatArray,
    *,
    ax: Axes | None = None,
    b: int = 2,
    max_bars: int = 512,
    log_y: bool = False,
) -> Axes:
    r"""The classic multiplicative-cascade bar plot.

    A :math:`b`-adic cascade measure on :math:`[0,1]` is drawn as
    :math:`n=b^{k}` contiguous bars of width :math:`\delta=1/n` and height
    :math:`\mu_i`. The visual signature of multifractality is that the mass is
    not merely uneven but *hierarchically* uneven: the same alternation of
    dense and sparse appears at every magnification, which is the geometric
    content of :math:`\mu_{i_1\ldots i_k}=\prod_j M_{i_1\ldots i_j}`.

    When the measure has more cells than ``max_bars`` it is coarse-grained by
    repeatedly summing blocks of :math:`b` neighbouring cells. That is not
    downsampling: summing adjacent cells is exactly the box partition at the
    coarser resolution :math:`\delta'=b^{-(k-j)}`, so the drawn object is the
    same measure viewed at a lower resolution, and the total mass is preserved
    to machine precision.

    ``log_y=True`` is the right choice for cascades with a strong intermittency
    parameter, where the largest cell can exceed the median by many orders of
    magnitude and a linear axis shows one spike and a flat line.
    """
    ax = _axes(ax, figsize=(7.0, 2.8))
    mu = validate_measure(measure, name="cascade measure", normalise=False)
    if b < 2:
        raise ValueError(f"b must be at least 2, got {b}")

    level = 0
    while mu.size > max_bars and mu.size % b == 0:
        mu = np.asarray(mu.reshape(-1, b).sum(axis=1), dtype=np.float64)
        level += 1

    n = mu.size
    width = 1.0 / n
    edges = np.arange(n, dtype=np.float64) * width
    ax.bar(edges, mu, width=width, align="edge", color=_colour(0), linewidth=0.0)
    if log_y:
        positive = mu[mu > 0.0]
        if positive.size == 0:  # pragma: no cover - validate_measure forbids zero total mass
            raise ValueError("cannot use a logarithmic axis: every cell is empty")
        ax.set_yscale("log")
        ax.set_ylim(float(positive.min()) * 0.5, float(mu.max()) * 2.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\mu_i$")
    coarse = f", coarse-grained {level} level(s)" if level else ""
    if n > max_bars:
        # The loop can only halt early because n stopped being divisible by b, i.e.
        # the measure is not b-adic. Say so rather than quietly drawing 2187 bars
        # after being asked for 100.
        coarse += f"; cap of {max_bars} not reached ({n} is not a power of {b})"
    ax.set_title(rf"cascade measure: {n} cells of width $\delta$ = 1/{n}, $b$ = {b}{coarse}")
    return ax


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


def _blank(ax: Axes, message: str) -> None:
    """Say why a panel is empty instead of leaving an unexplained white box."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _summary_table(ax: Axes, report: MultifractalReport) -> None:
    ax.axis("off")
    rows: list[tuple[str, str]] = []
    for k, v in report.summary().items():
        if isinstance(v, float):
            rows.append((k, f"{v:.5g}"))
        elif isinstance(v, str):
            rows.append((k, v if len(v) <= 24 else v[:21] + "..."))
        elif isinstance(v, int):
            rows.append((k, str(v)))
        else:
            rows.append((k, str(v)[:24]))
    table = ax.table(cellText=rows, colLabels=("quantity", "value"), loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax.set_title("summary")


def multifractal_dashboard(report: MultifractalReport, *, fig: Figure | None = None) -> Figure:
    r"""Six panels: the whole analysis in one call.

    ``(1)`` structure functions with the fitted window shaded, ``(2)``
    :math:`\zeta(q)` against the monofractal line, ``(3)`` :math:`h(q)`,
    ``(4)`` :math:`f(\alpha)` with its descriptors, ``(5)`` the log--log
    residuals of the fit nearest :math:`q=2`, ``(6)`` the scalar summary
    together with the provenance root.

    Panels 1 and 5 are drawn only if the report retained the raw scaling
    surface and the per-:math:`q` fits; when it did not, the panel says so
    rather than being silently dropped, because "the estimator did not keep its
    evidence" is itself a finding about the run.

    The provenance short root is printed in the summary panel. Two dashboards
    that carry the same root were produced by the same computation, which is
    the point of the whole audit layer.

    Parameters
    ----------
    fig
        An **empty** figure to populate with a 2x3 grid; a new one of size
        ``(13.5, 8)`` is created when ``None``.
    """
    fig = _figure((13.5, 8.0)) if fig is None else fig
    axes = [cast("Axes", a) for a in np.ravel(fig.subplots(2, 3))]
    exps = report.exponents

    if exps.structure is not None:
        plot_structure_functions(exps.structure, ax=axes[0], scaling_range=exps.scaling_range)
    else:
        _blank(axes[0], "no structure functions retained\n(estimator did not keep S(q, delta))")

    plot_exponents(exps, ax=axes[1])
    plot_hq(exps, axes[2])
    plot_spectrum(report.spectrum, ax=axes[3])

    if exps.fits:
        aligned = len(exps.fits) == exps.q.size
        idx = int(np.argmin(np.abs(exps.q - 2.0))) if aligned else 0
        plot_fit_residuals(exps.fits[idx], axes[4])
        # Only claim a q when fits[i] provably corresponds to q[i]. With a
        # mismatched length the correspondence is unknown, and printing q[0]
        # would attach a fabricated label to a real regression.
        axes[4].set_title(
            f"residuals at $q$ = {exps.q[idx]:g}"
            if aligned
            else f"residuals, fit 1 of {len(exps.fits)} ($q$ unknown: "
            f"{len(exps.fits)} fits for {exps.q.size} moment orders)"
        )
    else:
        _blank(axes[4], "no per-q fits retained")

    _summary_table(axes[5], report)
    root = report.provenance.short_root if report.provenance is not None else "unrecorded"
    fig.text(0.995, 0.005, f"provenance root: {root}", ha="right", va="bottom", fontsize=7)
    fig.suptitle(f"multifractal report — {exps.estimator}", x=0.01, ha="left")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    return fig
