r"""Contract tests for :mod:`mmar.viz`.

What is asserted here is the *contract*, never the pixels: a renderer that
returns the right object, draws the number of marks it promised, creates
exactly one figure, and refuses to display anything by itself. Rendering
fidelity is not testable without a reference image, and image comparison is a
maintenance liability that catches font-version changes rather than bugs.

Three invariants are worth naming because breaking any of them makes the
library unusable rather than merely ugly:

* ``plt.show()`` is never called. Enforced by an autouse fixture that replaces
  it with a function that fails the test.
* One call, one figure. The autouse fixture empties the figure manager before
  each test, so ``plt.get_fignums()`` afterwards counts exactly what the call
  under test created — a renderer that quietly opens a second canvas fails.
* Without matplotlib every entry point raises :exc:`ImportError` naming the
  extra, not :exc:`ModuleNotFoundError` from somewhere in the middle of a draw.

Numerical assertions are made against the closed-form oracles in
:mod:`mmar.models.theory`, not against recorded output: the drawn monofractal
line must pass through :math:`\zeta(2)`, the drawn fit must recover the exact
:math:`\zeta(3)` of a noiseless power law, and the drawn cascade bars must
conserve mass. A plot that draws the wrong numbers beautifully still fails.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.regression import loglog_fit, select_scaling_range
from mmar.core.types import (
    Convention,
    FitMethod,
    FloatArray,
    HolderSpectrum,
    LogLogFit,
    MultifractalReport,
    PointwiseHolder,
    ScalingExponents,
    ScalingRange,
    StructureFunctions,
)
from mmar.models.cascade import deterministic_cascade
from mmar.models.theory import fbm_exponents, lognormal_exponents
from mmar.viz import diagnostics, spectra

Q_GRID = np.linspace(-4.0, 4.0, 17)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_show_clean_slate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Forbid ``plt.show`` and start every test with an empty figure manager.

    Closing at *entry* rather than only at exit is what makes the figure-count
    assertions below meaningful: each test measures figures it created itself.
    """

    def _forbidden(*_: object, **__: object) -> None:
        raise AssertionError("a mmar.viz function called plt.show(); display is the caller's job")

    monkeypatch.setattr(plt, "show", _forbidden)
    plt.close("all")
    assert plt.get_fignums() == []
    yield
    plt.close("all")


@pytest.fixture
def exponents() -> ScalingExponents:
    """Lognormal-cascade exponents with plausible, q-dependent standard errors."""
    base = lognormal_exponents(Q_GRID, lam=1.10)
    stderr = 0.004 * (1.0 + np.abs(base.q))
    return ScalingExponents(
        q=base.q,
        zeta=base.zeta,
        d_support=base.d_support,
        zeta_stderr=stderr,
        estimator="test.lognormal",
        provenance=base.provenance,
    )


@pytest.fixture
def spectrum(exponents: ScalingExponents) -> HolderSpectrum:
    return spectrum_from_exponents(exponents, method="fenchel", n_alpha=81)


def _structure_from(exps: ScalingExponents, *, n_scales: int = 12) -> StructureFunctions:
    r"""Exact scaling surface :math:`S(q,\delta)=\delta^{\zeta(q)}` on a geometric grid.

    Noiseless by construction, so any deviation a test sees comes from the code
    under test rather than from a random draw.
    """
    scales = np.geomspace(4.0, 512.0, n_scales)
    values = scales[None, :] ** exps.zeta[:, None]
    counts = np.maximum(np.round(8192.0 / scales), 4.0).astype(np.int64)
    return StructureFunctions(q=exps.q, scales=scales, values=values, counts=counts)


@pytest.fixture
def structure(exponents: ScalingExponents) -> StructureFunctions:
    return _structure_from(exponents)


@pytest.fixture
def scaling_range(structure: StructureFunctions) -> ScalingRange:
    row = structure.row(2.0)
    return select_scaling_range(structure.scales, row, min_points=5).scaling_range


@pytest.fixture
def fit(structure: StructureFunctions) -> LogLogFit:
    return loglog_fit(structure.scales, structure.row(2.0))


@pytest.fixture
def pointwise() -> PointwiseHolder:
    """A regularity field with three planted singular episodes."""
    t = np.arange(600, dtype=np.float64)
    h = 0.55 + 0.05 * np.sin(2.0 * np.pi * t / 200.0)
    h[[100, 250, 251, 400]] = 0.18
    return PointwiseHolder(t=t, h=h, stderr=np.full(t.size, 0.02), method="oscillation")


@pytest.fixture
def report(
    exponents: ScalingExponents,
    spectrum: HolderSpectrum,
    structure: StructureFunctions,
    scaling_range: ScalingRange,
) -> MultifractalReport:
    fits = tuple(loglog_fit(structure.scales, structure.values[i]) for i in range(exponents.q.size))
    exps = ScalingExponents(
        q=exponents.q,
        zeta=exponents.zeta,
        d_support=exponents.d_support,
        zeta_stderr=exponents.zeta_stderr,
        fits=fits,
        scaling_range=scaling_range,
        structure=structure,
        estimator="test.lognormal",
        provenance=exponents.provenance,
    )
    return MultifractalReport(exponents=exps, spectrum=spectrum, provenance=exponents.provenance)


@dataclass(frozen=True)
class _SurrogateResult:
    """Stand-in for the concurrently developed inference result type."""

    statistic: float
    null_distribution: FloatArray
    p_value: float
    name: str = "delta-alpha vs shuffled surrogates"


@pytest.fixture
def surrogate() -> _SurrogateResult:
    rng = np.random.default_rng(20260727)
    null = rng.normal(0.10, 0.02, size=400)
    return _SurrogateResult(statistic=0.42, null_distribution=null, p_value=1.0 / 401.0)


def _legend_labels(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None, "a multi-series axis must carry a legend"
    return [text.get_text() for text in legend.get_texts()]


def _bars(ax: Axes) -> list[Rectangle]:
    return [patch for patch in ax.patches if isinstance(patch, Rectangle)]


# --------------------------------------------------------------------------- #
# spectra.py — return types and mark counts
# --------------------------------------------------------------------------- #


def test_plot_spectrum_returns_axes_with_curve_and_apex_marker(spectrum: HolderSpectrum) -> None:
    ax = spectra.plot_spectrum(spectrum)
    assert isinstance(ax, Axes)
    # the f(alpha) curve plus the dotted marker at alpha_0
    assert len(ax.lines) == 2
    assert ax.get_ylabel() == r"$f(\alpha)$"


def test_plot_spectrum_without_descriptors_drops_the_apex_marker(
    spectrum: HolderSpectrum,
) -> None:
    ax = spectra.plot_spectrum(spectrum, show_descriptors=False)
    assert len(ax.lines) == 1
    assert not ax.texts


def test_plot_spectrum_stderr_band_is_a_collection_not_a_line(
    exponents: ScalingExponents,
) -> None:
    spec = spectrum_from_exponents(exponents, method="parametric")
    assert spec.f_stderr is not None, "the parametric route must propagate delta-method errors"
    ax = spectra.plot_spectrum(spec)
    assert len(ax.lines) == 2
    assert len(ax.collections) == 1


def test_plot_spectrum_accepts_a_reference_spectrum_and_a_raw_pair(
    spectrum: HolderSpectrum,
) -> None:
    ax = spectra.plot_spectrum(spectrum, reference=spectrum)
    assert len(ax.lines) == 3
    ax2 = spectra.plot_spectrum(spectrum, reference=(spectrum.alpha, spectrum.f_alpha))
    assert len(ax2.lines) == 3


def test_plot_spectrum_rejects_an_everywhere_infinite_spectrum() -> None:
    degenerate = HolderSpectrum(
        alpha=np.linspace(0.4, 0.6, 5), f_alpha=np.full(5, -np.inf), method="degenerate"
    )
    with pytest.raises(ValueError, match="no finite"):
        spectra.plot_spectrum(degenerate)
    plt.close("all")


def test_plot_spectrum_draws_on_a_supplied_axes(spectrum: HolderSpectrum) -> None:
    fig, ax = plt.subplots()
    returned = spectra.plot_spectrum(spectrum, ax=ax)
    assert returned is ax
    assert len(fig.axes) == 1


def test_plot_exponents_shows_the_monofractal_line(exponents: ScalingExponents) -> None:
    ax = spectra.plot_exponents(exponents)
    assert isinstance(ax, Axes)
    # errorbar data line, the q*h(2) null line, and the y = 0 rule
    assert len(ax.lines) == 3
    assert any("monofractal" in label for label in _legend_labels(ax))


def test_plot_exponents_monofractal_line_passes_through_zeta_of_two(
    exponents: ScalingExponents,
) -> None:
    ax = spectra.plot_exponents(exponents)
    null_line = ax.lines[1]
    q, y = null_line.get_xdata(), null_line.get_ydata()
    assert np.allclose(np.interp(2.0, np.asarray(q), np.asarray(y)), exponents.at(2.0))


def test_plot_exponents_measure_convention_offsets_both_curves(
    exponents: ScalingExponents,
) -> None:
    ax_sig = spectra.plot_exponents(exponents, convention=Convention.SIGNAL)
    ax_mea = spectra.plot_exponents(exponents, convention=Convention.MEASURE)
    y_sig = np.asarray(ax_sig.lines[0].get_ydata(), dtype=np.float64)
    y_mea = np.asarray(ax_mea.lines[0].get_ydata(), dtype=np.float64)
    assert np.allclose(y_sig - y_mea, exponents.d_support)
    null_sig = np.asarray(ax_sig.lines[1].get_ydata(), dtype=np.float64)
    null_mea = np.asarray(ax_mea.lines[1].get_ydata(), dtype=np.float64)
    assert np.allclose(null_sig - null_mea, exponents.d_support)


def test_plot_exponents_with_reference_adds_one_line(exponents: ScalingExponents) -> None:
    ax = spectra.plot_exponents(exponents, reference=fbm_exponents(Q_GRID, H=0.55))
    assert len(ax.lines) == 4


def test_plot_hq_is_flat_for_a_monofractal() -> None:
    ax = spectra.plot_hq(fbm_exponents(Q_GRID, H=0.62))
    assert isinstance(ax, Axes)
    # h(q) itself plus the h(2) rule
    assert len(ax.lines) == 2
    h = np.asarray(ax.lines[0].get_ydata(), dtype=np.float64)
    finite = h[np.isfinite(h)]
    assert np.allclose(finite, 0.62)


def test_plot_hq_leaves_q_zero_as_a_gap(exponents: ScalingExponents) -> None:
    ax = spectra.plot_hq(exponents)
    q = np.asarray(ax.lines[0].get_xdata(), dtype=np.float64)
    h = np.asarray(ax.lines[0].get_ydata(), dtype=np.float64)
    assert not np.isfinite(h[np.argmin(np.abs(q))])


def test_plot_hq_with_reference_adds_one_line(exponents: ScalingExponents) -> None:
    ax = spectra.plot_hq(exponents, reference=fbm_exponents(Q_GRID, H=0.55))
    assert len(ax.lines) == 3


def test_plot_structure_functions_draws_data_and_fit_per_q(
    structure: StructureFunctions,
) -> None:
    ax = spectra.plot_structure_functions(structure, q_subset=[-2.0, 1.0, 3.0])
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 6
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_plot_structure_functions_recovers_the_exact_slope(
    structure: StructureFunctions, exponents: ScalingExponents
) -> None:
    """The surface is exactly a power law, so the drawn fit must reproduce zeta."""
    ax = spectra.plot_structure_functions(structure, q_subset=[3.0], show_fits=True)
    fitted = ax.lines[1]
    x = np.log(np.asarray(fitted.get_xdata(), dtype=np.float64))
    y = np.log(np.asarray(fitted.get_ydata(), dtype=np.float64))
    slope = float(np.polyfit(x, y, 1)[0])
    assert slope == pytest.approx(exponents.at(3.0), rel=1e-9)


def test_plot_structure_functions_shades_the_selected_window(
    structure: StructureFunctions, scaling_range: ScalingRange
) -> None:
    ax = spectra.plot_structure_functions(structure, q_subset=[2.0], scaling_range=scaling_range)
    assert len(ax.patches) == 1
    assert "fitted window" in ax.get_title()


def test_plot_structure_functions_defaults_to_five_curves(
    structure: StructureFunctions,
) -> None:
    ax = spectra.plot_structure_functions(structure)
    assert len(ax.lines) == 10


def test_plot_structure_functions_without_fits_halves_the_lines(
    structure: StructureFunctions,
) -> None:
    ax = spectra.plot_structure_functions(structure, q_subset=[0.5, 2.0], show_fits=False)
    assert len(ax.lines) == 2


def test_plot_structure_functions_refuses_a_non_positive_row(
    exponents: ScalingExponents,
) -> None:
    base = _structure_from(exponents)
    values = np.array(base.values, copy=True)
    values[0, :] = -1.0
    broken = StructureFunctions(q=base.q, scales=base.scales, values=values)
    with pytest.raises(ValueError, match="positive finite moments"):
        spectra.plot_structure_functions(broken, q_subset=[base.q[0]])
    plt.close("all")


def test_plot_comparison_one_line_per_estimator(spectrum: HolderSpectrum) -> None:
    ax = spectra.plot_comparison({"mfdfa": spectrum, "wtmm": spectrum, "leaders": spectrum})
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 3
    assert all(r"\Delta\alpha" in label for label in _legend_labels(ax))


def test_plot_comparison_rejects_an_empty_mapping() -> None:
    with pytest.raises(ValueError, match="empty"):
        spectra.plot_comparison({})
    plt.close("all")


# --------------------------------------------------------------------------- #
# diagnostics.py
# --------------------------------------------------------------------------- #


def test_plot_fit_residuals_draws_zero_rule_points_and_trend(fit: LogLogFit) -> None:
    ax = diagnostics.plot_fit_residuals(fit)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 3
    assert ax.lines[1].get_linestyle() == "None"


def test_plot_fit_residuals_rejects_a_fit_without_residuals() -> None:
    hollow = LogLogFit(
        slope=1.0,
        intercept=0.0,
        slope_stderr=0.1,
        intercept_stderr=0.1,
        slope_ci=(0.9, 1.1),
        r_squared=0.99,
        n_points=8,
    )
    with pytest.raises(ValueError, match="no residuals"):
        diagnostics.plot_fit_residuals(hollow)
    plt.close("all")


def test_plot_scaling_range_selection_marks_window_and_extrapolation(
    structure: StructureFunctions, scaling_range: ScalingRange
) -> None:
    ax = diagnostics.plot_scaling_range_selection(structure, scaling_range)
    assert isinstance(ax, Axes)
    # data points, the extrapolated fit, and the fit inside the window
    assert len(ax.lines) == 3
    assert len(ax.patches) == 1
    assert "decades" in ax.get_title()


def test_plot_scaling_range_selection_rejects_a_too_narrow_window(
    structure: StructureFunctions,
) -> None:
    narrow = ScalingRange(s_min=4.0, s_max=4.5, n_scales=1, selection="manual")
    with pytest.raises(ValueError, match="usable scales"):
        diagnostics.plot_scaling_range_selection(structure, narrow)
    plt.close("all")


def test_plot_surrogate_null_marks_the_observation(surrogate: _SurrogateResult) -> None:
    ax = diagnostics.plot_surrogate_null(surrogate)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1
    marked = np.asarray(ax.lines[0].get_xdata(), dtype=np.float64)
    assert marked[0] == pytest.approx(surrogate.statistic)
    assert len(ax.patches) > 1  # the histogram bars
    note = ax.texts[0].get_text()
    assert "min attainable" in note


def test_plot_surrogate_null_accepts_a_mapping_and_derives_the_p_value() -> None:
    rng = np.random.default_rng(7)
    null = rng.normal(size=200)
    ax = diagnostics.plot_surrogate_null({"null": null, "statistic": 0.0})
    # The plug-in estimator (1 + #{T_i >= T}) / (m + 1) at the median of a
    # symmetric null must land near one half.
    text = ax.texts[0].get_text()
    p_line = next(line for line in text.splitlines() if line.startswith("$p$"))
    assert 0.4 < float(p_line.split("=")[1]) < 0.6


def test_plot_surrogate_null_rejects_an_object_without_a_null() -> None:
    with pytest.raises(TypeError, match="does not expose a surrogate null"):
        diagnostics.plot_surrogate_null(object())
    plt.close("all")


def test_plot_pointwise_holder_returns_a_two_panel_figure(
    pointwise: PointwiseHolder,
) -> None:
    x = np.cumsum(np.random.default_rng(1).standard_normal(pointwise.t.size))
    fig = diagnostics.plot_pointwise_holder(pointwise, x=x)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    series_ax, h_ax = fig.axes
    assert len(series_ax.lines) == 2  # the series and the singular points
    assert len(h_ax.lines) == 3  # h(t), the threshold rule, the singular points
    assert h_ax.get_ylabel() == "$h(t)$"


def test_plot_pointwise_holder_threshold_selects_the_planted_singularities(
    pointwise: PointwiseHolder,
) -> None:
    fig = diagnostics.plot_pointwise_holder(pointwise, threshold=0.3)
    flagged = np.asarray(fig.axes[1].lines[2].get_xdata(), dtype=np.float64)
    assert sorted(flagged.tolist()) == [100.0, 250.0, 251.0, 400.0]
    # The highlighted set must be exactly the one the type itself defines,
    # so the figure cannot drift from PointwiseHolder.singular_times.
    expected = pointwise.t[pointwise.singular_times(0.3)]
    assert np.array_equal(np.sort(flagged), np.sort(expected))
    # ... and the rule is drawn at the threshold, not near it.
    assert float(np.asarray(fig.axes[1].lines[1].get_ydata())[0]) == pytest.approx(0.3)


def test_plot_pointwise_holder_rejects_a_misaligned_series(
    pointwise: PointwiseHolder,
) -> None:
    with pytest.raises(ValueError, match="must align"):
        diagnostics.plot_pointwise_holder(pointwise, x=np.zeros(3))
    plt.close("all")


def test_plot_cascade_bars_preserve_the_total_mass() -> None:
    mu = deterministic_cascade(12, (0.65, 0.35))
    ax = diagnostics.plot_cascade(mu, b=2, max_bars=128)
    assert isinstance(ax, Axes)
    heights = np.array([bar.get_height() for bar in _bars(ax)])
    assert heights.size == 128
    assert float(heights.sum()) == pytest.approx(float(mu.sum()), rel=1e-12)
    assert ax.get_xlim() == (0.0, 1.0)


def test_plot_cascade_log_axis() -> None:
    mu = deterministic_cascade(8, (0.7, 0.3))
    ax = diagnostics.plot_cascade(mu, b=2, max_bars=512, log_y=True)
    assert ax.get_yscale() == "log"


def test_plot_cascade_rejects_a_signed_array() -> None:
    with pytest.raises(ValueError, match="negative"):
        diagnostics.plot_cascade(np.array([0.5, -0.5, 1.0]))
    plt.close("all")


def test_multifractal_dashboard_has_six_panels(report: MultifractalReport) -> None:
    fig = diagnostics.multifractal_dashboard(report)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 6
    footer = [t.get_text() for t in fig.texts]
    assert any("provenance root" in text for text in footer)


def test_multifractal_dashboard_explains_missing_evidence(
    exponents: ScalingExponents, spectrum: HolderSpectrum
) -> None:
    bare = MultifractalReport(exponents=exponents, spectrum=spectrum)
    fig = diagnostics.multifractal_dashboard(bare)
    messages = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("no structure functions retained" in m for m in messages)
    assert any("no per-q fits retained" in m for m in messages)


def test_multifractal_dashboard_accepts_a_supplied_figure(report: MultifractalReport) -> None:
    fig = plt.figure(figsize=(12.0, 7.0))
    returned = diagnostics.multifractal_dashboard(report, fig=fig)
    assert returned is fig
    assert len(fig.axes) == 6


# --------------------------------------------------------------------------- #
# Numerical contracts that a noiseless power law cannot discriminate
#
# Every test above uses S(q, delta) = delta^zeta exactly, so a fit taken over
# the wrong scale window returns the *same* slope and the window logic is
# unfalsifiable. The fixture below plants a hard crossover so that respecting
# the window and ignoring it give demonstrably different numbers.
# --------------------------------------------------------------------------- #

_BREAK: float = 64.0
_ZETA_LO: float = 0.60
_ZETA_HI: float = 0.20


@pytest.fixture
def crossover() -> tuple[StructureFunctions, ScalingRange]:
    r"""A surface with slope 0.60 below :math:`\delta=64` and 0.20 above it."""
    scales = np.geomspace(4.0, 1024.0, 17)
    s = np.where(
        scales <= _BREAK,
        scales**_ZETA_LO,
        (_BREAK**_ZETA_LO) * (scales / _BREAK) ** _ZETA_HI,
    )
    inside = scales <= _BREAK
    struct = StructureFunctions(q=np.array([1.0, 2.0]), scales=scales, values=np.vstack([s, s]))
    window = ScalingRange(
        s_min=float(scales[inside].min()),
        s_max=float(scales[inside].max()),
        n_scales=int(inside.sum()),
        selection="manual",
        mask=tuple(bool(v) for v in inside),
    )
    return struct, window


def _drawn_slope(line: Any) -> float:
    x = np.log(np.asarray(line.get_xdata(), dtype=np.float64))
    y = np.log(np.asarray(line.get_ydata(), dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def test_structure_function_fit_is_confined_to_the_scaling_window(
    crossover: tuple[StructureFunctions, ScalingRange],
) -> None:
    """Rule 1 of the module: the drawn fit is the fit over the shaded window."""
    struct, window = crossover
    ax = spectra.plot_structure_functions(struct, q_subset=[2.0], scaling_range=window)
    fitted = ax.lines[1]
    assert _drawn_slope(fitted) == pytest.approx(_ZETA_LO, rel=1e-10)
    xs = np.asarray(fitted.get_xdata(), dtype=np.float64)
    assert xs.min() >= window.s_min and xs.max() <= window.s_max
    # ... and the whole-axis fit is a different number, so the test can fail.
    assert _drawn_slope(ax.lines[0]) != pytest.approx(_ZETA_LO, rel=1e-3)


def test_scaling_range_selection_fits_inside_and_extrapolates_outside(
    crossover: tuple[StructureFunctions, ScalingRange],
) -> None:
    struct, window = crossover
    ax = diagnostics.plot_scaling_range_selection(struct, window, q=2.0)
    solid = ax.lines[2]
    dashed = ax.lines[1]
    assert _drawn_slope(solid) == pytest.approx(_ZETA_LO, rel=1e-10)
    assert np.asarray(solid.get_xdata()).max() == pytest.approx(window.s_max)
    assert np.asarray(dashed.get_xdata()).max() > window.s_max
    assert f"slope = {_ZETA_LO:.4g}" in _legend_labels(ax)[2]


def test_structure_function_fit_is_weighted_by_the_window_counts(
    structure: StructureFunctions,
) -> None:
    """The drawn line must be the WLS fit, not an unweighted one drawn beside it."""
    assert structure.counts is not None
    heteroscedastic = np.array(structure.values, copy=True)
    row = int(np.argmin(np.abs(structure.q - 2.0)))
    heteroscedastic[row] *= 1.0 + 0.35 * np.linspace(-1.0, 1.0, structure.scales.size) ** 3
    tilted = StructureFunctions(
        q=structure.q,
        scales=structure.scales,
        values=heteroscedastic,
        counts=structure.counts,
    )
    ax = spectra.plot_structure_functions(tilted, q_subset=[2.0])
    drawn = _drawn_slope(ax.lines[1])
    wls = loglog_fit(
        tilted.scales,
        tilted.row(2.0),
        method=FitMethod.WLS,
        weights=np.asarray(tilted.counts, dtype=np.float64),
    ).slope
    ols = loglog_fit(tilted.scales, tilted.row(2.0)).slope
    assert drawn == pytest.approx(wls, rel=1e-12)
    assert abs(wls - ols) > 1e-4, "the fixture must separate WLS from OLS"


def test_spectrum_band_is_exactly_one_standard_error_wide(
    exponents: ScalingExponents,
) -> None:
    spec = spectrum_from_exponents(exponents, method="parametric")
    assert spec.f_stderr is not None
    ax = spectra.plot_spectrum(spec)
    poly = np.asarray(ax.collections[0].get_paths()[0].vertices, dtype=np.float64)
    ok = np.isfinite(spec.alpha) & np.isfinite(spec.f_alpha) & np.isfinite(spec.f_stderr)
    order = np.argsort(spec.alpha[ok], kind="stable")
    expected_lo = (spec.f_alpha[ok] - spec.f_stderr[ok])[order]
    # fill_between builds the polygon as the lower boundary forwards, then the
    # upper boundary back; vertex 0 is the closing duplicate of vertex 1.
    drawn_lo = poly[1 : 1 + expected_lo.size, 1]
    assert np.allclose(drawn_lo, expected_lo, rtol=0, atol=1e-12)
    # A 2-sigma band would be twice as far from the curve, so this can fail.
    assert not np.allclose(drawn_lo, (spec.f_alpha[ok] - 2.0 * spec.f_stderr[ok])[order])


def test_hq_error_bars_are_the_delta_method_ratio(exponents: ScalingExponents) -> None:
    r""":math:`h=\zeta/q` is a deterministic scaling, so :math:`\sigma_h=\sigma_\zeta/|q|`."""
    ax = spectra.plot_hq(exponents)
    assert exponents.zeta_stderr is not None
    bars = cast(LineCollection, ax.collections[0])
    segments = [np.asarray(s, dtype=np.float64) for s in bars.get_segments()]
    q = exponents.q
    # The removable singularity at q = 0 is a NaN error bar, so matplotlib emits
    # an empty segment there; that gap is itself part of the contract.
    assert [i for i, s in enumerate(segments) if s.size == 0] == [int(np.argmin(np.abs(q)))]
    drawn = np.array([abs(s[1, 1] - s[0, 1]) for s in segments if s.size])
    expected = 2.0 * exponents.zeta_stderr[q != 0.0] / np.abs(q[q != 0.0])
    assert drawn.size == expected.size
    assert np.allclose(drawn, expected, rtol=1e-10)
    # sigma_zeta / q^2 would be a different set of bars, so this can fail.
    assert not np.allclose(drawn, 2.0 * exponents.zeta_stderr[q != 0.0] / q[q != 0.0] ** 2)


def test_surrogate_null_uses_the_unbiased_monte_carlo_p_value() -> None:
    r""":math:`\hat p=(1+\#\{T_i\ge T\})/(m+1)`, never the biased :math:`\#/m`."""
    null = np.arange(100.0)  # #{null >= 60} = 40 exactly
    ax = diagnostics.plot_surrogate_null({"null": null, "statistic": 60.0})
    text = ax.texts[0].get_text()
    p = float(next(ln for ln in text.splitlines() if ln.startswith("$p$")).split("=")[1])
    # The box prints 4 significant figures, which still separates the two forms.
    assert p == pytest.approx(41.0 / 101.0, rel=1e-3)
    assert p != pytest.approx(40.0 / 100.0, rel=1e-3)
    assert f"min attainable $p$ = {1.0 / 101.0:.3g}" in text


def test_surrogate_null_omits_an_empty_name_line() -> None:
    """``TestResult.name`` defaults to ``""``; the box must not open with a blank."""
    ax = diagnostics.plot_surrogate_null({"null": np.arange(50.0), "statistic": 10.0, "name": ""})
    assert ax.texts[0].get_text().splitlines()[0].startswith("observed")


def test_pointwise_default_threshold_is_the_five_percent_quantile(
    pointwise: PointwiseHolder,
) -> None:
    fig = diagnostics.plot_pointwise_holder(pointwise)
    cut = float(np.asarray(fig.axes[1].lines[1].get_ydata())[0])
    expected = float(np.quantile(pointwise.h[np.isfinite(pointwise.h)], 0.05))
    assert cut == pytest.approx(expected, rel=1e-12)
    flagged = np.asarray(fig.axes[1].lines[2].get_xdata(), dtype=np.float64)
    assert np.array_equal(np.sort(flagged), np.sort(pointwise.t[pointwise.singular_times(cut)]))


def test_cascade_bars_tile_the_unit_interval_from_the_left_edge() -> None:
    """``align='edge'``: bar *i* occupies :math:`[i/n,(i+1)/n)`, not a centred box."""
    ax = diagnostics.plot_cascade(np.array([0.25, 0.75]), b=2, max_bars=512)
    bars = sorted(_bars(ax), key=lambda r: r.get_x())
    assert [r.get_x() for r in bars] == pytest.approx([0.0, 0.5])
    assert [r.get_width() for r in bars] == pytest.approx([0.5, 0.5])


def test_cascade_says_so_when_the_bar_cap_cannot_be_met() -> None:
    """A non-b-adic measure cannot be coarse-grained; the title must not stay silent."""
    mu = np.linspace(1.0, 2.0, 3**5)
    ax = diagnostics.plot_cascade(mu, b=2, max_bars=100)
    assert len(_bars(ax)) == 3**5
    assert "cap of 100 not reached" in ax.get_title()


def test_cascade_coarse_graining_is_the_b_adic_box_partition() -> None:
    r"""Summing :math:`b` neighbours *is* the coarser partition, for any base."""
    mu = deterministic_cascade(7, (0.5, 0.2, 0.3))
    ax = diagnostics.plot_cascade(mu, b=3, max_bars=100)
    heights = np.array([bar.get_height() for bar in sorted(_bars(ax), key=lambda r: r.get_x())])
    assert heights.size == 81
    assert np.allclose(heights, mu.reshape(81, -1).sum(axis=1), rtol=1e-14)
    assert float(heights.sum()) == pytest.approx(float(mu.sum()), rel=1e-14)


def test_dashboard_refuses_to_label_a_q_it_cannot_know(
    exponents: ScalingExponents, spectrum: HolderSpectrum, structure: StructureFunctions
) -> None:
    """With fewer fits than moment orders the panel's q is unknown; say so."""
    exps = ScalingExponents(
        q=exponents.q,
        zeta=exponents.zeta,
        d_support=exponents.d_support,
        fits=(loglog_fit(structure.scales, structure.row(-4.0)),),
        estimator="test.partial",
    )
    fig = diagnostics.multifractal_dashboard(MultifractalReport(exponents=exps, spectrum=spectrum))
    title = fig.axes[4].get_title()
    assert "$q$ unknown" in title
    assert "residuals at" not in title


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #


def test_use_style_is_scoped_by_default() -> None:
    before = matplotlib.rcParams["font.family"]
    with spectra.use_style():
        assert matplotlib.rcParams["font.family"] == ["serif"]
        assert matplotlib.rcParams["axes.grid"] is True
    assert matplotlib.rcParams["font.family"] == before


def test_use_style_permanent_mutates_and_is_restorable() -> None:
    snapshot = matplotlib.rcParams.copy()
    try:
        with spectra.use_style(permanent=True):
            pass
        assert matplotlib.rcParams["font.family"] == ["serif"]
        cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        assert cycle == list(spectra.CVD_PALETTE)
    finally:
        matplotlib.rcParams.update(snapshot)


def test_palette_is_exported_and_distinct() -> None:
    assert len(set(spectra.CVD_PALETTE)) == len(spectra.CVD_PALETTE)
    assert diagnostics.use_style is spectra.use_style


# --------------------------------------------------------------------------- #
# The optional-dependency contract
# --------------------------------------------------------------------------- #


@pytest.fixture
def renderers(
    *,
    spectrum: HolderSpectrum,
    exponents: ScalingExponents,
    structure: StructureFunctions,
    scaling_range: ScalingRange,
    fit: LogLogFit,
    pointwise: PointwiseHolder,
    surrogate: _SurrogateResult,
    report: MultifractalReport,
) -> dict[str, Callable[[], Any]]:
    """Every public entry point, bound to valid arguments and no target Axes."""
    return {
        "plot_spectrum": lambda: spectra.plot_spectrum(spectrum),
        "plot_comparison": lambda: spectra.plot_comparison({"a": spectrum}),
        "plot_exponents": lambda: spectra.plot_exponents(exponents),
        "plot_hq": lambda: spectra.plot_hq(exponents),
        "plot_structure_functions": lambda: spectra.plot_structure_functions(structure),
        "plot_fit_residuals": lambda: diagnostics.plot_fit_residuals(fit),
        "plot_scaling_range_selection": lambda: diagnostics.plot_scaling_range_selection(
            structure, scaling_range
        ),
        "plot_surrogate_null": lambda: diagnostics.plot_surrogate_null(surrogate),
        "plot_pointwise_holder": lambda: diagnostics.plot_pointwise_holder(pointwise),
        "plot_cascade": lambda: diagnostics.plot_cascade(deterministic_cascade(6, (0.7, 0.3))),
        "multifractal_dashboard": lambda: diagnostics.multifractal_dashboard(report),
        "use_style": spectra.use_style,
    }


RENDERER_NAMES = (
    "plot_spectrum",
    "plot_comparison",
    "plot_exponents",
    "plot_hq",
    "plot_structure_functions",
    "plot_fit_residuals",
    "plot_scaling_range_selection",
    "plot_surrogate_null",
    "plot_pointwise_holder",
    "plot_cascade",
    "multifractal_dashboard",
    "use_style",
)


def test_renderer_registry_covers_the_public_api() -> None:
    """The two contract tests below are only worth anything if nothing is missing."""
    from mmar import viz

    exported = {n for n in viz.__all__ if n.startswith(("plot_", "multifractal_", "use_"))}
    assert exported == set(RENDERER_NAMES)


@pytest.mark.parametrize("name", RENDERER_NAMES)
def test_renderer_returns_an_artist_and_owns_exactly_one_figure(
    name: str, renderers: dict[str, Callable[[], Any]]
) -> None:
    result = renderers[name]()
    if name == "use_style":
        assert plt.get_fignums() == []
        return
    assert isinstance(result, Axes | Figure), f"{name} returned {type(result).__name__}"
    assert len(plt.get_fignums()) == 1, f"{name} created more than one figure"


@pytest.mark.parametrize("name", RENDERER_NAMES)
def test_missing_matplotlib_raises_a_named_import_error(
    name: str, renderers: dict[str, Callable[[], Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry in sys.modules makes the import machinery raise ImportError,
    # which is exactly the state of an environment installed without the extra.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"mmar\[viz\]"):
        renderers[name]()
