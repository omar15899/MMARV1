r"""Adversarial probe of the moment grid and the scale grid at their limits.

The angle: everything that indexes a scaling surface. :math:`q` exactly at 0,
:math:`\pm1`, :math:`\pm2`; :math:`|q|` far past any moment that exists;
:math:`q` grids that are descending, duplicated, too short or non-finite; scale
grids of one element, of :math:`n` elements, exceeding :math:`n`, non-integer,
descending, duplicated; and the two float64 traps of the trade --- a level
series whose increments live below its own ulp, and a moment whose summands
underflow.

Three silently-wrong behaviours were found and fixed here --- a fit-quality
summary poisoned by the identically-degenerate :math:`q=0` row, a reversed
:math:`q` axis answered with a sign-flipped :math:`h(2)`, and a scaling surface
that returned the nearest row for a moment it had never computed. The rest of
the file is a regression fence around behaviour that is already right and would
be expensive to lose.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mmar.core.regression import loglog_fit
from mmar.core.types import (
    FitMethod,
    FloatArray,
    ScalingExponents,
    StructureFunctions,
    log_spaced_scales,
)
from mmar.core.validate import DataQualityError
from mmar.estimators.ghe import generalised_hurst, generalised_hurst_structure
from mmar.estimators.mfdfa import mfdfa
from mmar.estimators.mfdma import mfdma
from mmar.estimators.structure_function import (
    critical_moment_order,
    exponents_from_structure,
    mfc_exponents,
    structure_functions,
)
from mmar.models.fbm import fbm

SEED = 20260727
N = 4096

Estimator = Callable[..., ScalingExponents]

SERIES_ESTIMATORS: Mapping[str, tuple[Estimator, dict[str, Any]]] = {
    # Every entry is handed the LEVEL series, so the two profile-building
    # estimators are pinned to integrate=False (mmar.analysis does the same).
    "mfc": (mfc_exponents, {}),
    "mfdfa": (mfdfa, {"integrate": False}),
    "mfdma": (mfdma, {"integrate": False}),
    "ghe": (generalised_hurst, {"series_kind": "level"}),
}


@pytest.fixture(scope="module")
def level() -> FloatArray:
    """One exact fBm path, H = 0.7, used as the level series everywhere below."""
    return fbm(N, 0.7, seed=SEED)


def _one_point_exponent(x: FloatArray, scales: FloatArray) -> float:
    r"""Log--log slope of :math:`\max_j|\Delta_\delta X|` against :math:`\delta`.

    The limit that :math:`\zeta(q)/q` must approach as :math:`q\to\infty`: past
    the largest existing moment the sample average *is* its largest term. Fitted
    with the same disjoint-window weighting ``mfc_exponents`` uses, so the two
    slopes are comparable rather than merely similar.
    """
    peaks = np.empty(scales.size, dtype=np.float64)
    counts = np.empty(scales.size, dtype=np.float64)
    for i, s in enumerate(scales):
        step = int(s)
        j = np.arange((x.size - 1) // step, dtype=np.intp) * step
        peaks[i] = float(np.max(np.abs(x[j + step] - x[j])))
        counts[i] = float(j.size)
    fit = loglog_fit(scales, peaks, method=FitMethod.WLS, weights=counts)
    return float(fit.slope)


# --------------------------------------------------------------------------- #
# q = 0: the row that is an identity, not a measurement
# --------------------------------------------------------------------------- #


def test_q_zero_moment_is_exactly_one_and_its_r_squared_is_nan(level: FloatArray) -> None:
    r"""Attack: does the degenerate :math:`q=0` row exist, and is it recognised?

    :math:`S(0,\delta)=1` *bit-exactly* by construction, so the :math:`q=0`
    log--log fit has zero total sum of squares and an undefined :math:`R^2`.
    Anything that aggregates per-:math:`q` fit quality must therefore be
    NaN-aware. This test pins the premise of the next one.
    """
    q = np.array([-1.0, 0.0, 1.0, 2.0])
    surface = structure_functions(level, q=q, scales=[8, 16, 32, 64, 128])
    row_zero = surface.values[1]
    assert np.array_equal(row_zero, np.ones_like(row_zero))

    fits = exponents_from_structure(surface).fits
    r2 = np.array([f.r_squared for f in fits], dtype=np.float64)
    assert np.isnan(r2[1])
    assert np.all(np.isfinite(np.delete(r2, 1)))


def test_default_q_grid_leaves_a_finite_reported_r_squared(level: FloatArray) -> None:
    r"""Attack: the library default :math:`q` grid contains exactly 0.0.

    ``mfc_exponents`` and ``generalised_hurst`` summarised fit quality with
    ``np.median`` over the per-:math:`q` :math:`R^2`. The :math:`q=0` entry is
    NaN by construction (previous test), so ``scaling_range.r_squared`` came
    back NaN for **every default call** --- the one number a reader consults to
    decide whether the power law held at all, reported as "unknown" on a fit
    whose true median :math:`R^2` is 0.98 (mfc) and 0.998 (ghe).
    ``wavelet_leaders`` already used ``np.nanmedian``; these two now do too.
    """
    for name in ("mfc", "ghe"):
        fn, kwargs = SERIES_ESTIMATORS[name]
        exponents = fn(level, **kwargs)
        assert exponents.scaling_range is not None
        reported = exponents.scaling_range.r_squared
        expected = float(np.nanmedian([f.r_squared for f in exponents.fits]))
        assert np.isfinite(reported), f"{name}: scaling_range.r_squared is {reported}"
        assert reported == pytest.approx(expected, abs=1e-12)
        assert 0.9 < reported <= 1.0


@pytest.mark.parametrize("name", ["mfdfa", "mfdma"])
def test_the_q_zero_logarithmic_branch_is_the_continuous_limit(
    name: str, level: FloatArray
) -> None:
    r"""Attack: the :math:`0/0` branch that :math:`\zeta(0)=0` cannot see.

    MF-DFA and MF-DMA form :math:`F_q=\big[\langle F_v^q\rangle\big]^{1/q}`,
    which is :math:`0/0` at :math:`q=0` and must be replaced by
    :math:`\ln F_0=\overline{\ln F_v}` --- the geometric mean, and in MF-DFA's
    case :math:`\tfrac12\overline{\ln F_v^2}`, where dropping the one-half is
    the classical slip. **Testing** :math:`\zeta(0)=0` **does not test this**:
    both estimators fit :math:`h(q)` and carry :math:`\zeta=q\,h(q)`, so
    :math:`\zeta(0)=0` holds identically whatever the branch returns, and
    :attr:`~mmar.core.types.ScalingExponents.hq` is NaN at :math:`q=0` by
    construction. The fitted :math:`h(0)` survives only inside ``fits``.

    So the branch is probed where it is observable: against its own two-sided
    limit. Measured here, :math:`h(0)` against the midpoint of
    :math:`h(\pm10^{-3})`: 3.3e-10 (mfdfa, :math:`h(0)=0.622679`) and 2.5e-09
    (mfdma, :math:`h(0)=0.748742`). Dropping the one-half doubles
    :math:`\ln F_0` and with it the slope, which this tolerance rejects by seven
    orders of magnitude.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    eps = 1e-3
    exponents = fn(level, q=np.array([-eps, 0.0, eps]), **kwargs)
    below, at_zero, above = (float(f.slope) for f in exponents.fits)

    assert at_zero == pytest.approx(0.5 * (below + above), abs=1e-7)
    assert min(below, above) <= at_zero <= max(below, above)


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
def test_zeta_at_q_zero_is_exactly_zero(name: str, level: FloatArray) -> None:
    r"""Attack: is :math:`\zeta(0)=0` an identity here, or a fitted number?

    :math:`S(0,\delta)\equiv1`, so the slope must vanish *exactly*. A tiny
    non-zero value would mean the estimator evaluated the :math:`q\to0` limit
    of a :math:`0/0` form numerically instead of taking the logarithmic branch,
    and :math:`h(q)=\zeta(q)/q` near the apex would inherit the error divided
    by :math:`q`.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    exponents = fn(level, q=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), **kwargs)
    assert exponents.zeta[2] == 0.0


# --------------------------------------------------------------------------- #
# q grids that are not grids
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
@pytest.mark.parametrize(
    "grid",
    [
        pytest.param([2.0, 1.0, 0.0, -1.0, -2.0], id="descending"),
        pytest.param([1.0, -2.0, 2.0, 0.0, -1.0], id="unsorted"),
        pytest.param([-2.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0], id="duplicated"),
    ],
)
def test_q_grid_is_normalised_without_shuffling_the_labels(
    name: str, grid: list[float], level: FloatArray
) -> None:
    r"""Attack: the cascade_moments defect --- rows reordered against their labels.

    A descending, unsorted or duplicated :math:`q` grid must either raise or be
    normalised *explicitly*, with the returned ``q`` axis reporting the grid
    actually used. What must never happen is ``zeta[i]`` belonging to a
    different :math:`q` than ``q[i]``. Checked by requiring the answer to equal
    the one from the canonically sorted, de-duplicated grid, term for term.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    canonical = np.unique(np.asarray(grid, dtype=np.float64))
    got = fn(level, q=np.asarray(grid, dtype=np.float64), **kwargs)
    want = fn(level, q=canonical, **kwargs)
    assert np.array_equal(got.q, canonical)
    assert np.array_equal(got.zeta, want.zeta)


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
@pytest.mark.parametrize(
    ("grid", "pattern"),
    [
        pytest.param([2.0], "at least three", id="single"),
        pytest.param([1.0, 2.0], "at least three", id="two"),
        pytest.param([1.0, 1.0, 1.0], "fewer than three", id="all-identical"),
        pytest.param([-1.0, 0.0, np.nan, 2.0], "non-finite", id="nan"),
        pytest.param([-1.0, 0.0, 1.0, np.inf], "non-finite", id="inf"),
    ],
)
def test_degenerate_q_grids_are_refused(
    name: str, grid: list[float], pattern: str, level: FloatArray
) -> None:
    """Attack: can a spectrum be built from a grid that cannot carry one?

    Two points give a straight line through anything and no curvature at all;
    NaN or inf in the grid would propagate into every moment. Both must be a
    clear exception, not a shorter answer.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    with pytest.raises(ValueError, match=pattern):
        fn(level, q=np.asarray(grid, dtype=np.float64), **kwargs)


def test_hand_built_exponents_refuse_a_descending_q_axis(level: FloatArray) -> None:
    r"""Attack: reverse a scaling surface and refit it through the public API.

    ``exponents_from_structure`` is documented as the way to refit a stored
    surface, and ``StructureFunctions`` can be assembled by hand. Reversing
    ``q`` and the value rows *together* leaves the identical mathematical
    object, so every scalar summary must be unchanged. It was not:
    :attr:`ScalingExponents.h2` and :meth:`~ScalingExponents.at` interpolate
    with :func:`numpy.interp`, which on a descending abscissa returns a number
    rather than raising. Measured on this path before the fix,
    ``h2 = -1.206`` against the true ``+0.620`` --- a sign-flipped Hurst
    exponent, straight into ``MultifractalReport.summary()`` --- while
    ``hurst_tau_root`` leaked a bare scipy "`x` must be strictly increasing".
    """
    q = np.array([-2.0, -1.0, 1.0, 2.0, 3.0])
    surface = structure_functions(level, q=q, scales=[8, 16, 32, 64, 128])
    ascending = exponents_from_structure(surface)
    assert ascending.h2 == pytest.approx(ascending.zeta[3] / 2.0)

    reversed_surface = StructureFunctions(
        q=surface.q[::-1].copy(),
        scales=surface.scales,
        values=surface.values[::-1].copy(),
        counts=surface.counts,
        provenance=surface.provenance,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        exponents_from_structure(reversed_surface)


def test_structure_row_refuses_a_moment_that_is_not_on_the_grid(level: FloatArray) -> None:
    r"""Attack: ask a scaling surface for a moment it never computed.

    :meth:`StructureFunctions.row` located its row with ``argmin(|q_grid - q|)``
    and returned it, so it answered *every* request, including ones no row
    exists for. On ``q = arange(-5, 5.01, 0.3)`` --- a grid written without
    noticing it steps over 2 --- ``row(2.0)`` returned the :math:`q=1.9` row,
    and the :math:`h(2)` fitted from it came back **0.688337** against
    **0.722039** from a grid containing 2: a 4.7 % error in the headline Hurst
    exponent, under the label the caller asked for. Off the end of the grid the
    snap is unbounded --- ``row(1000.0)`` returned the :math:`q=4.9` row.

    This is the same defect class :meth:`ScalingExponents.at` was hardened
    against (NaN outside the fitted range, not the clamped endpoint): a moment
    that was never computed is not an interpolation problem.
    """
    q = np.arange(-5.0, 5.01, 0.3)
    assert not np.any(q == 2.0)
    scales = [8, 16, 32, 64, 128, 256]
    surface = structure_functions(level, q=q, scales=scales)

    with pytest.raises(ValueError, match="not on this surface's moment grid"):
        surface.row(2.0)
    with pytest.raises(ValueError, match="not on this surface's moment grid"):
        surface.row(1000.0)

    on_grid = structure_functions(level, q=np.array([-2.0, 0.0, 2.0]), scales=scales)
    h2 = float(loglog_fit(on_grid.scales, on_grid.row(2.0)).slope) / 2.0
    nearest = float(q[int(np.argmin(np.abs(q - 2.0)))])
    h2_snapped = float(loglog_fit(surface.scales, surface.row(nearest)).slope) / 2.0
    assert abs(h2 - h2_snapped) > 0.03, "the snapped row must be a materially different number"


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param([3.0, 2.0, 1.0], id="descending"),
        pytest.param([1.0, 1.0, 2.0], id="duplicated"),
        pytest.param([1.0, np.nan, 2.0], id="nan"),
    ],
)
def test_scaling_exponents_refuse_a_non_monotone_grid(grid: list[float]) -> None:
    r"""Attack: the carrier type itself, constructed directly.

    :class:`ScalingExponents` checked shape and length but not monotonicity,
    while three of its own accessors (``at``, ``h2``, ``hurst_tau_root``) read
    ``q`` as an increasing abscissa. A duplicated knot is just as fatal as a
    reversed one: PCHIP refuses it and ``np.interp`` does not.
    """
    q = np.asarray(grid, dtype=np.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        ScalingExponents(q=q, zeta=0.7 * q)


# --------------------------------------------------------------------------- #
# |q| beyond any moment that exists
# --------------------------------------------------------------------------- #


def test_large_q_collapses_onto_the_single_largest_increment(level: FloatArray) -> None:
    r"""Attack: at :math:`|q|=100`, is the estimator measuring the process?

    It is not. Past the largest existing moment,
    :math:`\hat S(q,\delta)\approx\max_j|\Delta_j|^q/N`, so
    :math:`\hat\zeta(q)/q` converges to the log--log exponent of the *sample
    maximum* --- one increment per scale. The log--log plots stay straight and
    nothing in the return value says so. Measured here on exact fBm
    (:math:`H=0.7`, :math:`N=4096`, this seed): the one-point exponent is
    0.5264, and :math:`h(q)` walks 0.7003, 0.5755, 0.5463, 0.5356 at
    :math:`q=2,20,50,100` --- 92 % of the way onto it by :math:`q=100`, on a
    process whose true :math:`h` is 0.7 at every :math:`q`. The estimator
    reports it as an ordinary exponent. This test pins the collapse so that a
    change in the large-:math:`|q|` moment path (an overflow, a dropped
    ``logsumexp``) shows up as a break rather than as a plausible curve.
    """
    scales = log_spaced_scales(level.size).astype(np.float64)
    one_point = _one_point_exponent(level, scales)
    exponents = mfc_exponents(level, q=np.array([2.0, 20.0, 50.0, 100.0]))
    gap = np.abs(exponents.hq - one_point)

    assert np.all(np.isfinite(exponents.zeta))
    assert np.all(np.diff(gap) < 0.0), f"h(q) must approach {one_point:.4f}, got {exponents.hq}"
    assert gap[-1] < 0.1 * gap[0]


def test_ghe_large_q_collapses_onto_the_overlapping_one_point_exponent(
    level: FloatArray,
) -> None:
    r"""Attack: the same collapse on the *overlapping* design, checked independently.

    ``mfc`` averages over disjoint windows; GHE averages over every :math:`t`.
    The large-:math:`q` limit is therefore the exponent of a different sample
    maximum --- :math:`\max_t|X(t+\tau)-X(t)|` over the full overlapping set ---
    and it is computed here from the series alone, without touching the
    estimator's moment path. Measured on this seed with GHE's own
    :math:`\tau\in[2,19]` grid and ``average_tau_max=False``: one-point exponent
    0.64737, against :math:`h(q)` = 0.67841, 0.65301, 0.65137, 0.64922, 0.64800
    at :math:`q=2,20,50,100,200`. The gap closes from 0.0310 to 0.0006, a factor
    of 49, on a process whose :math:`h` is 0.7 at every :math:`q`. Two
    independent window designs collapsing onto their own sample maximum is what
    makes the effect a property of the moment, not of one implementation.
    """
    q = np.array([2.0, 20.0, 50.0, 100.0, 200.0])
    surface = generalised_hurst_structure(level, q=q)
    tau = surface.scales.astype(np.int64)
    peaks = np.array(
        [float(np.max(np.abs(level[int(t) :] - level[: -int(t)]))) for t in tau],
        dtype=np.float64,
    )
    one_point = float(
        loglog_fit(
            tau.astype(np.float64),
            peaks,
            method=FitMethod.WLS,
            weights=surface.counts.astype(np.float64),
        ).slope
    )
    exponents = generalised_hurst(level, q=q, series_kind="level", average_tau_max=False)
    gap = np.abs(exponents.hq - one_point)

    assert np.all(np.isfinite(exponents.zeta))
    assert np.all(np.diff(gap) < 0.0), f"h(q) must approach {one_point:.5f}, got {exponents.hq}"
    assert gap[-1] < 0.05 * gap[0]


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
def test_the_unmeasurable_moment_is_still_reported_as_a_straight_line(
    name: str, level: FloatArray
) -> None:
    r"""Attack: at :math:`q` far past :math:`q^\star`, does the fit look bad?

    It does not, and that is the whole danger. :func:`critical_moment_order`
    puts :math:`q^\star=7.75` on this path, so :math:`q=100` is thirteen times
    past the largest moment the sample can carry; every estimator nonetheless
    returns a finite exponent whose log--log fit has :math:`R^2\ge0.956`
    (0.9666 mfc, 0.9563 mfdfa, 0.9924 mfdma, 0.9898 ghe). Meanwhile
    :math:`h(q)` has walked away from :math:`h(2)` by 0.165 (mfc), 0.051
    (mfdfa), 0.136 (mfdma) --- a fabricated multifractal spread on an exactly
    monofractal process, with a fit-quality number that certifies it.

    The moment order is a *precondition*, not something the regression can
    detect: this test exists so that any future claim that a fit-quality
    statistic guards the large-:math:`q` end is contradicted by a number.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    q = np.array([2.0, 5.0, 20.0, 50.0, 100.0])
    q_star = critical_moment_order(level)
    assert 5.0 < q_star < 12.0, f"q* moved to {q_star:.3f}; the premise needs rechecking"

    exponents = fn(level, q=q, **kwargs)
    r_squared = np.array([f.r_squared for f in exponents.fits], dtype=np.float64)

    assert np.all(np.isfinite(exponents.zeta))
    assert float(r_squared[-1]) > 0.95, f"{name}: R^2 at q=100 is {r_squared[-1]:.4f}"
    assert exponents.hq[-1] < exponents.hq[0]


def test_extreme_negative_moment_on_underflowing_increments_is_loud_or_exact(
    level: FloatArray,
) -> None:
    r"""Attack: :math:`q=-10` on increments of order :math:`10^{-40}`.

    Two acceptable outcomes and no third. Either the estimator refuses,
    because :math:`\log S = -10\log(10^{-40})\approx 921` is past float64 range
    (``mfc``, which fits the moment itself), or it is *exactly* invariant under
    :math:`x\mapsto cx`, because it works in the log domain throughout: a
    rescaling multiplies :math:`S` by :math:`c^q` and moves the intercept
    only, never the slope. Returning a merely-different number would mean the
    moment was formed as ``(a ** q).mean()`` somewhere.
    """
    unit = level / float(np.std(np.diff(level)))
    underflowing = 1e-40 * unit
    q = np.array([-10.0, -5.0, 2.0])

    with pytest.raises(ValueError, match=r"overflow float64|non-positive or non-finite"):
        mfc_exponents(underflowing, q=q)

    for name in ("mfdfa", "mfdma", "ghe"):
        fn, kwargs = SERIES_ESTIMATORS[name]
        big = fn(unit, q=q, **kwargs).zeta
        small = fn(underflowing, q=q, **kwargs).zeta
        assert np.allclose(big, small, rtol=0.0, atol=1e-9), name


# --------------------------------------------------------------------------- #
# Scale grids
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
@pytest.mark.parametrize(
    "scales",
    [
        pytest.param([1, 2, 4, 8, 16, 32, 64], id="scale-of-one"),
        pytest.param([0, 8, 16, 32], id="scale-of-zero"),
        pytest.param([-8, 8, 16, 32], id="negative-scale"),
    ],
)
def test_sub_two_scales_are_refused(name: str, scales: list[int], level: FloatArray) -> None:
    r"""Attack: a lag of 1, 0 or -8.

    :math:`\delta=1` has no window structure to weight and :math:`\delta\le0`
    has no logarithm. Silently dropping them would leave the fit anchored on a
    scale range the caller never sees.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    with pytest.raises(ValueError, match=r">= 2"):
        fn(level, q=np.array([-1.0, 0.0, 1.0, 2.0]), scales=scales, **kwargs)


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
@pytest.mark.parametrize("scales", [pytest.param([16], id="one"), pytest.param([16, 32], id="two")])
def test_one_and_two_scale_fits_are_refused(
    name: str, scales: list[int], level: FloatArray
) -> None:
    """Attack: a power law fitted through two points.

    Two points determine a line exactly: the residual is zero, :math:`R^2` is
    1 by definition and the standard error is undefined. Reporting that as a
    scaling exponent with a confidence interval is the archetype of the defect
    this library exists to avoid, so every entry point must refuse before it
    reaches the regression.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    with pytest.raises(ValueError, match=r"3|three"):
        fn(level, q=np.array([-1.0, 0.0, 1.0, 2.0]), scales=scales, **kwargs)


def test_loglog_fit_refuses_fewer_than_three_points() -> None:
    """Attack: call the regression primitive directly with two points."""
    with pytest.raises(ValueError, match="at least 3 scales"):
        loglog_fit(np.array([1.0, 2.0]), np.array([1.0, 4.0]))


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
def test_no_two_point_fit_survives_into_a_default_analysis(name: str, level: FloatArray) -> None:
    r"""Attack: does any :math:`q` end up fitted over fewer than three scales?

    A per-:math:`q` fit over a two-scale window would return :math:`R^2=1` with
    no error bar and be indistinguishable, in the returned object, from a
    genuinely perfect power law. Every :class:`LogLogFit` in a default run must
    therefore carry ``n_points >= 3``, and a reported :math:`R^2` of exactly 1
    is only admissible with exactly-vanishing residuals.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    exponents = fn(level, **kwargs)
    assert exponents.fits
    for fit in exponents.fits:
        assert fit.n_points >= 3
        assert fit.dof >= 1
        if fit.r_squared == 1.0:
            assert float(np.max(np.abs(fit.residuals))) == 0.0


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
def test_descending_and_duplicated_scales_match_the_canonical_grid(
    name: str, level: FloatArray
) -> None:
    """Attack: a scale list handed in backwards, or with every entry twice.

    Both must be normalised to the same sorted, de-duplicated grid and give a
    bit-identical answer. A duplicated scale entering the regression twice
    would double its weight for no extra information.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    q = np.array([-1.0, 0.0, 1.0, 2.0])
    canonical = fn(level, q=q, scales=[16, 32, 64, 128], **kwargs)
    for variant in ([128, 64, 32, 16], [16, 16, 32, 32, 64, 64, 128, 128]):
        got = fn(level, q=q, scales=variant, **kwargs)
        assert got.structure is not None and canonical.structure is not None
        assert np.array_equal(got.structure.scales, canonical.structure.scales)
        assert np.array_equal(got.zeta, canonical.zeta)


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
def test_scales_at_or_beyond_the_sample_length_are_dropped_and_recorded(
    name: str, level: FloatArray
) -> None:
    r"""Attack: ask for :math:`\delta=n` and :math:`\delta=4n`.

    No increment of lag :math:`n` exists, so the request cannot be honoured.
    What matters is that the surface reports the grid it actually used, so the
    caller can see that the fit does not span the range they asked for; and
    that a grid of *only* inadmissible scales is a hard error rather than an
    empty selection treated as success.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    q = np.array([-1.0, 0.0, 1.0, 2.0])
    got = fn(level, q=q, scales=[8, 16, 32, level.size, 4 * level.size], **kwargs)
    assert got.structure is not None
    assert np.array_equal(got.structure.scales, np.array([8.0, 16.0, 32.0]))

    with pytest.raises((ValueError, DataQualityError)):
        fn(level, q=q, scales=[level.size, 2 * level.size, 4 * level.size], **kwargs)


@pytest.mark.parametrize("name", sorted(SERIES_ESTIMATORS))
def test_non_integer_scales_are_truncated_not_rounded(name: str, level: FloatArray) -> None:
    r"""Attack: hand in ``np.geomspace`` output without rounding it.

    Scales are cast with ``np.asarray(..., dtype=np.int64)``, i.e. truncated
    toward zero, while the library's own :func:`log_spaced_scales` *rounds*.
    So ``[8.7, 16.9, 32.9, 64.9]`` becomes ``[8, 16, 32, 64]`` and not
    ``[9, 17, 33, 65]``. The fit stays self-consistent --- the moments are
    computed at the truncated lags and the surface reports them --- so this is
    pinned as behaviour rather than reported as a wrong number, but it is a
    silent reinterpretation of the request and the asymmetry with the library's
    own default grid is worth a fence.
    """
    fn, kwargs = SERIES_ESTIMATORS[name]
    q = np.array([-1.0, 0.0, 1.0, 2.0])
    got = fn(level, q=q, scales=[8.7, 16.9, 32.9, 64.9], **kwargs)
    assert got.structure is not None
    assert np.array_equal(got.structure.scales, np.array([8.0, 16.0, 32.0, 64.0]))


# --------------------------------------------------------------------------- #
# float64 traps
# --------------------------------------------------------------------------- #


def test_increments_below_the_level_ulp_are_refused_outright(level: FloatArray) -> None:
    r"""Attack: the log-price trap --- a huge level carrying a tiny fluctuation.

    With increments :math:`10^{-16}` of the level, every difference is annihilated
    by the float64 representation of the level itself and the array is exactly
    constant. Every estimator must refuse; returning an exponent computed from
    the round-off pattern would be the worst possible outcome.
    """
    unit = level / float(np.std(np.diff(level)))
    annihilated = 1e8 + 1e-16 * unit
    assert float(np.std(np.diff(annihilated))) == 0.0
    for fn, kwargs in SERIES_ESTIMATORS.values():
        with pytest.raises(DataQualityError, match="constant"):
            fn(annihilated, q=np.array([-1.0, 0.0, 1.0, 2.0]), **kwargs)


def test_partial_cancellation_wrecks_the_exponent_and_is_only_visible_in_provenance(
    level: FloatArray,
) -> None:
    r"""Attack: the same trap one decade short of annihilation.

    ``1e8 + 1e-9 * x`` keeps the series non-constant --- the increments survive,
    quantised onto the :math:`1.5\times10^{-8}` ulp grid of the level --- so
    nothing raises. Measured against the identical series without the offset:
    :math:`h(2)` moves from **0.724 to 0.386** for ``mfc`` (and 0.765 to 0.436
    for ``mfdma``, 0.712 to 0.350 for ``ghe``), because 77 % of the increments
    at the finest lag are now *exactly* zero and are dropped from the moment.
    No warning is emitted: the only trace is
    ``provenance.diagnostics['zeros_dropped']``. This test fences that trace,
    which is the sole defence a caller has against the wrong number.
    """
    unit = level / float(np.std(np.diff(level)))
    q = np.array([-1.0, 0.0, 1.0, 2.0])
    clean = mfc_exponents(1e-9 * unit, q=q)
    quantised = mfc_exponents(1e8 + 1e-9 * unit, q=q)

    assert abs(quantised.h2 - clean.h2) > 0.25

    surface = quantised.structure
    assert surface is not None and surface.provenance is not None
    dropped = np.asarray(surface.provenance.diagnostics["zeros_dropped"], dtype=np.float64)
    assert surface.counts is not None
    fraction = dropped / (dropped + surface.counts.astype(np.float64))
    assert float(fraction.max()) > 0.5

    clean_surface = clean.structure
    assert clean_surface is not None and clean_surface.provenance is not None
    assert int(np.sum(clean_surface.provenance.diagnostics["zeros_dropped"])) == 0


# --------------------------------------------------------------------------- #
# Warning hygiene
# --------------------------------------------------------------------------- #


def test_src_never_suppresses_warnings_globally() -> None:
    """Attack: is a RuntimeWarning being hidden rather than reasoned about?

    ``np.errstate`` scopes a numerical exception to the expression that is
    known to raise it. ``warnings.filterwarnings('ignore')`` or
    ``simplefilter('ignore')`` at import scope would silence the whole process,
    including the caller's own diagnostics, and would defeat this suite's
    ``filterwarnings = ["error::RuntimeWarning"]``.
    """
    src = Path(__file__).resolve().parents[2] / "src" / "mmar"
    pattern = re.compile(r"""(filterwarnings|simplefilter)\(\s*["']ignore["']""")
    offenders = [
        f"{path.relative_to(src)}:{lineno}"
        for path in sorted(src.rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]
    assert not offenders, "global warning suppression in src/: " + ", ".join(offenders)
