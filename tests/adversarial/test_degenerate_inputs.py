r"""Pathological-but-legal inputs: is the answer right, or is the refusal informative?

The contract attacked here is the one the package README and
:mod:`mmar.core.validate` both state --- *loud failures instead of quiet NaNs*.
For every entry point the admissible outcomes are exactly two: a correct number,
or an exception that names what is wrong with the input. A plausible curve
fitted through two surviving windows, a headline :math:`h(2)` read off a moment
that was never fitted, or a series rejected as "constant" because its variance
underflowed are all failures of that contract, and each is exercised below.

Every attack is on a legal object: a float64 array of finite reals, a
non-negative measure, or a numpy container the standard library hands out
(masked, read-only, non-contiguous). Nonsense inputs are not attacked --- a
clear exception on nonsense is correct behaviour, not a defect.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from mmar.analysis import ESTIMATORS, analyze
from mmar.core.rng import make_rng
from mmar.core.types import FloatArray
from mmar.core.validate import DataQualityError, validate_measure, validate_series
from mmar.estimators.box_counting import coarse_holder, generalised_dimensions, histogram_spectrum
from mmar.estimators.chhabra_jensen import chhabra_jensen
from mmar.estimators.ghe import generalised_hurst
from mmar.estimators.mfdma import mfdma
from mmar.estimators.pointwise import leader_holder, oscillation_holder, wavelet_holder
from mmar.estimators.structure_function import mfc_exponents
from mmar.inference.bootstrap import (
    bootstrap_descriptors,
    optimal_block_length,
    politis_white_block_length,
)
from mmar.inference.tests import monofractal_test
from mmar.models.fbm import fbm

pywt = pytest.importorskip("pywt", reason="the wavelet estimators need PyWavelets")

_N = 2048
"""Long enough for every estimator's default ladder, short enough to stay fast."""


def _fbm(h: float = 0.7, *, seed: int = 21, n: int = _N) -> FloatArray:
    return np.asarray(fbm(n, h, seed=make_rng(seed)), dtype=np.float64)


def _pathological_series() -> dict[str, FloatArray]:
    """Series that are legal float64 and hostile mathematics."""
    t = np.arange(_N, dtype=np.float64)
    half = _N // 2
    return {
        "two_valued": np.where(np.arange(_N) % 2 == 0, 0.0, 1.0),
        "single_spike": np.concatenate([np.zeros(half), [1.0], np.zeros(_N - half - 1)]),
        "linear_trend": t,
        "exponential": np.exp(t / _N * 20.0),
        "sine": np.sin(2.0 * np.pi * t / 128.0),
        "square_wave": np.sign(np.sin(2.0 * np.pi * t / 128.0)),
        "step": np.where(t < half, 0.0, 1.0),
        "near_overflow": np.full(_N, 1e300) + t * 1e290,
        "near_underflow": (t + 1.0) * 1e-300,
        "spans_both": np.concatenate([np.full(half, 1e-300), np.full(_N - half, 1e300)]),
    }


# --------------------------------------------------------------------------- #
# 1. The constant test must be a constant test, not a variance test
# --------------------------------------------------------------------------- #


def test_underflowing_ramp_is_not_rejected_as_constant() -> None:
    r"""Attack: a strictly monotone ramp scaled to :math:`10^{-300}`.

    ``validate_series`` gated on ``np.std(a) == 0.0``. The variance of a series
    of magnitude :math:`10^{-300}` is :math:`10^{-600}`, which underflows float64
    to exactly zero, so a perfectly well-conditioned linear ramp --- 2048
    distinct, strictly increasing values --- was refused with the message
    "series is constant". The refusal is not merely unhelpful, it is *false*: the
    series carries an exactly recoverable :math:`h(2)=1`, which is what the
    estimator returns once the gate is fixed. Constancy is ``min == max``, a
    comparison, not a floating-point reduction.
    """
    ramp = (np.arange(_N, dtype=np.float64) + 1.0) * 1e-300
    assert ramp.min() != ramp.max()

    out = validate_series(ramp)
    assert out.size == _N

    exponents = ESTIMATORS["ghe"](ramp)
    assert exponents.h2 == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("estimator", ["mfdfa", "mfdma"])
def test_variance_estimators_refuse_an_unsquarable_series_by_name(estimator: str) -> None:
    r"""Attack: a series of magnitude :math:`10^{300}`, squared before any logarithm.

    MF-DFA (``resid * resid``) and MF-DMA (``block * block``) form a variance
    *before* taking a logarithm, so a record beyond :math:`1.34\times10^{154}`
    overflowed to :math:`\infty` behind a bare ``RuntimeWarning: overflow
    encountered in multiply`` and then carried the infinity into
    :math:`F_q(n)`. Under this package's own
    ``filterwarnings = ["error::RuntimeWarning"]`` the caller saw an exception
    raised from inside the estimator's arithmetic, naming neither the input nor
    a remedy; with warnings merely printed the fit continued on infinities.

    The log-domain estimators (GHE, the structure functions) are genuinely
    unaffected and are checked below not to be gated by mistake --- the guard
    belongs where the squaring is, not in ``validate_series``.
    """
    x = _fbm(n=1024)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(DataQualityError, match="square overflows float64"):
            analyze(x * 1e300, estimator=estimator)
        # The remedy the message offers is exact, not a workaround: a merely
        # large but squarable multiple of the same path gives the same exponent.
        base = analyze(x, estimator=estimator).summary()["h2"]
        big = analyze(x * 1e20, estimator=estimator).summary()["h2"]
    assert math.isfinite(base)
    assert big == pytest.approx(base, abs=1e-9)


def test_log_domain_estimators_are_not_gated_by_the_squarability_guard() -> None:
    """Attack: the same :math:`10^{300}` series through the log-domain estimators.

    GHE never squares the data, so it must still answer; the structure functions
    do overflow at :math:`q=5` but say so themselves, in their own vocabulary.
    A guard placed in shared validation would have refused both, which is a
    refusal the mathematics does not license.
    """
    x = _fbm(n=1024)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert math.isfinite(analyze(x * 1e300, estimator="ghe").summary()["h2"])
        with pytest.raises(ValueError, match="overflow float64"):
            analyze(x * 1e300, estimator="structure_function")


# --------------------------------------------------------------------------- #
# 2. h(2) must come from a fitted moment
# --------------------------------------------------------------------------- #


def test_h2_is_not_fabricated_when_the_q_grid_stops_short_of_two() -> None:
    r"""Attack: ask for a spectrum on :math:`q\in[-1,1]`, then read ``h2``.

    A narrow moment window is legal and often deliberate --- wide :math:`|q|` is
    the classic way to fabricate a spectrum, and
    :func:`~mmar.core.validate.validate_q_grid` says so. But
    ``ScalingExponents.at`` used :func:`numpy.interp` with its default
    *clamping*, so ``at(2.0)`` silently returned :math:`\zeta(q_{\max})`. On the
    fBm below (:math:`H=0.7`, full-grid :math:`h(2)=0.654`) the narrow grid
    reported ``h2 = 0.329`` --- exactly :math:`\zeta(1)/2`, half the truth --- as
    a headline number in ``MultifractalReport.summary()`` and in
    ``consensus()``, with nothing to indicate the moment was never fitted.
    """
    x = _fbm()
    full = analyze(x, estimator="mfdfa", q=np.linspace(-5.0, 5.0, 41))
    narrow = analyze(x, estimator="mfdfa", q=np.linspace(-1.0, 1.0, 21))

    assert math.isfinite(full.summary()["h2"])

    h2_narrow = narrow.summary()["h2"]
    zeta_at_grid_edge = float(narrow.exponents.zeta[-1]) / 2.0
    assert not math.isfinite(h2_narrow), (
        f"h2={h2_narrow:.4f} was reported from a grid that stops at "
        f"q={float(narrow.exponents.q[-1])}; it is the clamped edge value "
        f"{zeta_at_grid_edge:.4f}, not h(2)"
    )
    assert not math.isfinite(narrow.exponents.at(2.0))
    assert narrow.exponents.at(0.5) == pytest.approx(
        float(np.interp(0.5, narrow.exponents.q, narrow.exponents.zeta))
    )


# --------------------------------------------------------------------------- #
# 3. A mask is not decoration
# --------------------------------------------------------------------------- #


def test_masked_entries_are_not_read_as_data() -> None:
    """Attack: a masked array whose masked cells hold a sentinel fill value.

    ``np.asarray`` on a :class:`numpy.ma.MaskedArray` returns the underlying
    buffer and throws the mask away, so the ``1e20`` the caller explicitly
    marked as missing was analysed as a genuine observation --- the single
    largest increment in the record, which is what every :math:`q>0` moment is
    dominated by. Masked entries must reach ``nan_policy``, which is where the
    caller already had to state what missing data means.
    """
    clean = _fbm(n=256)
    contaminated = clean.copy()
    contaminated[100] = 1e20
    masked = np.ma.masked_array(contaminated, mask=np.zeros(clean.size, dtype=bool))
    masked.mask[100] = True

    with pytest.raises(DataQualityError, match="non-finite"):
        validate_series(masked)

    dropped = validate_series(masked, nan_policy="drop")
    assert dropped.size == clean.size - 1
    assert float(np.abs(dropped).max()) < 1e3

    positive = np.ma.masked_array(np.abs(contaminated), mask=masked.mask)
    with pytest.raises(DataQualityError, match="non-finite"):
        validate_measure(positive)


# --------------------------------------------------------------------------- #
# 4. Nobody writes on the caller's array
# --------------------------------------------------------------------------- #


def test_validated_series_is_an_owned_writable_copy() -> None:
    """Attack: a read-only series, and a check that validation does not alias.

    ``np.ascontiguousarray`` returns the caller's own buffer for a contiguous
    float64 input. Two consequences, both real: every estimator sat one in-place
    operation away from rewriting the caller's data, and a read-only array --- a
    memory-mapped record, or one frozen on purpose --- propagated its flag all
    the way into PyWavelets, which died with ``ValueError: buffer source array
    is read-only``, naming neither the offending input nor any remedy.
    """
    x = _fbm(n=512)
    assert not np.shares_memory(validate_series(x), x)

    frozen = x.copy()
    frozen.flags.writeable = False
    for name, estimator in ESTIMATORS.items():
        estimator(frozen)  # must not raise
        assert not frozen.flags.writeable, f"{name} unfroze the input"


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_no_series_estimator_mutates_its_input(name: str) -> None:
    """Attack: hand each estimator a buffer and compare it bit-for-bit afterwards.

    In-place detrending or normalisation of the caller's array is a common and
    almost undetectable defect: the first analysis is right and every subsequent
    use of the same variable is wrong.
    """
    x = _fbm(n=1024)
    before = x.copy()
    ESTIMATORS[name](x)
    assert np.array_equal(x, before), f"{name} modified the caller's series in place"
    assert x.dtype == before.dtype


def test_no_measure_or_pointwise_routine_mutates_its_input() -> None:
    """Attack: the same check for the measure-domain and pointwise entry points."""
    rng = make_rng(5)
    mu = np.abs(rng.standard_normal(1024))
    x = _fbm(n=1024)

    measure_calls: Sequence[Callable[[FloatArray], Any]] = (
        chhabra_jensen,
        generalised_dimensions,
        lambda m: coarse_holder(m, 1.0 / 16.0),
        lambda m: histogram_spectrum(m, bins=8),
    )
    for call in measure_calls:
        before = mu.copy()
        call(mu)
        assert np.array_equal(mu, before), f"{call} modified the measure in place"

    for call in (oscillation_holder, wavelet_holder, leader_holder):
        before = x.copy()
        call(x)
        assert np.array_equal(x, before), f"{call.__name__} modified the series in place"


# --------------------------------------------------------------------------- #
# 5. A slope through two survivors is not an estimate
# --------------------------------------------------------------------------- #


def test_spike_on_a_flat_background_is_refused_not_fitted_through_two_segments() -> None:
    r"""Attack: 2047 zeros and one 1, handed to MF-DMA.

    The moving-average residual vanishes in every segment that does not contain
    the spike, and those segments are dropped (correctly --- at :math:`q=-5` they
    would contribute :math:`F_v^q=\infty`). What survived was **two** segments at
    every one of the 22 scales, and the estimator fitted the whole
    :math:`q`-surface through them, returning :math:`h(2)=-0.49` and
    :math:`\alpha_0=-0.75` --- negative Hölder exponents, which no bounded
    function has --- with ``segments_by_scale = [2, 2, ..., 2]`` buried in the
    diagnostics and no error.

    The module's own contract (docstring, bias 3) is that "scales leaving fewer
    than ``min_segments`` segments are refused"; ``_resolve_scales`` enforces it
    on the *nominal* count and the post-drop check must enforce the same bar.
    """
    n = 4096  # at this length two segments survive at every scale, which is the trap
    spike = np.concatenate([np.zeros(n // 2), [1.0], np.zeros(n // 2 - 1)])
    with pytest.raises(ValueError, match=r"fewer than min_segments=4"):
        mfdma(spike, integrate=False)

    # A genuine series still passes: the guard rejects degeneracy, not signal.
    assert math.isfinite(mfdma(_fbm(n=1024), integrate=False).h2)


# --------------------------------------------------------------------------- #
# 6. The contract, swept
# --------------------------------------------------------------------------- #

_HEADLINE = ("h2", "alpha_0", "width", "f_max")


@pytest.mark.parametrize("case", sorted(_pathological_series()))
@pytest.mark.parametrize("estimator", ["mfdfa", "mfdma", "ghe", "structure_function"])
def test_pathological_series_give_a_number_or_a_reason(case: str, estimator: str) -> None:
    """Attack: sweep every degenerate shape against every cheap estimator.

    The admissible outcomes are a finite summary or a ``ValueError`` that says
    something. What must never happen is a summary carrying NaN or infinity in
    a headline field: those are the numbers a caller quotes, and a NaN that
    reaches them has passed every guard the library has.

    ``hurst_tau_root`` is excluded deliberately --- it is *documented* to be NaN
    when :math:`\\tau` has no positive root, which is the honest answer and not a
    degradation.
    """
    series = _pathological_series()[case]
    try:
        summary = analyze(series, estimator=estimator).summary()
    except ValueError as exc:  # DataQualityError is a ValueError
        assert len(str(exc)) > 20, "an exception must explain which input property failed"
        return
    for field in _HEADLINE:
        value = summary[field]
        assert isinstance(value, float)
        assert math.isfinite(value), f"{estimator} on {case}: summary['{field}'] = {value}"


@pytest.mark.parametrize(
    ("case", "measure"),
    [
        ("all_zero", np.zeros(1024)),
        ("one_cell_holds_everything", np.concatenate([[1.0], np.zeros(1023)])),
        ("negative_cells", np.linspace(-1.0, 1.0, 1024)),
        ("total_mass_100", np.full(1024, 100.0 / 1024.0)),
        ("uniform", np.full(1024, 1.0)),
        ("length_not_a_power_of_two", np.abs(np.sin(np.arange(1021, dtype=np.float64)))),
        ("two_atoms", np.concatenate([[0.5], np.zeros(511), [0.5], np.zeros(511)])),
    ],
)
def test_pathological_measures_give_a_number_or_a_reason(case: str, measure: FloatArray) -> None:
    r"""Attack: the measure-domain entry points on degenerate mass distributions.

    Same contract as for series. The two structural anchors are checked wherever
    an answer *is* returned: :math:`f(\alpha)` may not exceed the dimension of
    the support by more than the estimator's own finite-size error (the largest
    excess observed over these cases is :math:`+0.006`, on the prime-length
    measure), and :math:`\tau(1)=0` for a conservative measure --- the box
    partition normalises to unit mass, so every measure here is conservative by
    construction.
    """
    for call in (chhabra_jensen, generalised_dimensions):
        try:
            result = call(measure)
        except ValueError as exc:
            assert len(str(exc)) > 20
            continue
        if hasattr(result, "descriptors"):
            descriptors = result.descriptors()
            assert math.isfinite(descriptors.alpha_0)
            assert math.isfinite(descriptors.width)
            assert descriptors.f_max <= 1.05, f"{case}: f_max = {descriptors.f_max} exceeds d=1"
        else:
            assert float(result.at(1.0) - result.d_support) == pytest.approx(0.0, abs=1e-9)


def test_measure_exponents_are_invariant_to_the_total_mass() -> None:
    r"""Attack: the same measure scaled by :math:`10^{7}`.

    :math:`Z(q,\delta)\mapsto M^{q}Z(q,\delta)` under :math:`\mu\mapsto M\mu`, so
    the total mass moves the *intercept* of every log--log fit and no slope. A
    normalisation applied at the wrong point --- per scale rather than once ---
    would break exactly this and shift every exponent by :math:`\tau(1)`.
    """
    rng = make_rng(2)
    mu = np.abs(rng.standard_normal(1024))
    base = chhabra_jensen(mu)
    scaled = chhabra_jensen(mu * 1e7)
    assert np.allclose(base.alpha, scaled.alpha, atol=1e-10)
    assert np.allclose(base.f_alpha, scaled.f_alpha, atol=1e-10)


# --------------------------------------------------------------------------- #
# 7. Robustness of the reported summary to one contaminated sample
# --------------------------------------------------------------------------- #


def test_one_outlier_leaves_alpha_0_alone_but_not_the_width() -> None:
    r"""Attack: insert a single :math:`10^{6}\sigma` increment into clean fBm.

    Measured at :math:`n=2048`, :math:`H=0.7`, MF-DFA, against a 60-resample
    moving-block bootstrap of the point estimate
    (:math:`\mathrm{sd}(\alpha_0)=0.070`, :math:`\mathrm{sd}(\Delta\alpha)=0.076`):

    ==========  ====================  =====================
    outlier     :math:`\Delta\alpha_0`  :math:`\Delta(\text{width})`
    ==========  ====================  =====================
    10 sigma    :math:`-0.04\,\hat\sigma`   :math:`-0.13\,\hat\sigma`
    100 sigma   :math:`+0.03\,\hat\sigma`   :math:`+8.96\,\hat\sigma`
    1e6 sigma   :math:`+1.13\,\hat\sigma`   :math:`+17.8\,\hat\sigma`
    ==========  ====================  =====================

    So the apex is robust --- one sample cannot move it beyond its own sampling
    error, which is the property asserted here. The **width** is not: a single
    point turns :math:`\Delta\alpha=0.072` into :math:`1.43`, twenty times
    wider, and at :math:`100\sigma` it does so while the ``concavity_defect``
    guard *falls* from :math:`3.8` to :math:`0.3` standard errors, i.e. the
    contaminated report looks better conditioned than the clean one. That
    asymmetry is recorded here rather than asserted: it is a property of the
    formalism at :math:`q=5`, not a coding defect, and the remedy is
    :func:`mmar.inference.tests.moment_existence`, which ``analyze`` does not
    run by default.
    """
    x = _fbm()
    sigma = float(np.std(np.diff(x)))
    clean = analyze(x, estimator="mfdfa").spectrum.descriptors()

    boot = bootstrap_descriptors(
        x,
        lambda s: ESTIMATORS["mfdfa"](s, integrate=False),
        n_resamples=60,
        seed=99,
        names=("alpha_0", "width"),
        interval="percentile",
        jackknife_blocks=None,
    )
    sd_alpha_0, sd_width = float(boot.stderr[0]), float(boot.stderr[1])
    assert sd_alpha_0 > 0.0 and sd_width > 0.0

    contaminated = x.copy()
    contaminated[_N // 2] += 1e6 * sigma
    hit = analyze(contaminated, estimator="mfdfa").spectrum.descriptors()

    shift_alpha_0 = abs(hit.alpha_0 - clean.alpha_0) / sd_alpha_0
    shift_width = abs(hit.width - clean.width) / sd_width
    print(
        f"\none 1e6-sigma outlier: alpha_0 {clean.alpha_0:+.4f} -> {hit.alpha_0:+.4f} "
        f"({shift_alpha_0:.2f} boot-sd), width {clean.width:.4f} -> {hit.width:.4f} "
        f"({shift_width:.2f} boot-sd)"
    )
    assert shift_alpha_0 < 3.0, (
        f"one sample moved alpha_0 by {shift_alpha_0:.2f} bootstrap standard errors; "
        "the apex of the spectrum is not supposed to be a function of a single point"
    )


# --------------------------------------------------------------------------- #
# 8. Scale invariance: the amplitude of a series is not one of its exponents
# --------------------------------------------------------------------------- #
#
# ``mmar.core.validate.require_squarable`` states the library-wide invariant in
# its own words: "Every exponent this library reports is invariant under
# x -> c*x". Rescaling shifts the intercept of every log--log fit and no slope.
# Each attack below rescales a *legal* series and asks whether the answer moved.


def _fgn(h: float = 0.8, *, seed: int = 3, n: int = 2048) -> FloatArray:
    """Increments of an exact fBm path: a stationary series with real memory."""
    return np.asarray(np.diff(fbm(n, h, seed=make_rng(seed))), dtype=np.float64)


def test_politis_white_block_length_survives_a_rescaled_series() -> None:
    r"""Attack: the Politis--White rule on the same fGn at eight amplitudes.

    Every quantity in the rule is a second or fourth power of the data, so it
    collapsed long before float64 did. Measured on fGn (:math:`H=0.8`,
    :math:`n=2047` increments), which needs :math:`b^\*=85`:

    ==============  =========  =====================================
    :math:`c`       :math:`b^\*`  mechanism
    ==============  =========  =====================================
    :math:`1`       85         --
    :math:`10^{-140}`  1       :math:`\hat g(0)^2` underflows although
                               :math:`\hat g(0)=5.3\times10^{-285}` is fine,
                               so ``_optimum`` sees ``d_hat = 0``
    :math:`10^{-160}`  1       :math:`\hat R(0)` itself underflows;
                               ``acov[1:]/acov[0]`` is 0/0 --- a bare
                               ``RuntimeWarning`` and an all-NaN correlogram
    :math:`10^{160}`   1       :math:`\hat R(0)` overflows to :math:`\infty`,
                               then :math:`\infty-\infty` in the FFT
    ==============  =========  =====================================

    ``b^\* = 1`` is not a mild degradation: it turns the moving-block bootstrap
    into an i.i.d. bootstrap of a series with memory, and
    ``tests/unit/test_bootstrap.py::test_block_length_one_destroys_long_memory_variance``
    measures that this understates the true sampling standard deviation by more
    than a factor of two. Every confidence interval downstream is then too
    narrow, and nothing in :class:`BlockLength` says so.

    The rule is exactly scale invariant --- :math:`\hat\rho` is a ratio of
    autocovariances and only :math:`\hat G^2/\hat g(0)^2` reaches
    :math:`b^\*` --- so the fix evaluates it on the series divided by a power of
    two (exact, introducing no round-off) and scales the two reported
    diagnostics back. On a well-scaled series the outputs are bit-identical.
    """
    x = _fgn()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        reference = politis_white_block_length(x)
        assert reference.moving > 8, "this fixture must need a real block to be a test"
        # The silent cases first: at 1e-140 nothing warns, b* is simply 1.
        for c in (1e-140, 1e-8, 1e8, 1e140, 1e-160, 1e-300, 1e160, 1e300):
            scaled = politis_white_block_length(x * c)
            assert scaled.moving == reference.moving, (
                f"block length {scaled.moving} at amplitude {c:g} against "
                f"{reference.moving} at amplitude 1, for the same series"
            )
            assert scaled.stationary == reference.stationary
            assert scaled.m_hat == reference.m_hat
            assert optimal_block_length(x * c) == reference.moving


def test_block_length_diagnostics_are_unchanged_on_a_well_scaled_series() -> None:
    """The scale-invariance repair must be a bit-for-bit no-op on ordinary data.

    ``g_hat`` and ``spectrum_at_zero`` carry units of variance and are reported
    in the caller's units. Dividing by a power of two and multiplying back is
    exact whenever neither operation over- or underflows, which is every series
    a user will actually pass.
    """
    x = _fgn()
    b = politis_white_block_length(x)
    assert b.g_hat == 0.000750717844660807
    assert b.spectrum_at_zero == 5.346479700262868e-05
    assert (b.moving, b.stationary, b.m_hat, b.bandwidth) == (85, 74, 51, 51)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN DEFECT: wavelet_leaders fits float64 round-off on a signal the "
        "wavelet annihilates, and reports alpha_0 = 0.28 for a straight line"
    ),
)
def test_wavelet_leaders_does_not_fit_round_off_on_a_polynomial() -> None:
    r"""Attack: the default estimator on a pure linear ramp, :math:`x_t = t`.

    A straight line is :math:`C^\infty`: :math:`\alpha(t)\equiv 1` everywhere
    and the spectrum is the single point :math:`\{(1,1)\}`. ``db3`` has three
    vanishing moments and annihilates it *exactly*, so the detail coefficients
    are algebraically zero and what survives is float64 round-off. Measured at
    :math:`n=4096`: interior :math:`|d_{j,k}|` runs from
    :math:`4.3\times10^{-14}` at :math:`j=1` to :math:`2.1\times10^{-12}` at
    :math:`j=7`, against a signal of size :math:`4.1\times10^{3}` --- a relative
    :math:`10^{-16}`, i.e. nothing.

    Round-off in a linear filter grows systematically with the length of the
    analysing support, so :math:`\log|d|` against :math:`j` is a clean straight
    line and the estimator reports a *confident* spectrum:

    ======================  =============  ==========
    quantity                reported        truth
    ======================  =============  ==========
    :math:`h(2)`            :math:`+0.178`  :math:`1`
    :math:`\alpha_0`        :math:`+0.281`  :math:`1`
    :math:`\Delta\alpha`    :math:`1.178`   :math:`0`
    ======================  =============  ==========

    ``support_truncated`` is set, but it is documented to mean the width is a
    *lower* bound --- an assertion in the wrong direction --- and
    :math:`\alpha_0` carries no qualification at all. The number also moves with
    the amplitude (:math:`\alpha_0 = 0.233 / 0.281 / 0.300` at :math:`c =
    10^{-6} / 1 / 10^{6}`), which is the proof that it is round-off and not
    signal: every exponent in this library is invariant under
    :math:`x\mapsto cx`.

    MF-DFA already refuses this input by name --- "scale s=20 leaves only 0
    windows with a detrended variance above the round-off floor" --- from an
    explicit floor :math:`(s\varepsilon)^2\langle w^2\rangle` in
    ``_window_variances``. The wavelet families have no equivalent, and neither
    does WTMM (:math:`\alpha_0=+0.106`, :math:`\Delta\alpha=0.849` on the same
    ramp). The remedy is that floor, transported to the multiresolution
    coefficient; it is a change to the estimator, not a tolerance.
    """
    ramp = np.arange(4096, dtype=np.float64) + 1.0
    d = analyze(ramp, estimator="wavelet_leaders").spectrum.descriptors()
    assert abs(d.alpha_0 - 1.0) < 0.2, (
        f"alpha_0 = {d.alpha_0:+.4f} for a C-infinity signal whose Holder "
        f"exponent is 1 everywhere (width {d.width:.4f}, truth 0)"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN DEFECT: leader_holder returns h = 0.047 for a straight line; "
        "the same round-off floor is missing from the pointwise field"
    ),
)
def test_pointwise_leaders_do_not_fabricate_an_exponent_for_a_straight_line() -> None:
    r"""Attack: the same ramp through the pointwise estimators.

    The module docstring promises exactly this case is handled: "a leader can
    vanish on a locally polynomial signal ... those cells are dropped ... and a
    point with fewer than ``min_points`` usable scales gets ``h = nan`` rather
    than a fabricated slope." The guard tests ``values <= 0.0``, and round-off
    is strictly positive, so it never fires:
    ``n_nonpositive_cells = 0`` on this input and 1280 samples receive a number.

    Measured at :math:`n=4096`: ``leader_holder`` reports a median
    :math:`\hat h = +0.047` (range :math:`-0.217` to :math:`+0.390`) with a
    median :math:`R^2` of :math:`0.93`, and ``wavelet_holder`` :math:`+0.210`.
    ``oscillation_holder`` --- which never divides by a vanishing quantity ---
    returns :math:`+1.040` uniformly, the right answer up to its documented
    saturation at 1. So the field is not merely noisy, it is wrong on the side
    that matters: a reader would locate singularities everywhere on a straight
    line.

    That the number is round-off and nothing else is settled by two controls.
    Adding noise of *relative* size :math:`10^{-15}` (amplitude
    :math:`10^{-12}` on a signal of size :math:`4\times10^{3}`) moves the median
    from :math:`+0.035` to :math:`-0.304`; and rescaling the ramp by
    :math:`10^{\pm6}` moves it to :math:`+0.071` / :math:`+0.046`. A cubic, which
    ``db3`` does *not* annihilate, is answered correctly and saturates at exactly
    :math:`+3.000`.
    """
    ramp = np.arange(4096, dtype=np.float64) + 1.0
    field = leader_holder(ramp)
    finite = field.h[np.isfinite(field.h)]
    median = float(np.median(finite)) if finite.size else float("nan")
    assert finite.size == 0 or median > 0.9, (
        f"median h = {median:+.4f} over {finite.size} samples of a C-infinity "
        "signal; the correct answer is NaN (undetermined) or h >= 1"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN DEFECT: monofractal_test rejects H0 for an exactly monofractal "
        "series, and the verdict depends on the amplitude of the input"
    ),
)
def test_monofractal_verdict_does_not_depend_on_the_amplitude() -> None:
    r"""Attack: the chi-square monofractality test on an exact straight line.

    A ramp has :math:`\zeta(q)=q` identically, and GHE recovers it to machine
    precision at every amplitude: :math:`\hat h(2)=1.000000000000` for all four
    below, with :math:`\max_q|\hat\zeta - q\hat H|` between
    :math:`1.8\times10^{-15}` and :math:`1.9\times10^{-13}`. It is as monofractal
    as a series can be. Yet the reported verdict is:

    ==================  ==============  ===================================
    :math:`c`           :math:`X^2`     :math:`p`
    ==================  ==============  ===================================
    :math:`1`           :math:`7.3`     :math:`1.0` (not rejected)
    :math:`10^{150}`    :math:`41.4`    :math:`0.37`
    :math:`10^{-150}`   :math:`48.9`    :math:`0.13`
    :math:`10^{-300}`   :math:`100.7`   :math:`2.4\times10^{-7}` (**rejected**)
    ==================  ==============  ===================================

    :math:`X^2=\sum_i (\hat\zeta_i-\hat Hq_i)^2/\hat\sigma_i^2` is a ratio of two
    quantities that are both pure round-off here --- the residuals *and* the
    regression standard errors, the latter at :math:`5\times10^{-16}`. The ratio
    of two round-off patterns is not scale invariant, so a categorical verdict
    ("H0 monofractal is rejected") is decided by the amplitude of the input.
    The docstring's ``.. warning::`` covers the *opposite* failure (the test is
    anti-conservative through correlated errors); it says nothing about an
    exactly-fitting regression, where the statistic is a disguised 0/0.
    """
    ramp = np.arange(4096, dtype=np.float64) + 1.0
    verdicts = {}
    for c in (1.0, 1e-150, 1e-300, 1e150):
        exponents = generalised_hurst(ramp * c)
        assert exponents.h2 == pytest.approx(1.0, abs=1e-9)
        verdicts[c] = monofractal_test(exponents).p_value
    assert min(verdicts.values()) > 0.05, (
        f"H0 (monofractal) rejected for an exactly monofractal series: {verdicts}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN DEFECT: mfc_exponents accepts n=65, fits 0.30 decades over four "
        "increments and reports h(2) = -0.006 with nothing to qualify it"
    ),
)
def test_structure_function_min_length_is_not_a_usable_scaling_range() -> None:
    r"""Attack: the shortest series the structure-function estimator accepts.

    ``mfc_exponents`` sets ``min_length=64``, which is the library declaring 64
    points sufficient. What that buys at :math:`n=65` is a scale ladder of
    :math:`8\ldots16` --- 0.301 decades --- whose coarsest scale is an average of
    **four** increments. ``ScalingRange.decades`` says in its own docstring
    "below ~1 be suspicious"; ``MultifractalReport.summary()`` reports neither
    it nor ``r_squared``.

    On exact fBm with :math:`H=0.7` (so :math:`h(q)\equiv0.7` and
    :math:`\Delta\alpha=0`), seven seeds at :math:`n=65`:

    ======  ==================  ==================  ====================  ======
    seed    :math:`\hat h(2)`   :math:`\Delta\alpha`  ``support_truncated``  :math:`R^2`
    ======  ==================  ==================  ====================  ======
    3       :math:`-0.0059`     4.186               ``False``             0.40
    21      :math:`+0.302`      1.469               ``True``              0.09
    100     :math:`+0.351`      2.226               ``True``              0.41
    101     :math:`+0.863`      2.292               ``True``              0.75
    102     :math:`+0.631`      2.627               ``False``             0.67
    103     :math:`-0.651`      0.000               ``True``              0.34
    104     :math:`+0.789`      0.000               ``True``              0.16
    ======  ==================  ==================  ====================  ======

    Two of seven are **negative** --- no continuous path has a negative
    generalised Hurst exponent --- and seed 3 reports a spectrum 4.19 wide with
    ``support_truncated = False``, i.e. as a fully resolved measurement, while
    ``analyze``'s ``concavity_defect`` (0.083 against a median
    :math:`\hat\sigma` of 1.89) says the fit is well conditioned. Over 20 seeds
    :math:`\hat h(2)` spans :math:`[-0.651, +0.958]` with 10% negative.

    MF-DFA raised its own floor to 256 for this reason. Either this estimator
    does the same, or a fit spanning under a decade is refused, or the summary
    carries the decades and :math:`R^2` that would let a reader discount it ---
    but a bare ``h2 = -0.0059`` is not one of the three.
    """
    x = np.asarray(fbm(65, 0.7, seed=make_rng(3)), dtype=np.float64)
    exponents = mfc_exponents(x)
    assert exponents.h2 > 0.0, (
        f"h(2) = {exponents.h2:+.6f} on fBm with H=0.7, from a fit over "
        f"{exponents.scaling_range.decades:.3f} decades "
        f"(R^2 = {exponents.scaling_range.r_squared:.3f})"
    )
