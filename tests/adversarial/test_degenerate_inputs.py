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
from mmar.estimators.mfdma import mfdma
from mmar.estimators.pointwise import leader_holder, oscillation_holder, wavelet_holder
from mmar.inference.bootstrap import bootstrap_descriptors
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
