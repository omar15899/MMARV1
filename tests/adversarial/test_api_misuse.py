"""What the library does when the *caller* is wrong.

A wrong answer is worse than a refusal, so every entry point attacked here is
one where a plausible user error was absorbed silently instead of being
reported: a two-dimensional array ravelled into an interleaved series, a
misspelt ``nan_policy`` falling through to the most destructive branch, a
complex array quietly stripped of its imaginary part, a confidence level given
in percent turning every interval into ``NaN``, a malformed CLI argument
escaping as a traceback, and two different results filed in one manifest under
one name.

The second half attacks *error quality* directly. An exception that names the
wrong cause is worse than no exception, because it sends the caller to fix
something that is not broken: a bare numpy ``TypeError`` from three frames down
that mentions neither the argument nor the remedy, a diagnosis blaming the
series length for a constraint the ``order`` argument imposed, a scalar reported
as a wide array, a string iterated into its characters, and a calibration
blaming :math:`\\tau` for a NaN the caller passed in.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pytest

from mmar.analysis import analyze, compare_estimators
from mmar.calibration import MMARCalibrationError, calibrate_from_spectrum
from mmar.cli import main
from mmar.core.provenance import InputRef, new_provenance
from mmar.core.regression import loglog_fit
from mmar.core.rng import make_rng
from mmar.core.types import FloatArray, ScalingExponents
from mmar.core.validate import DataQualityError, validate_series
from mmar.estimators.mfdfa import mfdfa
from mmar.inference.tests import moment_existence
from mmar.tracing import Run

_Q = np.linspace(-3.0, 3.0, 13)


def _series(n: int = 2048, *, seed: int = 11) -> FloatArray:
    """A level series (a random walk), the representation ``analyze`` documents."""
    rng = make_rng(seed)
    return np.asarray(np.cumsum(rng.standard_normal(n)), dtype=np.float64)


def _exponents(h: float = 0.7, *, tag: str = "a") -> ScalingExponents:
    q = np.linspace(-5.0, 5.0, 41)
    return ScalingExponents(
        q=q,
        zeta=q * h,
        estimator="synthetic",
        provenance=new_provenance(
            "synthetic",
            params={"h": h, "tag": tag},
            inputs=[InputRef.from_array("x", np.linspace(0.0, 1.0, 128))],
        ),
    )


# --------------------------------------------------------------------------- #
# Shape: a matrix is not a series
# --------------------------------------------------------------------------- #


def test_two_dimensional_input_is_rejected_rather_than_ravelled() -> None:
    """Attack: hand ``validate_series`` an ``(n, 2)`` array.

    ``np.asarray(x).ravel()`` flattens in C order, so two columns become one
    interleaved sequence ``[a0, b0, a1, b1, ...]`` of length ``2n`` — a series
    that does not exist. The ``ndim != 1`` guard sitting after the ``ravel``
    can never fire. Passing ``df[["Close"]]`` or ``np.column_stack`` is a
    routine mistake and it must not produce a number.
    """
    a = np.arange(200.0)
    b = -np.arange(200.0)
    with pytest.raises(DataQualityError, match="one-dimensional"):
        validate_series(np.column_stack([a, b]), min_length=64)


def test_stacked_columns_do_not_reach_an_estimator_as_one_series() -> None:
    """Attack: ``analyze`` on two stacked independent paths.

    Before the fix this returned a complete, plausible report: h(2) = 0.0039
    computed from the 4096-point interleaved sequence, against 0.6584 for the
    first column alone. A summary that is off by two orders of magnitude and
    carries no complaint is the worst failure mode in this library.
    """
    a, b = _series(seed=3), _series(seed=4)
    with pytest.raises(DataQualityError, match="one-dimensional"):
        analyze(np.column_stack([a, b]), estimator="mfdfa", q=_Q)


def test_a_single_column_frame_is_rejected_with_advice() -> None:
    """Attack: the ``(n, 1)`` column vector, the commonest pandas slip.

    Silently squeezing it is not safe either: it teaches the caller that a
    frame is an acceptable input, and the day the frame has two columns the
    answer is garbage. The error must say what to pass instead.
    """
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"Close": _series(512)})
    with pytest.raises(DataQualityError, match="one-dimensional"):
        validate_series(frame, min_length=64)


# --------------------------------------------------------------------------- #
# nan_policy: an unrecognised policy must not pick the destructive branch
# --------------------------------------------------------------------------- #


def test_unknown_nan_policy_is_rejected_not_treated_as_interpolate() -> None:
    """Attack: ``validate_series(x, nan_policy="drpo")`` on a gapped series.

    The dispatch was ``if raise ... elif drop ... else interpolate``, so every
    typo landed on interpolation — the one policy the module docstring says
    "manufactures regularity and biases Hoelder exponents *upward*", and the
    one a caller would never silently opt into. ``mmar.io.loaders`` validates
    the same argument explicitly; this path did not.
    """
    x = _series(512)
    x[100] = np.nan
    with pytest.raises(ValueError, match="nan_policy"):
        validate_series(x, nan_policy="drpo", min_length=64)


def test_unknown_nan_policy_is_rejected_even_when_the_series_is_clean() -> None:
    """Attack: the same typo on a series with no gaps.

    The check used to sit inside ``if bad.any()``, so a clean series accepted
    any string at all and recorded it in provenance. The manifest then claims a
    policy no code path implements, which makes the audit record false rather
    than merely incomplete.
    """
    with pytest.raises(ValueError, match="nan_policy"):
        validate_series(_series(512), nan_policy="banana", min_length=64)


def test_analyze_does_not_interpolate_behind_a_misspelt_policy() -> None:
    """Attack: ``analyze(x, nan_policy="ignore")`` on a series with one NaN.

    Observed before the fix: a finished report with h(2) = 0.5152 and
    ``provenance.params["nan_policy"] == "ignore"``. The gap had been filled by
    linear interpolation, which no reader of that manifest could know.
    """
    x = _series(2048)
    x[500] = np.nan
    with pytest.raises(ValueError, match="nan_policy"):
        analyze(x, estimator="mfdfa", q=_Q, nan_policy="ignore")


# --------------------------------------------------------------------------- #
# dtype: a complex array is not a real signal
# --------------------------------------------------------------------------- #


def test_complex_input_is_rejected_not_silently_truncated() -> None:
    """Attack: pass an analytic signal (complex) to ``validate_series``.

    ``np.asarray(z, dtype=np.float64)`` discards the imaginary part behind a
    ``ComplexWarning``, which is a warning and not an error outside this
    repository's own pytest configuration. Every exponent then describes
    ``z.real`` while the caller believes it describes ``z``. The choice of
    projection belongs to the caller.
    """
    z = np.exp(1j * np.arange(512.0)) * np.arange(512.0)
    with pytest.raises(DataQualityError, match="complex"):
        validate_series(z, min_length=64)


# --------------------------------------------------------------------------- #
# confidence: a probability, not a percentage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("confidence", [95.0, 1.0, 0.0, 1.5, -0.5])
def test_confidence_outside_the_open_unit_interval_is_rejected(confidence: float) -> None:
    """Attack: the percent-for-fraction slip and its neighbours in ``loglog_fit``.

    Measured before the fix on an exact power law of slope 0.7:
    ``confidence=95`` gave ``slope_ci=(nan, nan)``; ``confidence=1.0`` gave
    ``(-inf, inf)``; ``confidence=0.0`` gave a zero-width interval
    ``(0.7, 0.7)``; ``confidence=-0.5`` gave an *inverted* interval with
    ``ci[0] > ci[1]``. None of them raised. Those intervals propagate straight
    into ``ScalingExponents.zeta_ci`` and out into a published error bar.
    """
    s = np.array([8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
    y = 3.0 * s**0.7
    with pytest.raises(ValueError, match="confidence"):
        loglog_fit(s, y, confidence=confidence)


def test_moment_existence_rejects_a_percentage_confidence() -> None:
    """Attack: the same slip in the tail-index diagnostic.

    ``stats.norm.ppf(0.5 + 95/2)`` is ``nan``, so the reported interval on
    ``q_crit`` — the statistic that decides which moments may be fitted at all
    — became ``(nan, nan)`` with no complaint.
    """
    rng = make_rng(7)
    x = np.asarray(rng.standard_normal(1024), dtype=np.float64)
    with pytest.raises(ValueError, match="confidence"):
        moment_existence(x, q=_Q, confidence=95.0)


# --------------------------------------------------------------------------- #
# CLI: an error is an exit code, never a traceback
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec", ["1:2", "-5:5:2"])
def test_cli_malformed_q_spec_exits_with_the_error_code(
    monkeypatch: pytest.MonkeyPatch, spec: str
) -> None:
    """Attack: ``mmar analyze --q 1:2``.

    ``_q_grid`` raises ``argparse.ArgumentTypeError``, which derives from
    ``Exception`` and not from ``ValueError``, so it walked straight past
    ``main``'s handler and out as a traceback. ``--q garbage`` (a ``ValueError``
    from ``float``) exited 1 on the same code path, so two malformed grids
    behaved differently for no reason a caller could see.
    """

    def fake_load(_args: argparse.Namespace) -> tuple[np.ndarray, str]:
        return _series(512), "synthetic"

    monkeypatch.setattr("mmar.cli._load", fake_load)
    assert main(["analyze", f"--q={spec}"]) == 1


# --------------------------------------------------------------------------- #
# Tracing: a name in a manifest must identify one computation
# --------------------------------------------------------------------------- #


def test_a_run_refuses_two_results_under_one_name() -> None:
    """Attack: record two different exponent objects as ``name="exponents"``.

    The manifest accepted both. ``RunManifest.by_name`` walks the node list and
    returns the *first* match, so the second result became unreachable while
    still contributing to the session root; ``ancestors`` and
    ``diff_manifests`` key on the same name and inherit the ambiguity.
    Observed: two nodes named ``exponents`` with summaries h(2) = 0.6181 and
    h(2) = 0.2594, and ``by_name`` returning the first with no warning. An
    audit record whose keys are not unique is not an audit record.
    """
    with pytest.raises(ValueError, match="already"), Run("dup", store=None) as run:
        run.record(_exponents(0.7, tag="a"), name="exponents")
        run.record(_exponents(0.3, tag="b"), name="exponents")


# --------------------------------------------------------------------------- #
# Contradictory arguments must be diagnosed, not misdiagnosed
# --------------------------------------------------------------------------- #


def test_an_empty_estimator_list_is_not_blamed_on_a_missing_extra() -> None:
    """Attack: ``compare_estimators(x, estimators=())``.

    The empty result fell into the "nothing could run" branch and told the
    caller to ``pip install mmar[full]``. The dependencies were fine; the list
    was empty — typically because a filter comprehension matched nothing. An
    error that names the wrong cause costs more than no error.
    """
    with pytest.raises(ValueError, match="estimators"):
        compare_estimators(_series(512), estimators=())


def test_analyze_rejects_a_nonpositive_support_dimension() -> None:
    """Attack: ``analyze(x, d_support=0)``.

    ``d`` is the topological dimension of the support: 1 for a time series.
    Nothing validated it, and ``tau = zeta - d`` and ``f(alpha_0) = d`` are
    both linear in it, so ``d_support=0`` returned a spectrum whose apex sat at
    ``f_max = -2.4e-07`` instead of 1 and ``d_support=-2`` one at -2.0. Both
    look like an estimation failure rather than the argument error they are.
    """
    with pytest.raises(ValueError, match="d_support"):
        analyze(_series(2048), estimator="mfdfa", q=_Q, d_support=0.0)


def test_scaling_exponents_rejects_a_nonpositive_support_dimension() -> None:
    """The same guard on the carrier type, which callers construct directly."""
    q = np.linspace(-3.0, 3.0, 13)
    with pytest.raises(ValueError, match="d_support"):
        ScalingExponents(q=q, zeta=q * 0.5, d_support=-2.0)


# --------------------------------------------------------------------------- #
# Container type: numpy's own message names neither the argument nor the remedy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "container",
    [
        pytest.param((float(v) for v in range(200)), id="generator"),
        pytest.param({"2020-01-01": 1.0, "2020-01-02": 2.0}, id="dict"),
        pytest.param({1.0, 2.0, 3.0}, id="set"),
    ],
)
def test_a_non_sequence_container_is_diagnosed_by_this_library_not_by_numpy(
    container: Any,
) -> None:
    """Attack: hand ``validate_series`` a generator, a dict or a set.

    ``np.asarray(x, dtype=np.float64)`` raises
    ``TypeError: float() argument must be a string or a real number, not
    'generator'`` from inside ``validate.py``, three frames below the call the
    user made. It names no argument, no function and no remedy, and reads like a
    library defect rather than an input error — while every other rejection in
    this module is precise and prescriptive.

    ``(p for p in prices)`` and a ``{date: price}`` mapping are both routine
    slips, and the fix in each case is one call the message should quote.
    """
    with pytest.raises(DataQualityError, match="numeric sequence"):
        validate_series(container, min_length=64)


def test_the_container_diagnosis_survives_the_trip_through_analyze() -> None:
    """Attack: the same generator through the public entry point.

    ``analyze`` is where a caller actually meets this, and it must not surface
    numpy's message either: the exception type alone (``TypeError`` against
    ``DataQualityError``) decides whether a sweep script treats the row as bad
    data or as a bug in its own code.
    """
    with pytest.raises(DataQualityError, match="numeric sequence"):
        analyze((float(v) for v in range(4096)), estimator="mfdfa", q=_Q)


@pytest.mark.parametrize("scalar", [None, 3.5, np.float64(2.0)])
def test_a_scalar_is_not_reported_as_a_wide_array(scalar: Any) -> None:
    """Attack: ``validate_series(None)`` and ``validate_series(3.5)``.

    ``np.asarray(None, dtype=float)`` is a *zero*-dimensional ``array(nan)``, so
    both fell into the ``ndim != 1`` branch and were answered with "series must
    be one-dimensional, got shape (). Flattening a wider array would interleave
    its columns ... pass a single column (x[:, 0], df[col].to_numpy())". Every
    clause of that advice is about a matrix; none of it applies to ``None``, and
    a caller who passed ``df["Close"]`` from an empty selection is sent looking
    for a column that is not the problem.
    """
    with pytest.raises(DataQualityError, match="scalar"):
        validate_series(scalar, min_length=64)


# --------------------------------------------------------------------------- #
# MF-DFA order: an integer argument silently truncated
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("order", [2.5, 2.99, 1.5])
def test_mfdfa_refuses_a_non_integral_detrending_order(order: float) -> None:
    """Attack: ``mfdfa(x, order=2.5)``.

    ``m = int(order)`` truncated toward zero, so a fractional order ran the
    *lower* integer detrending with nothing said. Measured on a 2048-point
    random walk: ``order=2.5``, ``order=2.99`` and ``order=2`` all returned
    ``h(2) = 0.4915054972``, bit for bit, and the provenance recorded
    ``order = 2`` — so the manifest is honest about what ran while the caller
    believes something else did. DFA2 and DFA3 differ by a whole trend order.
    """
    x = _series(2048)
    with pytest.raises(ValueError, match="integer polynomial degree"):
        mfdfa(x, q=_Q, order=order, integrate=False)


@pytest.mark.parametrize("order", [-0.5, -0.9])
def test_a_fractional_negative_order_does_not_slip_past_the_sign_guard(
    order: float,
) -> None:
    """Attack: ``mfdfa(x, order=-0.5)``, where ``order=-1`` is already rejected.

    The guard read ``if int(order) < 0``, and ``int(-0.5) == 0``. So ``-1``
    raised "order must be non-negative" while ``-0.5`` ran DFA0 and returned
    ``h(2) = 0.4612081555`` — identical to ``order=0`` — with no complaint. A
    validation whose verdict flips between two neighbouring nonsense values is
    not a validation.
    """
    x = _series(2048)
    with pytest.raises(ValueError, match="integer polynomial degree"):
        mfdfa(x, q=_Q, order=order, integrate=False)


def test_a_boolean_order_is_not_silently_dfa1() -> None:
    """Attack: ``mfdfa(x, order=True)`` — the bool-is-an-int trap.

    ``int(True) == 1``, so the call ran DFA1 and returned
    ``h(2) = 0.4882759101``, exactly ``order=1``. ``True`` reaches this argument
    from a config parsed out of YAML or a mis-keyed flag, and a linear detrending
    silently substituted for whatever was intended is a shifted exponent, not a
    typo.
    """
    x = _series(2048)
    with pytest.raises(ValueError, match="integer polynomial degree"):
        mfdfa(x, q=_Q, order=True, integrate=False)


def test_the_default_scale_grid_blames_the_order_not_the_series_length() -> None:
    """Attack: ``mfdfa(x_256, order=12)`` with no ``scales`` argument.

    The default grid starts at ``s_min = 10(m+1)``, so a large ``order`` alone
    pushes every default scale past ``n/4``. ``check_scales`` then fired first
    and reported: "no scale in scales leaves >= 4 sample windows for n=256
    (largest admissible scale is 64); the series is too short".

    Three things are wrong with that. It names ``scales``, which the caller never
    passed — the grid was *derived* from ``order``. It says the series is too
    short when the same 256 points fit happily at orders 1 through 5. And the
    remedy that would have worked, "lower the order", sits eight lines below in a
    branch this path never reaches. A message that prescribes the wrong fix costs
    more than a bare traceback, because it gets followed.
    """
    x = _series(256, seed=21)
    with pytest.raises(DataQualityError) as excinfo:
        mfdfa(x, q=_Q, order=12, integrate=False)
    message = str(excinfo.value)
    assert "order=12" in message
    assert "too short" not in message
    # The remedy the message prescribes must actually work on this series.
    assert np.isfinite(mfdfa(x, q=_Q, order=1, integrate=False).h2)


# --------------------------------------------------------------------------- #
# A string is a Sequence[str]
# --------------------------------------------------------------------------- #


def test_compare_estimators_rejects_a_bare_string_instead_of_iterating_it() -> None:
    """Attack: ``compare_estimators(x, estimators="mfdfa")``.

    ``str`` satisfies ``Sequence[str]``, so the loop iterated *characters* and
    the caller was told ``unknown estimator 'm'; available: [...]`` — a name they
    never wrote, from an argument they spelled correctly. The empty-list guard
    directly above does not catch it either, because ``len("mfdfa") == 5``.
    """
    with pytest.raises(ValueError, match="single string"):
        compare_estimators(_series(512), estimators="mfdfa", q=_Q)


# --------------------------------------------------------------------------- #
# Calibration: do not blame the data for the caller's argument
# --------------------------------------------------------------------------- #


def test_a_nan_hurst_override_is_not_blamed_on_the_exponents() -> None:
    """Attack: ``calibrate_from_spectrum(exponents, hurst=float("nan"))``.

    The NaN check does not distinguish "``tau`` has no root" from "the caller
    supplied a NaN", so an explicit override was diagnosed as: "tau(q) has no
    root on the positive-q branch, so the MFC identity tau(1/H)=0 identifies no
    Hurst exponent. Widen the q grid".

    On these exponents ``tau`` *does* have a root — ``hurst_tau_root`` returns
    0.7 — so the diagnosis is factually false and it sends the caller to widen a
    ``q`` grid that is already fine. A NaN arrives here from an upstream
    estimator that failed silently, which is exactly when a truthful attribution
    matters.
    """
    exponents = _exponents(0.7)
    assert np.isclose(exponents.hurst_tau_root, 0.7, atol=1e-6)

    result = calibrate_from_spectrum(exponents, hurst=float("nan"))
    joined = " ".join(result.diagnosis)
    assert "hurst=" in joined
    assert "no root on the positive-q branch" not in joined


def test_a_failed_hurst_identification_is_not_reported_as_admissible() -> None:
    """Attack: calibrate exponents whose ``tau`` never crosses zero on ``q > 0``.

    The MFC identity ``tau(1/H) = 0`` is the whole basis of this calibration, and
    it identified nothing here: ``hurst_tau_root`` is NaN. The code says so in
    ``diagnosis``, falls back to ``h(2)``, and then reassigns ``H`` — after which
    ``fatal`` tested the *fallback* for finiteness and found it fine.

    Observed on ``zeta(q) = 0.35q - 0.015q^2`` over ``q in [-3, 3]`` (``tau < 0``
    throughout the positive branch, so no root): ``admissible = True``,
    ``H = 0.3200`` taken from ``h(2)``, ``lambda = 1.0937``, and
    ``to_spec_kwargs()`` — which documents that it refuses an inadmissible
    calibration — returned a complete ``MMARSpec`` payload. Through the CLI the
    same case exits 0.

    So the one field a caller gates on said success while the calibration's own
    diagnosis said the primary identity had failed, and the reported ``H`` came
    from a different estimator than the one the verdict is about.
    """
    q = np.linspace(-3.0, 3.0, 61)
    exponents = ScalingExponents(q=q, zeta=0.35 * q - 0.015 * q**2, estimator="synthetic")
    assert not np.isfinite(exponents.hurst_tau_root)

    result = calibrate_from_spectrum(exponents)
    assert any("no root" in d for d in result.diagnosis)
    assert not result.admissible
    with pytest.raises(MMARCalibrationError, match="inadmissible"):
        result.to_spec_kwargs()
