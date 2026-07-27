"""What the library does when the *caller* is wrong.

A wrong answer is worse than a refusal, so every entry point attacked here is
one where a plausible user error was absorbed silently instead of being
reported: a two-dimensional array ravelled into an interleaved series, a
misspelt ``nan_policy`` falling through to the most destructive branch, a
complex array quietly stripped of its imaginary part, a confidence level given
in percent turning every interval into ``NaN``, a malformed CLI argument
escaping as a traceback, and two different results filed in one manifest under
one name.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from mmar.analysis import analyze, compare_estimators
from mmar.cli import main
from mmar.core.provenance import InputRef, new_provenance
from mmar.core.regression import loglog_fit
from mmar.core.rng import make_rng
from mmar.core.types import FloatArray, ScalingExponents
from mmar.core.validate import DataQualityError, validate_series
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
