"""The high-level API must make the estimators comparable, not merely runnable."""

from __future__ import annotations

import numpy as np
import pytest

from mmar.analysis import (
    ESTIMATORS,
    analyze,
    compare_estimators,
    consensus,
    get_estimator,
    spectrum_agreement,
    spectrum_table,
)
from mmar.models.cascade import lognormal_cascade, trading_time
from mmar.models.fbm import fbm, fgn

Q = np.linspace(-5.0, 5.0, 33)


def test_analyze_returns_a_complete_report() -> None:
    rep = analyze(fbm(2**13, 0.6, seed=3), q=Q, seed=0)
    assert rep.exponents.q.size == Q.size
    assert rep.spectrum.f_alpha.size > 0
    assert rep.cumulants is not None
    assert "concavity_defect" in rep.tests
    assert rep.provenance is not None
    assert set(rep.summary()) >= {"h2", "alpha_0", "width", "c1", "c2", "root"}


def test_report_root_is_reproducible_and_parameter_sensitive() -> None:
    x = fbm(2**12, 0.6, seed=3)
    a, b = analyze(x, q=Q, seed=0), analyze(x, q=Q, seed=0)
    assert a.provenance is not None and b.provenance is not None
    assert a.provenance.root == b.provenance.root
    other = analyze(x, q=Q, seed=0, spectrum_method="parametric")
    assert other.provenance is not None
    assert other.provenance.root != a.provenance.root


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_every_registered_estimator_recovers_a_monofractal(name: str) -> None:
    """h(2) must land near H for every family. This is the floor, not the ceiling."""
    rep = analyze(fbm(2**14, 0.7, seed=11), estimator=name, kind="level", q=Q, seed=1)
    assert rep.exponents.h2 == pytest.approx(0.7, abs=0.09), (
        f"{name} reported h(2)={rep.exponents.h2:.4f} on an fBm with H=0.7"
    )


def test_level_and_increments_of_the_same_process_agree() -> None:
    """Regression: mfdfa shipped integrate=True and mfdma integrate=False.

    Comparing them without canonicalising the representation put h(2) a full 1.0
    apart -- an artefact of the flags, not of the data.
    """
    names = ["mfdfa", "mfdma", "structure_function"]
    level = compare_estimators(
        fbm(2**14, 0.7, seed=11), estimators=names, kind="level", q=Q, seed=1
    )
    incr = compare_estimators(
        fgn(2**14, 0.7, seed=11), estimators=names, kind="increments", q=Q, seed=1
    )
    for name in names:
        assert level[name].exponents.h2 == pytest.approx(incr[name].exponents.h2, abs=0.05), (
            f"{name}: level h(2)={level[name].exponents.h2:.4f} vs "
            f"increments h(2)={incr[name].exponents.h2:.4f}"
        )
        assert level[name].exponents.h2 == pytest.approx(0.7, abs=0.09)


def test_bad_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        analyze(fbm(2**12, 0.6, seed=3), kind="profile", q=Q)  # type: ignore[arg-type]


def test_c2_separates_monofractal_from_cascade_where_width_does_not() -> None:
    """c2 is the reliable multifractality statistic; the Legendre width is not.

    On a monofractal the increment-based structure function still yields a wide
    f(alpha) -- pure estimation noise amplified by the transform -- while c2
    stays near zero. Both numbers are reported so the discrepancy is visible.
    """
    mono = analyze(fbm(2**14, 0.7, seed=11), estimator="structure_function", q=Q, seed=1)
    theta = trading_time(lognormal_cascade(14, 1.3, seed=5))
    multi = analyze(theta, estimator="structure_function", q=Q, seed=1)

    assert mono.cumulants is not None and multi.cumulants is not None
    assert abs(mono.cumulants.c2) < 0.05, f"monofractal c2={mono.cumulants.c2:.4f}"
    assert multi.cumulants.c2 < mono.cumulants.c2 - 0.05, (
        f"cascade c2={multi.cumulants.c2:.4f} vs monofractal {mono.cumulants.c2:.4f}"
    )


def test_cascade_is_measurably_more_multifractal_than_fbm() -> None:
    mono = analyze(fbm(2**14, 0.6, seed=2), q=Q, seed=1)
    multi = analyze(trading_time(lognormal_cascade(14, 1.3, seed=5)), q=Q, seed=1)
    assert mono.cumulants is not None and multi.cumulants is not None
    assert multi.cumulants.c2 < mono.cumulants.c2


def test_consensus_flags_a_planted_disagreement() -> None:
    reps = compare_estimators(fbm(2**13, 0.6, seed=3), estimators=["mfdfa", "mfdma"], q=Q, seed=1)
    good = consensus(reps)
    assert "agree" in good.verdict.lower()

    # Force a disagreement by mis-declaring the representation for one estimator.
    reps["mfdfa_wrong"] = analyze(
        fbm(2**13, 0.6, seed=3), estimator="mfdfa", q=Q, seed=1, integrate=True
    )
    bad = consensus(reps)
    assert "DISAGREE" in bad.verdict
    assert bad.alpha_0_spread > 0.5


def test_consensus_on_a_single_estimator_says_so() -> None:
    reps = compare_estimators(fbm(2**12, 0.6, seed=3), estimators=["mfdfa"], q=Q, seed=1)
    assert "single estimator" in consensus(reps).verdict


def test_spectrum_table_covers_every_report() -> None:
    reps = compare_estimators(fbm(2**12, 0.6, seed=3), estimators=["mfdfa", "mfdma"], q=Q, seed=1)
    table = spectrum_table(reps)
    assert all(name in table for name in reps)
    assert len(table.splitlines()) == len(reps) + 2


def test_spectrum_agreement_is_zero_against_itself_and_nan_when_disjoint() -> None:
    rep = analyze(fbm(2**12, 0.6, seed=3), q=Q, seed=1)
    assert spectrum_agreement(rep.spectrum, rep.spectrum) == pytest.approx(0.0, abs=1e-12)

    from mmar.core.types import HolderSpectrum

    far = HolderSpectrum(alpha=rep.spectrum.alpha + 100.0, f_alpha=rep.spectrum.f_alpha)
    assert np.isnan(spectrum_agreement(rep.spectrum, far))


def test_unknown_estimator_names_the_alternatives() -> None:
    with pytest.raises(KeyError, match="wavelet_leaders"):
        get_estimator("wavelet-leaders")


def test_a_custom_callable_is_accepted() -> None:
    from mmar.estimators.mfdfa import mfdfa

    def my_estimator(x, **kwargs):  # type: ignore[no-untyped-def]
        return mfdfa(x, integrate=False, **kwargs)

    rep = analyze(fbm(2**12, 0.6, seed=3), estimator=my_estimator, q=Q, seed=1)
    assert rep.exponents.h2 == pytest.approx(0.6, abs=0.1)
    assert rep.provenance is not None and rep.provenance.params["estimator"] == "my_estimator"
