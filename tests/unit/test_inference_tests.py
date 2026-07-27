r"""Tests for :mod:`mmar.inference.tests`.

The reference estimator below is deliberately local: the hypothesis-test module
takes the estimator as a callable, and these tests must not depend on any
particular estimator implementation. It is the textbook structure-function
estimator, computed in log space so that heavy tails cannot overflow it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import logsumexp

from mmar.core.regression import loglog_fit
from mmar.core.rng import make_rng, spawn_rngs
from mmar.core.types import FloatArray, ScalingExponents, StructureFunctions
from mmar.inference.tests import (
    MomentDiagnostic,
    _rolling_std,
    _spectral_mismatch,
    _surrogate,
    concavity_test,
    empirical_p_value,
    moment_existence,
    monofractal_test,
    multifractality_report,
    multifractality_test,
    scaling_range_test,
    spectrum_statistic,
)
from mmar.inference.tests import TestResult as SurrogateTestResult  # aliased: pytest collection
from mmar.models.cascade import lognormal_cascade
from mmar.models.fbm import fbm, levy_stable_increments

Q_GRID = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
SCALES = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512], dtype=np.int64)
N = 16384
r"""Sample length and scale window used throughout.

Both matter for the *power* of the surrogate tests and were chosen deliberately.
At small scales the structure function of i.i.d. heavy-tailed increments is
dominated by the sample maximum and is itself curved, so a shuffle surrogate
looks as multifractal as a genuine cascade and the test has no power. Only once
the window reaches scales at which the i.i.d. aggregate has converged to the
Gaussian regime (:math:`\zeta=q/2`, exactly linear) does the cascade separate
from its own shuffle. Two decades of scale on 2**14 points is enough; half a
decade on 2**12 is not, and this is a property of surrogate testing, not of the
implementation.
"""


# --------------------------------------------------------------------------- #
# A self-contained reference estimator
# --------------------------------------------------------------------------- #


def _log_structure(x: FloatArray, scales: np.ndarray, q: FloatArray) -> FloatArray:
    r"""``out[i, j] = log <|X(t+s_j) - X(t)|^{q_i}>`` over overlapping increments.

    Evaluated by ``logsumexp`` rather than ``mean(|d|**q)``: with a stable-law
    sample the direct power overflows at moderate ``q``, and an overflow warning
    is an error under this project's pytest configuration.
    """
    out = np.empty((q.size, scales.size), dtype=np.float64)
    for j, s in enumerate(scales):
        d = np.abs(x[int(s) :] - x[: -int(s)])
        with np.errstate(divide="ignore"):
            ld = np.log(d)
        out[:, j] = logsumexp(q[:, None] * ld[None, :], axis=1) - math.log(d.size)
    return out


def structure_estimator(
    x: FloatArray, *, q: FloatArray = Q_GRID, scales: np.ndarray = SCALES
) -> ScalingExponents:
    """Plain structure-function estimator: one log--log slope per ``q``."""
    usable = scales[(x.size // scales) >= 8]
    log_s = _log_structure(x, usable, q)
    fits = tuple(
        loglog_fit(usable.astype(np.float64), log_s[i], log_y=False) for i in range(q.size)
    )
    return ScalingExponents(
        q=q,
        zeta=np.array([f.slope for f in fits]),
        zeta_stderr=np.array([f.slope_stderr for f in fits]),
        fits=fits,
        estimator="test.structure_function",
    )


def structure_functions(x: FloatArray, *, q: FloatArray = Q_GRID) -> StructureFunctions:
    usable = SCALES[(x.size // SCALES) >= 8]
    values = np.exp(_log_structure(x, usable, q))
    counts = np.asarray(x.size - usable, dtype=np.int64)
    return StructureFunctions(q=q, scales=usable.astype(np.float64), values=values, counts=counts)


# --------------------------------------------------------------------------- #
# Sample paths
# --------------------------------------------------------------------------- #


def _fbm_path(n: int = N, H: float = 0.6, seed: int = 11) -> FloatArray:
    return fbm(n, H, seed=seed)


def _levy_path(n: int = N, alpha: float = 1.5, seed: int = 7) -> FloatArray:
    """Cumulated i.i.d. stable increments: monofractal, but looks multifractal."""
    return np.cumsum(levy_stable_increments(n, alpha, seed=seed))


def _cascade_path(kmax: int = 14, lam: float = 1.10, seed: int = 3) -> FloatArray:
    r"""Multifractal random walk: :math:`\Delta X_i=\sqrt{\mu_i/\bar\mu}\,Z_i`.

    The lognormal cascade supplies the trading-time increments and an
    independent Gaussian supplies the sign, so the increments are genuinely
    multifractal in time while remaining sign-symmetric. Conditionally on the
    cascade, :math:`\Delta_\delta X \approx \sqrt{\Delta_\delta\theta}\,Z`,
    hence :math:`\zeta_X(q)=\zeta_\theta(q/2)=q\lambda/2-\kappa q^2/8` with
    :math:`\kappa=\sigma^2\ln 2`: a genuine parabola, with :math:`c_2=-\kappa/4
    = -0.05` at ``lam=1.10``.
    """
    rng_mu, rng_z = spawn_rngs(seed, 2)
    mu = lognormal_cascade(kmax, lam, seed=rng_mu)
    z = rng_z.standard_normal(mu.size)
    return np.asarray(np.cumsum(np.sqrt(mu / mu.mean()) * z), dtype=np.float64)


# --------------------------------------------------------------------------- #
# p-value machinery
# --------------------------------------------------------------------------- #


def test_empirical_p_value_never_zero_and_counts_the_observation() -> None:
    null = np.zeros(99)
    assert empirical_p_value(1.0, null) == pytest.approx(1.0 / 100.0)
    assert empirical_p_value(-1.0, null) == pytest.approx(1.0)


def test_empirical_p_value_drops_non_finite_surrogates() -> None:
    null = np.array([0.0, 1.0, np.nan, np.inf])
    # Two usable surrogates, one of which is >= 0.5.
    assert empirical_p_value(0.5, null) == pytest.approx(2.0 / 3.0)


def test_empirical_p_value_is_uniform_under_the_null() -> None:
    """Exchangeable draws must give a p-value with (at most) uniform tails."""
    rng = make_rng(1234)
    draws = rng.standard_normal((400, 65))
    p = np.array([empirical_p_value(float(row[0]), row[1:]) for row in draws])
    assert 0.02 <= float(np.mean(p <= 0.05)) <= 0.10
    assert float(np.mean(p)) == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_statistics_vanish_on_an_exactly_linear_zeta() -> None:
    q = np.linspace(-4.0, 4.0, 33)
    exps = ScalingExponents(q=q, zeta=0.6 * q, estimator="exact.monofractal")
    assert spectrum_statistic(exps, "width") == pytest.approx(0.0, abs=1e-9)
    assert spectrum_statistic(exps, "c2") == pytest.approx(0.0, abs=1e-12)
    assert spectrum_statistic(exps, "hq_range") == pytest.approx(0.0, abs=1e-12)


def test_statistics_recover_the_lognormal_parabola() -> None:
    r""":math:`\zeta = q\lambda - \tfrac12 c q^2` has :math:`c_2=-c` and
    :math:`\Delta\alpha = c\,(q_{\max}-q_{\min})` over the fitted window."""
    q = np.linspace(-3.0, 3.0, 25)
    c = 0.07
    exps = ScalingExponents(q=q, zeta=1.05 * q - 0.5 * c * q**2, estimator="exact.lognormal")
    assert spectrum_statistic(exps, "c2") == pytest.approx(c, rel=1e-9)
    assert spectrum_statistic(exps, "width") == pytest.approx(c * 6.0, rel=1e-3)


def test_unknown_statistic_is_rejected() -> None:
    q = np.linspace(1.0, 3.0, 5)
    exps = ScalingExponents(q=q, zeta=q * 0.5)
    with pytest.raises(ValueError, match="unknown statistic"):
        spectrum_statistic(exps, "delta_alpha")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Surrogate construction
# --------------------------------------------------------------------------- #


def test_shuffle_surrogate_preserves_the_increment_multiset() -> None:
    x = _fbm_path(1024)
    y = _surrogate(x, "shuffle", make_rng(0), basis="increments", iaaft_iterations=0)
    assert y.shape == x.shape
    assert y[0] == x[0]
    np.testing.assert_allclose(np.sort(np.diff(y)), np.sort(np.diff(x)), atol=1e-12)
    assert not np.allclose(y, x)


def test_iaaft_preserves_marginal_exactly_and_spectrum_closely() -> None:
    x = _fbm_path(1024, H=0.75, seed=5)
    y = _surrogate(x, "iaaft", make_rng(0), basis="increments", iaaft_iterations=200)
    np.testing.assert_allclose(np.sort(np.diff(y)), np.sort(np.diff(x)), atol=1e-12)
    shuffled = _surrogate(x, "shuffle", make_rng(0), basis="increments", iaaft_iterations=0)
    iaaft_err = _spectral_mismatch(x, y, "increments")
    shuffle_err = _spectral_mismatch(x, shuffled, "increments")
    assert iaaft_err < 0.5 * shuffle_err


def test_surrogates_are_reproducible_and_distinct_across_the_ensemble() -> None:
    x = _fbm_path(2048)
    a = multifractality_test(x, structure_estimator, n_surrogates=8, seed=99)
    b = multifractality_test(x, structure_estimator, n_surrogates=8, seed=99)
    c = multifractality_test(x, structure_estimator, n_surrogates=8, seed=100)
    np.testing.assert_allclose(a.null_distribution, b.null_distribution)
    assert a.provenance is not None and b.provenance is not None
    assert a.provenance.root == b.provenance.root
    assert not np.allclose(a.null_distribution, c.null_distribution)
    assert np.unique(a.null_distribution).size == a.n_surrogates


# --------------------------------------------------------------------------- #
# The three headline cases
# --------------------------------------------------------------------------- #


def test_fbm_is_not_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """Exact fBm is monofractal: neither null may be rejected."""
    x = _fbm_path(N, H=0.6, seed=11)
    report = multifractality_report(x, structure_estimator, n_surrogates=49, seed=2026)
    with capsys.disabled():
        print(
            f"\n[fBm H=0.6] width = {report.shuffle.statistic:.4f}  "
            f"p_shuffle = {report.shuffle.p_value:.3f}  p_iaaft = {report.iaaft.p_value:.3f}"
        )
    assert report.shuffle.p_value > 0.05
    assert report.iaaft.p_value > 0.05
    assert report.verdict.startswith("NO EVIDENCE")


def test_lognormal_cascade_is_rejected_decisively(capsys: pytest.CaptureFixture[str]) -> None:
    """A multifractal random walk over a lognormal cascade must reject both nulls."""
    x = _cascade_path(kmax=14, lam=1.10, seed=3)
    report = multifractality_report(x, structure_estimator, n_surrogates=49, seed=4242)
    with capsys.disabled():
        print(
            f"\n[lognormal cascade lam=1.10] width = {report.shuffle.statistic:.4f}  "
            f"p_shuffle = {report.shuffle.p_value:.4f} (z = {report.shuffle.z_score:.1f})  "
            f"p_iaaft = {report.iaaft.p_value:.4f} (z = {report.iaaft.z_score:.1f})"
        )
    assert report.shuffle.p_value <= 0.02
    assert report.iaaft.p_value <= 0.02
    assert report.shuffle.z_score > 3.0
    assert report.iaaft.z_score > 3.0
    assert report.verdict.startswith("NONLINEAR")


def test_levy_false_positive_is_caught_by_the_shuffle_null(
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""**The most important test in the module.**

    i.i.d. :math:`\alpha`-stable increments have no temporal structure at all,
    so the cumulated path is monofractal by construction. Because
    :math:`\mathbb E|\Delta X|^q=\infty` for :math:`q\ge\alpha`, the naive
    spectrum is nonetheless wide. The shuffle null must place that width in the
    middle of its ensemble and fail to reject.
    """
    x = _levy_path(N, alpha=1.5, seed=7)
    result = multifractality_test(
        x, structure_estimator, kind="shuffle", n_surrogates=199, statistic="width", seed=808
    )
    fbm_width = spectrum_statistic(structure_estimator(_fbm_path(N, 0.6, 11)), "width")
    with capsys.disabled():
        print(
            f"\n[Levy alpha=1.5] naive width = {result.statistic:.4f}  "
            f"(fBm reference width = {fbm_width:.4f})  "
            f"shuffle p = {result.p_value:.3f}  "
            f"null mean = {result.null_mean:.4f}, sd = {result.null_sd:.4f}, "
            f"z = {result.z_score:.2f}"
        )
    # The false positive is real: the naive width is far above the fBm baseline.
    assert result.statistic > 20.0 * fbm_width
    # And the shuffle surrogate catches it.
    assert result.p_value > 0.05
    assert abs(result.z_score) < 2.0
    assert "no evidence" in result.interpretation.lower()


@pytest.mark.slow
def test_levy_shuffle_test_is_calibrated_across_realisations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The Levy non-rejection is not a lucky seed: repeat over fresh paths."""
    p_values = []
    for k, seed in enumerate((21, 22, 23, 24, 25)):
        x = _levy_path(N, alpha=1.6, seed=seed)
        res = multifractality_test(
            x, structure_estimator, kind="shuffle", n_surrogates=99, seed=900 + k
        )
        p_values.append(res.p_value)
    with capsys.disabled():
        print(f"\n[Levy calibration] p = {[round(p, 3) for p in p_values]}")
    assert sum(p > 0.05 for p in p_values) >= 4


def test_statistic_choice_does_not_flip_the_cascade_verdict() -> None:
    x = _cascade_path(kmax=14, lam=1.10, seed=3)
    for statistic in ("c2", "hq_range"):
        res = multifractality_test(
            x,
            structure_estimator,
            kind="shuffle",
            n_surrogates=49,
            statistic=statistic,
            seed=17,
        )
        assert res.p_value <= 0.05, statistic


# --------------------------------------------------------------------------- #
# Guard rails on the test itself
# --------------------------------------------------------------------------- #


def test_degenerate_null_is_refused() -> None:
    q = Q_GRID

    def constant_estimator(_x: FloatArray) -> ScalingExponents:
        return ScalingExponents(q=q, zeta=0.5 * q, estimator="constant")

    with pytest.raises(ValueError, match="degenerate"):
        multifractality_test(_fbm_path(2048), constant_estimator, n_surrogates=5, seed=0)


def test_estimator_failures_above_the_tolerance_abort_the_test() -> None:
    calls = {"n": 0}

    def flaky(x: FloatArray) -> ScalingExponents:
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("no admissible scaling range found")
        return structure_estimator(x)

    with pytest.raises(ValueError, match="failed on"):
        multifractality_test(_fbm_path(2048), flaky, n_surrogates=10, seed=0)


def test_unknown_surrogate_kind_and_basis_are_rejected() -> None:
    x = _fbm_path(2048)
    with pytest.raises(ValueError, match="unknown surrogate kind"):
        multifractality_test(x, structure_estimator, kind="bootstrap", n_surrogates=2, seed=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown basis"):
        multifractality_test(x, structure_estimator, basis="levels", n_surrogates=2, seed=0)  # type: ignore[arg-type]


def test_test_result_carries_provenance_with_the_seed() -> None:
    res = multifractality_test(_fbm_path(2048), structure_estimator, n_surrogates=5, seed=31337)
    assert isinstance(res, SurrogateTestResult)
    assert res.provenance is not None
    assert res.provenance.seed is not None
    assert res.provenance.seed.entropy == 31337
    assert res.provenance.params["kind"] == "shuffle"
    assert len(res.provenance.root) == 64
    with pytest.raises(AttributeError):
        res.p_value = 0.5  # type: ignore[misc]


def test_a_reused_seed_sequence_still_reproduces_the_ensemble() -> None:
    """Regression: same provenance root must mean same numbers.

    ``SeedSequence.spawn`` advances a counter, so handing the *same* object to
    two calls used to give two different null ensembles under one content
    address -- the one thing the provenance layer promises cannot happen.
    """
    x = _fbm_path(2048)
    ss = np.random.SeedSequence(12345)
    a = multifractality_test(x, structure_estimator, n_surrogates=6, seed=ss)
    b = multifractality_test(x, structure_estimator, n_surrogates=6, seed=ss)
    assert a.provenance is not None and b.provenance is not None
    assert a.provenance.root == b.provenance.root
    np.testing.assert_allclose(a.null_distribution, b.null_distribution)


def test_report_gives_each_null_a_distinct_recoverable_seed() -> None:
    """The two sub-tests must be independent *and* individually reproducible."""
    report = multifractality_report(
        _fbm_path(2048), structure_estimator, n_surrogates=5, seed=555, iaaft_iterations=20
    )
    ps, pi = report.shuffle.provenance, report.iaaft.provenance
    assert ps is not None and pi is not None and report.provenance is not None
    assert ps.seed is not None and pi.seed is not None
    assert ps.seed.entropy == pi.seed.entropy == 555
    assert ps.seed.spawn_key != pi.seed.spawn_key
    assert ps.root != pi.root
    assert report.summary()["verdict"] == report.verdict


# --------------------------------------------------------------------------- #
# monofractal_test
# --------------------------------------------------------------------------- #


def test_monofractal_test_does_not_reject_a_straight_line() -> None:
    q = np.linspace(-3.0, 3.0, 13)
    exps = ScalingExponents(q=q, zeta=0.6 * q, zeta_stderr=np.full(q.size, 0.01), estimator="exact")
    res = monofractal_test(exps)
    assert res.statistic == pytest.approx(0.0, abs=1e-18)
    assert res.p_value == pytest.approx(1.0)
    # 13 grid points, minus the uninformative q = 0, minus one fitted H.
    assert res.dof == 11.0
    assert res.diagnostics["H"] == pytest.approx(0.6)


def test_monofractal_test_rejects_a_parabola() -> None:
    q = np.linspace(-3.0, 3.0, 13)
    zeta = 0.6 * q - 0.5 * 0.05 * q**2
    exps = ScalingExponents(q=q, zeta=zeta, zeta_stderr=np.full(q.size, 0.01))
    res = monofractal_test(exps)
    assert res.p_value < 1e-6
    assert "strongly positively correlated" in res.interpretation


def test_monofractal_test_needs_standard_errors() -> None:
    q = np.linspace(1.0, 3.0, 5)
    with pytest.raises(ValueError, match="no standard errors"):
        monofractal_test(ScalingExponents(q=q, zeta=0.5 * q))


def test_monofractal_test_falls_back_to_the_per_q_fits() -> None:
    x = _fbm_path(N, H=0.55, seed=8)
    exps = structure_estimator(x)
    stripped = ScalingExponents(q=exps.q, zeta=exps.zeta, fits=exps.fits, estimator=exps.estimator)
    assert monofractal_test(stripped).statistic == pytest.approx(monofractal_test(exps).statistic)


def test_monofractal_test_on_a_cascade_beats_the_fbm_statistic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    chi2_fbm = monofractal_test(structure_estimator(_fbm_path(N, 0.6, 11))).statistic
    chi2_mf = monofractal_test(structure_estimator(_cascade_path(14, 1.10, 3))).statistic
    with capsys.disabled():
        print(f"\n[monofractal chi2] fBm = {chi2_fbm:.1f}   cascade = {chi2_mf:.1f}")
    assert chi2_fbm < 10.0
    assert chi2_mf > 50.0


# --------------------------------------------------------------------------- #
# moment_existence
# --------------------------------------------------------------------------- #


def test_hill_recovers_the_stable_tail_index(capsys: pytest.CaptureFixture[str]) -> None:
    r"""For :math:`S_\alpha` increments the tail index is :math:`\alpha` itself."""
    x = _levy_path(20000, alpha=1.5, seed=42)
    diag = moment_existence(x, q=Q_GRID)
    with capsys.disabled():
        print(
            f"\n[Hill, Levy alpha=1.5] q_crit = {diag.q_crit:.3f} "
            f"CI ({diag.q_crit_ci[0]:.3f}, {diag.q_crit_ci[1]:.3f})  k = {diag.k_selected}"
        )
    assert isinstance(diag, MomentDiagnostic)
    # The Hill estimator is biased upward on stable data -- the slowly varying
    # factor in the tail is not constant -- so the interval need not cover 1.5.
    assert diag.q_crit == pytest.approx(1.5, abs=0.3)
    assert diag.q_crit_ci[0] < diag.q_crit < diag.q_crit_ci[1]
    # Everything at or above the tail index must be flagged as inadmissible.
    assert not diag.exists[Q_GRID >= 2.0].any()
    assert diag.exists[Q_GRID <= 1.0].all()
    assert set(diag.rejected_q().tolist()) == {2.0, 2.5, 3.0}


def test_gaussian_increments_admit_every_positive_moment() -> None:
    x = _fbm_path(N, H=0.5, seed=13)
    diag = moment_existence(x, q=np.array([1.0, 2.0, 4.0]))
    assert diag.q_crit > 4.0
    assert diag.exists.all()


def test_negative_moments_of_gaussian_increments_are_flagged() -> None:
    r""":math:`\mathbb E|Z|^{q}=\infty` for :math:`q\le-1` when the density at 0 is positive."""
    x = _fbm_path(N, H=0.5, seed=13)
    diag = moment_existence(x, q=np.array([-3.0, -2.0, -0.5, 0.0, 2.0]))
    assert diag.q_crit_negative == pytest.approx(1.0, abs=0.3)
    assert not diag.exists[0] and not diag.exists[1]
    assert diag.exists[2] and diag.exists[3] and diag.exists[4]


def test_hill_plot_is_returned_and_aligned() -> None:
    x = _levy_path(8192, alpha=1.2, seed=1)
    diag = moment_existence(x, q=np.array([1.0, 2.0]))
    assert diag.hill.shape == diag.k_grid.shape
    assert 1 <= diag.k_selected <= int(diag.k_grid[-1])
    assert diag.hill[diag.k_selected - 1] == pytest.approx(diag.q_crit)
    assert diag.admissible_range[1] == diag.q_crit


def test_moment_existence_refuses_a_sample_with_no_tail() -> None:
    x = np.cumsum(np.ones(200))
    with pytest.raises(ValueError, match=r"non-zero increments|Hill"):
        moment_existence(x, q=np.array([2.0]), k_min=10, k_max_fraction=0.01)


def test_rolling_std_confines_a_non_finite_entry_to_its_own_windows() -> None:
    """Regression: a cumsum formulation propagated one inf to every later window.

    The Hill plot returns ``inf`` whenever the top order statistics are tied, so
    this is the normal case for quantised (tick-sized) data, not a corner case.
    """
    a = np.array([1.0, 2.0, np.inf, 3.0, 1.5, 2.2, 1.9, 2.0, 2.1, 1.8, 2.0, 2.3])
    out = _rolling_std(a, 5)
    # Windows containing index 2 are NaN; the ones past it must be finite again.
    assert not np.isfinite(out[2:5]).any()
    assert np.isfinite(out[7:10]).all()
    assert out[8] == pytest.approx(float(np.std(a[6:11])))


def test_moment_existence_survives_quantised_increments() -> None:
    """Regression: tick-quantised data used to emit RuntimeWarnings and NaN CIs.

    ``|increments|`` is tied at 1 for a large fraction of the sample, so the
    reciprocal Hill plot is entirely infinite. That must produce a clean
    ``inf`` bound, not ``(nan, inf)``, and no numpy warning -- this project runs
    pytest with ``error::RuntimeWarning``.
    """
    rng = make_rng(5)
    inc = np.round(rng.standard_t(1.5, size=20000) * 3.0)
    inc[inc == 0.0] = 1.0
    diag = moment_existence(np.cumsum(inc), q=np.array([-2.0, 1.0, 2.0, 4.0]))
    assert math.isfinite(diag.q_crit)
    assert all(math.isfinite(b) for b in diag.q_crit_ci)
    assert diag.q_crit_ci[0] < diag.q_crit < diag.q_crit_ci[1]
    # Increments are bounded away from zero, so every negative moment exists.
    assert diag.q_crit_negative == float("inf")
    assert bool(diag.exists[0])


def test_moment_existence_reports_a_bounded_sample_as_no_tail() -> None:
    """All increments tied: q_crit is infinite, and the interval must not be NaN."""
    diag = moment_existence(np.cumsum(np.ones(4000)), q=np.array([1.0, 10.0]))
    assert diag.q_crit == float("inf")
    assert diag.q_crit_ci == (float("inf"), float("inf"))
    assert diag.exists.all()


def test_moment_existence_admits_light_tails_only_as_a_floor() -> None:
    """Documented failure mode, pinned so it cannot change silently.

    Every moment of a Gaussian is finite, but the Hill estimator has no plateau
    on light-tailed data and returns a finite sample-size-dependent number, so
    high q are flagged. ``rejected_q`` means "the sample cannot support this",
    not "the moment diverges".
    """
    x = np.cumsum(make_rng(11).standard_normal(20000))
    diag = moment_existence(x, q=np.array([2.0, 4.0, 6.0, 12.0]))
    assert 4.0 < diag.q_crit < 8.0
    assert set(diag.rejected_q().tolist()) == {6.0, 12.0}


# --------------------------------------------------------------------------- #
# scaling_range_test
# --------------------------------------------------------------------------- #


def test_scaling_range_test_accepts_an_exact_power_law() -> None:
    scales = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
    q = np.array([1.0, 2.0, 3.0])
    values = np.exp(np.outer(0.6 * q, np.log(scales)))
    out = scaling_range_test(StructureFunctions(q=q, scales=scales, values=values))
    np.testing.assert_allclose(out["r_squared"], 1.0, atol=1e-12)
    np.testing.assert_allclose(out["curvature"], 0.0, atol=1e-10)
    assert out["n_curved"] == 0
    assert out["verdict"].startswith("Linear")


def test_scaling_range_test_catches_curvature_that_r2_hides(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A crossover with R^2 > 0.99 must still fail the specification test."""
    scales = np.geomspace(2.0, 512.0, 14)
    lx = np.log(scales)
    q = np.array([1.0, 2.0, 3.0])
    # A quadratic in log-log: two regimes joined smoothly.
    log_s = np.outer(q, 0.6 * lx - 0.02 * lx**2)
    out = scaling_range_test(StructureFunctions(q=q, scales=scales, values=np.exp(log_s)))
    with capsys.disabled():
        print(
            f"\n[curvature] min R^2 = {out['min_r_squared']:.5f}  "
            f"beta2 = {out['curvature'][1]:.4f}  p = {out['curvature_p'][1]:.2e}"
        )
    assert out["min_r_squared"] > 0.99
    assert out["n_curved"] == 3
    assert out["verdict"].startswith("CURVED")
    np.testing.assert_allclose(out["curvature"], -0.02 * q, atol=1e-9)


def test_scaling_range_test_reports_a_common_window_on_real_data() -> None:
    out = scaling_range_test(structure_functions(_fbm_path(N, 0.6, 11)))
    assert out["weighted"] is True
    assert out["common_window"] is not None
    lo, hi, n_inside = out["common_window"]
    assert lo < hi and n_inside >= 5
    assert out["failed_q"].size == 0


def test_scaling_range_test_flags_its_own_correlated_residuals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""The curvature p-value is anti-conservative and must say so.

    Brownian motion is exactly self-similar with :math:`\zeta(q)=q/2`: there is
    no crossover. The RESET test nonetheless rejects linearity on most paths,
    because the log-moment errors are strongly autocorrelated across scales
    (overlapping windows), which the :math:`t_{n-3}` null forbids. Measured over
    30 fresh Brownian paths: 24 declared CURVED at a nominal FDR of 5%. This
    test pins the *diagnostic*, so a caller can never be told CURVED without
    also being told the p-value is not trustworthy.
    """
    x = np.cumsum(make_rng(3000).standard_normal(N))
    out = scaling_range_test(structure_functions(x))
    rho = float(np.nanmax(np.abs(out["residual_autocorr"])))
    with capsys.disabled():
        print(
            f"\n[scaling range, exact Brownian] n_curved = {out['n_curved']}/6  "
            f"min R^2 = {out['min_r_squared']:.5f}  max |lag-1 residual rho| = {rho:.2f}"
        )
    assert np.isfinite(out["residual_autocorr"]).all()
    assert rho > 0.3, "the residuals really are correlated; if not, the warning is miscalibrated"
    if out["n_curved"] > 0:
        assert "anti-conservative" in out["verdict"]


def test_scaling_range_test_needs_four_scales() -> None:
    scales = np.array([2.0, 4.0, 8.0])
    q = np.array([1.0, 2.0, 3.0])
    values = np.exp(np.outer(q, np.log(scales)))
    with pytest.raises(ValueError, match="at least 4 scales"):
        scaling_range_test(StructureFunctions(q=q, scales=scales, values=values))


# --------------------------------------------------------------------------- #
# concavity_test
# --------------------------------------------------------------------------- #


def test_concavity_test_passes_a_concave_zeta() -> None:
    q = np.linspace(-3.0, 3.0, 21)
    exps = ScalingExponents(
        q=q, zeta=1.0 * q - 0.5 * 0.06 * q**2, zeta_stderr=np.full(q.size, 0.01)
    )
    res = concavity_test(exps, n_surrogates=199, seed=5)
    assert res.statistic == pytest.approx(0.0, abs=1e-12)
    assert res.p_value > 0.5
    assert res.diagnostics["verdict"] == "concave"


def test_concavity_test_flags_a_convex_kink(capsys: pytest.CaptureFixture[str]) -> None:
    q = np.linspace(-3.0, 3.0, 21)
    zeta = 1.0 * q - 0.5 * 0.06 * q**2
    zeta[10] -= 0.25  # a dent the Legendre transform would silently discard
    exps = ScalingExponents(q=q, zeta=zeta, zeta_stderr=np.full(q.size, 0.01))
    res = concavity_test(exps, n_surrogates=199, seed=5)
    with capsys.disabled():
        print(
            f"\n[concavity] defect = {res.statistic:.4f} "
            f"({res.diagnostics['defect_over_median_stderr']:.1f} x SE)  p = {res.p_value:.4f}"
        )
    assert res.statistic > 0.2
    assert res.p_value <= 0.01
    assert res.diagnostics["verdict"] == "not-concave"
    assert "does not apply" in res.interpretation


def test_concavity_test_is_reproducible_from_the_seed() -> None:
    q = np.linspace(-2.0, 2.0, 11)
    rng = make_rng(0)
    zeta = 0.8 * q - 0.05 * q**2 + 0.01 * rng.standard_normal(q.size)
    exps = ScalingExponents(q=q, zeta=zeta, zeta_stderr=np.full(q.size, 0.01))
    a = concavity_test(exps, n_surrogates=99, seed=77)
    b = concavity_test(exps, n_surrogates=99, seed=77)
    np.testing.assert_allclose(a.null_distribution, b.null_distribution)
    assert a.provenance is not None and b.provenance is not None
    assert a.provenance.root == b.provenance.root


def test_concavity_test_on_an_estimated_fbm_curve() -> None:
    res = concavity_test(structure_estimator(_fbm_path(N, 0.6, 11)), n_surrogates=99, seed=1)
    assert res.diagnostics["verdict"] in {"concave", "marginal", "not-concave"}
    assert 0.0 < res.p_value <= 1.0
