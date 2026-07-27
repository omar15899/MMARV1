r"""Validation of :mod:`mmar.estimators.structure_function` against closed forms.

Every tolerance quoted here was measured before it was asserted; the observed
value is stated next to the bound so that a regression shows up as a change in
the number, not merely as a red test.

Measured on this machine (numpy 2.x, float64):

==========================================  =========================  ==========
Quantity                                    Observed                   Bound
==========================================  =========================  ==========
:math:`\hat\tau(0)`, any scale grid         exactly :math:`-1.0`       ``1e-12``
:math:`\hat\zeta(1)-1` on trading time      exactly :math:`0.0`        ``1e-12``
Binomial cascade, dyadic scales             :math:`1.3\times10^{-8}`   ``1e-6``
Binomial :math:`f(\alpha)`, dyadic scales   :math:`2.0\times10^{-7}`   ``1e-5``
:math:`\tau_{\text{sum}}-(\zeta-1)`         :math:`2.7\times10^{-15}`  ``1e-10``
fBm :math:`h(2)-H`, 3 pinned seeds          :math:`-0.007` … :math:`-0.026`  ``0.035``
fBm :math:`h(q)` spread, :math:`q\in[0.5,5]`  :math:`\le 0.011`        ``0.02``
fBm :math:`h(2)` **population** bias        :math:`-0.0016\pm0.0009`   ``0.006``
:math:`\mathrm{sd}(\hat\zeta)/\overline{\mathrm{se}}`  :math:`3.2`--:math:`5.7`  ``> 2``
==========================================  =========================  ==========

The last two rows exist because the first one is *not* a population statement.
The three seeds pinned in :func:`test_monofractal_recovery_on_exact_fbm` all
happen to sit in the lower tail of the sampling distribution (6th, 12th and 7th
percentile of 150 independent paths at :math:`H=0.7`, :math:`n=2^{16}`; the
chance of a random triple averaging that low is 0.4 %). The bias they show,
:math:`-0.024`, is roughly ten times the true one. See
:func:`test_h2_bias_is_a_population_statement_not_a_seed_statement`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.regression import loglog_fit, weighted_by_counts
from mmar.core.types import FitMethod, StructureFunctions, log_spaced_scales
from mmar.core.validate import DataQualityError
from mmar.estimators.structure_function import (
    ESTIMATOR_ID,
    critical_moment_order,
    exponents_from_structure,
    hurst_from_tau_root,
    increment_counts,
    mfc_exponents,
    mfc_exponents_structure,
    partition_function,
    structure_functions,
)
from mmar.models import theory
from mmar.models.cascade import deterministic_cascade, trading_time
from mmar.models.fbm import fbm, levy_stable_increments

M_BINOMIAL = (0.6, 0.4)
KMAX = 16
N_FBM = 1 << 16
SEEDS = (11, 4242, 7)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def theta() -> np.ndarray:
    r"""Trading time of the deterministic binomial cascade, :math:`m_0=0.6`.

    ``trading_time`` returns ``2**KMAX + 1`` points, so every dyadic lag divides
    :math:`n-1` exactly and the disjoint increments *are* the cascade boxes.
    """
    return trading_time(deterministic_cascade(KMAX, list(M_BINOMIAL)))


@pytest.fixture(scope="module")
def dyadic() -> np.ndarray:
    return np.array([2**j for j in range(1, KMAX - 3)], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Increment geometry and the D6 regression
# --------------------------------------------------------------------------- #


def test_increment_counts_match_the_legacy_slicing() -> None:
    """``X[dt::dt] - X[:-dt:dt]`` yields exactly ``floor((n-1)/dt)`` increments."""
    n = 7049
    scales = np.arange(2, 1000, dtype=np.int64)
    probe = np.zeros(n)
    legacy = np.array([probe[int(d) :: int(d)].size for d in scales], dtype=np.int64)
    assert np.array_equal(legacy, increment_counts(n, scales))
    assert np.array_equal(increment_counts(n, scales, overlapping=True), n - scales)


def test_d6_partition_slope_at_q0_is_not_minus_one_on_a_non_dividing_grid() -> None:
    r"""Legacy defect D6, reproduced: :math:`\hat\tau(0)\neq-1` from the count slope.

    :math:`Z(0,\delta)=\lfloor (n-1)/\delta\rfloor`. On the legacy linear grid
    ``arange(2, 1000)`` at :math:`n=7049` the floor tilts the slope to
    :math:`-1.0179` (the audit observed :math:`-1.0174` on the same data with
    the same grid). On the four lags that divide :math:`n-1=7048` the slope is
    :math:`-1` to :math:`10^{-15}`.
    """
    n = 7049
    linear = np.arange(2, 1000, dtype=np.int64)
    tilted = loglog_fit(
        linear.astype(np.float64), increment_counts(n, linear).astype(np.float64)
    ).slope
    assert tilted == pytest.approx(-1.0179, abs=5e-4)
    assert abs(tilted + 1.0) > 1e-2, "the floor bias must be visible, not marginal"

    divisors = np.array([d for d in range(2, 1000) if (n - 1) % d == 0], dtype=np.int64)
    assert divisors.size >= 3
    exact = loglog_fit(
        divisors.astype(np.float64), increment_counts(n, divisors).astype(np.float64)
    ).slope
    assert exact == pytest.approx(-1.0, abs=1e-12)


def test_d6_estimator_tau_zero_is_exactly_minus_one_on_the_same_bad_grid() -> None:
    r"""The corrected (mean-form) estimator gives :math:`\tau(0)=-1` regardless of grid.

    :math:`S(0,\delta)=1` identically, so no arithmetic involving the increment
    count enters :math:`\hat\zeta(0)` at all. This is the fix for D6: the
    identity holds by construction rather than by luck of divisibility.
    """
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.standard_normal(7049))
    scales = np.unique(np.arange(2, 1000, 7, dtype=np.int64))
    e = mfc_exponents(walk, q=np.linspace(-3.0, 3.0, 13), scales=scales)
    i0 = int(np.argmin(np.abs(e.q)))
    assert e.q[i0] == 0.0
    assert e.tau[i0] == pytest.approx(-1.0, abs=1e-9)
    assert e.zeta[i0] == pytest.approx(0.0, abs=1e-9)


def test_structure_function_zero_moment_is_bit_exactly_one(theta: np.ndarray) -> None:
    sf = structure_functions(theta, q=np.array([-1.0, 0.0, 1.0]), scales=np.array([4, 8, 16, 32]))
    assert np.all(sf.values[1] == 1.0)


def test_counts_are_the_actual_number_of_increments(theta: np.ndarray) -> None:
    scales = np.array([3, 7, 32, 100], dtype=np.int64)
    sf = structure_functions(theta, q=np.array([1.0, 2.0, 3.0]), scales=scales)
    assert sf.counts is not None
    assert np.array_equal(sf.counts, increment_counts(theta.size, scales))
    over = structure_functions(theta, q=np.array([1.0, 2.0, 3.0]), scales=scales, overlapping=True)
    assert over.counts is not None
    assert np.array_equal(over.counts, theta.size - scales)


# --------------------------------------------------------------------------- #
# Convention: the sum form is the mean form minus one
# --------------------------------------------------------------------------- #


def test_partition_function_slope_is_zeta_minus_one(theta: np.ndarray, dyadic: np.ndarray) -> None:
    r"""``slope_sum = slope_mean - 1 = zeta - 1 = tau``, to :math:`3\times10^{-15}`."""
    q = np.linspace(-5.0, 5.0, 41)
    mean_form = mfc_exponents(theta, q=q, scales=dyadic)
    pf = partition_function(theta, q=q, scales=dyadic)
    assert pf.counts is not None
    w = weighted_by_counts(pf.counts)
    tau_sum = np.array(
        [
            loglog_fit(pf.scales, pf.values[i], method=FitMethod.WLS, weights=w).slope
            for i in range(pf.q.size)
        ]
    )
    assert np.max(np.abs(tau_sum - mean_form.tau)) < 1e-10
    assert tau_sum[int(np.argmin(np.abs(q)))] == pytest.approx(-1.0, abs=1e-12)


def test_partition_function_equals_mean_times_count(theta: np.ndarray) -> None:
    q = np.array([-2.0, 0.0, 1.5])
    scales = np.array([8, 16, 64], dtype=np.int64)
    sf = structure_functions(theta, q=q, scales=scales)
    pf = partition_function(theta, q=q, scales=scales)
    assert sf.counts is not None
    np.testing.assert_allclose(pf.values, sf.values * sf.counts[None, :], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# (a) Monofractal recovery
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.stochastic
@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_monofractal_recovery_on_exact_fbm(H: float) -> None:
    r"""On exact fBm, :math:`h(q)` is flat and equals :math:`H`.

    Grid :math:`q\in[0.5,5]` (see
    :func:`test_negative_q_on_a_gaussian_signal_is_a_documented_failure` for why
    the negative branch is excluded here), :math:`n=2^{16}`, default geometric
    scale grid, WLS.

    Observed over seeds ``(11, 4242, 7)``:

    * :math:`h(2)-H`: :math:`H=0.3` :math:`[-0.0134,-0.0166,-0.0136]`;
      :math:`H=0.5` :math:`[-0.0184,-0.0201,-0.0163]`;
      :math:`H=0.7` :math:`[-0.0257,-0.0208,-0.0246]`.
    * spread :math:`\max_q h - \min_q h`: :math:`\le 0.0098`.

    These are three **pinned paths**, and they are a byte-for-byte regression
    anchor, not a measurement of the estimator's bias. All three sit in the
    lower tail (per-path :math:`\mathrm{sd}(\hat h(2))=0.0133` at
    :math:`H=0.7`), so the sign is a fact about these seeds; the population
    statement --- a genuine but roughly ten times smaller negative bias --- is
    asserted separately in
    :func:`test_h2_bias_is_a_population_statement_not_a_seed_statement`.
    Re-seeding this test will change the numbers and can change the signs.
    """
    q = np.linspace(0.5, 5.0, 19)
    biases, spreads = [], []
    for seed in SEEDS:
        e = mfc_exponents(fbm(N_FBM, H, seed=seed), q=q)
        h = e.hq
        assert np.all(np.isfinite(h))
        biases.append(e.h2 - H)
        spreads.append(float(h.max() - h.min()))
        assert e.is_monofractal(tol=0.02), f"h(q) not flat: spread {h.max() - h.min():.4f}"
    assert max(abs(b) for b in biases) < 0.035, f"h(2) biases {biases}"
    assert max(spreads) < 0.02, f"h(q) spreads {spreads}"


@pytest.mark.stochastic
def test_h2_bias_is_a_population_statement_not_a_seed_statement() -> None:
    r"""The Jensen bias is real, :math:`O(1/N)`, and an order of magnitude small.

    Over 120 *independent* paths at :math:`n=2^{13}` the mean
    :math:`\hat h(2)-H` is :math:`-0.008\pm0.002` --- negative, as the Jensen
    argument predicts, but tiny against the per-path scatter
    (:math:`\mathrm{sd}\approx0.026`). At :math:`n=2^{16}` it falls to
    :math:`-0.0016\pm0.0009` over 270 paths, i.e. it decays like the coarse-end
    window count. Trimming :math:`\delta_{\max}` removes it, which is the
    signature of the mechanism and is pinned here at the same :math:`n`.

    This test is what makes the three-seed table above honest: without it, the
    :math:`-0.024` observed at :math:`n=2^{16}` on seeds ``(11, 4242, 7)`` reads
    as the estimator's bias, and it is not.
    """
    n, reps = 1 << 13, 120
    q = np.array([1.0, 2.0, 3.0])
    full = np.array(
        [mfc_exponents(fbm(n, 0.5, seed=900_000 + r), q=q).h2 - 0.5 for r in range(reps)]
    )
    trimmed = np.array(
        [
            mfc_exponents(
                fbm(n, 0.5, seed=900_000 + r), q=q, scales=log_spaced_scales(n, s_max=64)
            ).h2
            - 0.5
            for r in range(reps)
        ]
    )
    se = full.std(ddof=1) / np.sqrt(reps)
    assert full.mean() < 0.0, (
        f"the Jensen bias must be negative in the mean, got {full.mean():+.5f}"
    )
    assert abs(full.mean()) < 0.02, f"mean bias {full.mean():+.5f} is far larger than measured"
    assert full.mean() < -2.0 * se, "the mean bias must be resolvable, not noise"
    assert trimmed.mean() > full.mean(), (
        f"trimming the coarse end must shrink the bias: full {full.mean():+.5f} "
        f"vs trimmed {trimmed.mean():+.5f}"
    )
    # And the headline three-seed number is NOT the population value.
    assert abs(full.mean()) < 0.5 * 0.024, (
        "the three pinned seeds report ~10x the population bias; if this ever "
        "becomes tight, the seed table has stopped being a tail draw"
    )


@pytest.mark.stochastic
def test_reported_stderr_is_a_line_quality_not_a_sampling_error() -> None:
    r"""``zeta_stderr`` understates the sampling error by a factor of several.

    Every scale is computed from the same series, so the log--log residuals are
    strongly dependent while :func:`~mmar.core.regression.loglog_fit` treats
    them as independent. Measured on exact fBm, :math:`H=0.5`:

    ==========  ============================================  ==============
    :math:`n`   :math:`\mathrm{sd}(\hat\zeta(2))/\overline{\mathrm{se}}`  95 % coverage
    ==========  ============================================  ==============
    :math:`2^{13}`  3.2--3.6                                  ~0.3
    :math:`2^{14}`  5.6                                       0.28
    :math:`2^{16}`  5.7                                       0.25
    ==========  ============================================  ==============

    The interval is a goodness-of-line diagnostic. This test exists so that a
    future "we now report honest error bars" claim has to move this number.
    """
    n, reps = 1 << 13, 120
    q = np.array([1.0, 2.0, 3.0])
    zeta, stderr, covered = [], [], 0
    for r in range(reps):
        e = mfc_exponents(fbm(n, 0.5, seed=800_000 + r), q=q)
        assert e.zeta_stderr is not None and e.zeta_ci is not None
        zeta.append(e.zeta[1])
        stderr.append(e.zeta_stderr[1])
        lo, hi = e.zeta_ci
        covered += bool(lo[1] <= 1.0 <= hi[1])
    ratio = float(np.std(zeta, ddof=1) / np.mean(stderr))
    coverage = covered / reps
    assert ratio > 2.0, f"sd/se = {ratio:.2f}; the SEs are no longer the understated ones"
    assert coverage < 0.75, f"nominal-95% coverage measured at {coverage:.2f}"


@pytest.mark.stochastic
def test_negative_q_on_a_gaussian_signal_is_a_documented_failure() -> None:
    r"""fBm has no :math:`q\le-1` moment, and the estimator must show it, not hide it.

    :math:`\mathbb E|Z|^q=\infty` for :math:`q\le-1` when :math:`Z` is Gaussian,
    so :math:`\hat S(q,\delta)` on the negative branch is a function of the
    sample minimum rather than of :math:`H`. Observed at :math:`H=0.7`,
    :math:`n=2^{16}`, seed 20260727: :math:`h(-5)=1.43` against a true
    :math:`H=0.7`, and the spread of :math:`h` over :math:`q\in[-5,5]` is
    :math:`0.63` where the true value is :math:`0`.

    This is why :func:`mfc_exponents` is not the recommended estimator for the
    right branch of :math:`f(\alpha)` on a *signal*, and why the multifractal
    recovery tests below use a *measure* (a trading time), whose increments are
    bounded away from zero.
    """
    H = 0.7
    x = fbm(N_FBM, H, seed=20260727)
    e = mfc_exponents(x, q=np.linspace(-5.0, 5.0, 41))
    h = e.hq[np.isfinite(e.hq)]
    assert e.h2 == pytest.approx(H, abs=0.035)
    h_neg = float(e.hq[0])
    assert h_neg > 1.2, f"expected the divergent negative branch to blow up, got h(-5)={h_neg}"
    assert (h.max() - h.min()) > 0.4
    assert not e.is_monofractal(tol=0.05)


# --------------------------------------------------------------------------- #
# (b) Multifractal recovery
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_binomial_cascade_tau_recovered_on_dyadic_scales(
    theta: np.ndarray, dyadic: np.ndarray
) -> None:
    r"""Deterministic binomial cascade: :math:`\hat\tau = \tau` to :math:`1.3\times10^{-8}`.

    The measure is exactly self-similar at dyadic box sizes, so the recovery is
    limited only by floating point, not by statistics: the tolerance
    :math:`10^{-6}` is a *round-off* budget. The residual comes from
    :func:`~mmar.models.cascade.trading_time`, which forms :math:`\theta` by
    ``cumsum`` over :math:`2^{16}` cells; differencing that at the finest lag
    recovers masses of order :math:`0.4^{16}\approx4\times10^{-7}` with absolute
    error :math:`O(\varepsilon)`, i.e. relative error :math:`\sim10^{-10}`,
    which at :math:`q=-5` becomes :math:`\sim10^{-9}` in :math:`\log S`.
    """
    q = np.linspace(-5.0, 5.0, 41)
    e = mfc_exponents(theta, q=q, scales=dyadic)
    err = np.max(np.abs(e.tau - theory.cascade_tau(q, list(M_BINOMIAL))))
    assert err < 1e-6, f"max |tau_hat - tau| = {err:.3e}"
    np.testing.assert_allclose(e.zeta, theory.cascade_zeta(q, list(M_BINOMIAL)), atol=1e-6)


@pytest.mark.theory
def test_binomial_cascade_falpha_recovered(theta: np.ndarray, dyadic: np.ndarray) -> None:
    r"""The Legendre transform of :math:`\hat\zeta` matches the entropy formula.

    Observed :math:`\max_\alpha|\hat f-f| = 2.0\times10^{-7}` over the 201-point
    :math:`\alpha` grid; bound :math:`10^{-5}`. The apex lands at
    :math:`\hat\alpha_0=1.0294468`, against the closed-form
    :math:`\tau'(0)=1.0294468445`.
    """
    q = np.linspace(-5.0, 5.0, 41)
    e = mfc_exponents(theta, q=q, scales=dyadic)
    sp = spectrum_from_exponents(e)
    f_true = theory.cascade_falpha(sp.alpha, list(M_BINOMIAL))
    ok = np.isfinite(f_true) & np.isfinite(sp.f_alpha)
    assert ok.sum() > 100
    err = float(np.max(np.abs(sp.f_alpha[ok] - f_true[ok])))
    assert err < 1e-5, f"max |f_hat - f| = {err:.3e}"
    d = sp.descriptors()
    assert d.f_max == pytest.approx(1.0, abs=1e-6)
    assert d.alpha_0 == pytest.approx(
        float(theory.cascade_alpha(np.array([0.0]), list(M_BINOMIAL))[0]), abs=1e-4
    )


@pytest.mark.theory
def test_non_dyadic_scales_on_a_dyadic_cascade_carry_log_periodic_error(
    theta: np.ndarray,
) -> None:
    r"""Known bias: a :math:`b`-adic cascade is only *discretely* scale invariant.

    A box of non-dyadic width straddles cells from different branches of the
    tree, so :math:`\log Z(q,\delta)` acquires a log-periodic oscillation in
    :math:`\log_2\delta` instead of a clean line. On the default geometric grid
    the error in :math:`\hat\tau` reaches :math:`0.157`, seven orders of
    magnitude worse than on dyadic lags. The remedy is
    :func:`~mmar.core.types.dyadic_scales`, not a wider grid.
    """
    q = np.linspace(-5.0, 5.0, 41)
    e = mfc_exponents(theta, q=q)
    err = float(np.max(np.abs(e.tau - theory.cascade_tau(q, list(M_BINOMIAL)))))
    assert err > 1e-2, f"expected log-periodic contamination, got {err:.3e}"
    assert err < 0.5


# --------------------------------------------------------------------------- #
# (c) Convention anchors
# --------------------------------------------------------------------------- #


def test_tau_zero_and_tau_one_anchors(theta: np.ndarray, dyadic: np.ndarray) -> None:
    r""":math:`\tau(0)=-1` always; :math:`\tau(1)=0` for a conservative measure.

    :math:`\tau(1)=0` is mass conservation: :math:`S(1,\delta)=\langle\Delta
    \theta\rangle = 1/N(\delta) \propto \delta`, so :math:`\zeta(1)=1` and
    :math:`\tau(1)=0` exactly on lags that divide :math:`n-1`.
    """
    q = np.array([-1.0, 0.0, 1.0, 2.0])
    e = mfc_exponents(theta, q=q, scales=dyadic)
    assert e.tau[1] == pytest.approx(-1.0, abs=1e-12)
    assert e.tau[2] == pytest.approx(0.0, abs=1e-12)
    assert e.d_support == 1.0
    assert e.estimator == ESTIMATOR_ID


# --------------------------------------------------------------------------- #
# (d) Determinism
# --------------------------------------------------------------------------- #


def test_determinism_is_bitwise(theta: np.ndarray, dyadic: np.ndarray) -> None:
    q = np.linspace(-4.0, 4.0, 17)
    a = mfc_exponents(theta, q=q, scales=dyadic, seed=1234)
    b = mfc_exponents(theta, q=q, scales=dyadic, seed=1234)
    assert np.array_equal(a.zeta, b.zeta)
    assert a.zeta_stderr is not None and b.zeta_stderr is not None
    assert np.array_equal(a.zeta_stderr, b.zeta_stderr)
    assert a.structure is not None and b.structure is not None
    assert np.array_equal(a.structure.values, b.structure.values)


# --------------------------------------------------------------------------- #
# (e) Provenance
# --------------------------------------------------------------------------- #


def test_provenance_root_is_stable_and_parameter_sensitive(
    theta: np.ndarray, dyadic: np.ndarray
) -> None:
    q = np.linspace(-4.0, 4.0, 17)
    base = mfc_exponents(theta, q=q, scales=dyadic)
    assert base.provenance is not None
    root = base.provenance.root
    assert mfc_exponents(theta, q=q, scales=dyadic).provenance.root == root  # type: ignore[union-attr]

    variants = {
        "overlapping": mfc_exponents(theta, q=q, scales=dyadic, overlapping=True),
        "fit_method": mfc_exponents(theta, q=q, scales=dyadic, fit_method=FitMethod.OLS),
        "d_support": mfc_exponents(theta, q=q, scales=dyadic, d_support=2.0),
        "q": mfc_exponents(theta, q=np.linspace(-4.0, 4.0, 19), scales=dyadic),
        "scales": mfc_exponents(theta, q=q, scales=dyadic[1:]),
        "select_range": mfc_exponents(theta, q=q, scales=dyadic, select_range=True),
        "seed": mfc_exponents(theta, q=q, scales=dyadic, seed=7),
        "min_windows": mfc_exponents(theta, q=q, scales=dyadic, min_windows=8),
        "x": mfc_exponents(theta * 2.0, q=q, scales=dyadic),
    }
    for name, e in variants.items():
        assert e.provenance is not None
        assert e.provenance.root != root, f"root did not change when {name} changed"

    inputs = {i.name for i in base.provenance.inputs}
    assert inputs == {"x"}
    assert base.provenance.diagnostics["n_samples"] == theta.size
    assert base.provenance.estimator == ESTIMATOR_ID


# --------------------------------------------------------------------------- #
# (f) Negative q: the direct regression test for legacy defect D2
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_negative_q_widens_the_recovered_alpha_range(theta: np.ndarray, dyadic: np.ndarray) -> None:
    r"""D2: with :math:`q\ge0` only, the right branch of :math:`f(\alpha)` is unreachable.

    True support for :math:`m_0=0.6`: :math:`\alpha\in[0.7370,1.3219]`,
    width :math:`0.5850`.

    ==================  ============================  ==================
    grid                recovered :math:`\alpha`      fraction of support
    ==================  ============================  ==================
    :math:`q\in[0,5]`   :math:`[0.8051, 1.0294]`      38 %
    :math:`q\in[-5,5]`  :math:`[0.8051, 1.2538]`      77 %
    ==================  ============================  ==================

    The left endpoint is identical --- it is set by :math:`q_{\max}` --- while the
    right endpoint moves from the apex :math:`\alpha_0` out towards
    :math:`\alpha_{\max}`, exactly doubling the width. Nothing but negative
    :math:`q` can produce that half of the spectrum.
    """
    a_min_true, a_max_true = theory.cascade_alpha_range(list(M_BINOMIAL))
    both = spectrum_from_exponents(
        mfc_exponents(theta, q=np.linspace(-5.0, 5.0, 41), scales=dyadic)
    ).descriptors()
    positive = spectrum_from_exponents(
        mfc_exponents(theta, q=np.linspace(0.0, 5.0, 21), scales=dyadic)
    ).descriptors()

    assert positive.alpha_max == pytest.approx(both.alpha_0, abs=1e-3), (
        "a q >= 0 grid can reach no exponent above the apex"
    )
    assert both.width > 1.9 * positive.width
    assert both.alpha_max > positive.alpha_max + 0.2
    assert both.alpha_min == pytest.approx(positive.alpha_min, abs=1e-3)
    assert a_min_true < both.alpha_min < both.alpha_max < a_max_true
    assert both.width / (a_max_true - a_min_true) > 0.7
    assert positive.width / (a_max_true - a_min_true) < 0.45


# --------------------------------------------------------------------------- #
# The MFC Hurst identity
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_hurst_from_tau_root_matches_the_oracle() -> None:
    r"""On the exact MMAR :math:`\tau`, the bracketed root returns :math:`H`."""
    q = np.linspace(-5.0, 5.0, 81)
    for H in (0.45, 0.55, 0.7):
        exact = theory.mmar_exponents_lognormal(q, H=H)
        assert hurst_from_tau_root(exact) == pytest.approx(H, abs=1e-6)
        assert hurst_from_tau_root(exact) == pytest.approx(
            theory.mmar_hurst_from_tau(exact.q, exact.tau), abs=1e-9
        )


def test_hurst_from_tau_root_on_the_estimated_trading_time(
    theta: np.ndarray, dyadic: np.ndarray
) -> None:
    r"""A conservative measure has :math:`\tau(1)=0`, i.e. :math:`q^\*=1`, :math:`H=1`."""
    e = mfc_exponents(theta, q=np.linspace(-5.0, 5.0, 41), scales=dyadic)
    assert hurst_from_tau_root(e) == pytest.approx(1.0, abs=1e-6)


def test_hurst_from_tau_root_returns_nan_loudly_without_a_sign_change(
    theta: np.ndarray,
) -> None:
    """No fallback, no exception, no silence: NaN plus a warning naming the range."""
    scales = np.array([2**j for j in range(1, 11)], dtype=np.int64)
    e = mfc_exponents(theta, q=np.linspace(-5.0, 0.5, 23), scales=scales)
    with pytest.warns(UserWarning, match="no sign change"):
        h = hurst_from_tau_root(e)
    assert np.isnan(h)


# --------------------------------------------------------------------------- #
# Moment existence
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
def test_critical_moment_order_separates_stable_from_gaussian() -> None:
    r"""Hill index of :math:`|\Delta X|`: :math:`\approx\alpha` for stable, large for fBm.

    Observed: :math:`\alpha`-stable motion with :math:`\alpha=1.5` gives
    :math:`\hat q_{\mathrm{crit}} = 1.56`; fBm with :math:`H=0.7` gives
    :math:`7.67`. The latter is *not* a critical order --- a Gaussian has every
    moment --- it is the documented upward-drifting Hill statistic in the Gumbel
    domain, and is to be read as "no evidence of divergence in the fitted
    :math:`q` window".
    """
    stable = np.cumsum(levy_stable_increments(1 << 15, alpha=1.5, seed=3))
    gaussian = fbm(N_FBM, 0.7, seed=7)
    q_stable = critical_moment_order(stable)
    q_gauss = critical_moment_order(gaussian)
    assert q_stable == pytest.approx(1.5, abs=0.35), f"stable q_crit = {q_stable:.3f}"
    assert q_gauss > 5.0, f"gaussian q_crit = {q_gauss:.3f}"
    assert q_gauss > 3.0 * q_stable


def test_critical_moment_order_rejects_bad_fractions(theta: np.ndarray) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        critical_moment_order(theta, fractions=(0.0, 0.5))
    with pytest.raises(ValueError, match="positive integer"):
        critical_moment_order(theta, lag=0)


# --------------------------------------------------------------------------- #
# Refitting, weighting, options
# --------------------------------------------------------------------------- #


def test_refit_from_a_precomputed_surface_reproduces_the_estimator(
    theta: np.ndarray, dyadic: np.ndarray
) -> None:
    q = np.linspace(-3.0, 3.0, 13)
    direct = mfc_exponents(theta, q=q, scales=dyadic)
    surface = mfc_exponents_structure(theta, q=q, scales=dyadic)
    refit = exponents_from_structure(surface)
    assert np.array_equal(direct.zeta, refit.zeta)
    assert np.array_equal(direct.structure.values, surface.values)  # type: ignore[union-attr]

    # The point of keeping the surface: refit over a narrower range for free.
    assert surface.counts is not None
    narrow = exponents_from_structure(
        StructureFunctions(
            q=surface.q,
            scales=surface.scales[:5],
            values=surface.values[:, :5],
            counts=surface.counts[:5],
            provenance=surface.provenance,
        )
    )
    assert narrow.scaling_range is not None
    assert narrow.scaling_range.n_scales == 5
    assert not np.array_equal(narrow.zeta, direct.zeta)


def test_refit_derives_provenance_instead_of_dropping_it(
    theta: np.ndarray, dyadic: np.ndarray
) -> None:
    """The refit entry point must not be the one hole in the audit chain.

    ``mfdfa_from_structure`` chains a child record onto the surface's; so does
    this. A refit that returned ``provenance=None`` would make
    ``report.summary()['root']`` disappear the moment a caller changed the
    scaling range, which is exactly when the record matters most.
    """
    surface = mfc_exponents_structure(theta, q=np.linspace(-3.0, 3.0, 13), scales=dyadic)
    assert surface.provenance is not None

    refit = exponents_from_structure(surface)
    assert refit.provenance is not None, "refit dropped provenance"
    assert refit.provenance.root != surface.provenance.root, "child must not alias its parent"
    assert "parent" in {i.name for i in refit.provenance.inputs}

    other = exponents_from_structure(surface, fit_method=FitMethod.OLS)
    assert other.provenance is not None
    assert other.provenance.root != refit.provenance.root, "fit_method must move the root"
    assert exponents_from_structure(surface).provenance.root == refit.provenance.root  # type: ignore[union-attr]


def test_weights_of_the_wrong_length_are_rejected_by_name(
    theta: np.ndarray, dyadic: np.ndarray
) -> None:
    surface = mfc_exponents_structure(theta, q=np.linspace(-2.0, 2.0, 9), scales=dyadic)
    with pytest.raises(ValueError, match="one weight per scale"):
        exponents_from_structure(surface, weights=np.ones(3))


def test_overlapping_increments_reduce_the_standard_error() -> None:
    """Lower variance, correlated windows; the weights use the disjoint count."""
    x = fbm(N_FBM, 0.7, seed=7)
    q = np.linspace(0.5, 5.0, 19)
    over = mfc_exponents(x, q=q, overlapping=True)
    disjoint = mfc_exponents(x, q=q, overlapping=False)
    assert np.median(over.zeta_stderr) < np.median(disjoint.zeta_stderr)  # type: ignore[arg-type]
    assert over.h2 == pytest.approx(disjoint.h2, abs=0.01)
    assert over.structure is not None and over.structure.counts is not None
    scales = np.rint(over.structure.scales).astype(np.int64)
    np.testing.assert_array_equal(over.structure.counts, x.size - scales)
    # The regression weight is the disjoint-equivalent count, not the raw one.
    weights = over.fits[0].weights
    assert weights is not None
    np.testing.assert_array_equal(weights, (x.size - 1) // scales)


def test_select_range_uses_one_common_window(theta: np.ndarray) -> None:
    q = np.linspace(-2.0, 2.0, 9)
    e = mfc_exponents(theta, q=q, select_range=True)
    assert e.scaling_range is not None
    assert e.scaling_range.selection.startswith("auto")
    windows = {(f.n_points, f.scaling_range.s_min, f.scaling_range.s_max) for f in e.fits}  # type: ignore[union-attr]
    assert len(windows) == 1, "every q must be fitted over the same scale window"


def test_fits_and_intervals_are_populated(theta: np.ndarray, dyadic: np.ndarray) -> None:
    q = np.linspace(-2.0, 2.0, 9)
    e = mfc_exponents(theta, q=q, scales=dyadic)
    assert len(e.fits) == q.size
    assert e.zeta_stderr is not None and e.zeta_ci is not None
    lo, hi = e.zeta_ci
    assert np.all(hi >= lo), "the confidence interval must not be inverted (legacy D4)"
    assert np.all(np.isfinite(e.zeta_stderr))
    assert e.structure is not None and e.structure.counts is not None
    for f in e.fits:
        assert f.method is FitMethod.WLS
        assert f.weights is not None
        np.testing.assert_array_equal(f.weights, e.structure.counts)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_signed_moments_require_integer_orders(theta: np.ndarray) -> None:
    with pytest.raises(ValueError, match="integer"):
        structure_functions(theta, q=np.array([-1.5, 0.0, 1.5]), absolute=False)
    sf = structure_functions(
        theta, q=np.array([1.0, 2.0, 3.0]), scales=np.array([4, 8, 16]), absolute=False
    )
    assert np.all(sf.values > 0.0), "increments of a monotone trading time are positive"
    assert sf.provenance is not None
    assert sf.provenance.diagnostics["n_nonfinite_values"] == 0
    assert "signed" in sf.provenance.diagnostics["drop_zeros_policy"]


def test_signed_moments_over_exact_zeros_go_infinite_loudly() -> None:
    r"""``absolute=False`` never sub-samples, so :math:`0^{q\le0}=\infty` survives.

    The absolute path drops zeros so that every :math:`q` shares one
    sub-sample; the signed path cannot, because that sub-sample would not be
    the one the absolute moments were taken over and the two would stop being
    comparable. The infinity is therefore deliberate --- and must be announced,
    not leaked as a bare numpy ``RuntimeWarning`` while the provenance record
    claims no zero handling was needed.
    """
    x = np.cumsum(np.abs(np.random.default_rng(0).standard_normal(4096)))
    x[100:200] = x[100]
    with pytest.warns(UserWarning, match="non-finite"):
        sf = structure_functions(
            x, q=np.array([-2.0, -1.0, 1.0]), scales=np.array([4, 8, 16]), absolute=False
        )
    assert sf.provenance is not None
    assert sf.provenance.diagnostics["n_nonfinite_values"] == 6
    assert np.all(np.isinf(sf.values[:2]))


def test_exact_zero_increments_are_dropped_and_recorded() -> None:
    """A flat stretch produces zero increments; they are removed, never ignored."""
    rng = np.random.default_rng(5)
    x = np.cumsum(np.abs(rng.standard_normal(4096)))
    x[1000:1400] = x[1000]
    sf = structure_functions(x, q=np.array([-2.0, 0.0, 2.0]), scales=np.array([4, 8, 16, 32]))
    assert sf.provenance is not None
    dropped = sf.provenance.diagnostics["zeros_dropped"]
    assert sum(dropped) > 0
    assert sf.counts is not None
    expected = increment_counts(x.size, np.rint(sf.scales).astype(np.int64))
    np.testing.assert_array_equal(sf.counts, expected - np.asarray(dropped))
    assert np.all(np.isfinite(sf.values))

    # With a strictly positive q grid the zeros are informative and kept.
    keep = structure_functions(x, q=np.array([1.0, 2.0, 3.0]), scales=np.array([4, 8, 16, 32]))
    assert keep.provenance is not None
    assert sum(keep.provenance.diagnostics["zeros_dropped"]) == 0
    assert keep.counts is not None
    np.testing.assert_array_equal(keep.counts, expected)


def test_scales_leaving_too_few_increments_are_rejected() -> None:
    x = fbm(4096, 0.5, seed=2)
    with pytest.raises((ValueError, DataQualityError)):
        mfc_exponents(x, q=np.array([1.0, 2.0, 3.0]), scales=np.array([2000, 3000]))


def test_the_spine_window_floor_is_tightened_to_the_true_increment_count() -> None:
    r"""``check_scales`` counts :math:`n/\delta`; this estimator needs :math:`(n-1)/\delta`.

    At :math:`n=2^{16}` and :math:`\delta=2^{14}` the spine sees four windows
    but only three increments exist, so the scale must be dropped rather than
    fitted from three points.
    """
    x = fbm(1 << 16, 0.5, seed=2)
    scales = np.array([1 << 12, 1 << 13, 1 << 14], dtype=np.int64)
    sf = structure_functions(x, q=np.array([1.0, 2.0, 3.0]), scales=scales, min_windows=4)
    assert np.array_equal(np.rint(sf.scales).astype(np.int64), scales[:2])


def test_constant_series_is_refused() -> None:
    with pytest.raises(DataQualityError, match="constant"):
        mfc_exponents(np.ones(4096))
