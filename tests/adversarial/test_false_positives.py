r"""The impostors: processes that are *not* multifractal but look like it.

Every claim this library makes rests on being able to separate a genuine cascade
from the mechanisms that counterfeit one --- a fat-tailed marginal, linear
memory, non-stationarity, a trend, a period, a handful of outliers. This module
attacks that separation directly. Each test builds a process whose true
singularity spectrum is known to be degenerate and asks whether the honest
verdict is reachable from the library's own outputs.

Measured on this tree (mfdfa unless stated, ``q = linspace(-5, 5, 21)``):

=========================================  =========  =========  ==============
Process (n = 4096--8192)                    width      c2         shuffle p
=========================================  =========  =========  ==============
exact fBm, H = 0.6, n = 32768               0.036      -0.002     --
exact fBm, H = 0.6, n = 512                 0.241      --         --
exact fBm, H = 0.5, structure functions     0.930      -0.044     --
   the same, restricted to existing q       0.099      --         --
iid Gaussian, structure functions           1.197      -0.200     0.425
iid Levy-stable, alpha = 1.2                0.90-1.05  -0.11      0.04-0.74
iid Levy-stable, alpha = 1.6                0.53-0.65  -0.12      0.62-0.66
iid Levy-stable, alpha = 1.8                0.32-0.52  -0.02      0.12-0.82
fBm H = 0.6 + 5 spikes in 8192              0.516      -0.038     0.300
iid Gaussian x (1 + 9t) volatility ramp     0.182      -0.021     --
GARCH(1,1), (0.10, 0.88)                    0.398      -0.019     0.020
fBm + sinusoid                              0.600      --         --
fBm(0.3) then fBm(0.8)                      0.000(!)   +0.112     1.000(!)
=========================================  =========  =========  ==============

The rows that matter most are the Levy block --- the naive width is large and the
shuffle surrogate correctly refuses to reject, which is the evidence the
library's central claim rests on --- and the last row, where the *width* of a
non-stationary sample collapses to exactly zero and was, before the fix in
:func:`~mmar.inference.tests.spectrum_statistic` carried here, handed to the
hypothesis test as a measurement.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from mmar.analysis import analyze, compare_estimators, consensus
from mmar.core.legendre import spectrum_from_exponents
from mmar.core.rng import make_rng, spawn_rngs
from mmar.core.types import FloatArray, ScalingExponents
from mmar.estimators.mfdfa import mfdfa, mfdfa_structure
from mmar.estimators.structure_function import mfc_exponents
from mmar.estimators.wavelet_leaders import wavelet_leaders
from mmar.inference.surrogates import spectrum_width
from mmar.inference.tests import (
    moment_existence,
    multifractality_report,
    multifractality_test,
    scaling_range_test,
    spectrum_statistic,
)
from mmar.models.fbm import fbm, fgn, levy_stable_increments

Q = np.linspace(-5.0, 5.0, 21)
LEVEL_MFDFA = functools.partial(mfdfa, q=Q, integrate=False)
LEVEL_SF = functools.partial(mfc_exponents, q=Q)
LEVEL_WL = functools.partial(wavelet_leaders, q=Q)


def _garch11(
    n: int,
    seed: int,
    *,
    omega: float = 1e-6,
    alpha: float = 0.10,
    beta: float = 0.88,
    burn: int = 1000,
) -> FloatArray:
    r"""GARCH(1,1) increments: volatility clustering with no cascade underneath.

    :math:`x_t=\sigma_t z_t`, :math:`\sigma_t^2=\omega+\alpha x_{t-1}^2+\beta
    \sigma_{t-1}^2`. Strictly stationary for :math:`\alpha+\beta<1` and *not*
    self-similar at any exponent, so its true spectrum is a point.
    """
    rng = make_rng(seed)
    z = rng.standard_normal(n + burn)
    out = np.empty(n + burn, dtype=np.float64)
    s2 = omega / (1.0 - alpha - beta)
    for i in range(n + burn):
        out[i] = np.sqrt(s2) * z[i]
        s2 = omega + alpha * out[i] ** 2 + beta * s2
    return np.ascontiguousarray(out[burn:])


def _two_regime(n: int, seed_a: int, seed_b: int) -> FloatArray:
    """fBm at H=0.3 followed by fBm at H=0.8, variance-matched at the join.

    Non-stationary in its scaling exponent, monofractal on each half, and not
    multifractal anywhere: no single self-similar law describes the whole path.
    """
    lo = fgn(n // 2, 0.3, seed=seed_a)
    hi = fgn(n // 2, 0.8, seed=seed_b)
    joined = np.concatenate([lo, hi * float(np.std(lo)) / float(np.std(hi))])
    return np.ascontiguousarray(np.cumsum(joined))


# --------------------------------------------------------------------------- #
# Impostor 1: a fat marginal with no temporal structure at all
# --------------------------------------------------------------------------- #


def test_iid_gaussian_width_is_a_marginal_artefact_the_shuffle_refuses_to_reject() -> None:
    r"""Attack: iid Gaussian increments through the structure-function estimator.

    The process has no temporal structure whatsoever, so its true spectrum is the
    single point :math:`\alpha=1/2`. The negative-:math:`q` branch of
    :math:`\langle|\Delta X|^q\rangle` is nevertheless dominated by the smallest
    increment (the Gaussian density is positive at the origin, so
    :math:`\mathbb E|Z|^q=\infty` for :math:`q\le-1`) and the estimator returns
    :math:`\Delta\alpha = 1.197`.

    Why it matters: 1.197 is wider than most published cascades. What has to
    hold is that the shuffle surrogate --- which preserves the marginal exactly
    and destroys nothing, because there is nothing to destroy --- reproduces it.
    Observed: null mean 1.117 against an observed 1.197, p = 0.425.
    """
    level = np.cumsum(make_rng(20260727).standard_normal(2048))
    naive = analyze(level, estimator="structure_function", kind="level", q=Q)
    width = naive.spectrum.descriptors().width
    assert width > 0.5, f"expected the known negative-q artefact, got width {width:.4f}"

    test = multifractality_test(
        level, LEVEL_SF, kind="shuffle", n_surrogates=39, seed=1, basis="increments"
    )
    assert test.p_value > 0.05, f"the iid null must not be rejected, p = {test.p_value:.3f}"
    assert test.null_mean == pytest.approx(width, rel=0.2), (
        f"the shuffle null must reproduce the width: observed {width:.4f}, "
        f"null mean {test.null_mean:.4f}"
    )


@pytest.mark.parametrize("alpha", [1.2, 1.6, 1.8])
def test_levy_stable_width_survives_shuffling_and_is_therefore_not_a_cascade(
    alpha: float,
) -> None:
    r"""Attack: iid :math:`\alpha`-stable increments, integrated. Heavy tails,
    zero temporal structure, monofractal by construction (:math:`H=1/\alpha`).

    Because :math:`\mathbb E|X|^q=\infty` for :math:`q\ge\alpha` the sample
    structure function saturates and MFDFA reports
    :math:`\Delta\alpha\in[0.32,1.05]` --- the textbook false positive. The
    decisive check is that the same width appears in the shuffle ensemble, whose
    members are exchangeable with the data because the data *are* iid.

    Observed over two independent paths per :math:`\alpha` (n = 4096, 49
    surrogates): null-mean/observed ratios 0.87, 1.03, 1.05, 1.02, 1.20, 0.89 and
    only one p-value below 0.05 out of six. This is the evidence for the README's
    central claim, so it is asserted as a ratio and not as a p-value alone.
    """
    ratios: list[float] = []
    rejections = 0
    for rng in spawn_rngs(int(alpha * 1000), 2):
        level = np.cumsum(levy_stable_increments(4096, alpha, seed=rng))
        width = analyze(level, estimator=LEVEL_MFDFA, kind="level").spectrum.descriptors().width
        assert width > 0.25, f"expected the heavy-tail artefact, got width {width:.4f}"
        test = multifractality_test(
            level, LEVEL_MFDFA, kind="shuffle", n_surrogates=49, seed=2, basis="increments"
        )
        ratios.append(test.null_mean / width)
        rejections += int(test.p_value <= 0.05)

    assert rejections <= 1, f"the iid heavy-tailed null was rejected {rejections}/2 times"
    for ratio in ratios:
        assert 0.75 < ratio < 1.35, (
            f"shuffling an iid series must not change the width; null/observed = {ratio:.3f}"
        )


def test_the_default_estimator_also_needs_the_shuffle_to_survive_levy_tails() -> None:
    r"""Attack: the same iid :math:`\alpha`-stable null through the estimator
    :func:`~mmar.analysis.analyze` actually uses when the caller names none.

    The previous test pins MFDFA and the structure functions. Wavelet leaders is
    the *default*, so it is the one a first-time user's number comes from, and
    the library's central claim has to hold there too. It manufactures the widest
    spectrum of the three: at :math:`\alpha=1.2` the naive report is
    :math:`\Delta\alpha=0.909` with :math:`c_2=-0.191` --- an intermittency
    coefficient six times that of fully developed turbulence --- for a series of
    independent draws.

    Observed (n = 4096, 49 shuffle surrogates, two paths per :math:`\alpha`):
    null-mean/observed ratios 1.03, 0.99 at :math:`\alpha=1.2`; 1.06, 1.04 at
    1.6; 1.02, 0.64 at 1.8, with five of the six p-values in [0.38, 0.76].
    """
    for alpha, width_floor in ((1.2, 0.6), (1.6, 0.4)):
        level = np.cumsum(levy_stable_increments(4096, alpha, seed=int(alpha * 100)))
        report = analyze(level, kind="level", q=Q)
        width = report.spectrum.descriptors().width
        assert report.exponents.estimator.startswith("wavelet"), (
            f"this test is about the default estimator; got {report.exponents.estimator!r}"
        )
        assert width > width_floor, (
            f"expected the heavy-tail artefact at alpha={alpha}, got width {width:.4f}"
        )
        assert report.cumulants is not None
        assert report.cumulants.c2 < -0.05, (
            f"expected a strongly negative c2 artefact, got {report.cumulants.c2:.4f}"
        )

        test = multifractality_test(
            level, LEVEL_WL, kind="shuffle", n_surrogates=49, seed=3, basis="increments"
        )
        assert test.p_value > 0.05, (
            f"the iid null must survive at alpha={alpha}, p = {test.p_value:.3f}"
        )
        assert 0.75 < test.null_mean / width < 1.35, (
            f"shuffling iid draws must not move the width: null/observed = "
            f"{test.null_mean / width:.3f}"
        )


def test_the_gaussian_width_is_carried_entirely_by_moments_that_do_not_exist() -> None:
    r"""Attack: exact Brownian motion (:math:`H=1/2`, the flattest possible null)
    through the structure functions, then ask the library's own guard rail why.

    :math:`\Delta\alpha=0.930` on a process whose true spectrum is the single
    point :math:`\alpha=1/2`. That number is not noise and it is not a small-
    sample floor: it is manufactured by :math:`q\le-1`, where
    :math:`\mathbb E|Z|^{q}=\infty` because the Gaussian density does not vanish
    at the origin. :func:`~mmar.inference.tests.moment_existence` says so
    independently --- it estimates the tail index of :math:`1/|\Delta X|` and
    reports ``q_crit_negative = 0.996``, ruling out every :math:`q\le-1`.

    This is the whole evidence chain for "the honest verdict is reachable from
    the library's own outputs", so it is asserted end to end. Observed at
    n = 8192:

    ==========================  ==========
    :math:`q` window            width
    ==========================  ==========
    ``linspace(-5, 5, 21)``     0.930
    the admissible subset       0.099
    :math:`q\ge0` only          0.011
    ==========================  ==========

    with :math:`h(2)=0.4828` unchanged to the last digit in all three: the
    exponent that *is* supported is fine, and 99% of the reported width comes
    from the moments that are not.
    """
    bm = fbm(8192, 0.5, seed=99)
    naive = analyze(bm, estimator="structure_function", kind="level", q=Q)
    wide = naive.spectrum.descriptors().width
    assert wide > 0.5, f"expected the negative-q artefact on exact BM, got {wide:.4f}"

    diagnostic = moment_existence(bm, q=Q)
    assert diagnostic.q_crit_negative < 1.05, (
        "the negative-moment bound must sit near 1 for Gaussian increments, got "
        f"{diagnostic.q_crit_negative:.4f}"
    )
    rejected = diagnostic.rejected_q()
    assert np.all(rejected < 0.0), f"only negative q should be ruled out, got {rejected}"
    assert rejected.min() == pytest.approx(-5.0)

    admissible = Q[diagnostic.exists]
    restricted = analyze(bm, estimator="structure_function", kind="level", q=admissible)
    narrow = restricted.spectrum.descriptors().width
    assert narrow < 0.2 * wide, (
        f"restricting to the moments that exist must collapse the width: {wide:.4f} -> {narrow:.4f}"
    )
    assert restricted.exponents.h2 == pytest.approx(naive.exponents.h2, abs=1e-9), (
        "the supported exponent must not move when the unsupported moments are dropped"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "moment_existence over-states q_crit for alpha-stable increments by O(1), not the "
        "O(0.1) its docstring claims, and the error does not shrink with n: at alpha = 1.8 "
        "the plateau rule selects k near the top of the admissible range, where the Hill "
        "plot has drifted to 2.6-2.9. The guard rail against the library's primary false "
        "positive therefore certifies q = 2.0 and q = 2.5 as having finite moments for a "
        "law whose moments diverge above 1.8."
    ),
)
def test_moment_existence_must_not_certify_moments_of_a_stable_law_that_diverge() -> None:
    r"""Attack: :math:`\mathbb E|X|^q<\infty \iff q<\alpha` holds *exactly* for an
    :math:`\alpha`-stable law, so ``q_crit`` has a known truth to be checked
    against --- and this is the diagnostic the module names as "the guard rail"
    for the heavy-tail false positive.

    Observed at :math:`\alpha=1.8` over six independent paths of n = 8192:
    ``q_crit`` in :math:`[2.616, 2.901]`, mean 2.816, against a truth of 1.8; the
    reported 95% interval at one path is :math:`(2.45, 2.99)`, which does not
    come close to covering. It is not a finite-sample effect --- 2.717 at
    n = 4096, 2.815 at n = 16384, 2.769 at n = 65536 --- but the Hill plot's own
    upward drift (1.50 at :math:`k=20`, 1.64 at 200, 2.35 at 2000, 2.78 at 6000)
    combined with a plateau rule that scores flatness by a rolling standard
    deviation, which is smallest exactly where the drift saturates.

    At :math:`\alpha=1.4` the bias is benign (1.49--1.56 against 1.4), so the
    defect is not visible on the heaviest tails; it appears precisely in the
    regime that matters for financial returns.

    The consequence is concrete: the width MFDFA reports on this process is
    0.32--0.52, the shuffle test correctly refuses to reject it, and a user
    reaching for the second line of defence is told the moments up to
    :math:`q=2.9` are sound.
    """
    alpha = 1.8
    level = np.cumsum(levy_stable_increments(8192, alpha, seed=31))
    diagnostic = moment_existence(level, q=Q)

    assert diagnostic.q_crit < alpha + 0.3, (
        f"q_crit = {diagnostic.q_crit:.3f} for a law with tail index {alpha}"
    )
    assert not bool(diagnostic.exists[Q == 2.5][0]), (
        "E|X|^2.5 is infinite for an alpha = 1.8 stable law; the diagnostic must not certify it"
    )


# --------------------------------------------------------------------------- #
# Impostor 2: the monofractal null, and its small-sample width floor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
def test_exact_fbm_is_reported_as_monofractal_by_every_estimator_but_one(hurst: float) -> None:
    r"""Attack: exact fBm, the monofractal null, at three Hurst exponents, through
    four estimator families at once.

    :math:`\zeta(q)=qH` exactly, so the true width is zero and any spread across
    estimators is bias. Measured at n = 8192, ``q = linspace(-5, 5, 21)``:

    ==========================  ==============  ==============  ==============
    estimator                    H = 0.3         H = 0.5         H = 0.7
    ==========================  ==============  ==============  ==============
    wavelet_leaders              0.216 / 0.082   0.453 / 0.072   0.667 / 0.108
    mfdfa                        0.292 / 0.096   0.486 / 0.093   0.683 / 0.087
    mfdma                        0.315 / 0.139   0.520 / 0.100   0.726 / 0.069
    structure_function           0.266 / 0.856   0.478 / 0.849   0.686 / 0.729
    ==========================  ==============  ==============  ==============

    (:math:`h(2)` / :math:`\Delta\alpha`.) Three of the four keep the width below
    0.14 and every one of them recovers :math:`h(2)` to within 0.09 of the truth.
    The fourth reports a width of order 0.8 at *every* :math:`H` --- the
    negative-:math:`q` artefact of
    :func:`test_the_gaussian_width_is_carried_entirely_by_moments_that_do_not_exist`,
    which is a property of the estimator and not of the process. That is exactly
    what :func:`~mmar.analysis.consensus` exists to catch, so its verdict is
    asserted to be a non-agreement.
    """
    path = fbm(8192, hurst, seed=11)
    reports = compare_estimators(
        path,
        kind="level",
        q=Q,
        estimators=("wavelet_leaders", "mfdfa", "mfdma", "structure_function"),
    )
    widths = {name: r.spectrum.descriptors().width for name, r in reports.items()}
    h2 = {name: r.exponents.h2 for name, r in reports.items()}

    for name in ("wavelet_leaders", "mfdfa", "mfdma"):
        assert widths[name] < 0.20, (
            f"{name} manufactures width {widths[name]:.4f} on an exact monofractal"
        )
    assert widths["structure_function"] > 0.5, (
        "if the structure-function negative-q artefact has been fixed, this test's "
        f"premise has changed: width {widths['structure_function']:.4f}"
    )
    for name, value in h2.items():
        assert abs(value - hurst) < 0.10, f"{name} reports h(2) = {value:.4f} for H = {hurst}"

    agreement = consensus(reports)
    assert agreement.width_spread > 0.5, (
        f"the outlier estimator must show up in the spread: {agreement.width_spread:.4f}"
    )
    assert "estimators agree" not in agreement.verdict, (
        "an estimator reporting a width an order of magnitude above the others must not "
        f"come back as unqualified agreement: {agreement.verdict}"
    )


def test_exact_fbm_is_not_called_multifractal_by_the_joint_surrogate_verdict() -> None:
    r"""Attack: the negative control for the whole inference layer. Exact fBm has
    long memory (so the shuffle null is *false*) but is Gaussian and linear (so
    the IAAFT null is *true*), and the pair must therefore never read NONLINEAR.

    This is the test that would catch the library calling its own monofractal
    null a cascade. Observed at :math:`H=0.7`, n = 4096, 49 surrogates each, over
    three independent paths: :math:`(p_{\text{shuffle}}, p_{\text{iaaft}})` =
    (0.060, 0.160), (0.900, 0.960), (0.740, 0.780) --- NO EVIDENCE every time.
    """
    for i, rng in enumerate(spawn_rngs(777, 2)):
        path = fbm(4096, 0.7, seed=rng)
        report = multifractality_report(
            path, LEVEL_MFDFA, n_surrogates=49, seed=100 + i, basis="increments"
        )
        assert not report.iaaft.rejects(), (
            "a Gaussian linear process must not reject the IAAFT null: "
            f"p = {report.iaaft.p_value:.3f}"
        )
        assert not report.verdict.startswith("NONLINEAR"), (
            f"exact fBm was called nonlinear-multifractal: {report.verdict[:80]}"
        )


def test_monofractal_width_floor_inflates_as_the_sample_shrinks() -> None:
    r"""Attack: exact fBm at :math:`H=0.6`, whose true width is exactly zero, at
    four sample sizes. Any measured :math:`\Delta\alpha` is finite-size bias.

    This is the number the library owes its users: a width smaller than the floor
    at the sample size in hand is not evidence of anything. Measured (MFDFA,
    ``q = linspace(-5, 5, 21)``, five independent paths, mean over paths):

    ======  ===============  ==================
    n        mean width      per-path range
    ======  ===============  ==================
    512      0.241           0.107 -- 0.422
    2048     0.139           0.049 -- 0.285
    8192     0.061           0.041 -- 0.088
    32768    0.036           0.006 -- 0.060
    ======  ===============  ==================

    At n = 512 the floor is comparable with the width of a genuine lognormal
    cascade, so a spectrum measured on 512 points carries no information about
    multifractality at all.
    """
    means: dict[int, float] = {}
    for n in (512, 2048, 8192, 32768):
        widths = [
            analyze(fbm(n, 0.6, seed=rng), estimator="mfdfa", kind="level", q=Q)
            .spectrum.descriptors()
            .width
            for rng in spawn_rngs(555, 5)
        ]
        means[n] = float(np.mean(widths))
        assert means[n] > 0.0, f"the monofractal floor at n={n} is not zero"

    assert means[512] > 2.5 * means[32768], (
        f"the small-sample floor must inflate: {means[512]:.4f} at n=512 against "
        f"{means[32768]:.4f} at n=32768"
    )
    assert means[512] > 0.12, f"floor at n=512 measured {means[512]:.4f}"
    assert means[32768] < 0.08, f"floor at n=32768 measured {means[32768]:.4f}"


# --------------------------------------------------------------------------- #
# Impostor 3: deterministic trends, level shifts, periods
# --------------------------------------------------------------------------- #


def test_mfdfa_order_annihilates_polynomial_trends_only_up_to_its_own_degree() -> None:
    r"""Attack: exact fBm plus a deterministic polynomial, through :func:`analyze`.

    MFDFA of order :math:`m` subtracts a degree-:math:`m` least-squares
    polynomial inside every window, so a global trend of degree :math:`\le m` is
    annihilated *algebraically*: the exponent must be bit-identical to the clean
    path, not merely close. Degree :math:`m+1` must not be removed, or the claim
    that ``order`` controls what is detrended is vacuous.

    Observed at n = 8192, trend amplitude :math:`500\sigma`:
    clean :math:`h(2)=0.585518` (DFA1) and :math:`0.581170` (DFA2); with a linear
    trend DFA1 returns 0.585518 (difference :math:`9\times10^{-15}`); with a
    quadratic trend DFA1 returns 0.809232 (a :math:`+0.224` bias, i.e. a
    manufactured near-persistent process) while DFA2 returns 0.581170
    (difference :math:`10^{-14}`).
    """
    n = 8192
    path = fbm(n, 0.6, seed=4242)
    t = np.arange(1, n + 1, dtype=np.float64) / n
    amp = 500.0 * float(np.std(path))

    def h2_of(x: FloatArray, order: int) -> float:
        return analyze(x, estimator="mfdfa", kind="level", q=Q, order=order).exponents.h2

    clean1, clean2 = h2_of(path, 1), h2_of(path, 2)
    linear, quadratic = path + amp * t, path + amp * t**2

    assert h2_of(linear, 1) == pytest.approx(clean1, abs=1e-10), "DFA1 must kill a linear trend"
    assert h2_of(quadratic, 2) == pytest.approx(clean2, abs=1e-10), (
        "DFA2 must kill a quadratic trend"
    )
    biased = h2_of(quadratic, 1)
    assert biased - clean1 > 0.15, (
        f"DFA1 must NOT remove a quadratic trend: h(2) went {clean1:.4f} -> {biased:.4f}"
    )


def test_a_level_shift_is_not_a_polynomial_and_no_detrending_order_removes_it() -> None:
    r"""Attack: exact fBm at :math:`H=0.6` plus one step discontinuity at the
    midpoint --- the third member of the trend family the module names, and the
    one polynomial detrending has no purchase on.

    A step is not in the span of any polynomial basis, so raising ``order``
    cannot annihilate it the way the previous test shows it annihilates
    :math:`t` and :math:`t^2`; it only lets the local fit chase the jump inside
    the windows that straddle it. Measured at n = 8192, jump height
    :math:`20\sigma`:

    ======  ===========  ===========  ========
    order    clean h(2)   shifted      bias
    ======  ===========  ===========  ========
    1        0.5855       0.4216       -0.164
    2        0.5812       0.4529       -0.128
    3        0.5802       0.4986       -0.082
    ======  ===========  ===========  ========

    The bias shrinks with ``order`` but never reaches the :math:`10^{-14}` of a
    genuine annihilation, and at every order the reported exponent is wrong by
    more than the width floor at this sample size. Worth stating because the
    natural reading of "MF-DFA removes trends" is that ``order=2`` is enough.
    """
    n = 8192
    path = fbm(n, 0.6, seed=4242)
    shift = np.zeros(n, dtype=np.float64)
    shift[n // 2 :] = 20.0 * float(np.std(path))

    biases = {}
    for order in (1, 2, 3):
        clean = analyze(path, estimator="mfdfa", kind="level", q=Q, order=order).exponents.h2
        shifted = analyze(
            path + shift, estimator="mfdfa", kind="level", q=Q, order=order
        ).exponents.h2
        biases[order] = shifted - clean

    for order, bias in biases.items():
        assert abs(bias) > 0.05, (
            f"order={order} appears to annihilate a level shift (bias {bias:+.4f}); if that "
            "is now true the docstring's table is stale"
        )
    assert abs(biases[3]) < abs(biases[1]), (
        f"raising the order must at least attenuate the step: {biases[1]:+.4f} -> {biases[3]:+.4f}"
    )


def test_a_strong_sinusoid_wrecks_h2_and_only_the_scaling_range_test_sees_it() -> None:
    r"""Attack: exact fBm at :math:`H=0.6` plus a strong periodic component.

    A sinusoid puts a crossover into the log--log plot: below the period the
    fluctuation grows like the sine's own smooth profile, above it like the fBm.
    The single slope fitted across the crossover is meaningless.

    Observed at n = 8192, amplitude :math:`2\sigma`, 40 cycles:
    :math:`h(2)=1.448` for a process with :math:`H=0.6`, width 0.600 --- and the
    only diagnostic :func:`analyze` runs by itself, the concavity defect, is
    exactly 0.00000. Nothing in the report says the number is wrong.
    :func:`~mmar.inference.tests.scaling_range_test`, which :func:`analyze` does
    not call, does see it: min :math:`R^2` falls to 0.9249 against 0.9964 on the
    clean path, and the common linear window shrinks from the full 24-scale grid
    to 10 scales, :math:`s\in[37,224]` --- the band between the sine period and
    the record length. The ``n_curved`` count is *not* used as the discriminator:
    the RESET test fires on exact Brownian motion too (the module already says
    so), so only the effect size separates the two.
    """
    n = 8192
    path = fbm(n, 0.6, seed=4242)
    t = np.arange(1, n + 1, dtype=np.float64) / n
    contaminated = path + 2.0 * float(np.std(path)) * np.sin(2.0 * np.pi * t * 40.0)

    report = analyze(contaminated, estimator="mfdfa", kind="level", q=Q)
    assert report.exponents.h2 > 1.0, (
        f"expected the crossover to inflate h(2) past 1, got {report.exponents.h2:.4f}"
    )
    assert float(report.tests["concavity_defect"]) < 1e-6, (
        "the concavity defect is blind to a crossover; if it stopped being blind, "
        "this test's premise has changed and the docstring must be rewritten"
    )

    diagnosed = scaling_range_test(mfdfa_structure(contaminated, q=Q, order=1, integrate=False))
    baseline = scaling_range_test(mfdfa_structure(path, q=Q, order=1, integrate=False))
    assert diagnosed["min_r_squared"] < baseline["min_r_squared"] - 0.03, (
        f"the crossover must separate on R^2: contaminated "
        f"{diagnosed['min_r_squared']:.4f} against clean {baseline['min_r_squared']:.4f}"
    )
    common = diagnosed["common_window"]
    assert common is not None
    assert common[2] <= diagnosed["n_scales"] // 2, (
        f"the common linear window must collapse, got {common[2]}/{diagnosed['n_scales']} scales"
    )


# --------------------------------------------------------------------------- #
# Impostor 4: non-stationarity, and the width that collapses to zero
# --------------------------------------------------------------------------- #


def test_a_spectrum_lying_wholly_below_the_level_is_not_reported_as_a_monofractal() -> None:
    r"""Attack: a strictly convex :math:`\zeta`, which no genuine multifractal has.

    Take :math:`\zeta(q)=\tfrac12 q+\tfrac1{20}q^2` on :math:`q\in[-5,5]`. It is
    convex, so :math:`q\alpha-\zeta(q)` is concave in :math:`q` and the infimum
    sits at an endpoint:

    .. math::
        f(\alpha) = \min(5\alpha - 3.75,\; -5\alpha + 1.25) + 1 ,

    a tent with apex :math:`f(0.5)=-0.25`. The whole curve lies below the
    :math:`f=0` level at which :attr:`SpectrumDescriptors` measures the support,
    so there is no crossing on either side, both endpoints collapse onto the apex
    and ``width`` comes back as **exactly 0.0** --- which the class documents as
    the signature of a monofractal.

    Before the fix ``support_truncated`` was ``False`` here, so the descriptors
    affirmed that the zero width was a measurement. The failure is not academic:
    the same collapse is produced by a real, plausible input (see
    :func:`test_two_regime_nonstationarity_collapses_the_width_to_zero`), and
    ``width`` is the default statistic of every surrogate test in the library.
    """
    exponents = ScalingExponents(q=Q, zeta=0.5 * Q + 0.05 * Q**2, estimator="synthetic.convex")
    d = spectrum_from_exponents(exponents).descriptors()

    assert d.f_max == pytest.approx(-0.25, abs=1e-9), f"closed form gives -0.25, got {d.f_max}"
    assert d.width == pytest.approx(0.0, abs=1e-12)
    assert d.support_truncated, (
        "a spectrum whose apex lies below the measurement level has no support there; "
        "reporting width = 0 with support_truncated = False makes a collapsed estimate "
        "indistinguishable from an exact monofractal"
    )


def test_two_regime_nonstationarity_collapses_the_width_to_zero() -> None:
    r"""Attack: fBm at :math:`H=0.3` concatenated with fBm at :math:`H=0.8`.

    Non-stationary, monofractal on each half, multifractal nowhere. MFDFA fits a
    single :math:`h(q)` across both regimes and returns an *increasing*
    :math:`h(q)` (0.40 at :math:`q=-5` rising to 0.81 at :math:`q=+5`) --- the
    wrong sign, since a real multifractal has :math:`h` decreasing. The
    consequences, measured at n = 8192:

    * :math:`c_2 = +0.129`. A positive :math:`c_2` is impossible for any genuine
      multifractal (:math:`c_2\le0` is what concavity of :math:`\zeta` means), so
      this is a self-diagnosing red flag and it is in ``summary()``.
    * concavity defect 1.203, i.e. 44 times the median :math:`\hat\sigma(\zeta)`.
    * ``width`` = **0.0** with ``f_max`` = -0.208: the collapse of the previous
      test, reached from a plausible input rather than a synthetic :math:`\zeta`.

    So the honest verdict *is* reachable --- from :math:`c_2`, from ``f_max``, and
    from the concavity defect --- but never from the headline width, which reads
    as a perfect monofractal.
    """
    report = analyze(_two_regime(8192, 5, 6), estimator="mfdfa", kind="level", q=Q)
    hq = report.exponents.hq
    finite = np.isfinite(hq)
    assert hq[finite][-1] > hq[finite][0], "the two-regime path must produce an increasing h(q)"

    assert report.cumulants is not None
    assert report.cumulants.c2 > 0.0, (
        f"c2 must come out positive (impossible for a multifractal), got {report.cumulants.c2}"
    )
    assert float(report.tests["concavity_defect_in_stderr"]) > 5.0, (
        "the concavity defect must dwarf the reported standard errors"
    )

    summary = report.summary()
    d = report.spectrum.descriptors()
    if d.width == 0.0:
        assert d.f_max < 0.0
        assert d.support_truncated, "a collapsed width must not be reported as a measurement"
        assert summary["support_truncated"] is True, (
            "summary() must carry the flag, or the headline width = 0 is unqualified"
        )


def test_a_collapsed_spectrum_must_not_enter_a_hypothesis_test_as_a_measurement() -> None:
    r"""Attack: hand the collapsed two-regime spectrum to the inference layer and
    watch a non-measurement become a p-value.

    :attr:`~mmar.core.types.SpectrumDescriptors.support_truncated` records that a
    ``width`` of zero from an apex below the level is *a collapse of the estimate,
    not the signature of a monofractal*. That distinction was made in
    :mod:`mmar.core.types` and then dropped one layer up:
    :func:`~mmar.inference.tests.spectrum_statistic` and
    :func:`~mmar.inference.surrogates.spectrum_width` returned ``descriptors().width``
    and nothing else, so the test statistic was ``0.0`` with no flag attached.

    Observed before the fix, on fBm(0.3) then fBm(0.8), n = 8192, 49 shuffle
    surrogates: :math:`f_{\max}=-0.2078`, observed statistic 0.0, null mean
    0.0594, null minimum 0.0155 --- *every* surrogate strictly larger --- hence
    :math:`p=1.0000` and an ``interpretation`` reading "A randomly reordered copy
    of the increments ... reproduces the observed statistic", which not one of the
    49 surrogates did, followed by "the width is a property of the increment
    distribution (heavy tails do exactly this)". The increments are Gaussian.

    After the fix the statistic is NaN and the existing "not finite on the data
    itself; the exponent estimate has failed before any surrogate was drawn"
    guard fires, which is the correct answer: there is nothing to test. Both
    ``c2`` (+0.112) and ``hq_range`` (0.481) stay finite and usable, so the fix
    does not blind the caller --- it only refuses to price a statistic that was
    never measured.
    """
    two = _two_regime(8192, 5, 6)
    exponents = LEVEL_MFDFA(two)
    d = spectrum_from_exponents(exponents).descriptors()
    assert d.f_max < 0.0 and d.width == 0.0, (
        f"premise: the spectrum must collapse here; f_max={d.f_max:.4f} width={d.width:.4f}"
    )

    assert np.isnan(spectrum_statistic(exponents, "width")), (
        "a width measured on a spectrum with no support is not a statistic"
    )
    assert np.isnan(spectrum_width(exponents)), (
        "the surrogate module's default statistic must apply the same rule"
    )
    assert np.isfinite(spectrum_statistic(exponents, "c2"))
    assert np.isfinite(spectrum_statistic(exponents, "hq_range"))

    with pytest.raises(ValueError, match="not finite on the data itself"):
        multifractality_test(
            two, LEVEL_MFDFA, kind="shuffle", n_surrogates=9, seed=11, basis="increments"
        )


def test_a_deterministic_volatility_ramp_manufactures_width_without_a_cascade() -> None:
    r"""Attack: iid Gaussian increments multiplied by a deterministic envelope.

    :math:`x_t=z_t\,\sigma(t)` with :math:`z` iid and :math:`\sigma` a known
    function of time has no dependence structure of any kind --- consecutive
    increments are independent --- and no cascade. It is simply non-stationary.
    The library reports it as increasingly multifractal as the envelope steepens.
    Measured at n = 8192, MFDFA:

    ==========================  ========  =========  ====================
    envelope                     width     c2         concavity defect
    ==========================  ========  =========  ====================
    :math:`1`                    0.088     -0.0089    0.0
    :math:`1+9t`                 0.182     -0.0205    0.0
    :math:`1+0.95\sin 6\pi t`    0.257     -0.0406    0.0
    ==========================  ========  =========  ====================

    A tripling of :math:`|c_2|` and of the width, with the only diagnostic
    :func:`analyze` runs on its own sitting at exactly zero throughout. This is
    the mechanism the surrogate layer exists for, and the reason a width quoted
    without one is not evidence.
    """
    n = 8192
    z = make_rng(4711).standard_normal(n)
    t = np.arange(1, n + 1, dtype=np.float64) / n

    def measure(increments: FloatArray) -> tuple[float, float, float]:
        report = analyze(np.cumsum(increments), estimator="mfdfa", kind="level", q=Q)
        assert report.cumulants is not None
        return (
            report.spectrum.descriptors().width,
            report.cumulants.c2,
            float(report.tests["concavity_defect"]),
        )

    flat_w, flat_c2, flat_defect = measure(z)
    ramp_w, _ramp_c2, ramp_defect = measure(z * (1.0 + 9.0 * t))
    sine_w, sine_c2, sine_defect = measure(z * (1.0 + 0.95 * np.sin(2.0 * np.pi * t * 3.0)))

    assert ramp_w > 1.5 * flat_w, f"the ramp must inflate the width: {flat_w:.4f} -> {ramp_w:.4f}"
    assert sine_w > 2.0 * flat_w, f"the envelope must inflate the width: {sine_w:.4f}"
    assert sine_c2 < 2.0 * flat_c2, (
        f"|c2| must at least double under a deterministic envelope: {flat_c2:.4f} -> {sine_c2:.4f}"
    )
    for defect in (flat_defect, ramp_defect, sine_defect):
        assert defect < 1e-6, (
            f"the premise is that analyze's own diagnostic stays silent; it reported {defect:.3e}"
        )


def test_five_outliers_in_eight_thousand_points_multiply_the_width_sevenfold() -> None:
    r"""Attack: exact fBm at :math:`H=0.6` with five increments enlarged to
    :math:`40\sigma`. Everything else is untouched.

    0.06% of the sample carries the whole result. Observed at n = 8192:
    :math:`\Delta\alpha` goes 0.0716 :math:`\to` 0.5158 and :math:`c_2` goes
    -0.0053 :math:`\to` -0.0380, while :math:`h(2)` barely moves (0.6168
    :math:`\to` 0.6007) --- so the second-order exponent gives no warning at all
    and the *shape* statistics carry it entirely.

    The rescue works and is worth recording as evidence: the shuffle and IAAFT
    nulls both preserve the marginal, hence the five spikes, hence the width;
    :math:`p=0.30` and :math:`0.32` respectively and the verdict is NO EVIDENCE.
    A width tripled by five points is reproduced by any reordering of those
    points, which is precisely what the surrogate is asking.
    """
    n = 8192
    path = fbm(n, 0.6, seed=17)
    increments = np.diff(np.concatenate([[0.0], path]))
    spiked = increments.copy()
    spiked[make_rng(5).choice(n, size=5, replace=False)] += 40.0 * float(np.std(increments))

    def measure(x: FloatArray) -> tuple[float, float, float]:
        report = analyze(np.cumsum(x), estimator="mfdfa", kind="level", q=Q)
        assert report.cumulants is not None
        return report.spectrum.descriptors().width, report.cumulants.c2, report.exponents.h2

    clean_w, clean_c2, clean_h2 = measure(increments)
    dirty_w, dirty_c2, dirty_h2 = measure(spiked)

    assert dirty_w > 5.0 * clean_w, (
        f"five spikes must dominate the width: {clean_w:.4f} -> {dirty_w:.4f}"
    )
    assert dirty_c2 < 3.0 * clean_c2, f"and |c2|: {clean_c2:.4f} -> {dirty_c2:.4f}"
    assert abs(dirty_h2 - clean_h2) < 0.05, (
        f"h(2) must stay quiet, or it would be the warning: {clean_h2:.4f} -> {dirty_h2:.4f}"
    )

    report = multifractality_report(
        np.cumsum(spiked), LEVEL_MFDFA, n_surrogates=39, seed=9, basis="increments"
    )
    assert report.verdict.startswith("NO EVIDENCE"), (
        f"a width carried by five exchangeable points must not survive: {report.verdict[:80]}"
    )


# --------------------------------------------------------------------------- #
# Impostor 5: volatility clustering without a cascade
# --------------------------------------------------------------------------- #


def test_garch_manufactures_a_width_that_analyze_alone_cannot_question() -> None:
    r"""Attack: GARCH(1,1) with :math:`(\alpha,\beta)=(0.10,0.88)`.

    Volatility clustering, genuinely nonlinear dependence, and no multiplicative
    cascade: the process is strictly stationary and its true spectrum is a point.

    Observed at n = 8192: width 0.398 against 0.095 for exact fBm of the same
    length and the same seed --- a fourfold inflation --- with :math:`c_2=-0.019`
    and a concavity defect of exactly 0.00000. Nothing :func:`analyze` computes
    on its own distinguishes this from a cascade.

    The surrogate pair does fire (p = 0.020 against both the shuffle and the
    IAAFT null on this path), and its verdict string reads "NONLINEAR ... the
    only combination that is evidence of genuine multifractal structure". That
    reading is too strong for GARCH: the rejection is correct --- nonlinear
    structure *is* required --- but nonlinear structure is not a cascade. The
    p-values are seed-dependent (2 of 4 paths at n = 4096 returned NO EVIDENCE),
    so only the deterministic half is asserted here.
    """
    n = 8192
    garch_level = np.cumsum(_garch11(n, 90210))
    garch_report = analyze(garch_level, estimator="mfdfa", kind="level", q=Q)
    fbm_width = (
        analyze(fbm(n, 0.5, seed=90210), estimator="mfdfa", kind="level", q=Q)
        .spectrum.descriptors()
        .width
    )
    garch_width = garch_report.spectrum.descriptors().width

    assert garch_width > 2.0 * fbm_width, (
        f"GARCH must manufacture a width above the monofractal floor: "
        f"{garch_width:.4f} against {fbm_width:.4f}"
    )
    assert float(garch_report.tests["concavity_defect"]) < 1e-6, (
        "analyze's own diagnostic is silent on GARCH; if that changes, so does the finding"
    )
