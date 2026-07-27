r"""Theory tests for the exact self-similar simulators in :mod:`mmar.models.fbm`.

These generators are the reference against which every estimator's bias is
measured, so they are held to closed-form truth rather than to each other:
the fGn autocovariance, the self-similarity relation
:math:`\operatorname{Var}B_H(t)=\sigma^2t^{2H}`, the collapse to white noise at
:math:`H=1/2`, and the Gaussian degeneracy of the stable law at
:math:`\alpha=2`.

Tolerance policy. Wherever a Monte Carlo mean is compared with a theoretical
value, the tolerance is *derived from the data*: the replicas produced by
:func:`~mmar.core.rng.spawn_rngs` are independent, so the across-replica
standard error is an honest estimate of the sampling noise, and the assertion is
made at four of them. This is preferable to a hand-tuned absolute tolerance,
which either hides a real bias (too loose) or fails on a different numpy
version (too tight). Every test is seeded, so "four sigma" is a statement about
one fixed realisation, not a flaky bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmar.core.provenance import array_digest
from mmar.core.rng import spawn_rngs
from mmar.models.fbm import (
    brownian_bridge,
    fbm,
    fbm_with_provenance,
    fgn,
    fgn_acf,
    fgn_hosking,
    levy_stable_increments,
    mbm,
)

pytestmark = [pytest.mark.theory, pytest.mark.stochastic]

HURSTS = (0.3, 0.5, 0.7, 0.9)
N_SIGMA = 4.0


def _sample_acov(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased lag-``k`` autocovariance of a known-zero-mean series, ``k=0..max_lag``.

    The sample mean is deliberately *not* subtracted: the processes here have
    mean zero by construction, and subtracting an estimated mean introduces the
    :math:`O(n^{2H-2})` bias that is precisely what long memory makes large.
    """
    n = x.size
    return np.array([float(np.mean(x[: n - k] * x[k:])) for k in range(max_lag + 1)])


# --------------------------------------------------------------------------- #
# Closed-form autocovariance
# --------------------------------------------------------------------------- #


def test_fgn_acf_closed_form_anchors() -> None:
    r"""Anchors of :math:`\gamma(k)=\tfrac{\sigma^2}{2}(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H})`."""
    for H in HURSTS:
        g = fgn_acf(8, H)
        assert g[0] == pytest.approx(1.0)

    # H = 1/2: exact white noise, gamma(k) = 0 for every k >= 1.
    assert fgn_acf(16, 0.5)[1:] == pytest.approx(np.zeros(15), abs=1e-12)

    # Sign of the correlations is decided by H - 1/2 (persistent vs antipersistent).
    assert np.all(fgn_acf(16, 0.8)[1:] > 0.0)
    assert np.all(fgn_acf(16, 0.2)[1:] < 0.0)

    # Asymptotics gamma(k) ~ sigma^2 H(2H-1) k^{2H-2}.
    H, k = 0.8, 4096
    g = fgn_acf(k + 1, H)[k]
    assert g == pytest.approx(H * (2 * H - 1) * k ** (2 * H - 2), rel=1e-3)

    # sigma enters quadratically.
    assert fgn_acf(8, 0.7, sigma=3.0) == pytest.approx(9.0 * fgn_acf(8, 0.7))


@pytest.mark.parametrize("H", HURSTS)
def test_davies_harte_matches_theoretical_acf(H: float) -> None:
    r"""Sample autocovariance of Davies--Harte fGn equals :func:`fgn_acf` at lags 0..10.

    2000 independent replicas of length 512. Each replica gives an unbiased
    estimate of :math:`\gamma(k)`; the across-replica standard error is exact
    (the streams are independent), and the deviation must sit inside four of
    them. For :math:`H=0.9` the per-replica estimator is itself long-range
    dependent, so its variance is inflated by :math:`O(L^{4H-3})` --- the
    empirical standard error absorbs that automatically, which is the reason for
    calibrating the tolerance from the replicas rather than from :math:`1/\sqrt{RL}`.
    """
    n_rep, length, max_lag = 2000, 512, 10
    paths = np.array([fgn(length, H, seed=g) for g in spawn_rngs(20240719, n_rep)])
    est = np.array([_sample_acov(x, max_lag) for x in paths])

    theory = fgn_acf(max_lag + 1, H)
    mean = est.mean(axis=0)
    stderr = est.std(axis=0, ddof=1) / np.sqrt(n_rep)
    z = np.abs(mean - theory) / stderr

    assert np.max(z) < N_SIGMA, (
        f"H={H}: worst lag {int(np.argmax(z))} deviates by {np.max(z):.2f} sigma; "
        f"mean={mean[np.argmax(z)]:.5f} theory={theory[np.argmax(z)]:.5f}"
    )
    # An absolute floor as well, so that a pathologically large stderr cannot
    # let a genuinely wrong covariance through.
    assert np.max(np.abs(mean - theory)) < 0.05


def test_h_half_is_iid_gaussian() -> None:
    """``H=0.5`` degenerates to white noise: sample acf at lags >= 1 is zero."""
    n_rep, length, max_lag = 1500, 1024, 10
    paths = np.array([fgn(length, 0.5, seed=g) for g in spawn_rngs(555, n_rep)])
    est = np.array([_sample_acov(x, max_lag) for x in paths])

    mean = est.mean(axis=0)
    stderr = est.std(axis=0, ddof=1) / np.sqrt(n_rep)
    assert mean[0] == pytest.approx(1.0, abs=N_SIGMA * stderr[0])
    assert np.all(np.abs(mean[1:]) < N_SIGMA * stderr[1:])
    # 1/sqrt(R*L) is the exact white-noise scale; the estimate must be at that order.
    assert np.max(np.abs(mean[1:])) < 5.0 / np.sqrt(n_rep * length)


@pytest.mark.parametrize("H", [0.3, 0.7])
def test_davies_harte_and_hosking_agree_in_distribution(H: float) -> None:
    """Two exact methods, two independent seeds: same second-order structure.

    A two-sample comparison, not a path comparison --- the algorithms consume
    randomness in completely different orders, so equality of realisations is
    neither expected nor meaningful. What must agree is the covariance, and both
    are separately checked against the closed form so that a shared error cannot
    cancel.
    """
    n_rep, length, max_lag = 600, 400, 8
    dh = np.array([_sample_acov(fgn(length, H, seed=g), max_lag) for g in spawn_rngs(31, n_rep)])
    hk = np.array(
        [_sample_acov(fgn_hosking(length, H, seed=g), max_lag) for g in spawn_rngs(97, n_rep)]
    )

    diff = dh.mean(axis=0) - hk.mean(axis=0)
    se = np.sqrt(dh.var(axis=0, ddof=1) / n_rep + hk.var(axis=0, ddof=1) / n_rep)
    assert np.max(np.abs(diff) / se) < N_SIGMA

    theory = fgn_acf(max_lag + 1, H)
    for name, est in (("davies-harte", dh), ("hosking", hk)):
        z = np.abs(est.mean(axis=0) - theory) / (est.std(axis=0, ddof=1) / np.sqrt(n_rep))
        assert np.max(z) < N_SIGMA, f"{name} deviates from theory by {np.max(z):.2f} sigma"


def test_hosking_selects_via_method_keyword() -> None:
    """``method='hosking'`` routes to the recursive generator, bit for bit."""
    assert np.array_equal(
        fgn(256, 0.65, seed=4, method="hosking"),
        fgn_hosking(256, 0.65, seed=4),
    )
    with pytest.raises(ValueError, match="unknown method"):
        fgn(64, 0.5, seed=0, method="circulant")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Self-similarity of the cumulated process
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("H", HURSTS)
def test_fbm_variance_scales_as_t_to_the_2h(H: float) -> None:
    r"""Recover :math:`H` from :math:`\operatorname{Var}B_H(t)\propto t^{2H}` to 0.02.

    The variance curve is estimated across 2000 independent paths and regressed
    on :math:`\log t`. This tests the *grid scaling* :math:`\Delta t^{H}` as much
    as the covariance: a wrong power would leave the slope intact but destroy the
    amplitude, so the terminal variance is checked separately against
    :math:`\sigma^2 t_{\max}^{2H}`.
    """
    n_rep, length = 2000, 512
    paths = np.array([fbm(length, H, seed=g) for g in spawn_rngs(20240720, n_rep)])

    t = np.arange(1, length + 1, dtype=float) / length
    var = paths.var(axis=0)
    slope = float(np.polyfit(np.log(t[8:]), np.log(var[8:]), 1)[0])
    assert slope / 2.0 == pytest.approx(H, abs=0.02)

    # Amplitude: Var B_H(t_max) = sigma^2 * t_max^{2H} = 1. The relative standard
    # error of a variance from R Gaussian samples is sqrt(2/R).
    rel_se = np.sqrt(2.0 / n_rep)
    assert float(var[-1]) == pytest.approx(1.0, rel=N_SIGMA * rel_se)


def test_fbm_scaling_is_invariant_to_grid_resolution_and_horizon() -> None:
    r""":math:`\operatorname{Var}B_H(t_{\max})=\sigma^2t_{\max}^{2H}` for any ``n``, ``t_max``."""
    H, n_rep = 0.7, 1500
    rel_se = np.sqrt(2.0 / n_rep)
    for length, t_max, sigma in ((256, 1.0, 1.0), (1024, 4.0, 1.0), (512, 0.25, 2.5)):
        term = np.array(
            [fbm(length, H, seed=g, sigma=sigma, t_max=t_max)[-1] for g in spawn_rngs(808, n_rep)]
        )
        target = sigma**2 * t_max ** (2 * H)
        assert float(term.var()) == pytest.approx(target, rel=N_SIGMA * rel_se)


def test_fbm_increments_are_the_generating_fgn() -> None:
    """The path is exactly ``cumsum(fgn) * dt**H``: no hidden detrending or shift."""
    H, length, t_max = 0.62, 300, 3.0
    path = fbm(length, H, seed=17, t_max=t_max)
    increments = fgn(length, H, seed=17)
    dt = t_max / length
    assert np.allclose(path, np.cumsum(increments) * dt**H, rtol=0, atol=1e-12)


def test_brownian_bridge_is_pinned() -> None:
    """The bridge vanishes at the right endpoint and keeps the fBm regularity."""
    for H in (0.5, 0.8):
        b = brownian_bridge(512, seed=6, H=H)
        assert b[-1] == pytest.approx(0.0, abs=1e-12)

    # Brownian bridge covariance: Var[b(t)] = t(1-t) for H = 1/2.
    n_rep, length = 2000, 256
    paths = np.array([brownian_bridge(length, seed=g) for g in spawn_rngs(64, n_rep)])
    t = np.arange(1, length + 1, dtype=float) / length
    var = paths.var(axis=0)
    interior = slice(8, length - 8)
    assert np.allclose(var[interior], (t * (1.0 - t))[interior], atol=6.0 * np.sqrt(2.0 / n_rep))


# --------------------------------------------------------------------------- #
# Reproducibility and stream independence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["davies-harte", "hosking"])
def test_same_seed_is_bitwise_identical(method: str) -> None:
    """Reproducibility is bitwise, not statistical: identical seed, identical bytes."""
    a = fgn(512, 0.7, seed=12345, method=method)  # type: ignore[arg-type]
    b = fgn(512, 0.7, seed=12345, method=method)  # type: ignore[arg-type]
    assert np.array_equal(a, b)
    assert array_digest(a) == array_digest(b)

    c = fgn(512, 0.7, seed=12346, method=method)  # type: ignore[arg-type]
    assert not np.array_equal(a, c)

    # A sequence seed is a different seed from the scalar with the same first entry.
    d = fgn(512, 0.7, seed=[12345, 1], method=method)  # type: ignore[arg-type]
    assert not np.array_equal(a, d)


def test_fbm_and_mbm_reproducibility() -> None:
    assert np.array_equal(fbm(400, 0.4, seed=2), fbm(400, 0.4, seed=2))
    assert not np.array_equal(fbm(400, 0.4, seed=2), fbm(400, 0.4, seed=3))

    def h(u: np.ndarray) -> np.ndarray:
        return 0.25 + 0.5 * u

    assert np.array_equal(mbm(400, h, seed=2), mbm(400, h, seed=2))
    assert not np.array_equal(mbm(400, h, seed=2), mbm(400, h, seed=3))


def test_spawned_streams_are_independent() -> None:
    """Paths from :func:`spawn_rngs` show no cross-correlation beyond sampling noise.

    Sequential integer seeds produce correlated PCG64 streams; ``spawn`` does
    not. For 50 white-noise paths of length 512 the sample correlation of a pair
    has standard deviation :math:`1/\\sqrt{512}=0.044`, and the maximum over
    1225 pairs of standard normals sits near 3.4 of those, so 4.5 is the
    threshold.
    """
    n_rep, length = 50, 512
    paths = np.array([fgn(length, 0.5, seed=g) for g in spawn_rngs(2024, n_rep)])
    z = (paths - paths.mean(axis=1, keepdims=True)) / paths.std(axis=1, keepdims=True)
    corr = (z @ z.T) / length
    off = corr[np.triu_indices(n_rep, k=1)]

    scale = 1.0 / np.sqrt(length)
    assert np.max(np.abs(off)) < 4.5 * scale
    assert abs(float(off.mean())) < 4.0 * scale / np.sqrt(off.size)


def test_provenance_root_is_the_recipe_not_the_realisation() -> None:
    """Same recipe, same root; different parameter, different root; path digested."""
    path_a, prov_a = fbm_with_provenance(256, 0.6, seed=99, sigma=2.0, t_max=1.5)
    path_b, prov_b = fbm_with_provenance(256, 0.6, seed=99, sigma=2.0, t_max=1.5)
    assert np.array_equal(path_a, path_b)
    assert prov_a.root == prov_b.root
    assert prov_a.diagnostics["output_digest"] == array_digest(path_a)
    assert prov_a.seed is not None and prov_a.seed.entropy == 99
    assert prov_a.params["H"] == 0.6
    assert prov_a.estimator == "models.fbm"

    _, prov_c = fbm_with_provenance(256, 0.61, seed=99, sigma=2.0, t_max=1.5)
    _, prov_d = fbm_with_provenance(256, 0.6, seed=100, sigma=2.0, t_max=1.5)
    assert len({prov_a.root, prov_c.root, prov_d.root}) == 3


# --------------------------------------------------------------------------- #
# Multifractional Brownian motion
# --------------------------------------------------------------------------- #


def test_mbm_with_constant_h_reduces_to_fbm() -> None:
    """Degenerate ``h`` must recover fBm bit for bit, not merely in distribution."""

    def constant(u: np.ndarray) -> np.ndarray:
        return np.full_like(u, 0.6)

    assert np.array_equal(mbm(2048, constant, seed=9), fbm(2048, 0.6, seed=9))


def test_mbm_marginal_variance_tracks_t_to_the_2h_of_t() -> None:
    r""":math:`\operatorname{Var}X(t)\approx t^{2h(t)}` under the krigeage construction.

    The interpolation in :math:`H` makes this only approximate (see the module
    docstring), so the tolerance is 15% relative: with 1500 replicas the pure
    Monte Carlo relative standard error is already 3.7% per point, and the
    maximum over 512 correlated points sits near three of them.
    """

    def h(u: np.ndarray) -> np.ndarray:
        return 0.3 + 0.5 * u

    n_rep, length = 1500, 512
    paths = np.array([mbm(length, h, seed=g, n_h=41) for g in spawn_rngs(3, n_rep)])
    t = np.arange(1, length + 1, dtype=float) / length
    ratio = paths.var(axis=0) / t ** (2.0 * h(t))
    assert np.max(np.abs(ratio - 1.0)) < 0.15


def test_mbm_local_regularity_follows_h() -> None:
    r"""Local exponent of :math:`|X(t+\delta)-X(t)|` recovers :math:`h(t)` to 0.05.

    This is the property the construction exists for: mBm is the ground truth
    for *pointwise* Hölder estimators. The residual bias is dominated by the
    variation of :math:`h` across the 512-point estimation window
    (:math:`0.5\times512/4096\approx0.06`), not by the simulator.
    """

    def h(u: np.ndarray) -> np.ndarray:
        return 0.3 + 0.5 * u

    length, n_rep, half = 4096, 40, 256
    scales = np.array([2, 4, 8, 16, 32])
    centres = np.array([0.15, 0.35, 0.55, 0.75])

    estimates = np.empty((n_rep, centres.size))
    for r, g in enumerate(spawn_rngs(21, n_rep)):
        x = mbm(length, h, seed=g, n_h=41)
        for c_idx, c in enumerate(centres):
            i0 = int(c * length) - half
            i1 = int(c * length) + half
            sf = [float(np.mean(np.abs(x[i0 + s : i1 + s] - x[i0:i1]))) for s in scales]
            estimates[r, c_idx] = np.polyfit(np.log(scales), np.log(sf), 1)[0]

    assert np.allclose(estimates.mean(axis=0), h(centres), atol=0.05)


def test_mbm_rejects_bad_h_func() -> None:
    with pytest.raises(ValueError, match="must be vectorised"):
        mbm(128, lambda u: np.array(0.5), seed=1)
    with pytest.raises(ValueError, match=r"open interval \(0, 1\)"):
        mbm(128, lambda u: 1.0 + 0.0 * u, seed=1)
    with pytest.raises(ValueError, match="n_h must be at least 2"):
        mbm(128, lambda u: 0.5 + 0.0 * u, seed=1, n_h=1)


# --------------------------------------------------------------------------- #
# alpha-stable: the spurious-multifractality null
# --------------------------------------------------------------------------- #


def _hill(x: np.ndarray, k: int) -> float:
    r"""Hill (1975) estimator of the tail index from the top ``k`` order statistics.

    :math:`\hat\alpha = \big[k^{-1}\sum_{i=1}^{k}\log(X_{(i)}/X_{(k+1)})\big]^{-1}`
    on :math:`|X|`.
    """
    y = np.sort(np.abs(x))[::-1]
    return float(1.0 / np.mean(np.log(y[:k] / y[k])))


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.2, 1.5, 1.8])
def test_levy_stable_tail_index(alpha: float) -> None:
    r"""Hill estimator recovers :math:`\alpha`, with a tolerance that grows with :math:`\alpha`.

    The Hill estimator is consistent but its bias is governed by the
    second-order regular variation of the stable tail, whose rate degrades as
    :math:`\alpha\uparrow2` (the tail constant
    :math:`C_\alpha=\frac{1-\alpha}{\Gamma(2-\alpha)\cos(\pi\alpha/2)}` vanishes
    there and the Gaussian bulk contaminates the top order statistics). The
    tolerance ``0.06 + 0.10*alpha`` reflects that honestly: it is roughly twice
    the observed bias across the whole range, and is a check that the CMS
    implementation produces the right tail *exponent*, not a claim about the
    estimator's efficiency.
    """
    x = levy_stable_increments(200_000, alpha, seed=1234)
    assert _hill(x, 2000) == pytest.approx(alpha, abs=0.06 + 0.10 * alpha)


def test_levy_stable_alpha_two_is_gaussian() -> None:
    r""":math:`\alpha=2` degenerates to :math:`\mathcal N(0,2c^2)`, not :math:`\mathcal N(0,c^2)`.

    The stable scale is not the standard deviation; the Gaussian member of the
    family has variance :math:`2c^2`. Checked here so that no downstream
    calibration silently absorbs a factor of :math:`\sqrt2`.
    """
    n = 400_000
    x = levy_stable_increments(n, 2.0, seed=3)
    assert float(x.std()) == pytest.approx(np.sqrt(2.0), rel=6.0 / np.sqrt(2.0 * n))
    assert float(x.mean()) == pytest.approx(0.0, abs=5.0 * np.sqrt(2.0 / n))
    # Gaussian kurtosis, i.e. no surviving heavy tail.
    assert float(np.mean((x / x.std()) ** 4)) == pytest.approx(3.0, rel=0.05)


def test_levy_stable_scale_and_skew() -> None:
    """``scale`` acts multiplicatively; ``beta > 0`` skews the law to the right."""
    a = levy_stable_increments(1000, 1.5, seed=8)
    b = levy_stable_increments(1000, 1.5, seed=8, scale=4.0)
    assert np.allclose(b, 4.0 * a, rtol=1e-12, atol=0.0)

    right = levy_stable_increments(200_000, 1.5, seed=4, beta=0.9)
    left = levy_stable_increments(200_000, 1.5, seed=4, beta=-0.9)
    assert float(np.quantile(right, 0.99)) > -float(np.quantile(right, 0.01))
    assert float(np.median(right)) < 0.0 < float(np.median(left))


def test_levy_stable_alpha_one_branch_is_finite() -> None:
    r"""The :math:`\alpha=1` (Cauchy) branch uses its own formula and stays finite."""
    x = levy_stable_increments(100_000, 1.0, seed=7)
    assert np.all(np.isfinite(x))
    # Cauchy: the interquartile range of S_1(1, 0, 0) is exactly 2.
    assert float(np.quantile(x, 0.75) - np.quantile(x, 0.25)) == pytest.approx(2.0, rel=0.05)


def _empirical_cf(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """:math:`\\hat\\varphi(t)=n^{-1}\\sum_j e^{itx_j}`, error :math:`O(n^{-1/2})`."""
    return np.asarray(np.mean(np.exp(1j * np.outer(t, x)), axis=1))


def _stable_cf(t: np.ndarray, alpha: float, beta: float, c: float) -> np.ndarray:
    r"""Characteristic function of :math:`S(\alpha,\beta,c,0;1)`, the 1-parametrisation.

    .. math::
        \log\varphi(t) = -c^\alpha|t|^\alpha
            \big(1 - i\beta\operatorname{sgn}(t)\tan\tfrac{\pi\alpha}{2}\big),
        \quad \alpha\neq1,
    .. math::
        \log\varphi(t) = -c|t|
            \big(1 + i\beta\tfrac{2}{\pi}\operatorname{sgn}(t)\log(c|t|)\big),
        \quad \alpha=1.
    """
    if alpha == 1.0:
        inner = 1.0 + 1j * beta * (2.0 / np.pi) * np.sign(t) * np.log(c * np.abs(t))
        return np.asarray(np.exp(-c * np.abs(t) * inner))
    inner = 1.0 - 1j * beta * np.sign(t) * np.tan(0.5 * np.pi * alpha)
    return np.asarray(np.exp(-(c**alpha) * np.abs(t) ** alpha * inner))


@pytest.mark.parametrize(
    ("alpha", "beta", "scale"),
    [
        (1.0, 0.0, 1.0),
        (1.0, 0.6, 1.0),
        (1.0, -0.9, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.5, 3.0),
        (1.5, 0.9, 1.0),
        (0.7, 0.5, 1.0),
        (1.3, 0.4, 2.5),
        (1.9, -0.6, 1.0),
    ],
)
def test_levy_stable_characteristic_function(alpha: float, beta: float, scale: float) -> None:
    r"""The empirical CF must match :math:`S(\alpha,\beta,c,0;1)` on a grid of :math:`t`.

    This is the only test that pins the *whole* law rather than a tail index or a
    quantile, and it is the one that catches a location error. The
    Chambers--Mallows--Stuck :math:`\alpha=1` branch carries a :math:`\pi/2`
    inside its logarithm; dropping it translates every draw by
    :math:`\tfrac{2}{\pi}\beta\log\tfrac\pi2\approx0.2875\beta`, a shift that is
    exactly zero at :math:`\beta=0` and therefore invisible to a symmetric test.
    The tolerance is 0.01, six times the :math:`n^{-1/2}=1.6\times10^{-3}`
    sampling error of the empirical CF, and the omitted-:math:`\pi/2` bug shows
    up at 0.03--0.10.
    """
    t = np.array([-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0])
    x = levy_stable_increments(400_000, alpha, seed=31337, beta=beta, scale=scale)
    deviation = float(np.max(np.abs(_empirical_cf(x, t) - _stable_cf(t, alpha, beta, scale))))
    assert deviation < 0.01, f"empirical CF deviates by {deviation:.4f}"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_H", [0.0, 1.0, -0.2, 1.5, float("nan")])
def test_hurst_outside_the_open_unit_interval_is_rejected(bad_H: float) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        fgn(128, bad_H, seed=0)


def test_other_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fgn(0, 0.5, seed=0)
    with pytest.raises(ValueError, match="sigma"):
        fgn(128, 0.5, seed=0, sigma=0.0)
    with pytest.raises(ValueError, match="t_max"):
        fbm(128, 0.5, seed=0, t_max=-1.0)
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 2\]"):
        levy_stable_increments(16, 2.5, seed=0)
    with pytest.raises(ValueError, match=r"beta must lie in \[-1, 1\]"):
        levy_stable_increments(16, 1.5, seed=0, beta=2.0)


def test_short_and_odd_lengths_are_handled() -> None:
    """The circulant embedding must be sized correctly for non-power-of-two ``n``."""
    for length in (1, 2, 3, 5, 17, 1000, 1023, 1025):
        x = fgn(length, 0.7, seed=1)
        assert x.shape == (length,)
        assert np.all(np.isfinite(x))
