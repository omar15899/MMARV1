"""The bootstrap layer is what makes every other number in the library quotable.

The tests are therefore calibration tests, not smoke tests: they measure the
*observed* coverage of the interval against a process whose exponent is known in
closed form, and they report the number. Where the observed coverage falls short
of nominal, that is recorded as a property of the method under long-range
dependence, not asserted away.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmar.core.provenance import InputRef, new_provenance
from mmar.core.regression import loglog_fit
from mmar.core.types import FloatArray, ScalingExponents
from mmar.inference.bootstrap import (
    DESCRIPTOR_NAMES,
    BootstrapResult,
    EstimatorContractError,
    bootstrap_descriptors,
    bootstrap_exponents,
    bootstrap_spectrum,
    circular_block_bootstrap,
    moving_block_bootstrap,
    optimal_block_length,
    politis_white_block_length,
    residual_bootstrap_fit,
)
from mmar.models.fbm import fbm, fgn, levy_stable_increments

# --------------------------------------------------------------------------- #
# A minimal structure-function estimator.
#
# Deliberately local: this module must work against any Callable, and the
# estimator packages are being written independently. Scales stay well below the
# block lengths used, which is the regime in which the block bootstrap is
# supposed to work.
# --------------------------------------------------------------------------- #

Q_POS = np.array([1.0, 2.0, 3.0])
Q_WIDE = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
SCALES = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64], dtype=np.int64)


def _structure_exponents(x: FloatArray, q: FloatArray = Q_POS) -> ScalingExponents:
    r""":math:`\hat\zeta(q)` from :math:`S(q,s)=\langle|X_{t+s}-X_t|^q\rangle`."""
    n = x.size
    moments = np.empty((q.size, SCALES.size), dtype=np.float64)
    for j, s in enumerate(SCALES):
        d = np.maximum(np.abs(x[int(s) :] - x[: n - int(s)]), 1e-300)
        moments[:, j] = np.mean(d[None, :] ** q[:, None], axis=1)
    zeta = np.array(
        [loglog_fit(SCALES.astype(np.float64), moments[i]).slope for i in range(q.size)],
        dtype=np.float64,
    )
    return ScalingExponents(
        q=q,
        zeta=zeta,
        estimator="test.structure_function",
        provenance=new_provenance(
            "test.structure_function",
            params={"scales": SCALES.tolist()},
            inputs=[InputRef.from_array("x", x)],
        ),
    )


def _wide_estimator(x: FloatArray) -> ScalingExponents:
    return _structure_exponents(x, Q_WIDE)


# --------------------------------------------------------------------------- #
# Resampling schemes
# --------------------------------------------------------------------------- #


def test_resamples_are_reproducible_from_the_seed_alone() -> None:
    x = np.arange(200, dtype=np.float64)
    a = list(moving_block_bootstrap(x, block=17, n_resamples=5, seed=42))
    b = list(moving_block_bootstrap(x, block=17, n_resamples=5, seed=42))
    c = list(moving_block_bootstrap(x, block=17, n_resamples=5, seed=43))
    for u, v in zip(a, b, strict=True):
        assert np.array_equal(u, v)
    assert not all(np.array_equal(u, v) for u, v in zip(a, c, strict=True))


def test_resample_k_does_not_depend_on_how_many_preceded_it() -> None:
    """spawn_rngs, not a single advancing stream: resample k is a function of (seed, k)."""
    x = np.arange(200, dtype=np.float64)
    long_run = list(moving_block_bootstrap(x, block=17, n_resamples=6, seed=1))
    short_run = list(moving_block_bootstrap(x, block=17, n_resamples=3, seed=1))
    for u, v in zip(long_run[:3], short_run, strict=True):
        assert np.array_equal(u, v)


def test_streams_are_spawned_children_not_incremented_integer_seeds() -> None:
    r"""``spawn_rngs``, not ``make_rng(seed + k)``.

    Reproducibility alone does not distinguish the two: ``seed + k`` is just as
    reproducible and just as independent of run length. It is however the
    documented mistake — consecutive integer seeds are *not* guaranteed to give
    non-overlapping PCG64 streams, which is the whole reason
    :func:`~mmar.core.rng.spawn_rngs` exists. The observable signature of the
    correct implementation is that resample :math:`k` under ``seed`` is *not*
    resample :math:`0` under ``seed + k``.
    """
    x = np.arange(400, dtype=np.float64)
    spawned = list(moving_block_bootstrap(x, block=13, n_resamples=4, seed=100))
    for k in range(1, 4):
        (shifted,) = moving_block_bootstrap(x, block=13, n_resamples=1, seed=100 + k)
        assert not np.array_equal(spawned[k], shifted), (
            f"resample {k} under seed 100 equals resample 0 under seed {100 + k}: "
            "the resamples are integer-incremented, not SeedSequence.spawn children"
        )


def test_moving_blocks_are_contiguous_runs_of_the_original_series() -> None:
    x = np.arange(500, dtype=np.float64)
    block = 25
    (y,) = moving_block_bootstrap(x, block=block, n_resamples=1, seed=7)
    assert y.size == x.size
    assert np.isin(y, x).all()
    # Every position that is not a block join must continue the original series.
    within = np.ones(y.size - 1, dtype=bool)
    within[np.arange(block - 1, y.size - 1, block)] = False
    assert np.all(np.diff(y)[within] == 1.0)


def test_moving_block_underweights_the_ends_and_circular_does_not() -> None:
    r"""Every observation belongs to exactly :math:`b` circular blocks; not so for MBB.

    The first observation lies in a single moving block, so its expected count in
    a resample is :math:`1/b` of an interior observation's. Politis & Romano
    (1992) wrap precisely to remove this.
    """
    n, block, reps = 120, 20, 400
    x = np.arange(n, dtype=np.float64)
    mbb = np.zeros(n)
    cbb = np.zeros(n)
    for y in moving_block_bootstrap(x, block=block, n_resamples=reps, seed=3):
        mbb += np.bincount(y.astype(int), minlength=n)
    for y in circular_block_bootstrap(x, block=block, n_resamples=reps, seed=3):
        cbb += np.bincount(y.astype(int), minlength=n)
    interior = slice(block, n - block)
    assert mbb[0] < 0.25 * mbb[interior].mean(), (mbb[0], mbb[interior].mean())
    ratio = cbb[0] / cbb[interior].mean()
    assert 0.8 < ratio < 1.25, f"circular inclusion ratio at the edge {ratio:.3f}"


def test_block_length_one_destroys_long_memory_variance() -> None:
    r"""The claim in the module docstring, measured.

    For fGn at :math:`H=0.8` the variance of the sample mean is
    :math:`O(n^{2H-2})`, not :math:`O(n^{-1})`. An iid bootstrap (``block=1``)
    reproduces the iid rate and so understates the true sampling standard
    deviation by a large factor; a long block recovers most of it.
    """
    n, reps = 2048, 200
    truth = np.std([fgn(n, 0.8, seed=4000 + r).mean() for r in range(120)], ddof=1)
    x = fgn(n, 0.8, seed=17)
    sd_iid = np.std(
        [y.mean() for y in moving_block_bootstrap(x, block=1, n_resamples=reps, seed=1)]
    )
    sd_blk = np.std(
        [y.mean() for y in circular_block_bootstrap(x, block=256, n_resamples=reps, seed=1)]
    )
    print(f"\nsd(mean) fGn H=0.8 n={n}: truth {truth:.5f} iid {sd_iid:.5f} block-256 {sd_blk:.5f}")
    assert sd_iid < 0.4 * truth, "the iid bootstrap should be badly too narrow here"
    assert sd_blk > 2.0 * sd_iid


def test_rejects_impossible_blocks() -> None:
    x = np.arange(50, dtype=np.float64)
    with pytest.raises(ValueError, match="exceeds the series length"):
        next(moving_block_bootstrap(x, block=51, n_resamples=1, seed=0))
    with pytest.raises(ValueError, match="single admissible block"):
        next(moving_block_bootstrap(x, block=50, n_resamples=1, seed=0))
    with pytest.raises(ValueError, match="at least 1"):
        next(circular_block_bootstrap(x, block=0, n_resamples=1, seed=0))
    with pytest.raises(ValueError, match="n_resamples must be positive"):
        next(moving_block_bootstrap(x, block=5, n_resamples=0, seed=0))


# --------------------------------------------------------------------------- #
# Politis-White block length
# --------------------------------------------------------------------------- #


def test_politis_white_block_length_grows_with_the_memory() -> None:
    """More persistence must buy a longer block; the rule is monotone in H."""
    lengths = {H: optimal_block_length(fgn(4096, H, seed=3)) for H in (0.5, 0.6, 0.7, 0.8, 0.9)}
    print("\nPolitis-White block length by H (n=4096):", lengths)
    ordered = [lengths[H] for H in (0.5, 0.6, 0.7, 0.8, 0.9)]
    assert ordered == sorted(ordered), lengths
    assert lengths[0.5] <= 8, "white noise needs no block"
    assert lengths[0.9] > 10 * max(lengths[0.5], 1)


def test_politis_white_diagnostics_flag_the_long_memory_regime() -> None:
    iid = politis_white_block_length(fgn(4096, 0.5, seed=11))
    memory = politis_white_block_length(fgn(4096, 0.9, seed=11))
    print(f"\niid: {iid}\nH=0.9: {memory}")
    assert iid.m_hat < memory.m_hat
    assert memory.g_hat > iid.g_hat, "G measures correlation mass at long lags"
    assert iid.spectrum_at_zero > 0.0
    assert iid.stationary >= 1 and iid.moving >= 1


def test_optimal_block_length_rejects_an_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="unknown block method"):
        optimal_block_length(fgn(256, 0.5, seed=0), method="stationary")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# bootstrap_exponents
# --------------------------------------------------------------------------- #


def _boot(x: FloatArray, **kw: object) -> BootstrapResult:
    params: dict[str, object] = {
        "n_resamples": 120,
        "block": 128,
        "seed": 11,
        "resample": "increments",
        "jackknife_blocks": 12,
    }
    params.update(kw)
    return bootstrap_exponents(x, _structure_exponents, **params)  # type: ignore[arg-type]


def test_bootstrap_exponents_is_reproducible_and_provenanced() -> None:
    x = fbm(2048, 0.7, seed=5)
    a, b = _boot(x), _boot(x)
    assert np.array_equal(a.replicates, b.replicates, equal_nan=True)
    assert a.provenance is not None and b.provenance is not None
    assert a.provenance.root == b.provenance.root
    assert _boot(x, seed=12).provenance.root != a.provenance.root  # type: ignore[union-attr]
    assert a.provenance.params["block"] == 128
    assert a.provenance.params["resample"] == "increments"
    assert a.provenance.seed is not None and a.provenance.seed.entropy == 11
    assert any(i.name == "point_estimate" for i in a.provenance.inputs)


def test_bootstrap_exponents_reports_a_usable_band() -> None:
    x = fbm(4096, 0.7, seed=5)
    r = _boot(x)
    print(f"\nzeta hat {np.round(r.point, 4)}  se {np.round(r.stderr, 4)}")
    print(f"band {np.round(r.lower, 4)} .. {np.round(r.upper, 4)}  bias {np.round(r.bias, 4)}")
    assert r.replicates.shape == (120, Q_POS.size)
    assert np.all(r.lower < r.upper)
    assert np.all(r.n_valid > 100)
    assert set(r.per_point_method) == {"bca"}
    assert np.all(np.isfinite(r.acceleration)) and np.all(np.isfinite(r.z0))
    point, lo, hi = r.at(2.0)
    assert lo < point < hi
    assert r.covers(2.0, point)
    assert not r.covers(2.0, hi + 1.0)


def test_percentile_and_bca_differ_and_both_are_recorded() -> None:
    x = fbm(2048, 0.7, seed=8)
    bca = _boot(x, interval="bca")
    pct = _boot(x, interval="percentile")
    assert set(pct.per_point_method) == {"percentile"}
    assert np.all(np.isnan(pct.acceleration))
    # BCa shifts the endpoints whenever the replicate distribution is skewed or
    # median-biased; identical endpoints would mean the correction did nothing.
    assert not np.allclose(bca.lower, pct.lower)
    print(f"\nBCa {bca.at(2.0)}  percentile {pct.at(2.0)}")


def test_bca_falls_back_to_percentile_and_says_so() -> None:
    """A degenerate statistic has no computable acceleration; that must be visible."""

    def constant(x: FloatArray) -> ScalingExponents:
        return ScalingExponents(
            q=Q_POS,
            zeta=np.array([0.7, 1.4, 2.1]),
            estimator="constant",
            provenance=new_provenance("constant", inputs=[InputRef.from_array("x", x)]),
        )

    r = bootstrap_exponents(
        fbm(1024, 0.7, seed=2), constant, n_resamples=30, block=64, seed=3, jackknife_blocks=8
    )
    assert set(r.per_point_method) == {"percentile"}
    assert any("percentile fallback" in n for n in r.notes)
    assert np.allclose(r.lower, r.point) and np.allclose(r.upper, r.point)


def test_percentile_endpoints_are_the_requested_quantiles_at_the_requested_level() -> None:
    r"""The nominal level must actually bind.

    A coverage study cannot see this: an interval built at 90% and *labelled* 95%
    still passes every "is the band sane" assertion. The endpoints are therefore
    pinned exactly against :func:`numpy.nanquantile` at
    :math:`((1-\text{ci})/2,\ 1-(1-\text{ci})/2)`, and the width is required to
    grow with the level at the Gaussian rate.
    """
    x = fbm(2048, 0.7, seed=21)
    widths = {}
    for ci in (0.50, 0.90, 0.95, 0.99):
        r = _boot(x, n_resamples=60, interval="percentile", ci=ci)
        assert r.ci_level == ci
        p = (1.0 - ci) / 2.0
        assert np.allclose(r.lower, np.nanquantile(r.replicates, p, axis=0), rtol=0, atol=1e-12)
        assert np.allclose(
            r.upper, np.nanquantile(r.replicates, 1.0 - p, axis=0), rtol=0, atol=1e-12
        )
        widths[ci] = float(np.mean(r.width))
    print(f"\nmean band width by level: { {k: round(v, 4) for k, v in widths.items()} }")
    assert widths[0.50] < widths[0.90] < widths[0.95] < widths[0.99]
    # z_{0.975}/z_{0.75} = 2.908; allow the empirical-quantile slop of B = 60.
    assert 2.0 < widths[0.95] / widths[0.50] < 4.2


def test_bca_endpoints_are_efrons_formula_evaluated_at_z0_and_a() -> None:
    r"""BCa is not "percentile with a nudge": pin the construction end to end.

    Recomputes :math:`\hat z_0=\Phi^{-1}(\#\{\hat\theta^*<\hat\theta\}/B)` from the
    replicates, feeds the reported :math:`\hat a` through Efron (1987) eq. 14.10,
    and requires the endpoints to be exactly the replicate quantiles at the
    resulting adjusted probabilities. A wrong nominal level, a dropped bias
    correction or a mis-signed tail all break this identity.
    """
    from scipy.stats import norm

    x = fbm(2048, 0.7, seed=13)
    ci = 0.95
    r = _boot(x, n_resamples=60, interval="bca", ci=ci, seed=2)
    assert set(r.per_point_method) == {"bca"}

    frac = np.count_nonzero(r.replicates < r.point[None, :], axis=0) / r.n_resamples
    assert np.allclose(r.z0, norm.ppf(frac), rtol=0, atol=1e-12)

    for i in range(r.q.size):
        for z, endpoint in (
            (norm.ppf((1 - ci) / 2), r.lower),
            (norm.ppf(1 - (1 - ci) / 2), r.upper),
        ):
            adj = r.z0[i] + (r.z0[i] + z) / (1.0 - r.acceleration[i] * (r.z0[i] + z))
            ref = float(np.nanquantile(r.replicates[:, i], float(norm.cdf(adj))))
            assert endpoint[i] == pytest.approx(ref, rel=0, abs=1e-12), (i, endpoint[i], ref)


def test_bca_acceleration_is_the_delete_one_block_jackknife() -> None:
    r"""The acceleration must be the Künsch (1989) delete-one-*block* jackknife.

    Every "is the band sane" assertion in this file survives replacing
    :math:`\hat a` by zero (BCa collapses to a bias-corrected percentile) or by
    the delete-one-*observation* jackknife (invalid under dependence — it holds
    the correlation structure fixed and understates the influence values). Both
    are therefore excluded explicitly: the reported :math:`\hat a` is rebuilt
    here from scratch and required to match bit for bit.
    """
    x = fbm(2048, 0.7, seed=13)
    block, n_deletions = 128, 12
    r = _boot(x, n_resamples=30, block=block, seed=2, interval="bca", jackknife_blocks=n_deletions)
    assert set(r.per_point_method) == {"bca"}

    units = np.diff(x)
    m = units.size
    n_blocks = m // block
    starts = (np.arange(n_blocks) * block)[
        np.linspace(0, n_blocks - 1, n_deletions).round().astype(np.intp)
    ]
    first = float(x[0])
    theta = []
    for s in starts:
        keep = np.ones(m, dtype=bool)
        keep[s : s + block] = False
        u = units[keep]
        theta.append(_structure_exponents(np.concatenate(([first], first + np.cumsum(u)))).zeta)
    dev = np.mean(theta, axis=0) - np.asarray(theta)
    a_block = (dev**3).sum(axis=0) / (6.0 * (dev**2).sum(axis=0) ** 1.5)

    theta_obs = []
    for s in np.linspace(0, m - 1, n_deletions).round().astype(np.intp):
        keep = np.ones(m, dtype=bool)
        keep[s] = False
        u = units[keep]
        theta_obs.append(_structure_exponents(np.concatenate(([first], first + np.cumsum(u)))).zeta)
    dev_obs = np.mean(theta_obs, axis=0) - np.asarray(theta_obs)
    a_obs = (dev_obs**3).sum(axis=0) / (6.0 * (dev_obs**2).sum(axis=0) ** 1.5)

    print(f"\na (delete-one-block) {np.round(r.acceleration, 5)}")
    print(f"a (delete-one-observation, rejected) {np.round(a_obs, 5)}")
    assert np.allclose(r.acceleration, a_block, rtol=0, atol=1e-12)
    assert np.all(np.abs(r.acceleration) > 1e-2), "a identically ~0 is not an acceleration"
    assert not np.allclose(r.acceleration, a_obs, rtol=0, atol=1e-3)


def test_estimator_failures_are_counted_not_hidden() -> None:
    calls = {"n": 0}

    def flaky(x: FloatArray) -> ScalingExponents:
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise ValueError("the moment does not exist on this resample")
        return _structure_exponents(x)

    r = bootstrap_exponents(
        fbm(2048, 0.7, seed=4),
        flaky,
        n_resamples=60,
        block=128,
        seed=5,
        interval="percentile",
        resample="increments",
    )
    print(f"\nfailed resamples: {r.n_failed}/60, notes {r.notes}")
    assert 0 < r.n_failed < 60
    assert np.all(r.n_valid == 60 - r.n_failed)
    assert any("failed the estimator" in n for n in r.notes)
    assert np.all(np.isfinite(r.lower))


def test_a_data_dependent_q_grid_is_rejected() -> None:
    def drifting(x: FloatArray) -> ScalingExponents:
        return _structure_exponents(x, Q_POS + float(x[0]) * 1e-3)

    with pytest.raises(EstimatorContractError, match="different q grid"):
        bootstrap_exponents(fbm(1024, 0.7, seed=1), drifting, n_resamples=4, block=64, seed=1)


def test_resampling_a_path_directly_is_the_documented_disaster() -> None:
    """Block-resampling a self-affine path injects jumps and destroys the exponent."""
    x = fbm(4096, 0.7, seed=6)
    good = _boot(x, resample="increments", interval="percentile")
    bad = _boot(x, resample="series", interval="percentile")
    print(f"\nzeta(2) increments {good.at(2.0)}  series {bad.at(2.0)}  truth 1.4")
    assert good.covers(2.0, 1.4)
    assert not bad.covers(2.0, 1.4), "resampling the path should visibly break the estimate"
    assert bad.at(2.0)[0] > bad.at(2.0)[2] or np.mean(bad.replicates[:, 1]) < 1.2


def test_argument_validation() -> None:
    x = fbm(1024, 0.7, seed=1)
    with pytest.raises(ValueError, match="ci must lie"):
        _boot(x, n_resamples=5, ci=1.5)
    with pytest.raises(ValueError, match="unknown block method"):
        _boot(x, n_resamples=5, method="stationary")
    with pytest.raises(ValueError, match="unknown resample target"):
        _boot(x, n_resamples=5, resample="levels")
    with pytest.raises(ValueError, match="unknown descriptor"):
        bootstrap_descriptors(
            x, _wide_estimator, n_resamples=5, block=64, seed=1, names=("kurtosis",)
        )


# --------------------------------------------------------------------------- #
# Calibration: does the interval do what it claims?
# --------------------------------------------------------------------------- #


def _coverage_study(
    *, n: int, block: int, n_rep: int, n_resamples: int, interval: str, H: float = 0.7
) -> tuple[float, float, float]:
    """Returns ``(coverage of 2H, mean interval width, sd of the point estimate)``."""
    hits = 0
    widths: list[float] = []
    points: list[float] = []
    for r in range(n_rep):
        x = fbm(n, H, seed=20_000 + r)
        res = bootstrap_exponents(
            x,
            _structure_exponents,
            n_resamples=n_resamples,
            block=block,
            seed=90_000 + r,
            resample="increments",
            interval=interval,  # type: ignore[arg-type]
            jackknife_blocks=12,
        )
        point, lo, hi = res.at(2.0)
        points.append(point)
        widths.append(hi - lo)
        hits += res.covers(2.0, 2.0 * H)
    return hits / n_rep, float(np.mean(widths)), float(np.std(points, ddof=1))


@pytest.mark.stochastic
def test_observed_coverage_of_zeta2_on_exact_fbm() -> None:
    r"""Coverage study at :math:`H=0.7`, nominal 95%.

    The interval is built by the block bootstrap and asked whether it contains
    the closed-form truth :math:`\zeta(2)=2H=1.4`. The observed coverage is
    *reported*; the assertion is only that the interval is not grossly
    miscalibrated, because a shortfall here is a documented property of the
    block bootstrap on a long-memory series, not a defect of the code.
    """
    n_rep = 40
    cov, width, sd = _coverage_study(
        n=4096, block=128, n_rep=n_rep, n_resamples=120, interval="bca"
    )
    cov_p, width_p, _ = _coverage_study(
        n=4096, block=128, n_rep=n_rep, n_resamples=120, interval="percentile"
    )
    nominal = 2.0 * 1.959964 * sd
    print(
        f"\ncoverage study, fBm H=0.7 n=4096, {n_rep} replicas, nominal 95%:"
        f"\n  BCa        coverage {cov:.2f}  mean width {width:.4f}"
        f"\n  percentile coverage {cov_p:.2f}  mean width {width_p:.4f}"
        f"\n  sd of zeta_hat(2) across replicas {sd:.4f} -> exact 95% width {nominal:.4f}"
    )
    assert 0.55 <= cov <= 1.0, f"observed BCa coverage {cov:.2f} is far from nominal"
    assert 0.55 <= cov_p <= 1.0, f"observed percentile coverage {cov_p:.2f}"
    # The interval must be the right order of magnitude, not merely non-empty.
    assert 0.4 * nominal < width < 2.5 * nominal


@pytest.mark.stochastic
def test_interval_width_shrinks_like_one_over_sqrt_n() -> None:
    r""":math:`\hat\zeta` is :math:`\sqrt n`-consistent, so the band must halve
    from :math:`n=2^{12}` to :math:`n=2^{14}`."""
    _, w12, _ = _coverage_study(n=4096, block=128, n_rep=12, n_resamples=100, interval="percentile")
    _, w14, _ = _coverage_study(
        n=16384, block=128, n_rep=12, n_resamples=100, interval="percentile"
    )
    ratio = w12 / w14
    print(f"\nwidth at n=2^12 {w12:.4f}, at n=2^14 {w14:.4f}, ratio {ratio:.3f} (expected 2.000)")
    assert 1.5 < ratio < 2.6, f"observed width ratio {ratio:.3f}"


# --------------------------------------------------------------------------- #
# Spectrum and descriptors
# --------------------------------------------------------------------------- #


def _levy_path(n: int = 4096, seed: int = 11) -> FloatArray:
    r"""An :math:`\alpha`-stable Lévy path: monofractal, but with an estimated
    :math:`\hat\zeta` that is a genuinely broken concave line, so its estimated
    spectrum is wide enough for a band to exist."""
    return np.asarray(np.cumsum(levy_stable_increments(n, 1.5, seed=seed)), dtype=np.float64)


def test_spectrum_band_is_nan_where_too_few_resamples_reach() -> None:
    r"""The central rule: no interval without support.

    Each resample has its own :math:`\alpha` support. Grid points reached by few
    resamples must come back NaN, and the coverage profile must fall away at the
    edges — a band that stayed narrow out to the tails would be the classic
    artefact.
    """
    x = _levy_path()
    s = bootstrap_spectrum(
        x, _wide_estimator, n_resamples=80, block=128, seed=3, resample="increments", n_alpha=41
    )
    resolved = int(s.resolved.sum())
    print(
        f"\nspectrum band: {resolved}/{s.alpha.size} grid points resolved; "
        f"coverage from {s.coverage.min():.2f} to {s.coverage.max():.2f}"
    )
    assert 5 < resolved < s.alpha.size, "some but not all grid points must be resolvable"
    assert np.all(np.isnan(s.lower[~s.resolved]))
    assert np.all(s.lower[s.resolved] <= s.upper[s.resolved])
    assert s.coverage.max() > 0.9 and s.coverage.min() < s.min_coverage
    assert np.all(np.isnan(s.lower[s.coverage < s.min_coverage]))
    assert any("coverage floor" in n for n in s.notes)
    assert s.provenance is not None and s.provenance.params["min_coverage"] == 0.8


def test_spectrum_of_a_monofractal_resolves_almost_nothing() -> None:
    """fBm has a degenerate spectrum; resampling it must not manufacture a curve."""
    s = bootstrap_spectrum(
        fbm(2048, 0.7, seed=9),
        _structure_exponents,
        n_resamples=40,
        block=128,
        seed=3,
        resample="increments",
        n_alpha=21,
    )
    print(
        f"\nfBm spectrum: {int(s.resolved.sum())}/21 resolved, max coverage {s.coverage.max():.2f}"
    )
    assert int(s.resolved.sum()) <= 3, "a point spectrum cannot support a band"


def test_spectrum_grid_is_honoured_and_replicates_kept() -> None:
    grid = np.linspace(0.35, 0.75, 17)
    s = bootstrap_spectrum(
        _levy_path(2048, seed=4),
        _wide_estimator,
        n_resamples=40,
        block=128,
        seed=1,
        resample="increments",
        alpha_grid=grid,
        min_coverage=0.5,
    )
    assert np.array_equal(s.alpha, grid)
    assert s.replicates.shape == (40, grid.size)
    assert np.all(s.coverage <= 1.0) and np.all(s.coverage >= 0.0)
    with pytest.raises(ValueError, match="min_coverage"):
        bootstrap_spectrum(
            _levy_path(1024, seed=4),
            _wide_estimator,
            n_resamples=4,
            block=64,
            seed=1,
            min_coverage=1.5,
        )


def test_descriptor_intervals_are_the_quotable_numbers() -> None:
    x = _levy_path()
    d = bootstrap_descriptors(
        x,
        _wide_estimator,
        n_resamples=80,
        block=128,
        seed=3,
        resample="increments",
        jackknife_blocks=12,
    )
    table = d.table()
    print("\ndescriptor intervals (Levy alpha=1.5 path):")
    for name, (p, lo, hi) in table.items():
        print(f"  {name:<10} {p:.4f}  [{lo:.4f}, {hi:.4f}]")
    assert set(table) == {"alpha_0", "width", "asymmetry"}
    for name in ("alpha_0", "width"):
        p, lo, hi = table[name]
        assert lo <= p <= hi, name
        assert math.isfinite(lo) and math.isfinite(hi)
    assert table["width"][1] > 0.0, "a wide estimated spectrum must exclude zero width"
    assert np.all(np.isfinite(d.stderr[:2]))
    with pytest.raises(KeyError, match="no descriptor"):
        d.interval_for("curvature")


def test_all_advertised_descriptors_can_be_requested() -> None:
    d = bootstrap_descriptors(
        _levy_path(1024, seed=2),
        _wide_estimator,
        n_resamples=20,
        block=64,
        seed=1,
        resample="increments",
        interval="percentile",
        names=DESCRIPTOR_NAMES,
    )
    assert d.names == DESCRIPTOR_NAMES
    assert d.point.size == len(DESCRIPTOR_NAMES)


# --------------------------------------------------------------------------- #
# Residual bootstrap
# --------------------------------------------------------------------------- #


def test_residual_bootstrap_agrees_with_the_ols_interval_on_a_clean_power_law() -> None:
    """Same model, same information: the two intervals must be close."""
    rng = np.random.default_rng(0)
    scales = np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
    y = np.exp(0.7 * np.log(scales) + 0.1 + 0.05 * rng.standard_normal(scales.size))
    fit = loglog_fit(scales, y)
    rb = residual_bootstrap_fit(fit, n_resamples=2000, seed=4)
    ols_width = fit.slope_ci[1] - fit.slope_ci[0]
    print(
        f"\nOLS  slope {fit.slope:.4f} CI [{fit.slope_ci[0]:.4f}, {fit.slope_ci[1]:.4f}]"
        f"\nboot slope {rb.slope:.4f} CI [{rb.lower:.4f}, {rb.upper:.4f}] "
        f"width ratio {rb.width / ols_width:.3f}"
    )
    assert rb.slope == fit.slope
    assert rb.lower < fit.slope < rb.upper
    # The bootstrap uses a normal-free reference distribution and no t correction,
    # so it is systematically a little narrower on seven points.
    assert 0.5 < rb.width / ols_width < 1.2
    assert any("optimistic" in n for n in rb.notes)


def test_residual_bootstrap_is_seeded_and_validated() -> None:
    scales = np.array([4.0, 8.0, 16.0, 32.0, 64.0])
    fit = loglog_fit(scales, scales**0.6)
    a = residual_bootstrap_fit(fit, n_resamples=100, seed=1)
    b = residual_bootstrap_fit(fit, n_resamples=100, seed=1)
    c = residual_bootstrap_fit(fit, n_resamples=100, seed=2)
    assert np.array_equal(a.replicates, b.replicates)
    assert not np.array_equal(a.replicates, c.replicates)
    assert a.provenance is not None and b.provenance is not None
    assert a.provenance.root == b.provenance.root
    # An exact power law has zero residuals, so every replicate is the same slope.
    assert a.stderr == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="ci must lie"):
        residual_bootstrap_fit(fit, n_resamples=10, seed=1, ci=0.0)
    with pytest.raises(ValueError, match="n_resamples must be at least 2"):
        residual_bootstrap_fit(fit, n_resamples=1, seed=1)
