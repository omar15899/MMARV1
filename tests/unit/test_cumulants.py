r"""Tests for :mod:`mmar.inference.cumulants`.

Observed numbers, recorded here because they are the calibration of the whole
library and not merely test fixtures (all reproduced by the assertions below):

*Lognormal cascade*, :math:`b=2`, :math:`\lambda=1.05`, conservative
(:math:`\sigma^2=0.144270`, so :math:`c_1=1.05`, :math:`c_2=-0.1` exactly),
one realisation at ``kmax=16`` (65 536 cells), seed 20260727, dyadic scales
:math:`2^1..2^{10}`:

======================  ==========  ===========  ===========
route                   :math:`c_1` :math:`c_2`  :math:`c_3`
======================  ==========  ===========  ===========
direct, overlapping     1.047074    -0.092384    -0.002896
direct, disjoint        1.043484    -0.085757    -0.002631
from :math:`\hat\zeta`  1.047099    -0.092934    -0.003013
from the moment surface 1.046357    -0.093056    -0.001771
**truth**               **1.05**    **-0.10**    **0**
======================  ==========  ===========  ===========

So a single cascade realisation recovers :math:`c_1` to 0.3 % and :math:`c_2` to
7 % low, with the three routes mutually consistent to ~0.001 in :math:`c_2`.
The 7 % is not estimator error: :math:`\tau` from :mod:`mmar.models.theory` is
the *annealed* exponent, and a single sample of a canonical cascade sits below
it at finite ``kmax``.

*Exact fBm*, :math:`H=0.7`, :math:`n=2^{16}`, overlapping increments, dyadic
scales :math:`2^2..2^{11}`, 8 independent seeds spawned from 20260727:

===================  =========================  ==========================
:math:`H`            :math:`\hat c_1`           :math:`\hat c_2`
===================  =========================  ==========================
0.5                  0.50374 (sd 0.00684)       -0.00012 (sd 0.00648)
0.7                  0.70716 (sd 0.01581)       -0.00339 (sd 0.00793)
===================  =========================  ==========================

with ``max |c2| = 0.01107`` (H=0.5) and ``0.01525`` (H=0.7) over the eight
replicas. **That 0.015 is the practical detection floor of the library** at
:math:`n=2^{16}`: it sits a factor of two below the turbulence value
:math:`-c_2\approx0.03` and a factor of two to five below the equity range
0.03--0.08, so those signals are detectable at this sample size — but only just,
and any reported :math:`-c_2 \lesssim 0.02` from a series of this length is
indistinguishable from monofractal noise.

Disjoint increments are markedly worse (:math:`\hat c_2 = -0.0252` on the same
path where overlapping gives :math:`-0.0038`), which is why overlapping is the
default: the variance reduction outweighs the correlation the weights already
account for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from mmar.core.legendre import legendre_fenchel
from mmar.core.provenance import new_provenance
from mmar.core.regression import loglog_fit
from mmar.core.rng import spawn_rngs
from mmar.core.types import (
    FitMethod,
    FloatArray,
    LogCumulants,
    ScalingExponents,
    StructureFunctions,
)
from mmar.inference.cumulants import (
    MAX_ORDER,
    _increments,
    _sample_cumulants,
    compare_cumulants,
    cumulant_spectrum,
    cumulants_from_exponents,
    cumulants_from_increments,
    cumulants_from_moments,
    cumulants_from_series,
    log_cumulants_by_scale,
    multifractality_index,
)
from mmar.models import cascade, theory
from mmar.models.fbm import fbm  # mmar.models re-exports the *function* under this name

LAM = 1.05
B = 2
KMAX = 16
CASCADE_SEED = 20260727
CASCADE_SCALES = np.array([2**j for j in range(1, 11)], dtype=np.int64)

FBM_H = 0.7
FBM_N = 1 << 16
FBM_SCALES = np.array([2**j for j in range(2, 12)], dtype=np.int64)
FBM_SEED = 99991


# --------------------------------------------------------------------------- #
# Local structure-function estimator
#
# Deliberately built from mmar.core.regression alone: this file must not depend
# on any module under mmar/estimators/, so the "estimator" is supplied as a
# callable, which is also how cumulants_from_series takes one.
# --------------------------------------------------------------------------- #


def _moment_surface(x: FloatArray, q: FloatArray, scales: NDArray[np.int64]) -> StructureFunctions:
    values = np.empty((q.size, scales.size), dtype=np.float64)
    counts = np.empty(scales.size, dtype=np.int64)
    for j, s in enumerate(scales.tolist()):
        d = np.abs(x[s:] - x[:-s])
        counts[j] = max((x.size - 1) // s, 1)
        values[:, j] = np.mean(d[None, :] ** q[:, None], axis=1)
    return StructureFunctions(q=q, scales=scales.astype(np.float64), values=values, counts=counts)


def _exponents_from_surface(surface: StructureFunctions) -> ScalingExponents:
    zeta = np.empty(surface.q.size, dtype=np.float64)
    stderr = np.empty(surface.q.size, dtype=np.float64)
    weights = np.asarray(surface.counts, dtype=np.float64)
    for i in range(surface.q.size):
        fit = loglog_fit(surface.scales, surface.values[i], method=FitMethod.WLS, weights=weights)
        zeta[i] = fit.slope
        stderr[i] = fit.slope_stderr
    return ScalingExponents(
        q=surface.q,
        zeta=zeta,
        zeta_stderr=stderr,
        structure=surface,
        estimator="test.structure_function",
    )


def _estimator(
    q: FloatArray, scales: NDArray[np.int64]
) -> Any:  # returns Callable[[FloatArray], ScalingExponents]
    def run(x: FloatArray) -> ScalingExponents:
        return _exponents_from_surface(_moment_surface(x, q, scales))

    return run


@pytest.fixture(scope="module")
def trading_time() -> FloatArray:
    """Cumulative lognormal cascade: a signal whose exact log-cumulants are known."""
    mu = cascade.lognormal_cascade(KMAX, LAM, b=B, seed=CASCADE_SEED, normalise=False)
    return cascade.trading_time(mu)


@pytest.fixture(scope="module")
def cascade_truth() -> tuple[float, float]:
    sigma2 = theory.lognormal_conservation_sigma2(LAM, float(B))
    return theory.cumulants_from_tau_lognormal(LAM, sigma2, float(B))


@pytest.fixture(scope="module")
def fbm_path() -> FloatArray:
    return fbm(FBM_N, FBM_H, seed=FBM_SEED)


# --------------------------------------------------------------------------- #
# Algebraic core
# --------------------------------------------------------------------------- #


def test_sample_cumulants_match_scipy_k_statistics() -> None:
    """``_sample_cumulants`` must be the unbiased k-statistics, not plug-in moments."""
    rng = np.random.default_rng(11)
    v = rng.standard_normal(257) * 1.7 + 0.3
    got = _sample_cumulants(v, MAX_ORDER)
    want = np.array([stats.kstat(v, n=p) for p in range(1, MAX_ORDER + 1)])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)
    # k2 is the ddof=1 variance; the plug-in m2 would differ by n/(n-1).
    assert got[1] == pytest.approx(float(np.var(v, ddof=1)), rel=1e-12)


def test_disjoint_increment_count_is_exact() -> None:
    """LEGACY_AUDIT D6: strided slicing miscounts when the scale does not divide n."""
    x = np.arange(1000, dtype=np.float64)
    for s in (3, 7, 64, 333):
        d = _increments(x, s, overlapping=False)
        assert d.size == (x.size - 1) // s
        np.testing.assert_allclose(d, float(s))
    assert _increments(x, 7, overlapping=True).size == x.size - 7


def test_exponents_route_is_exact_on_a_polynomial_zeta() -> None:
    """With an exactly cubic zeta the regression must return the coefficients."""
    c = np.array([0.62, -0.11, 0.037])
    q = np.linspace(-4.0, 4.0, 33)
    zeta = c[0] * q + c[1] * q**2 / 2.0 + c[2] * q**3 / 6.0
    got = cumulants_from_exponents(ScalingExponents(q=q, zeta=zeta, estimator="synthetic"), order=3)
    np.testing.assert_allclose(got.c, c, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Closed-form oracles
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_lognormal_theory_exponents_recover_c1_c2_exactly(
    cascade_truth: tuple[float, float],
) -> None:
    """On the oracle, (c1, c2) = (lam, -sigma2 ln b) to machine precision, c3 = 0."""
    c1_true, c2_true = cascade_truth
    exponents = theory.lognormal_exponents(np.linspace(-3.0, 3.0, 31), lam=LAM)
    got = cumulants_from_exponents(exponents, order=3)
    assert got.c1 == pytest.approx(c1_true, abs=1e-12)
    assert got.c2 == pytest.approx(c2_true, abs=1e-12)
    assert abs(got.c3) < 1e-12, f"c3 = {got.c3:.3e} on an exactly quadratic zeta"
    assert multifractality_index(got) == pytest.approx(0.1, abs=1e-12)


@pytest.mark.theory
def test_monofractal_oracle_gives_c2_exactly_zero() -> None:
    """The monofractal null is exact, not approximate: zeta = qH is affine."""
    exponents = theory.fbm_exponents(np.linspace(-4.0, 4.0, 41), H=FBM_H)
    got = cumulants_from_exponents(exponents, order=3)
    assert got.c1 == pytest.approx(FBM_H, abs=1e-13)
    assert abs(got.c2) < 1e-13
    assert abs(got.c3) < 1e-13
    assert multifractality_index(got) == pytest.approx(0.0, abs=1e-13)


@pytest.mark.theory
def test_measure_convention_input_is_not_silently_absorbed() -> None:
    r"""Passing :math:`\tau=\zeta-d` must corrupt the fit, not be absorbed.

    Observed: the offset :math:`-d` is a *constant*, so on a symmetric
    :math:`q` grid it is orthogonal to every odd column of the design and
    :math:`\hat c_1` survives untouched (1.05 to 3e-15). It lands entirely in
    the even columns: :math:`\hat c_2` moves from -0.100 to -0.448, and the
    residual sum of squares from 0 to 12.75. A convention error is therefore not
    detectable from :math:`c_1` — only from the fit quality and :math:`c_2`.
    """
    q = np.linspace(-3.0, 3.0, 31)
    good = theory.lognormal_exponents(q, lam=LAM)
    bad = ScalingExponents(q=q, zeta=good.tau, estimator="tau-by-mistake")
    fit_good = cumulants_from_exponents(good, order=3)
    fit_bad = cumulants_from_exponents(bad, order=3)
    # Without an intercept the missing -d cannot be absorbed, so the residuals blow up.
    chi_good = float(fit_good.provenance.diagnostics["chi2"])  # type: ignore[union-attr]
    chi_bad = float(fit_bad.provenance.diagnostics["chi2"])  # type: ignore[union-attr]
    assert chi_good < 1e-20
    assert chi_bad > 1.0
    assert fit_bad.c1 == pytest.approx(fit_good.c1, abs=1e-12)
    assert abs(fit_bad.c2 - fit_good.c2) > 0.3
    assert fit_bad.stderr is not None and fit_bad.stderr[1] > 0.05


# --------------------------------------------------------------------------- #
# Lognormal cascade: both routes, and their agreement
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
def test_direct_route_recovers_the_cascade_cumulants(
    trading_time: FloatArray, cascade_truth: tuple[float, float]
) -> None:
    """Observed: c1 = 1.047074 (truth 1.05), c2 = -0.092384 (truth -0.1), c3 = -0.0029."""
    c1_true, c2_true = cascade_truth
    got = cumulants_from_increments(trading_time, scales=CASCADE_SCALES, order=3, overlapping=True)
    assert got.c1 == pytest.approx(c1_true, abs=0.01), f"c1 = {got.c1:.6f}"
    assert got.c2 == pytest.approx(c2_true, abs=0.015), f"c2 = {got.c2:.6f}"
    assert abs(got.c3) < 0.01, f"c3 = {got.c3:.6f} on an exactly lognormal cascade"
    # The single-realisation estimate sits *inside* the annealed value: a canonical
    # cascade's sample exponent is below the ensemble one at finite kmax.
    assert c2_true < got.c2 < 0.0
    r2 = got.provenance.diagnostics["fit_r_squared"]  # type: ignore[union-attr]
    assert r2[0] > 0.999 and r2[1] > 0.99, f"C_p(delta) is not linear in ln delta: {r2}"


@pytest.mark.stochastic
def test_all_three_routes_agree_on_the_same_cascade(
    trading_time: FloatArray, cascade_truth: tuple[float, float]
) -> None:
    """Direct, exponent and moment-surface routes, same data, same increments.

    Observed discrepancies (order 1, 2, 3):
    direct vs exponents  |dc| = (2.5e-05, 5.5e-04, 1.2e-04)
    direct vs moments    |dc| = (7.2e-04, 6.7e-04, 1.1e-03)
    """
    q = np.linspace(-2.0, 3.0, 21)
    surface = _moment_surface(trading_time, q, CASCADE_SCALES)
    exponents = _exponents_from_surface(surface)

    direct = cumulants_from_increments(
        trading_time, scales=CASCADE_SCALES, order=3, overlapping=True
    )
    from_zeta = cumulants_from_exponents(exponents, order=3)
    from_moments = cumulants_from_moments(surface, order=3)

    c1_true, c2_true = cascade_truth
    for name, est in (("zeta", from_zeta), ("moments", from_moments)):
        assert est.c1 == pytest.approx(c1_true, abs=0.01), f"{name}: c1 = {est.c1:.6f}"
        assert est.c2 == pytest.approx(c2_true, abs=0.015), f"{name}: c2 = {est.c2:.6f}"

    for other, tol in ((from_zeta, 1e-3), (from_moments, 2e-3)):
        report = compare_cumulants(direct, other)
        diff = np.abs(np.asarray(report["difference"]))
        assert diff.max() < tol, report
        assert report["max_abs_z"] < 4.0, report

    # The routes are far closer to each other than either is to the annealed truth:
    # they share a sample, so the finite-kmax error is common, not independent.
    assert abs(direct.c2 - from_zeta.c2) < abs(direct.c2 - c2_true)


@pytest.mark.stochastic
def test_cumulants_from_series_threads_a_callable_estimator(
    trading_time: FloatArray, cascade_truth: tuple[float, float]
) -> None:
    q = np.linspace(-2.0, 3.0, 21)
    got = cumulants_from_series(trading_time, _estimator(q, CASCADE_SCALES), order=3)
    assert got.c1 == pytest.approx(cascade_truth[0], abs=0.01)
    assert got.c2 == pytest.approx(cascade_truth[1], abs=0.015)


@pytest.mark.stochastic
def test_scale_resolved_cumulants_are_linear_in_log_delta(
    trading_time: FloatArray, cascade_truth: tuple[float, float]
) -> None:
    """C_p(delta) = c_p ln(delta) + const is the model; check it directly."""
    scales, c_scale, counts, n_dropped = log_cumulants_by_scale(
        trading_time, scales=CASCADE_SCALES, order=2, overlapping=True
    )
    assert c_scale.shape == (2, scales.size)
    assert n_dropped == 0
    np.testing.assert_array_equal(counts, (trading_time.size - 1) // scales)
    ln_delta = np.log(scales.astype(np.float64))
    slope_c1 = float(np.polyfit(ln_delta, c_scale[0], 1)[0])
    slope_c2 = float(np.polyfit(ln_delta, c_scale[1], 1)[0])
    assert slope_c1 == pytest.approx(cascade_truth[0], abs=0.02)
    assert slope_c2 == pytest.approx(cascade_truth[1], abs=0.02)


# --------------------------------------------------------------------------- #
# The monofractal detection floor
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
def test_fbm_c1_is_the_hurst_exponent(fbm_path: FloatArray) -> None:
    """Observed: c1 = 0.708633 for H = 0.7 (n = 2**16, overlapping increments)."""
    got = cumulants_from_increments(fbm_path, scales=FBM_SCALES, order=3)
    assert got.c1 == pytest.approx(FBM_H, abs=0.02), f"c1 = {got.c1:.6f}"
    r2 = got.provenance.diagnostics["fit_r_squared"]  # type: ignore[union-attr]
    assert r2[0] > 0.999
    # C_2(delta) has no scale dependence to fit, and the regression says so.
    assert r2[1] < 0.5, f"spurious C_2 scaling on fBm: R^2 = {r2[1]:.3f}"


@pytest.mark.stochastic
def test_fbm_c2_detection_floor() -> None:
    r"""The floor: |c2| an fBm sample fakes at :math:`n=2^{16}`.

    Observed over 8 spawned seeds, overlapping increments, scales :math:`2^2..2^{11}`:

    ==========  ======================  =========================  ==========
    :math:`H`   mean :math:`\hat c_2`   sd                          max abs
    ==========  ======================  =========================  ==========
    0.5         -0.00012                0.00648                     0.01107
    0.7         -0.00339                0.00793                     0.01525
    ==========  ======================  =========================  ==========

    So :math:`-c_2 \approx 0.015` is the noise level; turbulence (0.03) and
    equity returns (0.03--0.08) are above it, but not by much.
    """
    for H, floor in ((0.5, 0.02), (FBM_H, 0.025)):
        c2 = np.array(
            [
                cumulants_from_increments(fbm(FBM_N, H, seed=rng), scales=FBM_SCALES, order=2).c2
                for rng in spawn_rngs(CASCADE_SEED, 8)
            ]
        )
        assert np.abs(c2).max() < floor, (
            f"H={H}: max |c2| = {np.abs(c2).max():.5f} over 8 replicas "
            f"(mean {c2.mean():+.5f}, sd {c2.std(ddof=1):.5f})"
        )
        assert abs(float(c2.mean())) < 0.01


@pytest.mark.stochastic
def test_overlapping_increments_lower_the_floor(fbm_path: FloatArray) -> None:
    """Observed on one path: |c2| = 0.0038 overlapping vs 0.0252 disjoint."""
    overlapping = cumulants_from_increments(fbm_path, scales=FBM_SCALES, order=2)
    disjoint = cumulants_from_increments(fbm_path, scales=FBM_SCALES, order=2, overlapping=False)
    assert abs(overlapping.c2) < abs(disjoint.c2)
    assert overlapping.stderr is not None and disjoint.stderr is not None
    assert overlapping.stderr[1] < disjoint.stderr[1]


# --------------------------------------------------------------------------- #
# Spectrum reconstruction
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_order2_spectrum_is_the_lognormal_parabola(
    cascade_truth: tuple[float, float],
) -> None:
    """The order-2 reconstruction *is* theory.lognormal_falpha, bit for bit."""
    sigma2 = theory.lognormal_conservation_sigma2(LAM, float(B))
    lo, hi = theory.lognormal_alpha_range(LAM, sigma2, float(B))
    alpha = np.linspace(lo, hi, 51)
    got = cumulant_spectrum(LogCumulants(c=np.array(cascade_truth), order=2), alpha)
    want = theory.lognormal_falpha(alpha, LAM, sigma2, float(B))
    np.testing.assert_allclose(got.f_alpha, want, rtol=0.0, atol=1e-15)
    assert got.alpha_0 == pytest.approx(LAM, abs=1e-12)
    assert float(np.nanmax(got.f_alpha)) == pytest.approx(1.0, abs=1e-12)
    # And it agrees with the dataclass's own parabola.
    np.testing.assert_allclose(
        got.f_alpha, LogCumulants(c=np.array(cascade_truth)).parabolic_spectrum(alpha)
    )


def test_cubic_correction_beats_the_parabola_near_the_apex() -> None:
    r"""Order 3 must be closer to the true Legendre transform of a cubic zeta.

    Observed on :math:`(c_1,c_2,c_3)=(1.05,-0.10,0.02)` over
    :math:`|\alpha-c_1|\le0.15`: max error 0.003735 (cubic) vs 0.014985
    (parabola), a factor 4.01. The reference is the numerical Legendre--Fenchel
    transform, which on this window agrees with the exact stationary branch
    :math:`\alpha=\zeta'(q)`, :math:`f=q\alpha-\zeta+d` to :math:`10^{-9}`
    (:math:`\zeta''=-0.1+0.02q<0` on :math:`|q|\le3`, so :math:`\zeta` is
    concave there and the infimum *is* the stationary point).
    """
    c = np.array([1.05, -0.10, 0.02])
    q = np.linspace(-3.0, 3.0, 601)
    zeta = c[0] * q + c[1] * q**2 / 2.0 + c[2] * q**3 / 6.0
    alpha = np.linspace(c[0] - 0.15, c[0] + 0.15, 41)
    _, f_exact = legendre_fenchel(q, zeta, d_support=1.0, alpha_grid=alpha, q_refine=20001)

    # Independent reference: invert alpha = c1 + c2 q + c3 q^2/2 in closed form on the
    # branch through q = 0, so the comparison does not rest on legendre_fenchel alone.
    u = alpha - c[0]
    root = np.sqrt(c[1] ** 2 + 2.0 * c[2] * u)
    q_star = np.where(
        np.abs((-c[1] + root) / c[2] - u / c[1]) < np.abs((-c[1] - root) / c[2] - u / c[1]),
        (-c[1] + root) / c[2],
        (-c[1] - root) / c[2],
    )
    f_branch = (
        q_star * alpha - (c[0] * q_star + c[1] * q_star**2 / 2.0 + c[2] * q_star**3 / 6.0) + 1.0
    )
    np.testing.assert_allclose(f_exact, f_branch, atol=1e-7)

    cum = LogCumulants(c=c, order=3)
    err3 = float(np.max(np.abs(cumulant_spectrum(cum, alpha, order=3).f_alpha - f_exact)))
    err2 = float(np.max(np.abs(cumulant_spectrum(cum, alpha, order=2).f_alpha - f_exact)))
    assert err3 == pytest.approx(0.003735, abs=2e-5), f"cubic error moved: {err3:.6f}"
    assert err2 == pytest.approx(0.014985, abs=2e-5), f"parabola error moved: {err2:.6f}"
    assert err3 < 0.3 * err2, f"cubic {err3:.6f} vs parabola {err2:.6f}"
    assert err3 < 0.005


def test_default_alpha_grid_spans_the_support() -> None:
    cum = LogCumulants(c=np.array([1.05, -0.1]), order=2)
    spec = cumulant_spectrum(cum, n_alpha=101, span=1.0)
    assert spec.alpha.size == 101
    half = math.sqrt(0.2)
    assert spec.alpha[0] == pytest.approx(1.05 - half)
    assert spec.alpha[-1] == pytest.approx(1.05 + half)
    assert float(spec.f_alpha[0]) == pytest.approx(0.0, abs=1e-12)


def test_non_concave_cumulants_give_nan_not_an_inverted_parabola() -> None:
    cum = LogCumulants(c=np.array([1.0, 0.05]), order=2)
    spec = cumulant_spectrum(cum, np.linspace(0.5, 1.5, 11))
    assert np.all(np.isnan(spec.f_alpha))
    with pytest.raises(ValueError, match="non-negative"):
        cumulant_spectrum(cum)
    assert multifractality_index(cum) == pytest.approx(-0.05)


# --------------------------------------------------------------------------- #
# Diagnostics and guards
# --------------------------------------------------------------------------- #


def test_multifractality_index_scale_is_documented_and_signed() -> None:
    mono = LogCumulants(c=np.array([0.7, 0.0]), order=2)
    turbulent = LogCumulants(c=np.array([0.36, -0.03]), order=2)
    assert multifractality_index(mono) == 0.0
    assert multifractality_index(turbulent) == pytest.approx(0.03)


def test_compare_cumulants_reports_z_scores() -> None:
    a = LogCumulants(c=np.array([1.0, -0.1]), stderr=np.array([0.01, 0.02]), order=2)
    b = LogCumulants(c=np.array([1.02, -0.1]), stderr=np.array([0.01, 0.02]), order=2)
    report = compare_cumulants(a, b, labels=("a", "b"))
    assert report["order"] == 2
    assert report["difference"][0] == pytest.approx(-0.02)
    assert report["z"][0] == pytest.approx(-0.02 / math.sqrt(2.0 * 1e-4))
    assert report["z"][1] == 0.0
    assert report["max_abs_z"] == pytest.approx(abs(report["z"][0]))


def test_compare_cumulants_does_not_hide_an_infinite_discrepancy() -> None:
    """A zero standard error makes z infinite; max_abs_z must not mask it away.

    Masking non-finite z and maximising over the rest reports the largest of the
    *agreeing* orders, so the summary that is supposed to flag a broken route
    reads 0.0 on the order that disagrees infinitely.
    """
    a = LogCumulants(c=np.array([1.0, -0.1]), stderr=np.array([0.0, 0.02]), order=2)
    b = LogCumulants(c=np.array([9.0, -0.1]), stderr=np.array([0.0, 0.02]), order=2)
    report = compare_cumulants(a, b)
    assert report["z"][0] == -math.inf
    assert report["z"][1] == 0.0
    assert report["max_abs_z"] == math.inf, report
    # stderr=None is the dataclass default, so the same path must hold there.
    no_se = compare_cumulants(
        LogCumulants(c=np.array([1.0, -0.1]), order=2),
        LogCumulants(c=np.array([1.1, -0.1]), order=2),
    )
    assert no_se["max_abs_z"] == math.inf, no_se
    # Genuine agreement still reports zero, not infinity.
    same = compare_cumulants(
        LogCumulants(c=np.array([1.0, -0.1]), order=2),
        LogCumulants(c=np.array([1.0, -0.1]), order=2),
    )
    assert same["max_abs_z"] == 0.0


def test_zero_increments_are_refused_by_default() -> None:
    x = np.repeat(np.arange(64, dtype=np.float64), 8)  # a staircase: many flat steps
    scales = np.array([2, 4, 8, 16], dtype=np.int64)
    with pytest.raises(ValueError, match="vanish exactly"):
        cumulants_from_increments(x, scales=scales, order=2)
    got = cumulants_from_increments(x, scales=scales, order=2, zero_policy="drop")
    assert math.isfinite(got.c1)
    assert got.provenance is not None
    assert int(got.provenance.diagnostics["zero_increments_dropped"]) > 0


def test_unknown_zero_policy_is_rejected_not_treated_as_drop() -> None:
    """Anything that is not ``"raise"`` used to fall through to the lossy branch."""
    x = np.repeat(np.arange(64, dtype=np.float64), 8)
    scales = np.array([2, 4, 8, 16], dtype=np.int64)
    for bad in ("ignore", "keep", "Raise", ""):
        with pytest.raises(ValueError, match="zero_policy must be one of"):
            cumulants_from_increments(x, scales=scales, order=2, zero_policy=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="zero_policy must be one of"):
            log_cumulants_by_scale(x, scales=scales, order=2, zero_policy=bad)  # type: ignore[arg-type]


def test_fluctuation_surface_is_rejected() -> None:
    """An MFDFA-style surface scales as delta^h(q); the guard must catch it."""
    q = np.linspace(-2.0, 3.0, 11)
    scales = np.array([4.0, 8.0, 16.0, 32.0, 64.0])
    values = np.exp(0.7 * np.log(scales)[None, :] + 0.3 * q[:, None])
    surface = StructureFunctions(q=q, scales=scales, values=values)
    with pytest.raises(ValueError, match="not a moment surface"):
        cumulants_from_moments(surface, order=2)


def test_moment_surface_without_q_zero_is_accepted() -> None:
    """The normalisation column is optional; the origin constraint still holds."""
    q = np.array([-1.5, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5])
    scales = np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    zeta = 0.55 * q - 0.04 * q**2 / 2.0
    values = np.exp(zeta[:, None] * np.log(scales)[None, :])
    got = cumulants_from_moments(StructureFunctions(q=q, scales=scales, values=values), order=2)
    assert got.c1 == pytest.approx(0.55, abs=1e-9)
    assert got.c2 == pytest.approx(-0.04, abs=1e-9)


def test_input_validation() -> None:
    q = np.linspace(-3.0, 3.0, 31)
    exponents = theory.lognormal_exponents(q, lam=LAM)
    with pytest.raises(ValueError, match="order must lie"):
        cumulants_from_exponents(exponents, order=0)
    with pytest.raises(ValueError, match="order must lie"):
        cumulants_from_exponents(exponents, order=MAX_ORDER + 1)
    with pytest.raises(ValueError, match="q_range"):
        cumulants_from_exponents(exponents, q_range=(2.0, 1.0))
    with pytest.raises(ValueError, match="usable q values"):
        cumulants_from_exponents(exponents, order=3, q_range=(0.99, 1.01))
    with pytest.raises(ValueError, match="non-finite"):
        cumulants_from_exponents(
            ScalingExponents(q=q, zeta=np.full_like(q, np.nan), estimator="broken")
        )
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        cumulants_from_exponents(
            ScalingExponents(
                q=q, zeta=exponents.zeta, zeta_stderr=np.zeros_like(q), estimator="broken"
            )
        )
    with pytest.raises(ValueError, match="at least the second cumulant"):
        cumulant_spectrum(LogCumulants(c=np.array([0.5]), order=1), np.linspace(0, 1, 5))


def test_q_range_narrows_the_fit_and_is_recorded() -> None:
    q = np.linspace(-4.0, 4.0, 41)
    exponents = theory.lognormal_exponents(q, lam=LAM)
    got = cumulants_from_exponents(exponents, order=2, q_range=(-2.0, 2.0))
    diag = got.provenance.diagnostics  # type: ignore[union-attr]
    assert diag["q_min"] == pytest.approx(-2.0)
    assert diag["q_max"] == pytest.approx(2.0)
    assert diag["n_points"] == int(np.sum((np.abs(q) > 1e-12) & (np.abs(q) <= 2.0)))
    # A quadratic oracle is recovered on any window.
    assert got.c2 == pytest.approx(-0.1, abs=1e-12)


def test_unweighted_fallback_matches_uniform_weights() -> None:
    q = np.linspace(-3.0, 3.0, 31)
    zeta = 0.6 * q - 0.05 * q**2 / 2.0
    se = np.full_like(q, 0.01)
    exponents = ScalingExponents(q=q, zeta=zeta, zeta_stderr=se, estimator="synthetic")
    weighted = cumulants_from_exponents(exponents, order=2)
    unweighted = cumulants_from_exponents(exponents, order=2, use_stderr=False)
    np.testing.assert_allclose(weighted.c, unweighted.c, atol=1e-14)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
def test_provenance_is_attached_deterministic_and_parameter_sensitive(
    trading_time: FloatArray,
) -> None:
    a = cumulants_from_increments(trading_time, scales=CASCADE_SCALES, order=3)
    b = cumulants_from_increments(trading_time, scales=CASCADE_SCALES, order=3)
    c = cumulants_from_increments(trading_time, scales=CASCADE_SCALES, order=3, overlapping=False)
    assert a.provenance is not None and c.provenance is not None
    assert a.provenance.root == b.provenance.root  # type: ignore[union-attr]
    assert a.provenance.root != c.provenance.root
    assert a.provenance.estimator == "inference.cumulants_from_increments"
    assert a.provenance.seed is None
    inputs = {i.name for i in a.provenance.inputs}
    assert inputs == {"x", "scales"}


@pytest.mark.stochastic
def test_provenance_chains_through_the_estimator(trading_time: FloatArray) -> None:
    """A parent provenance must appear as a Merkle edge, not be dropped."""
    q = np.linspace(-2.0, 3.0, 21)
    surface = _moment_surface(trading_time, q, CASCADE_SCALES)
    exponents = _exponents_from_surface(surface)
    parent = new_provenance("test.parent", params={"note": "upstream"})
    chained = ScalingExponents(
        q=exponents.q,
        zeta=exponents.zeta,
        zeta_stderr=exponents.zeta_stderr,
        estimator="test",
        provenance=parent,
    )
    got = cumulants_from_exponents(chained, order=2)
    assert got.provenance is not None
    assert parent.root in {i.digest for i in got.provenance.inputs}
    spectrum = cumulant_spectrum(got, np.linspace(0.9, 1.2, 11))
    assert spectrum.provenance is not None
    assert got.provenance.root in {i.digest for i in spectrum.provenance.inputs}


def _as_sequence(x: FloatArray) -> Sequence[float]:
    return [float(v) for v in x]


def test_plain_sequences_are_accepted() -> None:
    """Every public entry point coerces its input rather than requiring ndarray."""
    x = np.cumsum(np.random.default_rng(3).standard_normal(4096))
    got = cumulants_from_increments(_as_sequence(x), scales=[8, 16, 32, 64, 128, 256], order=2)
    assert got.c1 == pytest.approx(0.5, abs=0.1)
