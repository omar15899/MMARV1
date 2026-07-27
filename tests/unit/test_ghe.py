r"""Validation of :mod:`mmar.estimators.ghe` against closed-form oracles.

Everything numeric here is measured, never assumed. The accuracy table produced
by :func:`test_accuracy_table_on_exact_fbm` is printed (run with ``-s``) and its
tolerances are the *observed* biases rounded up, not tolerances chosen to make
the assertions pass. Where an estimator is genuinely biased --- R/S is, badly,
away from :math:`H=1/2` --- the test asserts the bias exists rather than hiding
it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.types import FitMethod, FloatArray
from mmar.estimators.ghe import (
    aggregated_variance,
    anis_lloyd_expected_rs,
    detrended_fluctuation,
    generalised_hurst,
    generalised_hurst_structure,
    hurst_consensus,
    periodogram_hurst,
    rescaled_range,
)
from mmar.models import theory
from mmar.models.cascade import deterministic_cascade, trading_time
from mmar.models.fbm import fbm

N = 2**14
BASE_SEED = 20260727
CASCADE_M = (0.6, 0.4)
CASCADE_KMAX = 14

# fBm increments are Gaussian, so E|Z|^q is finite only for q > -1 (see
# theory.fbm_zeta). Any flatness test on an fBm must therefore live strictly
# inside that band; asking for h(-5) on a Gaussian process is asking for the
# sample estimate of a divergent moment.
Q_FBM = np.array([-0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
Q_FULL = np.linspace(-5.0, 5.0, 21)


def _fbm(H: float, replica: int) -> FloatArray:
    """A reproducible fBm path; the seed is a tuple so replicas do not share a stream."""
    return fbm(N, H, seed=[BASE_SEED, round(H * 100), replica])


def _theta() -> FloatArray:
    r"""Trading time of the deterministic binomial cascade :math:`(0.6, 0.4)`.

    Its increments over a dyadic lag :math:`\tau=2^{j}` aligned at the origin are
    *exactly* the level-:math:`(k_{\max}-j)` box masses, so the partition
    function --- and therefore :math:`\zeta` --- is exact at every level, not
    asymptotic. That is what licenses a :math:`10^{-6}` tolerance below.
    """
    return trading_time(deterministic_cascade(CASCADE_KMAX, list(CASCADE_M)))


DYADIC = np.array([2**j for j in range(1, 12)], dtype=np.int64)


# --------------------------------------------------------------------------- #
# (a) Monofractal recovery
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_ghe_recovers_flat_hq_on_fbm(H: float, capsys: pytest.CaptureFixture[str]) -> None:
    """h(q) is flat and equal to H on exact fBm, over the band where the moments exist."""
    hq = np.array([generalised_hurst(_fbm(H, r), q=Q_FBM).hq for r in range(8)])
    mean_over_q = hq.mean(axis=1)
    flatness = float(np.mean(hq.max(axis=1) - hq.min(axis=1)))
    bias = float(mean_over_q.mean() - H)
    spread = float(mean_over_q.std(ddof=1))

    with capsys.disabled():
        print(
            f"\nGHE on fBm(H={H}), n={N}, 8 replicas, q in [-0.5, 5]: "
            f"bias={bias:+.4f}  sd={spread:.4f}  mean spread of h(q)={flatness:.4f}"
        )

    assert abs(bias) < 0.02, f"H={H}: bias {bias:+.4f}"
    assert spread < 0.02, f"H={H}: replica sd {spread:.4f}"
    # A flat h(q) is the definition of monofractality. Observed 0.015-0.027 over
    # q in [-0.5, 5]; the threshold is set at 0.045 to leave seed-to-seed slack,
    # and is still an order of magnitude below the cascade's range of 0.256.
    assert flatness < 0.045, f"H={H}: h(q) varies by {flatness:.4f} on a monofractal"


@pytest.mark.theory
def test_ghe_hq_decreases_on_a_cascade() -> None:
    """The counterpart of the flatness test: h(q) must be strictly decreasing."""
    res = generalised_hurst(_theta(), q=Q_FULL, scales=DYADIC, overlapping=False)
    hq = res.hq[np.isfinite(res.hq)]
    q_pos = res.q[res.q > 0]
    h_pos = res.hq[res.q > 0]
    assert np.all(np.diff(h_pos) < 0.0), f"h(q) not decreasing on q>0: {h_pos}"
    # Observed range over q in [-5, 5] is 0.256; the fBm null gives < 0.01.
    assert hq.max() - hq.min() > 0.2, f"cascade h(q) range only {hq.max() - hq.min():.3f}"
    # Sanity: the theoretical h(q) = zeta(q)/q of this cascade.
    theory_h = theory.cascade_zeta(q_pos, list(CASCADE_M)) / q_pos
    assert np.allclose(h_pos, theory_h, atol=1e-6)


# --------------------------------------------------------------------------- #
# (b) Multifractal recovery
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_cascade_zeta_is_exact_on_dyadic_disjoint_lags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""Machine-precision recovery of :math:`\zeta(q)=1-\log_2(m_0^q+m_1^q)`.

    Tolerance :math:`10^{-6}`, justified structurally rather than empirically:
    with disjoint dyadic lags the increments *are* the cascade cells, so
    :math:`\log S(q,\tau)` is exactly affine in :math:`\log\tau` and the
    regression residual is float round-off. Anything above :math:`10^{-6}` would
    be a genuine bug in the partition, not sampling noise.
    """
    res = generalised_hurst(_theta(), q=Q_FULL, scales=DYADIC, overlapping=False)
    expected = theory.cascade_zeta(Q_FULL, list(CASCADE_M))
    err = float(np.abs(res.zeta - expected).max())
    with capsys.disabled():
        print(f"\ncascade zeta, disjoint dyadic lags: max |zeta_hat - zeta| = {err:.3e}")
    assert err < 1e-6


@pytest.mark.theory
def test_cascade_zeta_overlapping_is_close(capsys: pytest.CaptureFixture[str]) -> None:
    r"""Di Matteo's overlapping average costs about 0.1 in :math:`\zeta` at :math:`|q|=5`.

    Overlapping increments straddle box boundaries, so at finite :math:`k_{\max}`
    they are mixtures of neighbouring cells; the mixture is heaviest exactly
    where the moment is most selective, i.e. at large :math:`|q|`. The tolerance
    is stated per band because a single number would hide that structure.
    """
    res = generalised_hurst(_theta(), q=Q_FULL, scales=DYADIC, overlapping=True)
    expected = theory.cascade_zeta(Q_FULL, list(CASCADE_M))
    err = np.abs(res.zeta - expected)
    core = err[np.abs(Q_FULL) <= 2.0].max()
    tail = err.max()
    with capsys.disabled():
        print(
            f"\ncascade zeta, overlapping lags: max err |q|<=2 = {core:.4f}, "
            f"max err |q|<=5 = {tail:.4f}"
        )
    assert core < 0.03
    assert tail < 0.12


@pytest.mark.theory
def test_cascade_falpha_recovered(capsys: pytest.CaptureFixture[str]) -> None:
    r"""The Legendre transform of the estimated :math:`\zeta` matches the entropy formula.

    Compared on the interior of the support only: near the endpoints
    :math:`f\to0` with infinite slope, so a finite :math:`q` window cannot reach
    them --- that is a property of the transform, not an estimator defect.
    """
    res = generalised_hurst(_theta(), q=Q_FULL, scales=DYADIC, overlapping=False)
    spec = spectrum_from_exponents(res, method="fenchel", n_alpha=201)
    a_min, a_max = theory.cascade_alpha_range(list(CASCADE_M), 2)
    inside = (spec.alpha > a_min + 0.05) & (spec.alpha < a_max - 0.05)
    assert int(inside.sum()) > 100, "the alpha grid does not cover the support interior"
    ref = theory.cascade_falpha(spec.alpha[inside], list(CASCADE_M))
    assert np.all(np.isfinite(ref))
    rms = float(np.sqrt(np.mean((spec.f_alpha[inside] - ref) ** 2)))
    with capsys.disabled():
        print(
            f"\ncascade f(alpha): {int(inside.sum())} interior points, "
            f"rms deviation = {rms:.3e}, peak f = {spec.f_alpha[inside].max():.6f}"
        )
    # 0.01 is ~3x the observed value; the residual is the Legendre-Fenchel grid
    # discretisation, not estimation error (zeta itself is exact to 1e-8).
    assert rms < 0.01


# --------------------------------------------------------------------------- #
# (c) Convention checks
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_tau_zero_is_minus_d_exactly() -> None:
    r""":math:`\zeta(0)=0` and hence :math:`\tau(0)=-d`, *exactly*, by construction.

    :math:`K_0(\tau)=\langle|\Delta X|^0\rangle/\langle|X|^0\rangle=1` for every
    :math:`\tau`, so the fitted slope is identically zero. This is the free
    self-consistency check the legacy library never performed (defect D6, where
    the measured value was :math:`-1.0174`).
    """
    for series, kwargs in (
        (_fbm(0.6, 0), {}),
        (_theta(), {"scales": DYADIC, "overlapping": False}),
    ):
        res = generalised_hurst(series, q=np.array([-1.0, 0.0, 1.0, 2.0]), **kwargs)  # type: ignore[arg-type]
        i0 = int(np.argmin(np.abs(res.q)))
        assert res.zeta[i0] == 0.0
        assert res.tau[i0] == -1.0


@pytest.mark.theory
def test_tau_one_is_zero_for_a_conservative_measure() -> None:
    r""":math:`\tau(1)=0` for the trading time of a mass-conserving cascade.

    :math:`\sum_i \mu_i = 1` at every level, so :math:`Z(1,\delta)=1` and
    :math:`\tau(1)=0`, i.e. :math:`\zeta(1)=d=1`.
    """
    res = generalised_hurst(_theta(), q=Q_FULL, scales=DYADIC, overlapping=False)
    i1 = int(np.argmin(np.abs(res.q - 1.0)))
    assert abs(res.tau[i1]) < 1e-10
    assert abs(res.zeta[i1] - 1.0) < 1e-10


def test_zeta_is_the_signal_convention_not_the_measure_one() -> None:
    """A convention slip of one would show up here and nowhere else.

    fBm has :math:`\\zeta(q)=qH` and :math:`\\tau(q)=qH-1`; the two differ by
    exactly ``d_support``, and ``d_support`` must be the only thing that moves
    when it changes.
    """
    x = _fbm(0.7, 0)
    a = generalised_hurst(x, q=Q_FBM, d_support=1.0)
    b = generalised_hurst(x, q=Q_FBM, d_support=2.0)
    assert np.array_equal(a.zeta, b.zeta)
    assert np.allclose(a.tau - b.tau, 1.0)
    assert abs(a.h2 - 0.7) < 0.03


# --------------------------------------------------------------------------- #
# (d) Determinism
# --------------------------------------------------------------------------- #


def test_bitwise_determinism() -> None:
    """Same input, same parameters, bitwise identical exponents."""
    x = _fbm(0.55, 3)
    a = generalised_hurst(x, q=Q_FBM)
    b = generalised_hurst(x, q=Q_FBM)
    assert np.array_equal(a.zeta, b.zeta)
    assert np.array_equal(a.zeta_stderr, b.zeta_stderr)  # type: ignore[arg-type]
    assert rescaled_range(x).slope == rescaled_range(x).slope
    assert detrended_fluctuation(x).slope == detrended_fluctuation(x).slope
    assert aggregated_variance(x).slope == aggregated_variance(x).slope
    assert periodogram_hurst(x).slope == periodogram_hurst(x).slope


# --------------------------------------------------------------------------- #
# (e) Provenance
# --------------------------------------------------------------------------- #


def test_provenance_root_is_stable_and_parameter_sensitive() -> None:
    """The root is a pure function of the identifying inputs, including every parameter."""
    x = _fbm(0.55, 4)
    base = generalised_hurst(x, q=Q_FBM)
    assert base.provenance is not None
    root = base.provenance.root
    assert generalised_hurst(x, q=Q_FBM).provenance.root == root  # type: ignore[union-attr]

    variants: list[dict[str, Any]] = [
        {"tau_max": 25},
        {"tau_min": 3},
        {"normalise": False},
        {"overlapping": False},
        {"drop_zeros": False},
        {"average_tau_max": False},
        {"min_scales_per_fit": 6},
        {"fit_method": FitMethod.OLS},
        {"select_range": True},
        {"d_support": 2.0},
        {"n_scales": 30},
        {"seed": 7},
        {"q": Q_FULL},
    ]
    seen = {root}
    for kw in variants:
        other = generalised_hurst(x, **{"q": Q_FBM, **kw})
        assert other.provenance is not None
        assert other.provenance.root != root, f"root unchanged by {kw}"
        seen.add(other.provenance.root)
    assert len(seen) == len(variants) + 1, "two different parameter sets share a root"

    # A different input series must also move the root.
    assert generalised_hurst(_fbm(0.55, 5), q=Q_FBM).provenance.root != root  # type: ignore[union-attr]


def test_provenance_records_dropped_zeros() -> None:
    """Zeros are removed for the negative-q moments and the count is reported, never hidden."""
    x = np.cumsum(np.repeat(np.array([1.0, 0.0, -1.0, 0.0]), 64))
    res = generalised_hurst(x, q=np.array([-1.0, 0.0, 1.0, 2.0]), tau_max=16)
    assert res.provenance is not None
    dropped = int(res.provenance.diagnostics["n_zero_increments_dropped"])
    assert dropped > 0
    assert "n_zero_levels_dropped" in res.provenance.diagnostics
    assert len(res.provenance.diagnostics["zero_fraction_by_scale"]) == len(
        res.structure.scales  # type: ignore[union-attr]
    )
    assert float(res.provenance.diagnostics["max_zero_fraction"]) > 0.0


@pytest.mark.theory
def test_dropping_zeros_cannot_move_a_positive_moment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""Regression test for the zero-divisor defect.

    Exact zeros must be removed before a negative moment is formed, but a zero
    contributes :math:`|0|^q=0` to a *positive* one, so :math:`K_q(\tau)` for
    :math:`q>0` is a mean over the whole sample and its divisor is
    :math:`N_{\text{tot}}(\tau)`. Dividing by the surviving count instead
    multiplies :math:`K_q(\tau)` by
    :math:`N_{\text{tot}}(\tau)/N_{\text{nz}}(\tau)`, which on a tick-discretised
    price is itself a power of :math:`\tau` --- the zero fraction falls as the
    lag grows. The damage is then a single constant :math:`\Delta` subtracted
    from *every* :math:`\zeta(q)`, i.e. :math:`-\Delta/q` added to :math:`h(q)`:
    a smooth, monotone, entirely fabricated multiscaling curve on a process that
    is exactly monofractal.

    Two independent statements are checked, neither of which depends on any
    claim about what rounding does to the physics:

    1. ``drop_zeros`` is *exactly inert* on the positive branch. It was not:
       on the series below it displaced every :math:`\zeta(q)` by
       :math:`-0.392`, taking :math:`h(2)` from 0.342 to 0.147 and the Legendre
       :math:`f_{\max}` from 1.00 to 0.37.
    2. The exported :math:`K_q(\tau)` equals the mean of :math:`|\Delta_\tau X|^q`
       over *all* increments, computed here by hand.
    """
    rng = np.random.default_rng(4242)
    walk = np.concatenate(([0.0], np.cumsum(0.3 * rng.standard_normal(N))))
    tick = np.round(walk)  # ~76 % of the lag-1 increments become exactly zero
    q = np.array([0.5, 1.0, 2.0, 3.0, 5.0])

    dropped = generalised_hurst(tick, q=q, drop_zeros=True)
    kept = generalised_hurst(tick, q=q, drop_zeros=False)
    assert dropped.provenance is not None
    assert float(dropped.provenance.diagnostics["max_zero_fraction"]) > 0.5
    displacement = float(np.abs(dropped.zeta - kept.zeta).max())

    surface = generalised_hurst_structure(tick, q=q, normalise=False)
    worst = 0.0
    for j, tau in enumerate(surface.scales.astype(int)):
        inc = tick[tau:] - tick[:-tau]
        for i, qi in enumerate(q):
            worst = max(worst, abs(surface.values[i, j] / float(np.mean(np.abs(inc) ** qi)) - 1.0))

    with capsys.disabled():
        print(
            f"\ntick-discretised walk (zero fraction "
            f"{float(dropped.provenance.diagnostics['max_zero_fraction']):.3f}): "
            f"|zeta(drop) - zeta(keep)| = {displacement:.2e}, "
            f"K_q relative error vs the full-sample mean = {worst:.2e}"
        )

    assert displacement < 1e-12, f"dropping zeros moved zeta by {displacement}"
    assert worst < 1e-10, f"K_q is not the mean over all increments (rel err {worst})"


def test_zero_dropping_is_inert_when_there_are_no_zeros() -> None:
    """The divisor repair must be bitwise invisible on a series without exact zeros."""
    x = _fbm(0.6, 12)
    assert not np.any(np.diff(x) == 0.0)
    a = generalised_hurst(x, q=Q_FBM, drop_zeros=True)
    b = generalised_hurst(x, q=Q_FBM, drop_zeros=False)
    assert np.array_equal(a.zeta, b.zeta)


# --------------------------------------------------------------------------- #
# (f) Negative q: the direct regression test for legacy defect D2
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_negative_q_widens_the_recovered_alpha_range(
    capsys: pytest.CaptureFixture[str],
) -> None:
    r"""The right branch of :math:`f(\alpha)` exists only for :math:`q<0`.

    Legacy defect D2: the v1 grid was :math:`q\in[0,3]`, which reaches 24 % of
    the true support of the binomial measure. Both spectra here come from the
    *same* estimator on the *same* data; the only difference is the sign of the
    moments probed.
    """
    theta = _theta()
    q_both = np.linspace(-5.0, 5.0, 41)
    q_pos = np.linspace(0.0, 5.0, 21)

    both = spectrum_from_exponents(
        generalised_hurst(theta, q=q_both, scales=DYADIC, overlapping=False)
    ).descriptors()
    pos = spectrum_from_exponents(
        generalised_hurst(theta, q=q_pos, scales=DYADIC, overlapping=False)
    ).descriptors()

    a_min, a_max = theory.cascade_alpha_range(list(CASCADE_M), 2)
    true_width = a_max - a_min
    with capsys.disabled():
        print(
            f"\nalpha support (true {a_min:.4f}..{a_max:.4f}, width {true_width:.4f}):"
            f"\n  q in [-5, 5]: {both.alpha_min:.4f}..{both.alpha_max:.4f} "
            f"width {both.width:.4f} ({100 * both.width / true_width:.0f}% of support)"
            f"\n  q in [ 0, 5]: {pos.alpha_min:.4f}..{pos.alpha_max:.4f} "
            f"width {pos.width:.4f} ({100 * pos.width / true_width:.0f}% of support)"
        )

    assert both.width > 1.8 * pos.width
    assert both.alpha_max > pos.alpha_max + 0.2
    # The q >= 0 grid cannot pass alpha_0 = 1 by more than the grid resolution.
    assert pos.alpha_max < 1.05
    assert both.alpha_max > 1.2


# --------------------------------------------------------------------------- #
# The accuracy panel
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.stochastic
def test_accuracy_table_on_exact_fbm(capsys: pytest.CaptureFixture[str]) -> None:
    """Mean bias and standard deviation of every estimator at six Hurst exponents.

    30 seeded replicas of exact Davies--Harte fBm at :math:`n=2^{14}`. The
    thresholds below are the *observed* numbers rounded up; each one is asserted
    so that a regression in any estimator fails loudly, and the R/S bias is
    asserted to be *present* rather than tolerated away.
    """
    hursts = (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
    replicas = 30
    names = ("ghe_h2", "rs_corrected", "rs_raw", "aggvar", "dfa1", "gph")
    table: dict[float, dict[str, tuple[float, float]]] = {}

    for H in hursts:
        acc: dict[str, list[float]] = {k: [] for k in names}
        for r in range(replicas):
            x = _fbm(H, r)
            acc["ghe_h2"].append(generalised_hurst(x, q=np.array([0.5, 1.0, 2.0])).h2)
            acc["rs_corrected"].append(rescaled_range(x).slope)
            acc["rs_raw"].append(rescaled_range(x, corrected=False).slope)
            acc["aggvar"].append(1.0 + 0.5 * aggregated_variance(x).slope)
            acc["dfa1"].append(detrended_fluctuation(x).slope)
            acc["gph"].append(0.5 - periodogram_hurst(x).slope)
        table[H] = {k: (float(np.mean(v) - H), float(np.std(v, ddof=1))) for k, v in acc.items()}

    with capsys.disabled():
        print(f"\n\nbias (mean H_hat - H) and sd, n={N}, {replicas} replicas")
        print("   H   " + "".join(f"{k:>18}" for k in names))
        for H in hursts:
            cells = "".join(f"{table[H][k][0]:>+9.4f}+-{table[H][k][1]:.4f}" for k in names)
            print(f"{H:6.2f} {cells}")

    for H in hursts:
        assert abs(table[H]["ghe_h2"][0]) < 0.015, f"GHE bias at H={H}"
        assert table[H]["ghe_h2"][1] < 0.03, f"GHE sd at H={H}"
        assert abs(table[H]["dfa1"][0]) < 0.03, f"DFA1 bias at H={H}"
        assert table[H]["dfa1"][1] < 0.015, f"DFA1 sd at H={H}"
        assert abs(table[H]["aggvar"][0]) < 0.09, f"aggregated-variance bias at H={H}"
        assert abs(table[H]["gph"][0]) < 0.03, f"GPH bias at H={H}"
        # GPH's variance is theoretical, not incidental: pi/sqrt(24 m) with
        # m = floor(n^0.5) = 128 gives 0.0567.
        assert 0.03 < table[H]["gph"][1] < 0.09, f"GPH sd at H={H}"

    # The known R/S upward bias at H = 1/2, stated as a positive assertion.
    assert table[0.5]["rs_raw"][0] > 0.05, "the classic R/S upward bias has vanished"
    assert abs(table[0.5]["rs_corrected"][0]) < 0.03, "Anis-Lloyd/Peters failed at the null"
    # ... and the price of that correction away from the null.
    assert table[0.95]["rs_corrected"][0] < -0.10, "the correction over-corrects at high H"
    assert table[0.2]["rs_corrected"][0] > 0.05, "the correction over-corrects at low H"


# --------------------------------------------------------------------------- #
# Estimator-specific properties
# --------------------------------------------------------------------------- #


def test_normalisation_is_exactly_slope_neutral() -> None:
    r"""Dividing by :math:`\langle|X|^q\rangle` moves the intercept and nothing else.

    The claim in the docstring is checkable, so it is checked: the denominator
    is :math:`\tau`-independent, so it can only add a constant to
    :math:`\log K_q`, and a constant cannot change a slope.
    """
    x = _fbm(0.65, 1)
    on = generalised_hurst(x, q=Q_FBM, normalise=True)
    off = generalised_hurst(x, q=Q_FBM, normalise=False)
    assert np.allclose(on.zeta, off.zeta, atol=1e-12, rtol=0.0)
    intercepts_on = np.array([f.intercept for f in on.fits])
    intercepts_off = np.array([f.intercept for f in off.fits])
    assert np.abs(intercepts_on - intercepts_off).max() > 0.1


def test_anis_lloyd_expected_rs_matches_its_asymptote() -> None:
    r""":math:`\mathbb{E}[(R/S)_n]/\sqrt{n\pi/2} \to 1`, and from below.

    The approach from below is the *cause* of the R/S upward bias: the finite-n
    expectation is smaller than the asymptote, so the empirical log--log slope
    of an iid series exceeds 1/2.
    """
    ratios = np.array(
        [anis_lloyd_expected_rs(n) / math.sqrt(0.5 * math.pi * n) for n in (16, 64, 256, 4096)]
    )
    assert np.all(ratios < 1.0)
    assert np.all(np.diff(ratios) > 0.0)
    assert ratios[-1] > 0.98
    # The n > 340 branch must not jump: continuity across the Stirling switch.
    assert abs(anis_lloyd_expected_rs(340) - anis_lloyd_expected_rs(341)) < 0.05


def test_gph_standard_error_matches_theory() -> None:
    r"""The GPH regression standard error is :math:`\pi/\sqrt{24m}` to within 15%."""
    fit = periodogram_hurst(_fbm(0.5, 2))
    m = fit.n_points
    predicted = math.pi / math.sqrt(24.0 * m)
    assert m == math.floor((N - 1) ** 0.5)
    assert abs(fit.slope_stderr - predicted) / predicted < 0.15


def test_dfa_is_trend_invariant_and_order_costs_bias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Linear-trend invariance is exact; higher detrending orders cost accuracy.

    Adding :math:`at` to the level adds the constant :math:`a/n` to every
    increment, which the profile's mean removal deletes exactly --- so DFA at
    any order is invariant to a linear trend up to float round-off, not merely
    "robust" to it. The price of a higher order is a positive small-scale bias,
    because a degree-``p`` polynomial absorbs a growing share of the genuine
    fluctuation inside each window.
    """
    x = _fbm(0.5, 6)
    t = np.arange(x.size, dtype=np.float64) / x.size
    trended = x + 100.0 * float(np.std(x)) * t

    clean = detrended_fluctuation(x).slope
    assert abs(detrended_fluctuation(trended).slope - clean) < 1e-12

    orders = {p: detrended_fluctuation(x, order=p).slope for p in (1, 2, 3)}
    with capsys.disabled():
        print(
            "\nDFA on fBm(H=0.5): " + "  ".join(f"order {p}: {v:+.4f}" for p, v in orders.items())
        )
    assert orders[1] < orders[2] < orders[3]
    assert abs(orders[1] - 0.5) < 0.02
    assert abs(orders[3] - 0.5) < 0.06


def test_series_kind_level_and_increments_agree() -> None:
    """Handing in the level or its increments must give the same answer."""
    x = _fbm(0.65, 7)
    inc = np.diff(x)
    for fn in (rescaled_range, aggregated_variance, detrended_fluctuation, periodogram_hurst):
        a = fn(x).slope
        b = fn(inc, series_kind="increments").slope
        assert abs(a - b) < 1e-9, fn.__name__
    ghe_level = generalised_hurst(x, q=Q_FBM).zeta
    ghe_inc = generalised_hurst(inc, q=Q_FBM, series_kind="increments").zeta
    # cumsum of diff reconstructs the level only up to its first value and up to
    # accumulated round-off, so this is not bitwise -- but the level enters zeta
    # only through the tau-independent normalisation, so the residual is
    # round-off and nothing else. Measured 2.2e-15; a tolerance of 0.01 here
    # would pass even with the level/increment convention inverted.
    assert np.abs(ghe_level - ghe_inc).max() < 1e-12


def test_structure_surface_matches_the_fitted_exponents() -> None:
    """The exported surface is the same object the estimator fitted."""
    x = _fbm(0.6, 8)
    surface = generalised_hurst_structure(x, q=Q_FBM)
    res = generalised_hurst(x, q=Q_FBM)
    assert res.structure is not None
    assert np.array_equal(surface.scales, res.structure.scales)
    assert np.allclose(surface.values, res.structure.values, rtol=1e-12, atol=0.0)
    assert surface.counts is not None
    assert np.array_equal(surface.counts, res.structure.counts)  # type: ignore[arg-type]
    # Refitting the exported surface by hand reproduces zeta(2) to plotting accuracy.
    i2 = int(np.argmin(np.abs(surface.q - 2.0)))
    slope = float(
        np.polyfit(
            np.log(surface.scales), np.log(surface.values[i2]), 1, w=np.sqrt(surface.counts)
        )[0]
    )
    assert abs(slope - res.zeta[i2]) < 0.02


def test_tau_max_averaging_reports_a_cut_off_spread() -> None:
    """The averaging device must actually run and must widen the error bar."""
    x = _fbm(0.7, 9)
    avg = generalised_hurst(x, q=Q_FBM, average_tau_max=True)
    single = generalised_hurst(x, q=Q_FBM, average_tau_max=False)
    assert avg.provenance is not None and single.provenance is not None
    assert avg.provenance.diagnostics["averaged_over_tau_max"] is True
    assert single.provenance.diagnostics["averaged_over_tau_max"] is False
    assert len(avg.provenance.diagnostics["tau_max_cutoffs"]) > 5
    assert max(float(v) for v in avg.provenance.diagnostics["cutoff_spread"]) > 0.0
    assert avg.zeta_stderr is not None and single.zeta_stderr is not None
    assert np.all(avg.zeta_stderr >= single.zeta_stderr - 1e-12)

    explicit = generalised_hurst(x, q=Q_FBM, tau_max_grid=[10, 14, 19])
    assert explicit.provenance is not None
    assert len(explicit.provenance.diagnostics["tau_max_cutoffs"]) == 3


def test_select_range_picks_a_sub_window() -> None:
    """Automatic scaling-range selection restricts the window and records how."""
    x = _fbm(0.6, 10)
    res = generalised_hurst(x, q=Q_FBM, tau_max=64, select_range=True)
    assert res.scaling_range is not None
    assert res.scaling_range.selection.startswith("auto-")
    assert res.scaling_range.n_scales <= len(res.scaling_range.mask)
    assert abs(res.h2 - 0.6) < 0.05


def test_consensus_panel_agrees_on_fbm(capsys: pytest.CaptureFixture[str]) -> None:
    """On exact fBm the five-member panel agrees to well under the 0.1 alarm level."""
    spreads = []
    for H in (0.4, 0.5, 0.6):
        panel = hurst_consensus(_fbm(H, 11))
        spreads.append(panel["spread"])
        with capsys.disabled():
            print(
                f"\nconsensus at H={H}: "
                + "  ".join(
                    f"{k}={panel[k]:+.4f}"
                    for k in ("ghe_h2", "rescaled_range", "aggregated_variance", "dfa1", "gph")
                )
                + f"  spread={panel['spread']:.4f}"
            )
        assert abs(panel["mean"] - H) < 0.05
        # 0.20 not 0.10: on a single realisation the GPH member alone has
        # sd ~= 0.06 (its theoretical pi/sqrt(24m)), and at H=0.4 it lands
        # 2.7 sigma high, giving an observed spread of 0.153 on data that is
        # exactly self-similar. The time-domain members agree far more tightly.
        assert panel["spread"] < 0.20
        assert panel["spread_time_domain"] < 0.08
        assert panel["min"] <= panel["median"] <= panel["max"]
    assert min(spreads) < 0.06


def test_consensus_flags_a_series_with_no_single_exponent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A regime switch has no single H, and the panel says so by disagreeing."""
    left = fbm(N // 2, 0.2, seed=[BASE_SEED, 1])
    right = fbm(N // 2, 0.9, seed=[BASE_SEED, 2])
    glued = np.concatenate([left, left[-1] + right])
    panel = hurst_consensus(glued)
    with capsys.disabled():
        print(
            f"\nconsensus on a spliced 0.2|0.9 series: spread={panel['spread']:.4f} "
            f"time-domain={panel['spread_time_domain']:.4f}"
        )
    assert panel["spread"] > 0.20
    assert panel["spread_time_domain"] > 0.10


def test_rejects_bad_input() -> None:
    """Loud failures, not quiet NaNs."""
    with pytest.raises(ValueError):
        generalised_hurst(np.ones(1024), q=Q_FBM)  # constant series
    with pytest.raises(ValueError):
        generalised_hurst(_fbm(0.5, 0), q=np.array([1.0]))  # q grid too small
    with pytest.raises(ValueError):
        generalised_hurst(_fbm(0.5, 0), q=Q_FBM, series_kind="nonsense")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generalised_hurst(_fbm(0.5, 0), q=Q_FBM, scales=[2, 3])  # too few scales for a slope
    with pytest.raises(ValueError):
        detrended_fluctuation(_fbm(0.5, 0), order=0)
    with pytest.raises(ValueError):
        periodogram_hurst(_fbm(0.5, 0), n_frequencies=2)
    with pytest.raises(ValueError):
        anis_lloyd_expected_rs(1)
