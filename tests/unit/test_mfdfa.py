r"""Validation of :mod:`mmar.estimators.mfdfa` against closed-form oracles.

Every tolerance below is justified in the test that uses it, and every number
quoted in a comment was observed by running this file --- not inferred.

Observed on 2026-07-27 (numpy 2.x, macOS arm64), all seeds fixed:

===========================================  ==========================
Quantity                                     Observed
===========================================  ==========================
fGn N=16384, bias of :math:`h(2)`            +0.0020 / -0.0025 / -0.0077
   at :math:`H=0.3/0.5/0.7`
spread :math:`\max h-\min h` over            0.027 / 0.025 / 0.034
   :math:`q\in[-5,5]` (true: 0)
mean bias of :math:`h(q)`, :math:`q<0`       +0.0148 / +0.0088 / +0.0085
mean bias of :math:`h(q)`, :math:`q>0`       +0.0008 / -0.0037 / -0.0091
binomial cascade, dyadic scales,             0.070
   :math:`\max_q|\hat\tau-\tau|`
   over :math:`q\in[-5,5]`
same, restricted to :math:`|q|\le3`          0.042
:math:`\hat\tau(1)` (true 0)                 +0.014
:math:`\max_q|\hat\alpha(q)-\alpha(q)|`      0.014
spectrum width, :math:`q\in[-5,5]`           0.449
spectrum width, :math:`q\in[0,5]`            0.224
===========================================  ==========================
"""

from __future__ import annotations

import numpy as np
import pytest

from mmar.core.legendre import concavity_defect, legendre_parametric, spectrum_from_exponents
from mmar.core.types import FitMethod, StructureFunctions, dyadic_scales
from mmar.core.validate import DataQualityError
from mmar.estimators.mfdfa import dfa, mfdfa, mfdfa_from_structure, mfdfa_structure
from mmar.models import theory
from mmar.models.cascade import deterministic_cascade
from mmar.models.fbm import fbm, fgn

SEED = 20260727
N_FBM = 16384
Q = np.linspace(-5.0, 5.0, 41)

CASCADE_M = (0.6, 0.4)
CASCADE_KMAX = 14
"""2**14 = 16384 cells: enough for eight dyadic scales inside [2**5, 2**12]."""


def _cascade_scales(n: int) -> np.ndarray:
    r"""Dyadic scales, the only grid on which a :math:`b`-adic cascade is exactly
    self-similar.

    A deterministic binomial measure is self-similar under :math:`\delta\to
    \delta/2` and *not* under any other ratio, so at non-dyadic :math:`s` the
    partition cuts across the cascade's own cells and :math:`F_q(s)` carries a
    log-periodic (lacunarity) oscillation of period :math:`\ln 2`. Sampling it
    on a generic geometric grid aliases that oscillation into the residuals:
    observed :math:`\max_q|\hat\tau-\tau|` degrades from 0.070 (dyadic) to
    0.566 (24-point geometric grid) at otherwise identical settings. The
    oscillation is a property of the *object*, not a defect of the estimator,
    and the right response is to sample on its period.
    """
    return dyadic_scales(n, j_min=5, j_max=int(np.log2(n)) - 2)


# --------------------------------------------------------------------------- #
# (a) Monofractal recovery
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.stochastic
@pytest.mark.parametrize(("H", "h2_tol"), [(0.3, 0.02), (0.5, 0.02), (0.7, 0.02)])
def test_fbm_recovers_flat_h_and_hurst(H: float, h2_tol: float) -> None:
    r"""On exact fGn, :math:`h(q)` must be flat at :math:`H` and :math:`\zeta=qH`.

    Tolerances. ``h2_tol=0.02`` on :math:`h(2)`: the observed biases are
    +0.0020, -0.0025, -0.0077 at :math:`H=0.3,0.5,0.7`, i.e. under half a
    percent absolute, and 0.02 leaves room for the sampling variability of a
    single :math:`N=16384` path (the fitted standard error on the slope is
    ~0.004). ``spread<0.06``: the true spread is exactly zero, so this bounds
    the *spurious multifractality* the estimator manufactures on a monofractal;
    observed 0.027 / 0.025 / 0.034 --- and it is dominated by the negative
    branch, quantified separately below.
    """
    x = fgn(N_FBM, H, seed=SEED)
    e = mfdfa(x, q=Q)

    h = e.hq
    finite = np.isfinite(h)
    spread = float(h[finite].max() - h[finite].min())

    assert abs(e.h2 - H) < h2_tol, f"h(2)={e.h2:.4f} vs H={H}"
    assert spread < 0.06, f"spurious multifractality: spread(h)={spread:.4f}"
    assert e.is_monofractal(tol=0.06)
    # zeta(q) = q*H is the SIGNAL-view anchor of the contract.
    assert float(np.max(np.abs(e.zeta - Q * H))) < 0.06 * float(np.max(np.abs(Q)))


@pytest.mark.theory
@pytest.mark.stochastic
def test_negative_q_overestimates_h_on_fbm_and_the_size_is_reported() -> None:
    r"""The documented :math:`q<0` bias, measured rather than asserted away.

    MFDFA's negative moments are tilted towards the smallest window
    fluctuations, which are undersampled on a finite record, and the deficit
    shrinks with :math:`s`; the fitted slope therefore comes out too steep.

    Observed mean bias of :math:`h(q)` over :math:`q\in[-5,-1]`, :math:`N=16384`:

    ======  ==========  ==========
    H       q < 0       q > 0
    ======  ==========  ==========
    0.3     +0.0159     +0.0008
    0.5     +0.0099     -0.0037
    0.7     +0.0101     -0.0091
    ======  ==========  ==========

    The test asserts the *sign* (positive on the left branch for all three
    :math:`H`) and an upper bound of 0.03, roughly twice the worst observed
    value. It does not assert that the bias is small: it is three to twenty
    times the positive-:math:`q` bias, and on a 4096-point record it grows.
    """
    biases = {}
    for H in (0.3, 0.5, 0.7):
        e = mfdfa(fgn(N_FBM, H, seed=SEED), q=Q)
        h = e.hq
        left = (e.q <= -1.0) & np.isfinite(h)
        biases[H] = float(np.mean(h[left]) - H)

    assert all(b > 0.0 for b in biases.values()), f"expected upward bias, got {biases}"
    assert max(biases.values()) < 0.03, f"negative-q bias exceeded its documented size: {biases}"


@pytest.mark.stochastic
def test_dfa_returns_the_hurst_exponent() -> None:
    """The :math:`q=2` special case is the classical DFA Hurst estimator.

    Observed slopes at :math:`N=16384`, seed 11: 0.3041, 0.5037, 0.7043 with
    standard errors 0.0023--0.0042 and :math:`R^2>0.998`.
    """
    for H in (0.3, 0.5, 0.7):
        fit = dfa(fgn(N_FBM, H, seed=11))
        assert abs(fit.slope - H) < 0.02, f"DFA slope {fit.slope:.4f} vs H={H}"
        assert fit.r_squared > 0.99
        lo, hi = fit.slope_ci
        assert lo < hi, "the confidence interval must not be inverted (legacy defect D4)"
        assert fit.slope_stderr > 0.0

    # dfa is the q=2 row of the full estimator, not an independent code path.
    x = fgn(4096, 0.6, seed=3)
    e = mfdfa(x, q=np.array([1.0, 2.0, 3.0]))
    assert dfa(x).slope == pytest.approx(e.h2, abs=1e-12)


@pytest.mark.stochastic
def test_integration_trap_shifts_h_by_one() -> None:
    r"""Double-integrating a motion shifts :math:`h` by one; ``integrate=False`` fixes it.

    Observed :math:`h(2)` on the fBm path itself: 0.3020 / 0.4975 / 0.6923 with
    ``integrate=False`` (correct), and 1.2631 / 1.4572 / 1.6471 with
    ``integrate=True``. The shift is :math:`+0.96` rather than exactly
    :math:`+1`: DFA1 applied to a twice-integrated process is close to its
    :math:`h\le m+1=2` saturation ceiling and drags the estimate down. That
    residual is *not* the point --- the point is that the number moves by
    roughly one and still looks like a legitimate exponent.
    """
    for H in (0.3, 0.5, 0.7):
        path = fbm(N_FBM, H, seed=SEED)
        correct = mfdfa(path, q=Q, integrate=False)
        trapped = mfdfa(path, q=Q, integrate=True)
        assert abs(correct.h2 - H) < 0.02
        assert abs(trapped.h2 - (H + 1.0)) < 0.10
        assert trapped.h2 - correct.h2 > 0.9


# --------------------------------------------------------------------------- #
# (b) Multifractal recovery
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_binomial_cascade_recovers_tau() -> None:
    r"""Deterministic binomial measure :math:`(0.6,0.4)`, :math:`k_{\max}=14`.

    The oracle is :math:`\tau(q)=-\log_2(m_0^q+m_1^q)`
    (:func:`mmar.models.theory.cascade_tau`), exact at every dyadic scale.

    Tolerance. ``0.10`` over the whole grid :math:`q\in[-5,5]`, ``0.06`` over
    :math:`|q|\le3`; observed 0.070 and 0.042. The residual is not sampling
    noise --- the cascade is deterministic and the estimate is bit-exact --- it
    is the estimator's own finite-scale bias, concentrated at :math:`|q|=5`
    where :math:`F_q` is controlled by the extreme windows. Loosening the
    tolerance to make a larger :math:`|q|` pass would be dishonest: the right
    statement is that MFDFA reproduces this cascade's :math:`\tau` to about
    0.04 for :math:`|q|\le3` and about 0.07 out to :math:`|q|=5`.
    """
    mu = deterministic_cascade(CASCADE_KMAX, CASCADE_M)
    e = mfdfa(mu, q=Q, scales=_cascade_scales(mu.size))

    tau_true = theory.cascade_tau(Q, CASCADE_M)
    err = np.abs(e.tau - tau_true)
    inner = np.abs(Q) <= 3.0

    assert float(err.max()) < 0.10, f"max|dtau| = {float(err.max()):.4f}"
    assert float(err[inner].max()) < 0.06, f"max|dtau| (|q|<=3) = {float(err[inner].max()):.4f}"
    # zeta = tau + d must hold identically; this guards a convention slip.
    np.testing.assert_allclose(e.zeta, e.tau + e.d_support, rtol=0, atol=1e-12)


@pytest.mark.theory
def test_binomial_cascade_recovers_falpha() -> None:
    r"""The singularity spectrum, both parametrically and by Legendre--Fenchel.

    Tolerances. :math:`\alpha(q)=\tau'(q)` to 0.03 (observed 0.014); the
    Legendre--Fenchel :math:`f(\alpha)` to an RMS of 0.06 against
    :math:`H_2\big((\alpha_{\max}-\alpha)/\Delta\alpha\big)`
    (:func:`mmar.models.theory.cascade_falpha`, observed RMS 0.036, max 0.075).
    :math:`f` is compared only where the true spectrum is finite, since outside
    :math:`[\alpha_{\min},\alpha_{\max}]` the oracle is :math:`-\infty` while
    any estimate is a finite number.
    """
    mu = deterministic_cascade(CASCADE_KMAX, CASCADE_M)
    e = mfdfa(mu, q=Q, scales=_cascade_scales(mu.size))

    _, alpha_hat, _ = legendre_parametric(e.q, e.zeta, d_support=e.d_support)
    alpha_true = theory.cascade_alpha(Q, CASCADE_M)
    assert float(np.max(np.abs(alpha_hat - alpha_true))) < 0.03

    spectrum = spectrum_from_exponents(e, method="fenchel", n_alpha=201)
    f_true = theory.cascade_falpha(spectrum.alpha, CASCADE_M)
    ok = np.isfinite(f_true) & np.isfinite(spectrum.f_alpha)
    assert int(ok.sum()) > 100
    rms = float(np.sqrt(np.mean((spectrum.f_alpha[ok] - f_true[ok]) ** 2)))
    assert rms < 0.06, f"rms f(alpha) error = {rms:.4f}"

    # The apex sits at the support dimension and the estimate stays concave.
    assert spectrum.descriptors().f_max == pytest.approx(1.0, abs=0.02)
    assert concavity_defect(e.q, e.zeta) < 1e-9


@pytest.mark.theory
def test_cascade_is_distinguished_from_the_monofractal_null() -> None:
    """Multifractality must not be a property the estimator invents.

    The cascade's recovered spread of :math:`h(q)` must exceed the fGn null's
    by a wide margin: observed 0.30 against 0.025.
    """
    mu = deterministic_cascade(CASCADE_KMAX, CASCADE_M)
    e_multi = mfdfa(mu, q=Q, scales=_cascade_scales(mu.size))
    e_mono = mfdfa(fgn(N_FBM, 0.5, seed=SEED), q=Q)

    h_multi = e_multi.hq[np.isfinite(e_multi.hq)]
    h_mono = e_mono.hq[np.isfinite(e_mono.hq)]
    spread_multi = float(h_multi.max() - h_multi.min())
    spread_mono = float(h_mono.max() - h_mono.min())

    assert not e_multi.is_monofractal(tol=0.06)
    assert e_mono.is_monofractal(tol=0.06)
    assert spread_multi > 5.0 * spread_mono, f"{spread_multi:.4f} vs {spread_mono:.4f}"


# --------------------------------------------------------------------------- #
# (c) Convention anchors
# --------------------------------------------------------------------------- #


def test_tau_zero_is_minus_d_identically() -> None:
    r""":math:`\tau(0)=-d`.

    For MFDFA this is an algebraic identity, :math:`\tau(0)=0\cdot h(0)-d`, so
    it is checked to machine precision rather than to a tolerance --- and it is
    therefore *not* evidence that the estimate is sound. What it does guard is
    the convention: v1 reported :math:`\hat\tau(0)=-1.017` because its
    increment counts drifted with :math:`\Delta t` (legacy defect D6), and any
    future refactor that made :math:`\zeta(0)\ne0` would be a convention slip.
    """
    e = mfdfa(fgn(4096, 0.55, seed=5), q=Q)
    i0 = int(np.argmin(np.abs(e.q)))
    assert e.q[i0] == 0.0
    assert e.zeta[i0] == 0.0
    assert e.tau[i0] == -1.0
    assert e.zeta_stderr is not None and e.zeta_stderr[i0] == 0.0

    e2 = mfdfa(fgn(4096, 0.55, seed=5), q=Q, d_support=2.0)
    assert e2.tau[i0] == -2.0


@pytest.mark.theory
def test_tau_one_vanishes_for_a_conservative_measure() -> None:
    r""":math:`\tau(1)=0` for a mass-conserving measure.

    The binomial cascade conserves mass pathwise, :math:`\sum_i m_i=1`, which
    is exactly the statement :math:`\tau(1)=0`, equivalently :math:`h(1)=1`.
    Observed :math:`\hat\tau(1)=+0.0141`; the tolerance is 0.03. Unlike
    :math:`\tau(0)`, this one is a genuine estimate and is the sharpest
    single-number check available on a measure.
    """
    mu = deterministic_cascade(CASCADE_KMAX, CASCADE_M)
    e = mfdfa(mu, q=Q, scales=_cascade_scales(mu.size))
    tau_1 = float(np.interp(1.0, e.q, e.tau))
    h_1 = float(np.interp(1.0, e.q, e.zeta))  # h(1) = zeta(1)/1
    assert abs(tau_1) < 0.03, f"tau(1) = {tau_1:+.4f}"
    assert abs(h_1 - 1.0) < 0.03, f"h(1) = {h_1:.4f}"


# --------------------------------------------------------------------------- #
# (d) Determinism
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
def test_determinism_is_bitwise() -> None:
    """Identical input and parameters must give bit-identical exponents."""
    x = fgn(4096, 0.6, seed=101)
    a = mfdfa(x, q=Q, seed=7)
    b = mfdfa(x, q=Q, seed=7)

    assert np.array_equal(a.zeta, b.zeta)
    assert np.array_equal(a.structure.values, b.structure.values)  # type: ignore[union-attr]
    assert [f.slope for f in a.fits] == [f.slope for f in b.fits]

    # The seed is inert: MFDFA draws no randomness, and only the audit record moves.
    c = mfdfa(x, q=Q, seed=8)
    assert np.array_equal(a.zeta, c.zeta)


# --------------------------------------------------------------------------- #
# (e) Provenance
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
def test_provenance_root_is_stable_and_parameter_sensitive() -> None:
    """The content address must be a pure function of the causally identifying inputs."""
    x = fgn(4096, 0.6, seed=202)
    base = mfdfa(x, q=Q)
    assert base.provenance is not None
    assert base.provenance.root == mfdfa(x, q=Q).provenance.root  # type: ignore[union-attr]

    variants = {
        "order": mfdfa(x, q=Q, order=2),
        "integrate": mfdfa(x, q=Q, integrate=False),
        "fit_method": mfdfa(x, q=Q, fit_method=FitMethod.OLS),
        "select_range": mfdfa(x, q=Q, select_range=True),
        "d_support": mfdfa(x, q=Q, d_support=2.0),
        "seed": mfdfa(x, q=Q, seed=1),
        "q": mfdfa(x, q=np.linspace(-4.0, 4.0, 41)),
        "scales": mfdfa(x, q=Q, scales=np.array([32, 64, 128, 256, 512])),
        "x": mfdfa(fgn(4096, 0.6, seed=203), q=Q),
    }
    roots = {k: v.provenance.root for k, v in variants.items()}  # type: ignore[union-attr]
    assert base.provenance.root not in roots.values(), (
        f"a parameter change left the root fixed: {roots}"
    )
    assert len(set(roots.values())) == len(roots), f"two distinct calls collided: {roots}"

    diagnostics = dict(base.provenance.diagnostics)
    assert diagnostics["n_samples"] == 4096
    assert diagnostics["degenerate_windows"] == 0
    assert "scaling_range" in diagnostics
    assert base.provenance.params["order"] == 1
    assert base.provenance.params["scales_used"]


def test_degenerate_windows_are_dropped_and_counted() -> None:
    r"""A flat stretch would send every :math:`q<0` moment to infinity.

    ``p[1000:1400]`` is set constant, so every window falling inside it has an
    algebraically zero detrended residual. Observed: 208 windows dropped across
    the 24 default scales, and the negative moments stay finite.
    """
    p = fbm(4096, 0.5, seed=1).copy()
    p[1000:1400] = p[1000]

    structure = mfdfa_structure(p, q=Q, integrate=False)
    assert structure.provenance is not None
    assert structure.provenance.diagnostics["degenerate_windows"] > 100
    assert np.all(np.isfinite(structure.values))
    assert np.all(structure.values > 0.0)
    assert structure.counts is not None
    assert np.all(structure.counts >= 4)


# --------------------------------------------------------------------------- #
# (f) Negative q — direct regression test for legacy defect D2
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_negative_q_widens_the_recovered_alpha_range() -> None:
    r"""The right branch of :math:`f(\alpha)` exists only for :math:`q<0`.

    v1 defaulted to :math:`q\in[0,5]` and recovered 24% of the binomial
    cascade's true support. Here, on the same measure and identical settings
    apart from the moment grid:

    ==================  ============  =================
    grid                width         :math:`\alpha_{\max}`
    ==================  ============  =================
    :math:`[0,5]`       0.224         1.043
    :math:`[-5,5]`      0.449         1.268
    true (:math:`b=2`)  0.585         1.322
    ==================  ============  =================

    Negative :math:`q` doubles the recovered width and is the *only* way to
    reach :math:`\alpha>\alpha_0`. Neither grid recovers the full 0.585,
    because the endpoints of the support are the limits
    :math:`q\to\pm\infty`; that is a property of a bounded moment window, not a
    defect, and it is why the test compares the two grids rather than either of
    them to the exact endpoints.
    """
    mu = deterministic_cascade(CASCADE_KMAX, CASCADE_M)
    scales = _cascade_scales(mu.size)

    full = spectrum_from_exponents(
        mfdfa(mu, q=np.linspace(-5.0, 5.0, 41), scales=scales), method="fenchel"
    ).descriptors()
    right_only = spectrum_from_exponents(
        mfdfa(mu, q=np.linspace(0.0, 5.0, 21), scales=scales), method="fenchel"
    ).descriptors()

    alpha_min_true, alpha_max_true = theory.cascade_alpha_range(CASCADE_M)

    assert full.width > 1.8 * right_only.width
    assert full.alpha_max > right_only.alpha_max + 0.15
    assert full.alpha_max < alpha_max_true, "cannot exceed the true support"
    assert full.alpha_min > alpha_min_true
    # The q >= 0 grid cannot see beyond the apex, by construction.
    assert right_only.alpha_max == pytest.approx(right_only.alpha_0, abs=0.02)


# --------------------------------------------------------------------------- #
# Structure surface, refitting, and guards
# --------------------------------------------------------------------------- #


@pytest.mark.stochastic
def test_structure_surface_is_refittable_without_recomputation() -> None:
    """``mfdfa_structure`` + ``mfdfa_from_structure`` must reproduce ``mfdfa``."""
    x = fgn(4096, 0.65, seed=13)
    direct = mfdfa(x, q=Q)
    structure = mfdfa_structure(x, q=Q)
    refit = mfdfa_from_structure(structure)

    np.testing.assert_array_equal(direct.zeta, refit.zeta)
    assert structure.values.shape == (Q.size, structure.scales.size)
    assert structure.counts is not None
    # Window counts must fall monotonically with scale: that is what the WLS
    # weighting exists to exploit (legacy defect D5).
    assert np.all(np.diff(structure.counts) <= 0)

    # The point of keeping the surface: refit a strictly narrower band of scales
    # without redoing the O(N * n_s) detrending.
    keep = structure.scales <= 200.0
    narrower = StructureFunctions(
        q=structure.q,
        scales=structure.scales[keep],
        values=structure.values[:, keep],
        counts=structure.counts[keep],
        provenance=structure.provenance,
    )
    band = mfdfa_from_structure(narrower)
    assert band.scaling_range is not None
    assert band.scaling_range.s_max == 185.0
    assert band.scaling_range.n_scales < structure.scales.size
    assert abs(band.h2 - direct.h2) < 0.05

    auto = mfdfa_from_structure(structure, select_range=True)
    assert auto.scaling_range is not None
    assert auto.scaling_range.selection.startswith("auto")
    assert 5 <= auto.scaling_range.n_scales <= structure.scales.size


@pytest.mark.stochastic
def test_result_is_fully_populated() -> None:
    """The contract: every field the standard signature promises is set."""
    x = fgn(4096, 0.6, seed=17)
    e = mfdfa(x, q=Q)

    assert e.estimator == "mfdfa"
    assert e.provenance is not None
    assert e.structure is not None and e.structure.counts is not None
    assert e.scaling_range is not None and e.scaling_range.decades > 1.5
    assert len(e.fits) == Q.size
    assert e.zeta_stderr is not None and e.zeta_stderr.shape == Q.shape
    # se(zeta) = |q| se(h): the absolute value is not decoration. Without it every
    # q < 0 reports a negative standard error, which no downstream consumer checks.
    assert np.all(e.zeta_stderr >= 0.0), "standard errors must be non-negative for q < 0"
    h_se = np.array([f.slope_stderr for f in e.fits])
    np.testing.assert_allclose(e.zeta_stderr, np.abs(Q) * h_se, rtol=0, atol=1e-15)
    assert e.zeta_ci is not None
    lo, hi = e.zeta_ci
    assert np.all(lo <= hi), "confidence intervals must not invert for q < 0"
    assert np.all(np.isfinite(e.zeta))
    assert all(f.method is FitMethod.WLS for f in e.fits)
    assert all(f.weights is not None for f in e.fits)


def test_q_zero_uses_the_logarithmic_limit() -> None:
    r""":math:`F_0(s)` must be the geometric mean, not 1.

    The naive substitution :math:`q=0` in
    :math:`(\langle (F^2)^{q/2}\rangle)^{1/q}` gives :math:`1^\infty=1` at every
    scale, so :math:`h(0)` would be exactly zero. Here :math:`F_0(s)` must
    instead sit between the neighbouring moments and its slope must be a
    plausible Hurst exponent.
    """
    x = fgn(4096, 0.7, seed=23)
    grid = np.array([-0.5, 0.0, 0.5])
    structure = mfdfa_structure(x, q=grid)
    f0 = structure.values[1]

    assert not np.allclose(f0, 1.0)
    # F_q is non-decreasing in q (a generalised mean), strictly so here.
    assert np.all(structure.values[0] < f0)
    assert np.all(f0 < structure.values[2])

    e = mfdfa(x, q=grid)
    h0 = e.fits[1].slope
    assert 0.6 < h0 < 0.8, f"h(0) = {h0:.4f}"

    # Continuity: the two branches agree as q -> 0, and the residual gap is the
    # predicted (q/8)*Var(ln F^2). At q = 1e-4 with Var(ln F^2) ~ 0.7 that is
    # ~9e-6 relative, which is exactly what is observed -- so the tolerance is a
    # measurement of the expansion, not a fudge.
    near = mfdfa_structure(x, q=np.array([-1e-4, 0.0, 1e-4]))
    np.testing.assert_allclose(near.values[0], near.values[1], rtol=2e-5)
    np.testing.assert_allclose(near.values[2], near.values[1], rtol=2e-5)
    gap = float(np.max(np.abs(near.values[2] / near.values[1] - 1.0)))
    assert 1e-7 < gap < 2e-5, f"q->0 branch gap {gap:.2e}"


def test_both_ends_of_the_series_are_used() -> None:
    r"""Step 2's backward sweep must actually double the window count.

    With :math:`N=1000` and :math:`s=30` there are :math:`N_s=33` forward
    windows and 10 discarded tail samples; Kantelhardt's second sweep recovers
    them, giving 66 windows. When :math:`s\mid N` the two tilings coincide and
    the duplicate is dropped, so the count is :math:`N_s`, not :math:`2N_s`.
    """
    x = fgn(1000, 0.5, seed=29)
    structure = mfdfa_structure(x, q=Q, scales=np.array([20, 25, 30]))
    assert structure.counts is not None
    assert int(structure.counts[0]) == 50  # 1000 = 20 * 50 exactly -> one sweep
    assert int(structure.counts[1]) == 40  # 1000 = 25 * 40 exactly -> one sweep
    assert int(structure.counts[2]) == 66  # floor(1000/30) = 33 -> two sweeps


def test_scale_guards_reject_inadmissible_grids() -> None:
    """Scales that cannot support a fluctuation estimate are refused, not fitted."""
    x = fgn(4096, 0.5, seed=31)

    with pytest.raises(DataQualityError):
        # s = order + 1 makes the detrending polynomial interpolate the window.
        mfdfa(x, q=Q, order=1, scales=np.array([2, 2, 2]))
    with pytest.raises(DataQualityError):
        # Fewer than four windows at every scale.
        mfdfa(x, q=Q, scales=np.array([2048, 3000, 4000]))
    with pytest.raises(DataQualityError):
        mfdfa(np.zeros(4096), q=Q)
    with pytest.raises(DataQualityError):
        mfdfa(fgn(100, 0.5, seed=33), q=Q)
    with pytest.raises(ValueError, match="order must be non-negative"):
        mfdfa(x, q=Q, order=-1)


@pytest.mark.stochastic
def test_higher_order_detrending_removes_a_polynomial_trend() -> None:
    r"""DFA-:math:`m` must annihilate a superimposed polynomial of degree :math:`\le m`.

    A quadratic trend added to the *profile* leaves DFA1 badly biased (the
    trend dominates :math:`F^2` at large :math:`s`) but is removed exactly by
    DFA2. This is the entire reason MFDFA exists.
    """
    H = 0.5
    path = fbm(N_FBM, H, seed=37)
    t = np.linspace(0.0, 1.0, path.size)
    contaminated = path + 3000.0 * float(np.std(path)) * t**2

    clean = mfdfa(path, q=Q, integrate=False, order=1).h2
    biased = mfdfa(contaminated, q=Q, integrate=False, order=1).h2
    corrected = mfdfa(contaminated, q=Q, integrate=False, order=2).h2

    # Observed: clean DFA1 0.4848, contaminated DFA1 0.7096, contaminated DFA2
    # 0.47757212548667 -- identical to the clean DFA2 value to 1e-15, i.e. the
    # quadratic is annihilated exactly rather than merely attenuated.
    assert abs(clean - H) < 0.02
    assert biased - H > 0.15, f"the quadratic trend should bias DFA1 upward, got {biased:.4f}"
    assert abs(corrected - H) < 0.03, f"DFA2 should remove it, got {corrected:.4f}"
    assert corrected == pytest.approx(mfdfa(path, q=Q, integrate=False, order=2).h2, abs=1e-12)
