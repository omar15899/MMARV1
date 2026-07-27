r"""Validation of :mod:`mmar.estimators.wtmm` against closed-form oracles.

Every tolerance below is stated with the number actually observed when the test
was written, so a future regression is visible as a shrinking margin rather
than as a silently loosened bound.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.regression import loglog_fit, weighted_by_counts
from mmar.core.types import FitMethod
from mmar.estimators.wtmm import (
    WTMMSkeleton,
    cwt_dog,
    line_exponent,
    maxima_lines,
    wtmm,
    wtmm_structure,
)
from mmar.models import theory
from mmar.models.cascade import deterministic_cascade, trading_time
from mmar.models.fbm import fbm

N_FBM = 2**14
SEEDS = (20250727, 20250728, 20250729)
Q = np.linspace(-5.0, 5.0, 41)

BINOMIAL = (0.6, 0.4)
CASCADE_KMAX = 14
CASCADE_SCALES = np.array([8, 16, 32, 64, 128, 256], dtype=np.int64)
"""Dyadic. The binomial measure is exactly 2-adic, so its partition function
carries a log-periodic (lacunary) prefactor of period :math:`\\log 2`; a
non-dyadic geometric grid samples that oscillation at drifting phase and the
local slope of :math:`\\log Z` swings between :math:`0` and :math:`-2` at
:math:`q=0`. Dyadic scales sample it at fixed phase, which is why WTMM papers
on deterministic cascades always use them. The window stops at :math:`n/64`:
the coarse-scale line count collapses and with it the accuracy (see the module
docstring)."""


@pytest.fixture(scope="module")
def cascade_primitive() -> np.ndarray:
    """Devil's staircase of the binomial measure: 16 385 points, exact."""
    return trading_time(deterministic_cascade(CASCADE_KMAX, BINOMIAL))


# --------------------------------------------------------------------------- #
# The transform itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("n_vanishing", "degree"), [(2, 1), (3, 1), (3, 2)])
def test_cwt_annihilates_low_order_polynomials(n_vanishing: int, degree: int) -> None:
    r"""A DOG wavelet with :math:`n_\psi` vanishing moments kills polynomials of
    degree :math:`< n_\psi`. Observed interior maximum: :math:`<10^{-15}`."""
    t = np.linspace(0.0, 1.0, 4096)
    x = t**degree
    coeffs = cwt_dog(x, np.array([16.0, 64.0]), n_vanishing=n_vanishing)
    interior = coeffs[:, 1024:3072]
    assert np.abs(interior).max() < 1e-12


def test_cwt_shape_and_validation() -> None:
    x = fbm(1024, 0.5, seed=1)
    assert cwt_dog(x, [8.0, 16.0, 32.0]).shape == (3, 1024)
    with pytest.raises(ValueError, match="strictly positive"):
        cwt_dog(x, [0.0, 8.0])
    with pytest.raises(ValueError, match="n_vanishing"):
        cwt_dog(x, [8.0], n_vanishing=0)


# --------------------------------------------------------------------------- #
# (a) Monofractal recovery
# --------------------------------------------------------------------------- #


@pytest.mark.theory
@pytest.mark.stochastic
@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_fbm_flat_hq_on_the_positive_branch(H: float) -> None:
    r"""On exact fBm, :math:`h(q)=\zeta(q)/q` is flat and equals :math:`H`.

    The claim is made on :math:`q\ge 1` only, and deliberately so: a monofractal
    has no right branch of :math:`f(\alpha)`, so :math:`q<0` probes exponents
    that do not exist and diverges by construction. That divergence is itself
    tested, quantitatively, in
    :func:`test_fbm_negative_q_follows_the_pathology_law`.

    Observed over 24 seeds at :math:`n=2^{14}`, worst case of each statistic:

    ======  ===============  ==========================
    ``H``   :math:`|h_2-H|`  spread of :math:`h(q\ge1)`
    ======  ===============  ==========================
    0.3     0.088            0.093
    0.5     0.052            0.073
    0.7     0.034            0.057
    ======  ===============  ==========================

    The bias is positive and largest for small :math:`H`. **The margin is
    thin**: at :math:`H=0.3` the worst of 24 seeds is 0.088 against the 0.09
    bound below, so this test's pass is a property of the three fixed seeds,
    not a probabilistic guarantee. It is left at 0.09 deliberately --- the
    honest statement is that WTMM's :math:`h(2)` is off by up to 0.09 at
    :math:`H=0.3`, and a looser bound would hide that. The mean bias, which is
    the reproducible quantity, is pinned separately in
    :func:`test_positive_q_bias_is_the_running_supremum`.
    """
    biases = []
    spreads = []
    for seed in SEEDS:
        res = wtmm(fbm(N_FBM, H, seed=seed), q=Q)
        h_pos = res.hq[res.q >= 1.0]
        biases.append(res.h2 - H)
        spreads.append(float(h_pos.max() - h_pos.min()))
        assert abs(res.h2 - H) < 0.09, f"h(2)={res.h2:.4f} at H={H}, seed={seed}"
        assert h_pos.max() - h_pos.min() < 0.12
    assert np.mean(biases) > 0.0, "the documented sign of the WTMM h(2) bias flipped"
    assert max(spreads) < 0.12


def _tau2_from(skeleton: WTMMSkeleton, field: np.ndarray) -> float:
    """WLS slope of ``log sum(field**2)`` vs ``log a``, on the estimator's own grid.

    Reproduces exactly what :func:`wtmm` does at :math:`q=2`, but lets the
    caller substitute the per-node quantity that is raised to the power.
    """
    keep = np.nonzero(skeleton.n_lines >= 4)[0]
    log_a = np.log(skeleton.scales[keep])
    log_z = np.array([logsumexp(2.0 * np.log(field[skeleton.scale_index == j])) for j in keep])
    w = weighted_by_counts(skeleton.n_lines[keep])
    return loglog_fit(np.exp(log_a), log_z, method=FitMethod.WLS, weights=w, log_y=False).slope


@pytest.mark.theory
@pytest.mark.stochastic
def test_positive_q_bias_is_the_running_supremum() -> None:
    r"""The :math:`h(2)` bias is the supremum's, and it does not shrink with :math:`n`.

    Two claims, both of which the module docstring now makes and neither of
    which is free:

    1. *It is asymptotic.* The mean bias at :math:`H=0.5` is :math:`+0.036` at
       :math:`n=2^{14}` and :math:`+0.040` at :math:`n=2^{16}` --- it does not
       decay, while MFDFA and the structure function reach
       :math:`|{\rm bias}|<0.005` on the same paths at :math:`n=2^{18}`.
       Calling this a small-sample effect, as the first version of this module
       did, tells a user to fix it with more data. More data does not fix it.
    2. *It is the running supremum.* Rebuilding :math:`\mathcal Z` from the node
       moduli on the **same** skeleton --- same chaining, same edge exclusion,
       same weights, only :math:`\sup_{a'\le a}|T|` replaced by
       :math:`|T(a,b)|` --- removes the bias entirely. The supremum's inflation
       factor grows from 1 at :math:`a=a_{\min}` to an :math:`a`-independent
       asymptote, and that transient tilts :math:`\log\mathcal Z`.

    This is *not* an argument for dropping the supremum: without it the whole
    :math:`q<0` branch collapses (:func:`test_running_supremum_is_what_makes_negative_q_work`).
    """
    H = 0.5
    for n in (2**14, 2**16):
        with_sup, without_sup = [], []
        for seed in SEEDS[:2]:
            sk = maxima_lines(fbm(n, H, seed=seed))
            with_sup.append(0.5 * (_tau2_from(sk, sk.running_sup) + 1.0))
            without_sup.append(0.5 * (_tau2_from(sk, sk.modulus) + 1.0))
        bias_sup = float(np.mean(with_sup)) - H
        bias_mod = float(np.mean(without_sup)) - H
        assert bias_sup > 0.02, f"n={n}: the documented upward bias vanished ({bias_sup:+.4f})"
        assert abs(bias_mod) < 0.015, (
            f"n={n}: dropping the supremum no longer removes the bias "
            f"({bias_mod:+.4f}); the diagnosis in the module docstring is stale"
        )
        assert bias_sup - abs(bias_mod) > 0.015


def test_running_supremum_is_what_makes_negative_q_work() -> None:
    r"""Replace :math:`\sup_{a'\le a}|T|` by :math:`|T(a,b)|` and :math:`q<0` breaks.

    The commonest WTMM implementation bug. On fBm at :math:`n=2^{14}` the
    partition function built from the suprema gives
    :math:`\tau(-5)\approx-6.6`; built from the bare moduli it gives
    :math:`\approx-2.3`, i.e. :math:`\hat h(-5)` collapses from 1.1 to 0.3 and
    the right branch of :math:`f(\alpha)` disappears. At :math:`q=0` the two
    coincide identically, which is the control.
    """
    sk = maxima_lines(fbm(N_FBM, 0.5, seed=SEEDS[0]))
    keep = np.nonzero(sk.n_lines >= 4)[0]
    w = weighted_by_counts(sk.n_lines[keep])

    def tau(field: np.ndarray, q: float) -> float:
        log_z = np.array([logsumexp(q * np.log(field[sk.scale_index == j])) for j in keep])
        return loglog_fit(
            sk.scales[keep], log_z, method=FitMethod.WLS, weights=w, log_y=False
        ).slope

    assert tau(sk.running_sup, 0.0) == pytest.approx(tau(sk.modulus, 0.0), abs=1e-12)
    assert tau(sk.running_sup, -5.0) < tau(sk.modulus, -5.0) - 3.0


@pytest.mark.theory
@pytest.mark.stochastic
@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_fbm_negative_q_follows_the_pathology_law(H: float) -> None:
    r"""The monofractal negative branch diverges, and it diverges *predictably*.

    Writing the line suprema as :math:`s_l = C_l a^{H}`, on a monofractal the
    density of :math:`C` is positive at the origin, so :math:`\mathbb E[C^q]`
    diverges for :math:`q\le-1` and :math:`\mathcal Z(q,a)` collapses onto its
    smallest term. The minimum of :math:`N_a\sim a^{-1}` draws is of order
    :math:`a`, giving :math:`\mathcal Z \sim a^{qH}a^{q}a^{-1}` and hence

    .. math:: \hat h(q) \approx H + 1 + 1/q .

    This test pins that law, because the alternative --- quietly reporting
    :math:`\hat h(-5)\approx 1.2` as an estimate of :math:`H` --- is exactly how
    a spurious multifractal width gets published. Observed worst deviation from
    the law: 0.21 on the three seeds below, 0.27 over 36 fresh seeds; the
    smallest observed :math:`\hat h(q)-H` was 0.46 (0.42 over the 36).
    """
    for seed in SEEDS:
        res = wtmm(fbm(N_FBM, H, seed=seed), q=Q)
        for q_neg in (-5.0, -4.0, -3.0):
            h = float(np.interp(q_neg, res.q, res.zeta)) / q_neg
            assert h - H > 0.35, "the negative branch stopped diverging; law changed"
            assert abs(h - (H + 1.0 + 1.0 / q_neg)) < 0.40, (
                f"h({q_neg})={h:.4f} departs from the H+1+1/q pathology law "
                f"({H + 1.0 + 1.0 / q_neg:.4f}) at H={H}, seed={seed}"
            )


# --------------------------------------------------------------------------- #
# (b) Multifractal recovery
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_binomial_cascade_tau(cascade_primitive: np.ndarray) -> None:
    r"""Recover :math:`\tau(q)=-\log_2(m_0^q+m_1^q)` on the binomial measure.

    Tolerance 0.06 over the whole of :math:`q\in[-5,5]`; observed maximum error
    0.036, attained at :math:`q=5` where the partition function is dominated by
    the single most singular maxima line and the coarse-scale line count is
    smallest. The negative branch is *better* than the positive one here
    (error :math:`+0.001` at :math:`q=-5`) because the binomial spectrum has a
    genuine right branch: :math:`\alpha \le \alpha_{\max}=-\log_2 0.4`, so the
    negative moments of the line suprema converge.
    """
    res = wtmm(cascade_primitive, q=Q, scales=CASCADE_SCALES, n_vanishing=1)
    err = res.tau - theory.cascade_tau(Q, BINOMIAL)
    assert np.abs(err).max() < 0.06, f"max |tau error| = {np.abs(err).max():.4f}"
    assert np.abs(err[Q <= 0.0]).max() < 0.01
    assert np.median([f.r_squared for f in res.fits]) > 0.999


@pytest.mark.theory
def test_binomial_cascade_falpha(cascade_primitive: np.ndarray) -> None:
    r"""Legendre--Fenchel spectrum against the closed-form binary entropy.

    Compared only on the interior of :math:`[\alpha_{\min},\alpha_{\max}]`,
    where the closed form is finite. Observed RMS error 0.012, maximum 0.037.
    The recovered support is :math:`[0.799, 1.254]` against the true
    :math:`[0.737, 1.322]`; the shortfall is *not* estimator bias but the
    :math:`|q|\le5` window --- the endpoints are the :math:`q\to\pm\infty`
    limits and no finite grid reaches them.
    """
    res = wtmm(cascade_primitive, q=Q, scales=CASCADE_SCALES, n_vanishing=1)
    spectrum = spectrum_from_exponents(res)
    a_min, a_max = theory.cascade_alpha_range(BINOMIAL, 2)

    finite = np.isfinite(spectrum.f_alpha)
    alpha = spectrum.alpha[finite]
    inside = (alpha > a_min + 1e-9) & (alpha < a_max - 1e-9)
    residual = spectrum.f_alpha[finite][inside] - theory.cascade_falpha(alpha[inside], BINOMIAL)
    assert np.sqrt(np.mean(residual**2)) < 0.03
    assert np.abs(residual).max() < 0.06

    descr = spectrum.descriptors()
    assert abs(descr.f_max - 1.0) < 0.02
    assert abs(descr.alpha_0 - 0.5 * (a_min + a_max)) < 0.02
    assert a_min < descr.alpha_min and descr.alpha_max < a_max


# --------------------------------------------------------------------------- #
# (c) Convention checks
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_tau_zero_is_minus_one_and_tau_one_is_zero(cascade_primitive: np.ndarray) -> None:
    r""":math:`\tau(0)=-d` counts maxima lines; :math:`\tau(1)=0` is conservation.

    :math:`\mathcal Z(0,a)` is the number of lines, which falls like
    :math:`a^{-d}`, so :math:`\tau(0)=-1` and :math:`\zeta(0)=0`. Because the
    binomial measure is conservative, :math:`\tau(1)=0` as well. Observed:
    :math:`\tau(0)=-1.00000`, :math:`\tau(1)=-0.0034`.
    """
    res = wtmm(cascade_primitive, q=Q, scales=CASCADE_SCALES, n_vanishing=1)
    assert abs(float(np.interp(0.0, res.q, res.tau)) + 1.0) < 5e-3
    assert abs(float(np.interp(0.0, res.q, res.zeta))) < 5e-3
    assert abs(float(np.interp(1.0, res.q, res.tau))) < 0.02


@pytest.mark.theory
def test_tau_zero_on_fbm() -> None:
    """The same anchor holds on a signal, not just on a measure: observed 0.007."""
    for seed in SEEDS:
        res = wtmm(fbm(N_FBM, 0.5, seed=seed), q=Q)
        assert abs(float(np.interp(0.0, res.q, res.tau)) + 1.0) < 0.02


def test_fitted_slope_is_tau_not_zeta(cascade_primitive: np.ndarray) -> None:
    """The stored regressions carry the measure exponent; ``zeta = slope + d``."""
    res = wtmm(cascade_primitive, q=Q, scales=CASCADE_SCALES, n_vanishing=1, d_support=1.0)
    slopes = np.array([f.slope for f in res.fits])
    assert np.allclose(res.zeta, slopes + 1.0, rtol=0, atol=0)
    assert np.allclose(res.tau, slopes, rtol=0, atol=1e-12)

    shifted = wtmm(cascade_primitive, q=Q, scales=CASCADE_SCALES, n_vanishing=1, d_support=2.0)
    assert np.allclose(shifted.zeta - res.zeta, 1.0)
    assert np.allclose(shifted.tau, res.tau)


# --------------------------------------------------------------------------- #
# (d) Determinism
# --------------------------------------------------------------------------- #


def test_bitwise_determinism() -> None:
    x = fbm(2**12, 0.6, seed=7)
    a = wtmm(x, q=Q, seed=123)
    b = wtmm(x, q=Q, seed=123)
    assert np.array_equal(a.zeta, b.zeta)
    assert np.array_equal(a.structure.values, b.structure.values)  # type: ignore[union-attr]

    sk1 = maxima_lines(x)
    sk2 = maxima_lines(x)
    assert np.array_equal(sk1.position, sk2.position)
    assert np.array_equal(sk1.running_sup, sk2.running_sup)


# --------------------------------------------------------------------------- #
# (e) Provenance
# --------------------------------------------------------------------------- #


def test_provenance_root_is_stable_and_parameter_sensitive() -> None:
    x = fbm(2**12, 0.6, seed=7)
    base = wtmm(x, q=Q)
    assert base.provenance is not None
    root = base.provenance.root
    assert wtmm(x, q=Q).provenance.root == root  # type: ignore[union-attr]

    variants = {
        "chain_tol": wtmm(x, q=Q, chain_tol=0.6),
        "edge_factor": wtmm(x, q=Q, edge_factor=0.25),
        "n_vanishing": wtmm(x, q=Q, n_vanishing=3),
        "min_lines": wtmm(x, q=Q, min_lines=8),
        "fit_method": wtmm(x, q=Q, fit_method=FitMethod.OLS),
        "select_range": wtmm(x, q=Q, select_range=True),
        "d_support": wtmm(x, q=Q, d_support=2.0),
        "q": wtmm(x, q=np.linspace(-4.0, 4.0, 41)),
        "scales": wtmm(x, q=Q, scales=[8, 16, 32, 64, 128]),
        "x": wtmm(fbm(2**12, 0.6, seed=8), q=Q),
    }
    for name, res in variants.items():
        assert res.provenance is not None
        assert res.provenance.root != root, f"root did not change when {name} changed"

    diag = base.provenance.diagnostics
    assert diag["n_zero_modulus_dropped"] == 0
    assert len(diag["n_lines_per_fitted_scale"]) == base.structure.scales.size  # type: ignore[union-attr]
    assert all(v > 0 for v in diag["n_maxima_per_scale"])
    assert diag["scaling_range"]["n_scales"] == base.structure.scales.size  # type: ignore[union-attr]


def test_provenance_records_every_parameter() -> None:
    x = fbm(2**12, 0.6, seed=7)
    res = wtmm(x, q=Q, chain_tol=0.8, edge_factor=0.25, min_lines=6, n_vanishing=3)
    assert res.provenance is not None
    params = dict(res.provenance.params)
    for key in (
        "n_vanishing",
        "chain_tol",
        "edge_factor",
        "min_lines",
        "fit_method",
        "select_range",
        "d_support",
        "n",
    ):
        assert key in params, f"parameter {key} missing from provenance"
    names = {ref.name for ref in res.provenance.inputs}
    assert {"x", "q", "scales"} <= names
    assert res.provenance.seed is None


# --------------------------------------------------------------------------- #
# (f) Negative q actually works  --  regression test for legacy defect D2
# --------------------------------------------------------------------------- #


@pytest.mark.theory
def test_negative_q_widens_the_recovered_alpha_range(cascade_primitive: np.ndarray) -> None:
    r"""Direct regression against LEGACY_AUDIT D2 (``q`` grid started at zero).

    The right branch :math:`\alpha>\alpha_0` is reachable only from :math:`q<0`.
    Observed on the binomial measure: the full grid recovers
    :math:`\Delta\alpha = 0.455` of the true :math:`0.585`, whereas a
    :math:`q\ge0` grid recovers :math:`0.228` --- it cannot exceed
    :math:`\alpha_0` even in principle.
    """
    full = spectrum_from_exponents(
        wtmm(cascade_primitive, q=Q, scales=CASCADE_SCALES, n_vanishing=1)
    ).descriptors()
    positive = spectrum_from_exponents(
        wtmm(
            cascade_primitive,
            q=np.linspace(0.0, 5.0, 21),
            scales=CASCADE_SCALES,
            n_vanishing=1,
        )
    ).descriptors()

    a_min, a_max = theory.cascade_alpha_range(BINOMIAL, 2)
    assert positive.alpha_max <= full.alpha_0 + 0.02, "q >= 0 reached past the apex"
    assert full.width > 1.8 * positive.width
    assert full.width > 0.7 * (a_max - a_min)
    assert full.alpha_max > full.alpha_0 + 0.15


# --------------------------------------------------------------------------- #
# The skeleton and the isolated singularity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9])
def test_isolated_cusp_exponent_from_one_maxima_line(alpha: float) -> None:
    r"""WTMM's unique strength: one line, one cusp, one exponent.

    For :math:`x(t)=|t-t_0|^{\alpha}` the maxima line converging onto
    :math:`t_0` satisfies :math:`|T(a,b(a))| = C a^{\alpha}` exactly, so a
    single regression along that line returns :math:`\alpha` with no partition
    function and no ensemble. Observed at :math:`n=2^{13}`: errors
    :math:`-0.021,\,-0.007,\,-0.003,\,-0.001` for
    :math:`\alpha=0.3,\,0.5,\,0.7,\,0.9`, with :math:`R^2\ge0.997`. The
    residual bias is discretisation of the cusp itself and shrinks with
    :math:`\alpha` (a rougher cusp is worse resolved on a fixed grid).
    """
    n = 2**13
    t = np.arange(n, dtype=np.float64) / n
    t0 = 0.5
    x = np.abs(t - t0) ** alpha

    skeleton = maxima_lines(x)
    line_id = skeleton.line_nearest(t0 * n)
    scales, positions, modulus = skeleton.line(line_id)

    # The line does not drift: it sits on the cusp at every scale.
    assert np.abs(positions - t0 * n).max() <= 1.0
    assert scales.size == skeleton.scales.size
    assert np.all(np.diff(scales) > 0)
    assert np.all(np.diff(modulus) > 0)

    fit = line_exponent(skeleton, line_id)
    assert abs(fit.slope - alpha) < 0.05, f"recovered {fit.slope:.4f} for alpha={alpha}"
    assert fit.r_squared > 0.99


def test_skeleton_structure_and_helpers() -> None:
    """Structural invariants, plus the two facts that are *not* free.

    ``n_nodes == n_lines.sum()``, ``running_sup >= modulus`` and monotone line
    counts hold by construction of :func:`maxima_lines` and would survive a
    completely wrong transform, so they are smoke tests only. The two checks
    that can actually fail are done against an independent recomputation from
    :func:`cwt_dog`: every node must sit on a genuine local maximum of
    :math:`|T(a,\\cdot)|` with the stored modulus, and ``running_sup`` must
    equal the max of that modulus along the line up to the node's scale.
    """
    x = fbm(2**12, 0.6, seed=5)
    sk = maxima_lines(x)
    assert isinstance(sk, WTMMSkeleton)
    assert sk.wavelet == "dog2"
    assert sk.n_series == 2**12
    assert sk.n_nodes == int(sk.n_lines.sum())
    assert sk.n_lines[0] == sk.n_lines_total
    assert np.all(np.diff(sk.n_lines) <= 0)
    assert np.all(sk.n_lines <= sk.n_maxima)
    assert set(np.unique(sk.line_id).tolist()) <= set(range(sk.n_lines_total))

    # Independent recomputation of the transform.
    coeffs = np.abs(cwt_dog(x, sk.scales))
    idx = sk.position.astype(np.int64)
    assert np.array_equal(sk.position, idx.astype(np.float64))
    assert np.allclose(coeffs[sk.scale_index, idx], sk.modulus, rtol=0, atol=0)
    assert np.all(coeffs[sk.scale_index, idx] > coeffs[sk.scale_index, idx - 1])
    assert np.all(coeffs[sk.scale_index, idx] > coeffs[sk.scale_index, idx + 1])

    # running_sup is the max along the line from the finest scale up to here.
    for line_id in sk.longest_lines(5):
        sel = np.nonzero(sk.line_id == line_id)[0]
        order = sel[np.argsort(sk.scale_index[sel])]
        assert np.allclose(sk.running_sup[order], np.maximum.accumulate(sk.modulus[order]))

    longest = sk.longest_lines(3)
    assert longest.size == 3
    scales, _, _ = sk.line(int(longest[0]))
    assert scales.size >= 3
    with pytest.raises(KeyError):
        sk.line(10**9)


def test_tighter_chaining_thins_the_skeleton() -> None:
    """``chain_tol`` is the method's most influential free parameter; show it."""
    x = fbm(2**12, 0.6, seed=5)
    loose = maxima_lines(x, chain_tol=2.0)
    tight = maxima_lines(x, chain_tol=0.1)
    assert tight.n_lines[-1] < loose.n_lines[-1]
    assert tight.n_lines_total == loose.n_lines_total  # the finest scale is unchanged


# --------------------------------------------------------------------------- #
# Structure surface: refitting without recomputation
# --------------------------------------------------------------------------- #


def test_structure_surface_supports_refitting(cascade_primitive: np.ndarray) -> None:
    """``wtmm_structure`` is the contract that makes a refit cheap and faithful.

    Refitting the stored surface over the four finest scales must reproduce
    ``wtmm`` called on exactly those scales. The two agree to ~1e-9 rather than
    bitwise: a shorter scale list changes the FFT padding length, hence the
    wrap-around terms, at the 1e-15 level.
    """
    surface = wtmm_structure(cascade_primitive, q=Q, scales=CASCADE_SCALES, n_vanishing=1)
    assert surface.values.shape == (Q.size, CASCADE_SCALES.size)
    assert surface.counts is not None
    assert np.all(surface.counts > 0)

    sub = slice(0, 4)
    weights = weighted_by_counts(surface.counts[sub])
    refit = np.array(
        [
            loglog_fit(
                surface.scales[sub],
                np.log(surface.values[i, sub]),
                method=FitMethod.WLS,
                weights=weights,
                log_y=False,
            ).slope
            for i in range(Q.size)
        ]
    )
    direct = wtmm(cascade_primitive, q=Q, scales=CASCADE_SCALES[sub], n_vanishing=1).tau
    assert np.allclose(refit, direct, rtol=1e-8, atol=1e-8)


def test_weighting_is_by_line_count(cascade_primitive: np.ndarray) -> None:
    """WLS is the default and the weights are the line counts, not ones."""
    res = wtmm(cascade_primitive, q=Q, scales=CASCADE_SCALES, n_vanishing=1)
    fit = res.fits[0]
    assert fit.method is FitMethod.WLS
    assert fit.weights is not None
    assert np.array_equal(fit.weights, res.structure.counts.astype(float))  # type: ignore[union-attr]

    ols = wtmm(
        cascade_primitive,
        q=Q,
        scales=CASCADE_SCALES,
        n_vanishing=1,
        fit_method=FitMethod.OLS,
    )
    assert not np.allclose(ols.zeta, res.zeta)


def test_select_range_picks_a_contiguous_better_fitting_window() -> None:
    r"""``select_range`` must earn its window, and must apply it to every ``q``.

    ``n_scales <= full``, ``s_min >= full.s_min`` and ``sum(mask) == n_scales``
    are all true by construction of :func:`select_scaling_range` and cannot
    fail; they are not asserted. What *can* fail, and is asserted:

    * the window is a **contiguous** block (a set of scales that happen to fit
      well is not a scaling range);
    * it is **strictly better** at the pivot :math:`q=2` than the full grid,
      measured by :math:`R^2` on the same rows --- otherwise selection is
      buying nothing for the degrees of freedom it spends;
    * the **same** window is used for every :math:`q`, which is the whole
      design decision (per-:math:`q` selection would make :math:`\hat\tau` a
      function of selection noise).

    Observed at :math:`H=0.6`, seed 11: 18 of 20 scales, :math:`R^2` 0.9844 at
    the pivot against 0.9604 on the full grid.
    """
    x = fbm(N_FBM, 0.6, seed=11)
    full = wtmm(x, q=Q)
    auto = wtmm(x, q=Q, select_range=True)
    assert auto.scaling_range is not None and full.scaling_range is not None
    assert auto.scaling_range.selection.startswith("auto-")

    mask = np.asarray(auto.scaling_range.mask, dtype=bool)
    on = np.nonzero(mask)[0]
    assert on.size >= 3
    assert np.all(np.diff(on) == 1), "the selected scaling range is not contiguous"
    assert on.size < mask.size, "selection kept the whole grid; the test proves nothing"

    pivot = int(np.argmin(np.abs(Q - 2.0)))
    assert auto.fits[pivot].r_squared > full.fits[pivot].r_squared

    # One window for all q: every stored fit uses the same number of points,
    # over the same scales.
    assert {f.n_points for f in auto.fits} == {int(on.size)}
    assert all(f.scaling_range is not None for f in auto.fits)
    assert {(f.scaling_range.s_min, f.scaling_range.s_max) for f in auto.fits} == {  # type: ignore[union-attr]
        (auto.scaling_range.s_min, auto.scaling_range.s_max)
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_rejects_degenerate_inputs() -> None:
    x = fbm(2**12, 0.5, seed=1)
    with pytest.raises(ValueError, match="chain_tol"):
        maxima_lines(x, chain_tol=0.0)
    with pytest.raises(ValueError, match="edge_factor"):
        maxima_lines(x, edge_factor=-1.0)
    with pytest.raises(ValueError, match="min_lines"):
        wtmm_structure(x, q=Q, min_lines=0)
    with pytest.raises(ValueError, match="constant"):
        wtmm(np.ones(2**12), q=Q)
    with pytest.raises(ValueError, match="at least three"):
        wtmm(x, q=Q, scales=[64, 128])


def test_min_lines_drops_line_starved_scales() -> None:
    """A partition function over three lines is not an estimate; refuse it."""
    x = fbm(2**12, 0.6, seed=5)
    generous = wtmm_structure(x, q=Q, min_lines=1)
    strict = wtmm_structure(x, q=Q, min_lines=200)
    assert strict.scales.size < generous.scales.size
    assert strict.scales.max() < generous.scales.max()
    assert int(strict.counts.min()) >= 200  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="maxima lines"):
        wtmm_structure(x, q=Q, min_lines=10**6)
