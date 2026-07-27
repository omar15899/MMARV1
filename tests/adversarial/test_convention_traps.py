r"""Adversarial probes of the library's central claim: the :math:`\pm1` and the log base cannot drift.

Every test here attacks one of the four places a multifractal library silently
loses an offset or a logarithm:

* the anchor :math:`\tau(0)=-d` / :math:`\zeta(0)=0`, free of sampling error;
* the conservation anchor :math:`\tau(1)=0`;
* the bridge :math:`\zeta=\tau+d` between the measure and signal views;
* the base of the logarithm in a log--log slope.

Plus the two transforms that must be exact --- the Legendre--Fenchel involution
and the ``kind=`` canonicalisation --- and the two places where the measure
convention was actually being violated.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest

from mmar.analysis import ESTIMATORS, analyze
from mmar.core.legendre import (
    concavity_defect,
    legendre_fenchel,
    legendre_parametric,
    spectrum_from_exponents,
)
from mmar.core.types import FloatArray, PointwiseHolder, ScalingExponents
from mmar.estimators.box_counting import generalised_dimensions
from mmar.estimators.chhabra_jensen import chhabra_jensen
from mmar.estimators.pointwise import oscillation_holder
from mmar.estimators.structure_function import mfc_exponents
from mmar.models.cascade import deterministic_cascade
from mmar.models.fbm import fbm, fgn, levy_stable_increments
from mmar.models.theory import (
    cascade_tau,
    lognormal_conservation_sigma2,
    lognormal_falpha,
    lognormal_tau,
)

_KMAX = 13
_M = (0.65, 0.35)
_Q = np.linspace(-3.0, 3.0, 25)

# WTMM sums over maxima *lines*, not boxes: Z(0, a) counts the surviving lines,
# whose number scales like a^{-1} only up to the boundary exclusion band, so its
# tau(0) is an estimate rather than an identity. Its own module docstring quotes
# an observed |tau(0) + 1| <= 0.008 on fBm; on the cascade primitive with the
# default Mexican hat this run measures 0.0117. Everything else is exact.
_TAU0_TOL: dict[str, float] = {"wtmm": 0.05}
_EXACT = 1e-9


def _binomial_measure() -> FloatArray:
    return deterministic_cascade(m=_M, kmax=_KMAX)


def _trading_time(mu: FloatArray) -> FloatArray:
    """The primitive of ``mu``: a monotone level series whose increments are the cells."""
    return np.asarray(np.concatenate([[0.0], np.cumsum(mu)]), dtype=np.float64)


def _series_exponents(x: FloatArray, **kwargs: Any) -> dict[str, ScalingExponents]:
    out: dict[str, ScalingExponents] = {}
    for name in ESTIMATORS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = analyze(x, estimator=name, q=_Q, cumulants=False, **kwargs).exponents
    return out


def _wls_slope(lx: FloatArray, ly: FloatArray, w: FloatArray) -> float:
    sw = w / w.sum()
    xb = float(np.sum(sw * lx))
    yb = float(np.sum(sw * ly))
    return float(np.sum(w * (lx - xb) * (ly - yb)) / np.sum(w * (lx - xb) ** 2))


# --------------------------------------------------------------------------- #
# The anchors
# --------------------------------------------------------------------------- #


def test_zeta_zero_vanishes_and_tau_zero_is_minus_d_for_every_estimator() -> None:
    r"""Attack: a dropped or doubled :math:`d` in the :math:`\zeta\leftrightarrow\tau`
    conversion of any one estimator.

    :math:`S(0,\delta)=1` for every :math:`\delta`, so :math:`\zeta(0)=0` carries
    no sampling error whatsoever and :math:`\tau(0)=-d` follows. Run on one
    object through the one entry point, so a family that converts the wrong way
    round shows up as a whole unit of disagreement with its peers rather than as
    a plausible curve.
    """
    theta = _trading_time(_binomial_measure())
    mu = _binomial_measure()
    exponents = _series_exponents(theta)
    exponents["chhabra_jensen"] = chhabra_jensen(mu, q=_Q).exponents
    exponents["box_counting"] = generalised_dimensions(mu, q=_Q)

    i0 = int(np.argmin(np.abs(_Q)))
    for name, e in exponents.items():
        tol = _TAU0_TOL.get(name, _EXACT)
        assert abs(float(e.zeta[i0])) < tol, f"{name}: zeta(0) = {e.zeta[i0]!r}, must be 0"
        assert abs(float(e.tau[i0]) + 1.0) < tol, f"{name}: tau(0) = {e.tau[i0]!r}, must be -1"


def test_tau_one_vanishes_on_a_conservative_cascade() -> None:
    r"""Attack: a lost conservation anchor, i.e. a :math:`\pm1` hiding in
    :math:`\tau(1)`.

    :math:`\sum_i\mu_i=1` at every scale, so :math:`\tau(1)=0` exactly for a
    measure-domain estimator. A signal-domain estimator applied to the primitive
    reaches it only up to its own bias; observed here at ``kmax=13``:
    ``structure_function +0.004``, ``ghe -0.004``, ``mfdma +0.025``,
    ``mfdfa -0.047``, ``wtmm -0.059``, ``wavelet_leaders -0.190`` (the leader
    construction is the least suited to a devil's staircase). The bound below is
    set at 0.3 --- deliberately looser than any of those and deliberately
    tighter than 0.5, so that it can only ever fire on a genuine unit shift and
    never on estimator quality, which is not what this test is about.
    """
    mu = _binomial_measure()
    theta = _trading_time(mu)
    i1 = int(np.argmin(np.abs(_Q - 1.0)))

    for name, e in (
        ("chhabra_jensen", chhabra_jensen(mu, q=_Q).exponents),
        ("box_counting", generalised_dimensions(mu, q=_Q)),
    ):
        assert abs(float(e.tau[i1])) < 1e-10, f"{name}: tau(1) = {e.tau[i1]!r}, must be 0"

    for name, e in _series_exponents(theta).items():
        assert abs(float(e.tau[i1])) < 0.3, f"{name}: tau(1) = {e.tau[i1]!r}, must be ~0"


def test_measure_and_signal_estimators_meet_across_the_zeta_equals_tau_plus_d_bridge() -> None:
    r"""Attack: the bridge :math:`\zeta=\tau+d` applied in the wrong direction, or not
    at all.

    A binomial measure :math:`\mu` and its primitive :math:`\theta` are the same
    object seen from the two views: the increment of :math:`\theta` over a box is
    that box's mass. A measure-domain estimator on :math:`\mu` and a
    signal-domain estimator on :math:`\theta` must therefore land on the *same*
    :math:`\zeta`, and both on the closed form :math:`\tau_{\text{exact}}+1`.
    Scaling is exact at every dyadic scale for this object, so the agreement is
    at floating-point level and a unit offset is unmissable.
    """
    mu = _binomial_measure()
    theta = _trading_time(mu)
    exact_zeta = cascade_tau(_Q, _M) + 1.0

    cj = chhabra_jensen(mu, q=_Q).exponents
    bc = generalised_dimensions(mu, q=_Q)
    sf = mfc_exponents(theta, q=_Q, scales=np.array([2**j for j in range(1, _KMAX - 2)]))

    assert np.max(np.abs(cj.zeta - exact_zeta)) < 1e-9
    assert np.max(np.abs(bc.zeta - exact_zeta)) < 1e-9
    assert np.max(np.abs(sf.zeta - exact_zeta)) < 1e-9
    assert np.max(np.abs(sf.zeta - cj.zeta)) < 1e-9
    assert np.max(np.abs(sf.tau - cascade_tau(_Q, _M))) < 1e-9


# --------------------------------------------------------------------------- #
# The log base
# --------------------------------------------------------------------------- #


def test_the_fitted_exponent_is_independent_of_the_log_base() -> None:
    r"""Attack: a mixed-base log--log fit (``log10`` on one axis, ``ln`` on the other).

    A slope is base-invariant only when *both* axes use the same base. This
    recomputes the library's :math:`\zeta` by hand from the stored structure
    surface in natural, binary and decimal logarithms; all three must return the
    same number to floating point. The final assertion pins the teeth of the
    test: a deliberately mixed-base fit of the same data is off by
    :math:`\ln 10 \approx 2.30`.
    """
    x = fbm(4096, H=0.65, seed=17)
    q = np.array([-2.0, -1.0, 0.5, 1.0, 2.0, 3.0])
    exponents = mfc_exponents(x, q=q)
    structure = exponents.structure
    assert structure is not None and structure.counts is not None
    s = structure.values
    delta = structure.scales
    w = np.asarray(structure.counts, dtype=np.float64)

    for base in (math.e, 2.0, 10.0):
        lx = np.log(delta) / math.log(base)
        by_hand = np.array(
            [_wls_slope(lx, np.log(s[i]) / math.log(base), w) for i in range(q.size)]
        )
        assert np.allclose(by_hand, exponents.zeta, rtol=0.0, atol=1e-12), (
            f"base {base}: {by_hand} vs library {exponents.zeta}"
        )

    mixed = _wls_slope(np.log10(delta), np.log(s[3]), w)
    assert abs(mixed - float(exponents.zeta[3])) > 0.5


# --------------------------------------------------------------------------- #
# d_support
# --------------------------------------------------------------------------- #


def test_d_support_shifts_tau_and_f_by_exactly_one_and_leaves_alpha_alone() -> None:
    r"""Attack: :math:`d` leaking into :math:`\alpha`, or into the wrong side of
    the transform.

    For a signal-native estimator :math:`\zeta` is what is measured, so declaring
    :math:`d=2` instead of :math:`d=1` must move :math:`\tau=\zeta-d` down by
    exactly one and :math:`f=\inf_q\{q\alpha-\zeta\}+d` up by exactly one, while
    :math:`\alpha` --- a Hölder exponent, the same in both views --- must not
    move at all.

    WTMM is excluded and checked separately: it is measure-native (it sums over
    maxima lines), so for it :math:`\tau` is the measured quantity and
    :math:`\zeta` is the one that shifts.
    """
    x = fbm(2048, H=0.6, seed=7)
    signal_native = [n for n in ESTIMATORS if n != "wtmm"]

    for name in signal_native:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = analyze(x, estimator=name, q=_Q, cumulants=False, d_support=1.0)
            r2 = analyze(x, estimator=name, q=_Q, cumulants=False, d_support=2.0)
        assert np.max(np.abs(r2.exponents.zeta - r1.exponents.zeta)) < 1e-12, name
        assert np.allclose(r2.exponents.tau - r1.exponents.tau, -1.0, atol=1e-12), name
        assert np.max(np.abs(r2.spectrum.alpha - r1.spectrum.alpha)) < 1e-12, name
        assert np.allclose(r2.spectrum.f_alpha - r1.spectrum.f_alpha, 1.0, atol=1e-12), name

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w1 = analyze(x, estimator="wtmm", q=_Q, cumulants=False, d_support=1.0)
        w2 = analyze(x, estimator="wtmm", q=_Q, cumulants=False, d_support=2.0)
    assert np.allclose(w2.exponents.zeta - w1.exponents.zeta, 1.0, atol=1e-12)
    assert np.max(np.abs(w2.exponents.tau - w1.exponents.tau)) < 1e-12


# --------------------------------------------------------------------------- #
# Legendre
# --------------------------------------------------------------------------- #


def test_legendre_fenchel_is_an_involution_on_the_exact_lognormal_tau() -> None:
    r"""Attack: an offset or a sign inside the transform that a one-way check would miss.

    The Legendre--Fenchel transform of a closed, proper, concave function is an
    involution. Feeding it the exact lognormal :math:`\tau` and transforming the
    resulting :math:`f` back --- :math:`\tau(q)=\inf_\alpha\{q\alpha-f(\alpha)\}`,
    which is the *measure*-view inverse and therefore also a check that
    ``f_alpha`` really is measure-view --- must return the analytic parabola. A
    stray :math:`d` on either leg survives the forward direction (it just moves
    the apex) but not the round trip.

    The forward leg is only grid-exact: the infimum runs over ``q_refine=2001``
    points spanning :math:`[-5,5]`, so it overshoots by
    :math:`\tfrac12|\tau''|(\Delta q/2)^2 \approx 4.5\times10^{-7}`, and the
    measured maximum deviation is :math:`3.1\times10^{-7}`. The round trip is
    exact to :math:`1.8\times10^{-15}` because the two discretisations are the
    same one, which is precisely why the involution is the sharper test.
    """
    lam, b = 1.05, 2.0
    sigma2 = lognormal_conservation_sigma2(lam, b)
    q = np.linspace(-5.0, 5.0, 101)
    zeta = lognormal_tau(q, lam, sigma2, b) + 1.0

    alpha, f = legendre_fenchel(q, zeta, d_support=1.0, n_alpha=401)
    apex = int(np.argmax(f))
    assert abs(float(f[apex]) - 1.0) < 1e-6
    assert abs(float(alpha[apex]) - lam) < 5e-3
    assert np.max(np.abs(f - lognormal_falpha(alpha, lam, sigma2, b))) < 1e-6

    q_back = np.linspace(-3.0, 3.0, 61)
    tau_back = np.min(np.outer(q_back, alpha) - f[None, :], axis=1)
    assert np.max(np.abs(tau_back - lognormal_tau(q_back, lam, sigma2, b))) < 1e-12

    # tau is analytic and strictly concave here, so the tangent construction must
    # land on the identical parabola -- offset included.
    _, alpha_p, f_p = legendre_parametric(q, zeta, d_support=1.0)
    assert np.max(np.abs(f_p - lognormal_falpha(alpha_p, lam, sigma2, b))) < 1e-9


def test_legendre_fenchel_of_a_non_concave_zeta_is_still_concave_and_the_defect_is_reported() -> (
    None
):
    r"""Attack: a transform that inherits the wiggles of :math:`\hat\zeta`.

    :math:`\inf_q\{q\alpha-\zeta(q)\}` is a pointwise infimum of affine functions
    of :math:`\alpha` and is therefore concave *whatever* :math:`\zeta` is. A
    deliberately non-concave (indeed strictly convex) :math:`\zeta` must still
    produce a concave :math:`f`, ``concavity_defect`` must be strictly positive,
    and the tangent construction --- which assumes what it should be testing ---
    must visibly fail, or the diagnostic is measuring nothing.
    """
    q = np.linspace(-4.0, 4.0, 33)
    wiggly = 0.6 * q + 0.15 * np.sin(2.0 * q)
    convex = 0.5 * q + 0.05 * q**2

    for zeta in (wiggly, convex):
        assert concavity_defect(q, zeta) > 1e-3
        _, f = legendre_fenchel(q, zeta, d_support=1.0, n_alpha=401)
        assert np.max(np.diff(f, 2)) < 1e-12, "Legendre-Fenchel returned a non-concave f"

    assert concavity_defect(q, 0.6 * q - 0.05 * q**2) < 1e-12

    _, alpha_p, f_p = legendre_parametric(q, wiggly, d_support=1.0)
    order = np.argsort(alpha_p)
    assert np.max(np.diff(f_p[order], 2)) > 0.1, "the tangent construction should fold here"

    exponents = ScalingExponents(q=q, zeta=wiggly, d_support=1.0, estimator="synthetic")
    spectrum = spectrum_from_exponents(exponents, method="fenchel")
    finite = np.isfinite(spectrum.f_alpha)
    assert np.max(np.diff(spectrum.f_alpha[finite], 2)) < 1e-12


# --------------------------------------------------------------------------- #
# kind=
# --------------------------------------------------------------------------- #


def test_kind_canonicalisation_is_exact_and_mislabelling_moves_the_exponent() -> None:
    r"""Attack: the commonest applied error --- feeding returns where a path is expected.

    Two halves. First, ``kind="increments"`` must be *exactly* the same
    computation as integrating by hand and passing ``kind="level"``: if the
    canonicalisation itself drifted, every downstream anchor would still pass
    while every exponent was wrong by one. Second, mislabelling must actually
    move the answer --- fGn read as a level series measures the regularity of
    noise (:math:`h\approx0`, not :math:`0.7`), and an fBm path read as
    increments is integrated twice, which MF-DFA (the only family here that can
    resolve :math:`h>1`) reports as :math:`H+1`.
    """
    g = fgn(4096, H=0.7, seed=11)
    by_hand = np.cumsum(g - float(np.mean(g)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = analyze(g, estimator="mfdfa", kind="increments", q=_Q, cumulants=False)
        b = analyze(by_hand, estimator="mfdfa", kind="level", q=_Q, cumulants=False)
    assert np.array_equal(a.exponents.zeta, b.exponents.zeta)

    for name, e in _series_exponents(g, kind="increments").items():
        assert abs(e.h2 - 0.7) < 0.12, f"{name}: h(2) = {e.h2!r} on fGn(H=0.7)"
    for name, e in _series_exponents(g, kind="level").items():
        assert e.h2 < 0.2, f"{name}: mislabelled fGn gave h(2) = {e.h2!r}, trap not triggered"

    path = fbm(4096, H=0.7, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        right = analyze(path, estimator="mfdfa", kind="level", q=_Q, cumulants=False)
        wrong = analyze(path, estimator="mfdfa", kind="increments", q=_Q, cumulants=False)
    assert abs((wrong.exponents.h2 - right.exponents.h2) - 1.0) < 0.1


# --------------------------------------------------------------------------- #
# f_max == d
# --------------------------------------------------------------------------- #


def test_f_max_is_the_support_dimension_for_a_space_filling_monofractal() -> None:
    r"""Attack: an ``f_alpha`` that is not in the measure convention.

    :math:`f(\alpha)=\inf_q\{q\alpha-\zeta(q)\}+d` is bounded above by the
    :math:`q=0` term, :math:`d-\zeta(0)=d`, and attains it at
    :math:`\alpha=\zeta'(0)`. So for any estimator on a space-filling monofractal
    the apex sits at :math:`d`, and the residual is the resolution of the
    :math:`\alpha` grid --- not a convention. A spectrum reported in the signal
    convention would peak at :math:`0`, and one built from a mis-signed
    transform would exceed :math:`d`.
    """
    x = fbm(4096, H=0.7, seed=11)
    for name, e in _series_exponents(x).items():
        d = spectrum_from_exponents(e).descriptors()
        assert 0.95 <= d.f_max <= 1.0 + 5e-3, f"{name}: f_max = {d.f_max!r}, expected ~d = 1"

    uniform = np.full(1 << 13, 1.0)
    spectrum = chhabra_jensen(uniform, q=_Q)
    assert abs(float(np.nanmax(spectrum.f_alpha)) - 1.0) < 1e-6


def test_pointwise_histogram_spectrum_stays_inside_the_measure_convention() -> None:
    r"""Attack: :meth:`PointwiseHolder.to_spectrum` normalising by the log of a *density*.

    ``HolderSpectrum`` promises that ``f_alpha`` is always the measure-view
    spectrum, so :math:`0 \le f \le d` and the apex is the *most common*
    exponent. Normalising :math:`\ln(\text{density})` by :math:`\ln` of its own
    maximum breaks both as soon as the field spreads over more than about
    :math:`e` units of :math:`h`: the denominator turns negative and the curve
    inverts.

    Reproduced on a real field --- ``oscillation_holder`` of a Levy path,
    :math:`h\in[0, 4.61]`, ``bins=20`` --- where the pre-fix code returned
    :math:`f_{\max}=140.15` with the apex at :math:`\alpha_0=+3.81` (the rarest
    exponent) instead of the mode near :math:`0.9`, the minimum :math:`f=1.0`
    sitting at the mode. Changing ``bins`` from 20 to 50 flipped the sign of the
    whole curve. Any width, asymmetry or :math:`\alpha_0` read off that object is
    a function of the bin count.
    """
    x = np.cumsum(levy_stable_increments(4096, alpha=1.2, seed=5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = oscillation_holder(x)
    h = field.h[np.isfinite(field.h)]
    assert float(h.max() - h.min()) > math.e, "the repro needs a field wider than e"
    mode = float(np.median(h))

    for bins in (5, 10, 20, 50):
        spectrum = field.to_spectrum(bins=bins)
        f = spectrum.f_alpha[np.isfinite(spectrum.f_alpha)]
        assert f.max() <= spectrum.d_support + 1e-12, (
            f"bins={bins}: f_max = {f.max()!r} exceeds d = {spectrum.d_support}"
        )
        assert f.min() >= -1e-12, f"bins={bins}: f_min = {f.min()!r} is negative"
        assert abs(spectrum.descriptors().alpha_0 - mode) < 1.0, (
            f"bins={bins}: apex at {spectrum.descriptors().alpha_0!r}, mode is {mode!r}"
        )

    flat = PointwiseHolder(
        t=np.arange(4096, dtype=np.float64),
        h=np.linspace(0.0, 6.0, 4096),
    ).to_spectrum(bins=12)
    finite = flat.f_alpha[np.isfinite(flat.f_alpha)]
    assert finite.size > 0
    assert float(finite.max()) <= 1.0 + 1e-12
    assert float(finite.min()) >= -1e-12


# --------------------------------------------------------------------------- #
# The surrogate ensemble must use the reported configuration
# --------------------------------------------------------------------------- #


def test_surrogate_test_measures_the_statistic_the_report_quotes() -> None:
    r"""Attack: a surrogate ensemble estimated through a different code path than the report.

    :func:`analyze` handed :func:`multifractality_report` the bare registry
    callable, which drops both the caller's ``q``/``scales`` and the
    ``_LEVEL_KWARGS`` pin that stops MF-DFA integrating an already-integrated
    series. The p-value was internally coherent but attached to the wrong
    number: on ``fbm(2048, H=0.6, seed=5)`` with ``q = linspace(-3, 3, 25)`` the
    report quoted ``width = 0.0663`` (``h(2) = 0.677``) while the tested observed
    statistic was ``0.3386`` --- the width of the doubly-integrated series'
    spectrum, ``h(2) = 1.554``, one whole exponent away, on the default
    :math:`[-5,5]` grid.
    """
    x = fbm(2048, H=0.6, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyze(
            x,
            estimator="mfdfa",
            kind="level",
            q=_Q,
            cumulants=False,
            surrogate_test=True,
            n_surrogates=6,
            seed=1,
        )
    reported = float(report.spectrum.descriptors().width)
    surrogates = report.tests["surrogates"]
    assert float(surrogates.shuffle.statistic) == pytest.approx(reported, rel=1e-12)
    assert float(surrogates.iaaft.statistic) == pytest.approx(reported, rel=1e-12)
    assert report.exponents.h2 < 1.0
