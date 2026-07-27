r"""Tests for :mod:`mmar.estimators.box_counting`.

The deterministic binomial measure is the natural adversary for this family:
its partition function is *exactly* :math:`(\sum_i m_i^q)^l` at every level, so
the box-counting :math:`\tau(q)` is not merely close to the oracle, it is equal
to it to machine precision. Anything less than :math:`10^{-10}` agreement on
:math:`\tau` therefore signals a bug, not sampling error, and the tolerances are
set accordingly.

The histogram method is held to a completely different, and much lower,
standard --- see :func:`test_histogram_spectrum_converges_with_depth`, which
reports the observed error at each depth rather than asserting a tolerance
pulled out of the air.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np
import pytest

from mmar.core.legendre import spectrum_from_exponents
from mmar.core.types import FitMethod, FloatArray
from mmar.estimators.box_counting import (
    box_partition,
    coarse_holder,
    dq,
    generalised_dimensions,
    generalised_dimensions_structure,
    histogram_spectrum,
    information_dimension,
    multifractal_measure_spectrum,
)
from mmar.models import theory
from mmar.models.cascade import deterministic_cascade
from mmar.models.fbm import fgn

M = (0.6, 0.4)
"""Binomial weights used throughout; alpha in [0.7370, 1.3219], f = binary entropy."""

SEED = 20260727


def _binomial_alpha(k: int, m0: float = 0.6, m1: float = 0.4) -> FloatArray:
    r"""Exact coarse Hölder exponent of every cell of the level-``k`` binomial.

    Cell ``i`` is built by ``np.kron`` from the base-2 digits of ``i``, digit 0
    selecting :math:`m_0`; with :math:`j` zero digits its mass is
    :math:`m_0^j m_1^{k-j}` and its exponent is
    :math:`-(j\ln m_0 + (k-j)\ln m_1)/(k\ln 2)`.
    """
    idx = np.arange(2**k, dtype=np.int64)
    j = k - np.array([int(v).bit_count() for v in idx], dtype=np.int64)
    return np.asarray(
        -(j * math.log(m0) + (k - j) * math.log(m1)) / (k * math.log(2.0)),
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def binomial14() -> FloatArray:
    return deterministic_cascade(14, M)


@pytest.fixture(scope="module")
def cantor9() -> FloatArray:
    """Triadic Cantor measure: base 3, mass split evenly over the outer thirds."""
    return deterministic_cascade(9, (0.5, 0.0, 0.5))


# --------------------------------------------------------------------------- #
# Coarse Hölder exponents, and legacy defect D8
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kmax", [10, 14, 18])
def test_coarse_holder_is_the_exact_combinatorial_identity(kmax: int) -> None:
    r"""Cell exponents equal :math:`-(j\ln m_0+(k-j)\ln m_1)/(k\ln b)` exactly.

    An identity, not an estimate: asserted to 1e-12, observed at 2.2e-16.
    """
    mu = deterministic_cascade(kmax, M)
    alpha = coarse_holder(mu, 2.0**-kmax)

    assert alpha.size == 2**kmax  # the binomial has no empty cells
    assert np.max(np.abs(alpha - _binomial_alpha(kmax))) < 1e-12

    # Multiplicities are the binomial coefficients, exactly.
    _, counts = np.unique(np.round(alpha, 12), return_counts=True)
    expected = np.array([comb(kmax, j) for j in range(kmax + 1)], dtype=np.int64)
    assert np.array_equal(np.sort(counts), np.sort(expected))

    # And the support is the theoretical one.
    a_min, a_max = theory.cascade_alpha_range(M, 2)
    assert alpha.min() == pytest.approx(a_min, abs=1e-12)
    assert alpha.max() == pytest.approx(a_max, abs=1e-12)


@pytest.mark.parametrize("kmax", [10, 12, 14])
def test_coarse_holder_off_by_one_denominator_regression_d8(kmax: int) -> None:
    r"""Normalising by :math:`b^{-(k-1)}` inflates every exponent by :math:`k/(k-1)`.

    Legacy defect D8. At :math:`k=12` the inflation is
    :math:`12/11 = 1.0909`, a 9% multiplicative bias on the whole spectrum.
    """
    mu = deterministic_cascade(kmax, M)
    correct = coarse_holder(mu, 2.0**-kmax)
    wrong = np.log(mu) / math.log(2.0 ** -(kmax - 1))

    ratio = wrong / correct
    assert np.allclose(ratio, kmax / (kmax - 1), rtol=1e-12)
    assert correct.max() == pytest.approx(theory.cascade_alpha_range(M, 2)[1], abs=1e-12)
    if kmax == 12:
        assert float(ratio[0]) == pytest.approx(12 / 11, rel=1e-12)
        assert float(ratio[0]) == pytest.approx(1.0909, abs=1e-4)


def test_coarse_holder_rejects_inconsistent_box_size(binomial14: FloatArray) -> None:
    with pytest.raises(ValueError, match="strictly in"):
        coarse_holder(binomial14, 1.0)
    with pytest.raises(ValueError, match="does not divide"):
        coarse_holder(binomial14, 1.0 / 3.0)
    with pytest.raises(ValueError, match="integer number of boxes"):
        coarse_holder(binomial14, 1.0 / 2.5)
    with pytest.raises(ValueError, match="does not divide"):
        # 2**15 boxes for 2**14 cells: the partition is finer than the measure.
        coarse_holder(binomial14, 2.0**-15)


def test_coarse_holder_requires_unit_mass(binomial14: FloatArray) -> None:
    r"""A total mass :math:`M\neq1` shifts every exponent by :math:`\ln M/\ln\delta`."""
    delta = 2.0**-14
    scaled = 7.0 * binomial14
    assert np.allclose(coarse_holder(scaled, delta), coarse_holder(binomial14, delta))
    shifted = coarse_holder(scaled, delta, normalise=False)
    offset = math.log(7.0) / math.log(delta)
    assert np.allclose(shifted - offset, coarse_holder(binomial14, delta), atol=1e-12)


# --------------------------------------------------------------------------- #
# The partition function and tau(q)
# --------------------------------------------------------------------------- #


def test_box_partition_is_exact_on_the_binomial(binomial14: FloatArray) -> None:
    r""":math:`Z(q,b^{-l}) = (\sum_i m_i^q)^l` exactly, at every level."""
    q = np.array([-2.0, -0.5, 0.0, 1.0, 2.0, 3.0])
    surface = box_partition(binomial14, q=q)
    assert surface.counts is not None
    n = binomial14.size
    for j, delta in enumerate(surface.scales):
        level = round(-math.log(float(delta), 2.0))
        assert surface.counts[j] == round(1.0 / float(delta))
        assert round(1.0 / float(delta)) == 2**level <= n
        for i, qi in enumerate(q):
            expected = (M[0] ** qi + M[1] ** qi) ** level
            assert surface.values[i, j] == pytest.approx(expected, rel=1e-12)

    alias = generalised_dimensions_structure(binomial14, q=q)
    assert np.array_equal(alias.values, surface.values)


def test_tau_matches_the_oracle_to_machine_precision(binomial14: FloatArray) -> None:
    """Multifractal recovery (b). Tolerance 1e-10; observed 3.1e-15."""
    result = generalised_dimensions(binomial14)
    err = float(np.max(np.abs(result.tau - theory.cascade_tau(result.q, M))))
    assert err < 1e-10, f"tau error {err:.3e}"

    # zeta = tau + d, and nothing else.
    assert np.allclose(result.zeta, result.tau + 1.0, atol=0.0, rtol=0.0)
    assert np.allclose(result.zeta, theory.cascade_zeta(result.q, M), atol=1e-10)

    # Contract: everything the standard signature promises is populated.
    assert result.zeta_stderr is not None
    assert result.zeta_ci is not None
    assert len(result.fits) == result.q.size
    assert result.scaling_range is not None
    assert result.structure is not None and result.structure.counts is not None
    assert result.estimator == "box_counting.generalised_dimensions"
    assert result.provenance is not None
    assert all(f.method is FitMethod.WLS for f in result.fits)
    # R^2 is undefined at q = 1, where ln Z == 0 at every scale by mass
    # conservation and the total sum of squares vanishes; everywhere else the
    # scaling is exact.
    informative = np.abs(result.q - 1.0) > 0.1
    assert all(
        f.r_squared > 1.0 - 1e-9 for f, keep in zip(result.fits, informative, strict=True) if keep
    )
    assert result.fits[int(np.argmin(np.abs(result.q - 1.0)))].slope == pytest.approx(
        0.0, abs=1e-12
    )


def test_convention_anchors_tau0_and_tau1(binomial14: FloatArray) -> None:
    r""":math:`\tau(0)=-D_0=-1` and :math:`\tau(1)=0` for a conservative measure."""
    result = generalised_dimensions(binomial14)
    i0 = int(np.argmin(np.abs(result.q - 0.0)))
    i1 = int(np.argmin(np.abs(result.q - 1.0)))
    assert result.q[i0] == 0.0
    assert result.q[i1] == 1.0
    assert result.tau[i0] == pytest.approx(-1.0, abs=1e-12)
    assert result.tau[i1] == pytest.approx(0.0, abs=1e-12)
    # In the stored (signal) convention the same anchors read zeta(0) = 0, zeta(1) = 1.
    assert result.zeta[i0] == pytest.approx(0.0, abs=1e-12)
    assert result.zeta[i1] == pytest.approx(1.0, abs=1e-12)


def test_cantor_support_dimension_and_empty_box_accounting(cantor9: FloatArray) -> None:
    r"""A measure with holes: :math:`D_q\equiv\log_3 2`, and the holes are counted.

    The empty boxes are what make :math:`Z(q\le0)` infinite if they are kept;
    dropping them is the box-counting convention and the number dropped is
    recorded rather than silently absorbed.
    """
    result = dq(cantor9, b=3)
    d_theory = math.log(2.0, 3.0)
    assert result.d0 == pytest.approx(d_theory, abs=1e-12)
    assert result.d1 == pytest.approx(d_theory, abs=1e-12)
    assert result.d2 == pytest.approx(d_theory, abs=1e-12)
    assert np.max(np.abs(result.dq - d_theory)) < 1e-10

    tau_theory = theory.cascade_tau(result.q, (0.5, 0.5), 3)
    assert np.max(np.abs(result.exponents.tau - tau_theory)) < 1e-12

    prov = result.exponents.provenance
    assert prov is not None
    empty = prov.diagnostics["n_empty_boxes"]
    assert empty[0] > 0 and all(e > 0 for e in empty)
    # The b-adic ladder for 3**9 cells stops at s = 3**7, i.e. level l = 2,
    # where the Cantor set occupies 2**l = 4 of the 3**l = 9 boxes.
    occupied = prov.diagnostics["occupied_boxes"]
    assert occupied[0] == 2**9
    assert occupied[-1] == 2**2
    assert empty[-1] == 3**2 - 2**2


def test_generalised_dimensions_landmarks_and_monotonicity(binomial14: FloatArray) -> None:
    r""":math:`D_0=1`, :math:`D_1` analytic, :math:`q\mapsto D_q` non-increasing."""
    result = dq(binomial14)
    d1_theory = float(theory.cascade_dq(1.0, M)[0])

    assert result.d0 == pytest.approx(1.0, abs=1e-12)
    assert result.d1 == pytest.approx(d1_theory, abs=1e-12)
    assert result.d1 == pytest.approx(0.9709505944546686, abs=1e-12)
    assert np.max(np.abs(result.dq - theory.cascade_dq(result.q, M))) < 1e-9
    assert float(np.max(np.diff(result.dq))) <= 1e-12
    assert np.all(np.isfinite(result.dq))
    assert result.dq_stderr.shape == result.q.shape


def test_information_dimension_uses_the_entropy_slope(binomial14: FloatArray) -> None:
    r""":math:`D_1` is the slope of :math:`\sum\mu\ln\mu` against :math:`\ln\delta`.

    Exact at every level for a self-similar measure, so :math:`R^2=1`; and the
    substitution window around :math:`q=1` is what keeps :math:`D_q` finite
    where :math:`\tau(1)/(1-1)` is :math:`0/0`.
    """
    fit = information_dimension(binomial14)
    d1_theory = float(theory.cascade_dq(1.0, M)[0])
    assert fit.slope == pytest.approx(d1_theory, abs=1e-12)
    assert fit.r_squared > 1.0 - 1e-12

    result = dq(binomial14)
    i1 = int(np.argmin(np.abs(result.q - 1.0)))
    assert result.q[i1] == 1.0
    assert math.isfinite(result.dq[i1])
    assert result.dq[i1] == pytest.approx(d1_theory, abs=1e-12)

    # The naive quotient at q = 1 is exactly the 0/0 this avoids.
    with np.errstate(invalid="ignore"):
        naive = result.exponents.tau[i1] / (result.q[i1] - 1.0)
    assert not math.isfinite(naive)


# --------------------------------------------------------------------------- #
# Monofractal null
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("H", [0.3, 0.5, 0.7])
def test_monofractal_null_on_fbm_increment_measure(H: float) -> None:
    r"""Monofractal recovery (a), stated in the convention this family works in.

    Box counting estimates the exponents of a **measure**. The monofractal null
    for a measure is Lebesgue-like: :math:`\alpha\equiv d`, :math:`\tau(q)=(q-1)d`,
    :math:`h(q)=\zeta(q)/q\equiv d=1`, :math:`D_q\equiv 1`.

    The measure canonically attached to a path is the absolute-variation measure
    :math:`\mu_i\propto|\Delta_i X|`. For fBm the increments are stationary with
    a finite mean, so the box masses are asymptotically proportional to the box
    size and the measure is *exactly* the monofractal null --- independently of
    :math:`H`. So the correct assertion here is :math:`h(q)\equiv1`, **not**
    :math:`h(q)\equiv H`: no box-counting statistic of the increment measure of
    fBm carries information about :math:`H`, and asserting otherwise would be
    asserting something false. Recovering :math:`H` from a path is the job of
    the fluctuation-based estimators (structure functions, MFDFA, wavelet
    leaders), which are separate modules.

    Observed on n = 2**16 with box scales 2**6..2**13:
    ``max|h(q) - 1| = 0.0112 / 0.0101 / 0.0135`` at ``H = 0.3 / 0.5 / 0.7``,
    with ``h`` spread ``0.0180 / 0.0164 / 0.0224`` across ``q`` in [-5, 5].
    """
    measure = np.abs(fgn(2**16, H, seed=SEED))
    scales = [2**j for j in range(6, 14)]
    result = generalised_dimensions(measure, scales=scales)

    h = result.hq[np.isfinite(result.hq)]
    spread = float(h.max() - h.min())
    bias = float(np.max(np.abs(h - 1.0)))

    assert spread < 0.03, f"h(q) not flat: spread {spread:.4f}"
    assert bias < 0.02, f"h(q) off the monofractal null by {bias:.4f}"
    assert result.is_monofractal(tol=0.05)

    dims = dq(measure, scales=scales)
    assert dims.d0 == pytest.approx(1.0, abs=0.01)
    assert dims.d1 == pytest.approx(1.0, abs=0.01)


def test_monofractal_null_is_independent_of_hurst() -> None:
    """The same null, three Hurst exponents: the measure view cannot see H."""
    scales = [2**j for j in range(6, 14)]
    curves = [
        generalised_dimensions(np.abs(fgn(2**16, H, seed=SEED)), scales=scales).hq
        for H in (0.3, 0.5, 0.7)
    ]
    stack = np.vstack(curves)
    finite = np.all(np.isfinite(stack), axis=0)
    across_h = float(np.max(np.ptp(stack[:, finite], axis=0)))
    assert across_h < 0.02, f"h(q) varied by {across_h:.4f} across H"


# --------------------------------------------------------------------------- #
# Spectrum: Legendre route
# --------------------------------------------------------------------------- #


def test_legendre_spectrum_matches_the_closed_form(binomial14: FloatArray) -> None:
    r""":math:`f(\alpha)` from the exact :math:`\tau`, against the entropy formula.

    Tolerance 1e-5, justified: :math:`\hat\tau` is exact to 3e-15, so the whole
    residual is the discretisation of the Legendre--Fenchel infimum over the
    refined :math:`q` grid. Observed 2.0e-7.
    """
    spectrum = multifractal_measure_spectrum(binomial14)
    f_theory = theory.cascade_falpha(spectrum.alpha, M)
    ok = np.isfinite(f_theory)
    err = float(np.max(np.abs(spectrum.f_alpha[ok] - f_theory[ok])))
    assert err < 1e-5, f"spectrum error {err:.3e}"
    assert spectrum.descriptors().f_max == pytest.approx(1.0, abs=1e-5)


def test_negative_q_widens_the_recoverable_alpha_range(binomial14: FloatArray) -> None:
    r"""Direct regression test for legacy defect D2.

    The right branch :math:`\alpha>\alpha_0` is generated by :math:`q<0` alone.
    A :math:`q\ge0` grid cannot reach it *in principle*: the reachable range is
    :math:`[\tau'(q_{\max}),\tau'(0)]`, which stops at the apex.
    """
    full = generalised_dimensions(binomial14, q=np.linspace(-5.0, 5.0, 41))
    right_only = generalised_dimensions(binomial14, q=np.linspace(0.0, 5.0, 21))

    s_full = spectrum_from_exponents(full)
    s_half = spectrum_from_exponents(right_only)
    width_full = float(np.ptp(s_full.alpha))
    width_half = float(np.ptp(s_half.alpha))

    assert width_full > 1.9 * width_half, f"{width_full:.4f} vs {width_half:.4f}"

    alpha_0 = float(theory.cascade_alpha(0.0, M)[0])
    a_min, a_max = theory.cascade_alpha_range(M, 2)
    # The q >= 0 grid tops out at the apex; the full grid reaches into the right branch.
    assert s_half.alpha.max() == pytest.approx(alpha_0, abs=1e-3)
    assert s_full.alpha.max() > alpha_0 + 0.2
    assert a_min < s_full.alpha.min() and s_full.alpha.max() < a_max


# --------------------------------------------------------------------------- #
# Spectrum: direct histogram route
# --------------------------------------------------------------------------- #


def _histogram_error(kmax: int, *, bins: int = 8, locator: str = "weighted") -> tuple[float, float]:
    """Max |f_hat - f_theory| over all bins, and over the core (f_theory >= 0.5)."""
    mu = deterministic_cascade(kmax, M)
    scales = [2 ** (kmax - level) for level in range(kmax // 2, kmax + 1)]
    spectrum = histogram_spectrum(
        mu,
        scales=scales,
        bins=bins,
        min_boxes=8,
        alpha_locator=locator,  # type: ignore[arg-type]
    )
    f_theory = theory.cascade_falpha(spectrum.alpha, M)
    ok = np.isfinite(f_theory)
    err = np.abs(spectrum.f_alpha - f_theory)
    core = ok & (f_theory >= 0.5)
    return float(np.max(err[ok])), float(np.max(err[core]))


def test_histogram_spectrum_converges_with_depth() -> None:
    r"""Direct method: convergence measured, not assumed.

    Box scales span levels :math:`k_{\max}/2 \ldots k_{\max}`, 8 fixed
    :math:`\alpha` bins, weighted locator. Observed:

    ==========  ==================  ==================
    kmax        max err (all bins)  max err (core)
    ==========  ==================  ==================
    10          0.4573              0.2223
    14          0.1747              0.1588
    18          0.1301              0.0849
    ==========  ==================  ==================

    The core error falls by 2.6x for a 1.8x increase in :math:`k_{\max}` --- a
    256-fold increase in the number of boxes. That is the
    :math:`1/\sqrt{\#\text{boxes}}` statistical rate degraded by the lattice
    aliasing described in the docstring, and it is *catastrophically* worse than
    the Legendre route, which is exact to 3e-15 on the very same measure. Any
    claim that a histogram spectrum has converged at :math:`k_{\max}\le 18`
    should be read as a claim about the plot, not the number.
    """
    errors = {kmax: _histogram_error(kmax) for kmax in (10, 14, 18)}

    all_errs = [errors[k][0] for k in (10, 14, 18)]
    core_errs = [errors[k][1] for k in (10, 14, 18)]

    assert all_errs[0] > all_errs[1] > all_errs[2], all_errs
    assert core_errs[0] > core_errs[1] > core_errs[2], core_errs
    assert core_errs[2] < 0.10, f"core error at kmax=18 is {core_errs[2]:.4f}"
    assert all_errs[2] < 0.15, f"tail error at kmax=18 is {all_errs[2]:.4f}"

    # The apex is the one place the method is even nearly accurate, and it
    # still overshoots d = 1 by 6.7e-2 at kmax = 18.
    mu = deterministic_cascade(18, M)
    scales = [2 ** (18 - level) for level in range(9, 19)]
    spectrum = histogram_spectrum(mu, scales=scales, bins=8, min_boxes=8)
    assert spectrum.descriptors().f_max == pytest.approx(1.0667, abs=1e-3)


def test_histogram_weighted_locator_beats_the_bin_centre() -> None:
    r"""The :math:`O(\mathrm{d}\alpha)` bin-width bias, isolated.

    A bin's occupancy is dominated by its edge nearest the apex, so the growth
    rate measured for the bin is :math:`f` at that edge. Reporting the bin
    centre mismatches abscissa and ordinate by
    :math:`\tfrac12|f'|\,\mathrm{d}\alpha`. Observed at kmax=18: 0.1301 (all
    bins) / 0.0849 (core) weighted, against 0.2591 / 0.1295 centred.
    """
    weighted_all, weighted_core = _histogram_error(18, locator="weighted")
    centre_all, centre_core = _histogram_error(18, locator="centre")
    assert weighted_all < centre_all
    assert weighted_core < centre_core


def test_bin_width_and_prefactor_cancel_only_in_the_slope() -> None:
    r"""Why :math:`f` is fitted as a slope and not read off one :math:`\delta`.

    A single-scale estimate carries two :math:`O(1/l)` corrections of opposite
    sign, the bin-width term :math:`\ln(\mathrm{d}\alpha)/\ln\delta` and the
    Stirling prefactor :math:`\ln\sqrt{l}/(l\ln b)`. Correcting only the first
    --- the "fix" that looks obviously right --- makes the estimate *worse*.
    Both are intercepts in the slope form and cancel exactly.

    Observed at :math:`k_{\max}=18` with 8 bins
    (:math:`\mathrm{d}\alpha=0.0731`, predicted bin-width term 0.2096), as mean
    absolute error against the entropy formula:

    ========================================  =====
    single scale, no bin-width correction     0.064
    single scale, bin-width corrected         0.183
    slope, bin centres                        0.096
    slope, weighted locator (the default)     0.060
    ========================================  =====
    """
    kmax = 18
    mu = deterministic_cascade(kmax, M)
    delta = 2.0**-kmax
    alpha = coarse_holder(mu, delta)
    edges = np.linspace(alpha.min(), alpha.max(), 9)
    d_alpha = float(edges[1] - edges[0])
    counts = np.histogram(alpha, bins=edges)[0].astype(np.float64)
    occupied = counts > 0

    f_naive = -np.log(counts[occupied]) / math.log(delta)
    f_corrected = -np.log(counts[occupied] / d_alpha) / math.log(delta)
    bin_term = math.log(d_alpha) / math.log(delta)

    assert bin_term > 0.0
    assert np.allclose(f_corrected - f_naive, bin_term, atol=1e-12)
    assert bin_term == pytest.approx(0.2096, abs=1e-3)
    # It decays only logarithmically in delta: 0.377 at 2**-10, 0.210 at 2**-18.
    assert math.log(d_alpha) / math.log(2.0**-10) == pytest.approx(0.3774, abs=1e-3)

    centres = (0.5 * (edges[:-1] + edges[1:]))[occupied]
    f_theory = theory.cascade_falpha(centres, M)
    ok = np.isfinite(f_theory)
    mae_naive = float(np.mean(np.abs(f_naive - f_theory)[ok]))
    mae_corrected = float(np.mean(np.abs(f_corrected - f_theory)[ok]))
    assert mae_corrected > mae_naive  # correcting one of two terms is worse

    scales = [2 ** (kmax - level) for level in range(kmax // 2, kmax + 1)]
    slope = histogram_spectrum(mu, scales=scales, bins=8, min_boxes=8)
    f_slope_theory = theory.cascade_falpha(slope.alpha, M)
    ok_slope = np.isfinite(f_slope_theory)
    mae_slope = float(np.mean(np.abs(slope.f_alpha - f_slope_theory)[ok_slope]))
    assert mae_slope < mae_naive < mae_corrected
    assert mae_slope == pytest.approx(0.060, abs=0.005)


def test_histogram_degrades_when_bins_are_finer_than_the_lattice() -> None:
    r"""Aliasing is visible in the diagnostics, and eventually fatal.

    At level :math:`l` the coarse exponents of a binomial live on a lattice of
    :math:`l+1` points. Bins narrower than that lattice are empty at most
    scales: with 120 bins over levels 5..10 only 3 of 120 survive, and past
    ``min_scales=4`` nothing but the two endpoints does, which is refused.
    """
    mu = deterministic_cascade(10, M)
    scales = [2 ** (10 - level) for level in range(5, 11)]

    aliased = histogram_spectrum(mu, scales=scales, bins=120, min_boxes=8)
    assert aliased.provenance is not None
    assert aliased.provenance.diagnostics["bins_dropped"] == 117
    assert aliased.alpha.size == 3

    with pytest.raises(ValueError, match="alpha bins survived"):
        histogram_spectrum(mu, scales=scales, bins=200, min_boxes=8, min_scales=4)


# --------------------------------------------------------------------------- #
# Determinism, provenance, guards
# --------------------------------------------------------------------------- #


def test_repeated_calls_are_bitwise_identical(binomial14: FloatArray) -> None:
    """Determinism (d): same input and seed give bit-for-bit identical zeta."""
    a = generalised_dimensions(binomial14, seed=SEED)
    b = generalised_dimensions(binomial14, seed=SEED)
    assert a.zeta.tobytes() == b.zeta.tobytes()
    assert a.tau.tobytes() == b.tau.tobytes()
    assert a.zeta_stderr is not None and b.zeta_stderr is not None
    assert a.zeta_stderr.tobytes() == b.zeta_stderr.tobytes()

    sa = histogram_spectrum(binomial14, bins=6, min_boxes=8)
    sb = histogram_spectrum(binomial14, bins=6, min_boxes=8)
    assert sa.f_alpha.tobytes() == sb.f_alpha.tobytes()


def test_provenance_root_is_stable_and_parameter_sensitive(binomial14: FloatArray) -> None:
    """Provenance (e): identical calls share a root; any parameter change moves it."""
    base = generalised_dimensions(binomial14)
    again = generalised_dimensions(binomial14)
    assert base.provenance is not None and again.provenance is not None
    root = base.provenance.root
    assert root == again.provenance.root

    variants = [
        generalised_dimensions(binomial14, q=np.linspace(-4.0, 4.0, 41)),
        generalised_dimensions(binomial14, scales=[2**j for j in range(0, 11)]),
        generalised_dimensions(binomial14, min_boxes=8),
        generalised_dimensions(binomial14, fit_method=FitMethod.OLS),
        generalised_dimensions(binomial14, select_range=True),
        generalised_dimensions(binomial14, d_support=2.0),
        generalised_dimensions(binomial14, seed=1),
        generalised_dimensions(deterministic_cascade(14, (0.7, 0.3))),
    ]
    roots = {root}
    for v in variants:
        assert v.provenance is not None
        assert v.provenance.root != root
        roots.add(v.provenance.root)
    assert len(roots) == len(variants) + 1

    # Diagnostics are non-identifying: attaching them must not move the root.
    assert base.provenance.with_diagnostics(extra=1).root == root


def test_scale_guards(binomial14: FloatArray) -> None:
    """Bad scale grids fail loudly rather than producing a plausible number."""
    with pytest.raises(ValueError, match="do not divide"):
        generalised_dimensions(binomial14, scales=[3, 6, 12])
    with pytest.raises(ValueError, match=r"(no scale|admissible box scales)"):
        # 2**13 cells per box leaves 2 boxes out of 2**14: below the four-box floor.
        generalised_dimensions(binomial14, scales=[2**12, 2**13])
    with pytest.raises(ValueError, match="min_boxes"):
        generalised_dimensions(binomial14, min_boxes=2)
    with pytest.raises(ValueError, match="at least three"):
        generalised_dimensions(binomial14, q=[1.0])


@pytest.mark.parametrize("q_grid", [(-60.0, 5.0, 21), (-5.0, 300.0, 11)])
def test_unrepresentable_moments_are_refused_by_every_entry_point(
    binomial14: FloatArray, q_grid: tuple[float, float, int]
) -> None:
    r""":math:`Z` outside float64 must raise, not be stored as ``inf`` or ``0``.

    ``StructureFunctions`` stores the moments themselves, so the stored surface
    is what a later refit takes the logarithm of. Overflow leaves ``inf`` behind
    a bare ``RuntimeWarning``; underflow leaves an exact ``0`` behind *nothing*.
    Both make ``structure`` unrefittable while ``tau`` still looks perfect,
    because the fit works on ``log Z`` and never sees the damage. The two entry
    points that build the surface must therefore behave identically.
    """
    lo, hi, n = q_grid
    q = np.linspace(lo, hi, n)
    with pytest.raises(ValueError, match="does not fit in float64"):
        box_partition(binomial14, q=q)
    with pytest.raises(ValueError, match="does not fit in float64"):
        generalised_dimensions(binomial14, q=q)

    # And what is stored on an admissible grid really is refittable.
    ok = generalised_dimensions(binomial14, q=np.linspace(-5.0, 5.0, 41))
    assert ok.structure is not None
    assert np.all(np.isfinite(ok.structure.values)) and np.all(ok.structure.values > 0.0)


def test_scaling_range_selection_and_alternative_fit_methods(
    binomial14: FloatArray,
) -> None:
    """The window is chosen once, from q = 2, and shared by every q."""
    auto = generalised_dimensions(binomial14, select_range=True)
    assert auto.scaling_range is not None
    assert auto.scaling_range.selection.startswith("auto-")
    assert all(f.n_points == auto.scaling_range.n_scales for f in auto.fits)
    assert np.max(np.abs(auto.tau - theory.cascade_tau(auto.q, M))) < 1e-10

    for method in (FitMethod.OLS, FitMethod.THEILSEN, FitMethod.HUBER):
        result = generalised_dimensions(binomial14, fit_method=method)
        err = float(np.max(np.abs(result.tau - theory.cascade_tau(result.q, M))))
        assert err < 1e-8, f"{method}: {err:.3e}"


def test_structure_surface_supports_refitting(binomial14: FloatArray) -> None:
    """The scaling surface is kept so exponents can be refitted without recomputing."""
    full = generalised_dimensions(binomial14)
    assert full.structure is not None
    surface = full.structure
    assert surface.values.shape == (full.q.size, surface.scales.size)
    assert surface.counts is not None and np.all(surface.counts > 0)

    # Refit q = 2 by hand over a sub-window and reproduce the reported exponent.
    i = int(np.argmin(np.abs(full.q - 2.0)))
    slope = np.polyfit(np.log(surface.scales), np.log(surface.values[i]), 1)[0]
    assert slope == pytest.approx(full.tau[i], abs=1e-10)
