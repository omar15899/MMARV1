r"""Self-consistency of the closed-form oracles in :mod:`mmar.models.theory`.

Nothing here validates an *estimator*: it validates the oracle itself. Every
assertion is either an analytic anchor (:math:`\tau(0)=-1`, :math:`\tau(1)=0`,
:math:`f(\alpha_0)=1`), a cross-check of two independent derivations of the same
object (closed-form vs parametric vs numerical Legendre), or a numerical
derivative confirming an analytic one.

If this file fails, every downstream tolerance in the library is meaningless.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmar.core.legendre import legendre_fenchel
from mmar.core.types import FloatArray
from mmar.models.theory import (
    cascade_alpha,
    cascade_alpha_range,
    cascade_dq,
    cascade_exponents,
    cascade_falpha,
    cascade_falpha_parametric,
    cascade_spectrum,
    cascade_tau,
    cascade_zeta,
    cumulants_from_tau_lognormal,
    fbm_exponents,
    fbm_falpha,
    fbm_tau,
    fbm_zeta,
    lognormal_alpha_range,
    lognormal_conservation_sigma2,
    lognormal_exponents,
    lognormal_falpha,
    lognormal_lambda_from_sigma2,
    lognormal_spectrum,
    lognormal_tau,
    mmar_alpha0,
    mmar_exponents_lognormal,
    mmar_falpha_lognormal,
    mmar_hurst_from_tau,
    mmar_lambda_from_alpha0,
    mmar_spectrum_lognormal,
    mmar_tau,
    mmar_tau_lognormal,
)

pytestmark = pytest.mark.theory

M_BINOM = (0.6, 0.4)


def _central_diff(fn, x: FloatArray, h: float) -> FloatArray:  # type: ignore[no-untyped-def]
    """Fourth-order central difference; error :math:`O(h^4)`."""
    return (fn(x - 2 * h) - 8 * fn(x - h) + 8 * fn(x + h) - fn(x + 2 * h)) / (12 * h)


# --------------------------------------------------------------------------- #
# Deterministic binomial cascade
# --------------------------------------------------------------------------- #


class TestBinomialCascade:
    def test_tau_anchors(self) -> None:
        tau = cascade_tau([0.0, 1.0], M_BINOM)
        assert abs(tau[0] - (-1.0)) < 1e-12
        assert abs(tau[1] - 0.0) < 1e-12

    def test_zeta_is_tau_plus_d(self) -> None:
        q = np.linspace(-5.0, 5.0, 41)
        for d in (1.0, 2.0):
            assert np.allclose(cascade_zeta(q, M_BINOM, d=d), cascade_tau(q, M_BINOM) + d, atol=0)

    def test_alpha_range(self) -> None:
        a_min, a_max = cascade_alpha_range(M_BINOM)
        assert abs(a_min - (-math.log2(0.6))) < 1e-15
        assert abs(a_max - (-math.log2(0.4))) < 1e-15
        assert a_min < a_max

    def test_falpha_apex_and_endpoints(self) -> None:
        a_min, a_max = cascade_alpha_range(M_BINOM)
        a0 = 0.5 * (a_min + a_max)
        assert abs(float(cascade_falpha(a0, M_BINOM)[0]) - 1.0) < 1e-14
        edges = cascade_falpha([a_min, a_max], M_BINOM)
        assert abs(float(edges[0])) < 1e-14
        assert abs(float(edges[1])) < 1e-14

    def test_falpha_apex_matches_q_zero(self) -> None:
        alpha0, f0 = cascade_falpha_parametric([0.0], M_BINOM)
        a_min, a_max = cascade_alpha_range(M_BINOM)
        assert abs(float(alpha0[0]) - 0.5 * (a_min + a_max)) < 1e-14
        assert abs(float(f0[0]) - 1.0) < 1e-14

    def test_falpha_closed_form_matches_parametric(self) -> None:
        q = np.linspace(-25.0, 25.0, 501)
        alpha, f_par = cascade_falpha_parametric(q, M_BINOM)
        f_closed = cascade_falpha(alpha, M_BINOM)
        err = float(np.max(np.abs(f_par - f_closed)))
        assert err < 1e-10, f"closed-form vs parametric f(alpha) differ by {err:.3e}"

    @pytest.mark.parametrize("base", [2, 3, 4, 8])
    def test_falpha_closed_form_matches_parametric_for_every_base(self, base: int) -> None:
        r"""Two branches out of ``base``: the entropy must be measured in base-b digits.

        Counting is binary (:math:`\binom{k}{tk}` strings) but boxes shrink by
        :math:`b`, so :math:`f=H_2(t)\log_b 2`. Dropping the :math:`\log_b 2`
        puts the apex at 1 for a support of dimension :math:`\log_b 2 < 1`.
        """
        m = (0.7, 0.3)
        q = np.linspace(-25.0, 25.0, 501)
        alpha, f_par = cascade_falpha_parametric(q, m, base)
        err = float(np.max(np.abs(f_par - cascade_falpha(alpha, m, base))))
        assert err < 1e-10, f"b={base}: closed form vs parametric differ by {err:.3e}"

    @pytest.mark.parametrize("base", [2, 3, 4, 8])
    def test_falpha_apex_is_the_box_dimension_of_the_support(self, base: int) -> None:
        m = (0.7, 0.3)
        a_min, a_max = cascade_alpha_range(m, base)
        apex = float(cascade_falpha([0.5 * (a_min + a_max)], m, base)[0])
        d0 = float(-cascade_tau([0.0], m, base)[0])
        assert abs(apex - d0) < 1e-14
        assert abs(d0 - math.log(2.0) / math.log(base)) < 1e-14

    def test_falpha_tolerates_a_parametric_alpha_at_the_support_edge(self) -> None:
        """alpha(q) saturates and lands one ULP outside; -inf there would be a lie."""
        m = (0.7, 0.3)
        alpha, f_par = cascade_falpha_parametric(np.linspace(-60.0, 60.0, 2401), m)
        f_closed = cascade_falpha(alpha, m)
        assert np.all(np.isfinite(f_closed)), (
            f"{int(np.sum(~np.isfinite(f_closed)))} of {alpha.size} parametric alphas "
            "fall outside the closed-form support"
        )
        assert float(np.max(np.abs(f_par - f_closed))) < 1e-12

    def test_falpha_outside_support_is_minus_inf(self) -> None:
        a_min, a_max = cascade_alpha_range(M_BINOM)
        out = cascade_falpha([a_min - 0.1, a_max + 0.1], M_BINOM)
        assert np.all(np.isneginf(out))

    def test_alpha_matches_finite_difference_of_tau(self) -> None:
        q = np.linspace(-6.0, 6.0, 121)
        num = _central_diff(lambda z: cascade_tau(z, M_BINOM), q, 1e-3)
        err = float(np.max(np.abs(cascade_alpha(q, M_BINOM) - num)))
        assert err < 1e-8, f"alpha(q) vs d(tau)/dq differ by {err:.3e}"

    def test_alpha_is_positive_and_decreasing(self) -> None:
        q = np.linspace(-20.0, 20.0, 401)
        a = cascade_alpha(q, M_BINOM)
        a_min, a_max = cascade_alpha_range(M_BINOM)
        assert np.all(a > 0.0)
        assert np.all(np.diff(a) <= 1e-15)
        assert a_min - 1e-12 <= a.min() and a.max() <= a_max + 1e-12

    def test_tau_is_concave(self) -> None:
        q = np.linspace(-8.0, 8.0, 321)
        second = np.diff(cascade_tau(q, M_BINOM), 2)
        assert np.all(second < 1e-14)

    def test_symmetric_weights_are_monofractal(self) -> None:
        q = np.linspace(-4.0, 4.0, 33)
        assert np.allclose(cascade_tau(q, (0.5, 0.5)), q - 1.0, atol=1e-14)
        with pytest.raises(ValueError, match="Lebesgue"):
            cascade_falpha([1.0], (0.5, 0.5))

    def test_high_q_does_not_overflow(self) -> None:
        tau = cascade_tau([-400.0, 400.0], (0.05, 0.95))
        assert np.all(np.isfinite(tau))


class TestGeneralisedDimensions:
    def test_d0_and_d1(self) -> None:
        m = np.asarray(M_BINOM)
        d = cascade_dq([0.0, 1.0], M_BINOM)
        assert abs(float(d[0]) - 1.0) < 1e-14
        d1_analytic = float(-np.sum(m * np.log2(m)))
        assert abs(float(d[1]) - d1_analytic) < 1e-14

    def test_d1_is_the_limit_of_the_quotient(self) -> None:
        d1 = float(cascade_dq([1.0], M_BINOM)[0])
        for eps in (1e-3, 1e-4, 1e-5):
            near = float(cascade_tau([1.0 + eps], M_BINOM)[0]) / eps
            assert abs(near - d1) < 5.0 * eps

    def test_dq_monotone_non_increasing(self) -> None:
        q = np.linspace(-5.0, 5.0, 81)
        assert 1.0 in set(q.tolist())
        d = cascade_dq(q, M_BINOM)
        assert np.all(np.diff(d) <= 1e-12)

    def test_reduced_support_lowers_d0(self) -> None:
        # Two surviving branches out of four: a Cantor-type support of dimension 1/2.
        d0 = float(cascade_dq([0.0], (0.5, 0.5), b=4)[0])
        assert abs(d0 - 0.5) < 1e-14


class TestCascadeValidation:
    def test_non_conservative_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="conservative"):
            cascade_tau([1.0], (0.6, 0.5))

    def test_zero_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero cells"):
            cascade_tau([1.0], (0.6, 0.4, 0.0))

    def test_base_smaller_than_branches_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be smaller"):
            cascade_tau([1.0], (0.25, 0.25, 0.25, 0.25), b=2)

    def test_falpha_rejects_non_binomial(self) -> None:
        with pytest.raises(ValueError, match="binomial"):
            cascade_falpha([1.0], (0.4, 0.4, 0.2))

    def test_non_finite_q_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            cascade_tau([0.0, np.nan], M_BINOM)


# --------------------------------------------------------------------------- #
# Lognormal cascade
# --------------------------------------------------------------------------- #

LAM = 1.05
B = 2.0
S2 = lognormal_conservation_sigma2(LAM, B)


class TestLognormalCascade:
    def test_conservation_round_trip(self) -> None:
        assert abs(lognormal_lambda_from_sigma2(S2, B) - LAM) < 1e-15
        assert abs(S2 - 2.0 * (LAM - 1.0) / math.log(B)) < 1e-15

    def test_conservation_requires_lam_above_one(self) -> None:
        with pytest.raises(ValueError, match="lam > 1"):
            lognormal_conservation_sigma2(1.0, B)

    def test_tau_anchors(self) -> None:
        tau = lognormal_tau([0.0, 1.0], LAM, S2, B)
        assert abs(float(tau[0]) - (-1.0)) < 1e-15
        assert abs(float(tau[1])) < 1e-15

    def test_falpha_apex(self) -> None:
        assert abs(float(lognormal_falpha([LAM], LAM, S2, B)[0]) - 1.0) < 1e-15

    def test_falpha_vanishes_at_the_analytic_edges(self) -> None:
        lo, hi = lognormal_alpha_range(LAM, S2, B)
        f = lognormal_falpha([lo, hi], LAM, S2, B)
        assert float(np.max(np.abs(f))) < 1e-14
        assert abs((hi - lo) - 2.0 * math.sqrt(2.0 * S2 * math.log(B))) < 1e-15

    def test_alpha_of_q_matches_derivative(self) -> None:
        q = np.linspace(-6.0, 6.0, 121)
        num = _central_diff(lambda z: lognormal_tau(z, LAM, S2, B), q, 1e-3)
        assert float(np.max(np.abs(num - (LAM - q * S2 * math.log(B))))) < 1e-10

    def test_parametric_legendre_reproduces_the_parabola(self) -> None:
        r"""alpha(q) = lam - q sigma^2 ln b, f = q alpha - tau, against the closed form."""
        q = np.linspace(-10.0, 10.0, 401)
        kappa = S2 * math.log(B)
        alpha = LAM - q * kappa
        f_par = q * alpha - lognormal_tau(q, LAM, S2, B)
        err = float(np.max(np.abs(f_par - lognormal_falpha(alpha, LAM, S2, B))))
        assert err < 1e-10, f"parametric vs closed-form parabola differ by {err:.3e}"

    def test_cumulants(self) -> None:
        c1, c2 = cumulants_from_tau_lognormal(LAM, S2, B)
        assert abs(c1 - LAM) < 1e-15
        assert abs(c2 + S2 * math.log(B)) < 1e-15
        assert c2 < 0.0
        # The apex sits above D_0 by exactly half the intermittency.
        assert abs(c1 - (1.0 - 0.5 * c2)) < 1e-15

    def test_curvature_matches_c2(self) -> None:
        c1, c2 = cumulants_from_tau_lognormal(LAM, S2, B)
        a = np.linspace(c1 - 0.2, c1 + 0.2, 9)
        f = lognormal_falpha(a, LAM, S2, B)
        assert np.allclose(f, 1.0 + (a - c1) ** 2 / (2.0 * c2), atol=1e-15)

    def test_zero_sigma2_rejected(self) -> None:
        with pytest.raises(ValueError, match="fbm_falpha"):
            lognormal_falpha([1.0], LAM, 0.0, B)

    @pytest.mark.parametrize("base", [2.0, math.e, 4.0, 10.0])
    def test_anchors_hold_for_every_base(self, base: float) -> None:
        s2 = lognormal_conservation_sigma2(1.08, base)
        tau = lognormal_tau([0.0, 1.0], 1.08, s2, base)
        assert abs(float(tau[0]) + 1.0) < 1e-14
        assert abs(float(tau[1])) < 1e-14


class TestLegendreRoundTrip:
    r"""Feed the exact zeta into the library's numerical Legendre-Fenchel transform."""

    def test_fenchel_recovers_the_parabola(self) -> None:
        q = np.linspace(-8.0, 8.0, 161)
        zeta = lognormal_tau(q, LAM, S2, B) + 1.0
        kappa = S2 * math.log(B)
        # Restrict to the alpha range the q window can resolve: alpha = lam - q*kappa.
        a_lo, a_hi = LAM - 6.0 * kappa, LAM + 6.0 * kappa
        grid = np.linspace(a_lo, a_hi, 401)
        alpha, f_hat = legendre_fenchel(q, zeta, d_support=1.0, alpha_grid=grid)
        err = float(np.max(np.abs(f_hat - lognormal_falpha(alpha, LAM, S2, B))))
        assert err < 1e-3, f"Legendre-Fenchel round-trip error {err:.3e}"
        print(f"\n[legendre round-trip] max |f_fenchel - f_exact| = {err:.3e}")

    def test_fenchel_recovers_the_binomial_spectrum(self) -> None:
        q = np.linspace(-12.0, 12.0, 241)
        zeta = cascade_tau(q, M_BINOM) + 1.0
        alpha, f_hat = legendre_fenchel(q, zeta, d_support=1.0, n_alpha=201)
        err = float(np.max(np.abs(f_hat - cascade_falpha(alpha, M_BINOM))))
        assert err < 1e-3, f"binomial Legendre-Fenchel round-trip error {err:.3e}"
        print(f"\n[legendre round-trip, binomial] max |f_fenchel - f_exact| = {err:.3e}")


# --------------------------------------------------------------------------- #
# Monofractal
# --------------------------------------------------------------------------- #


class TestFbm:
    @pytest.mark.parametrize("H", [0.2, 0.5, 0.75])
    def test_zeta_is_linear(self, H: float) -> None:
        q = np.linspace(-4.0, 4.0, 33)
        assert np.allclose(fbm_zeta(q, H), q * H, atol=0)
        assert np.allclose(fbm_tau(q, H, d=1.0), q * H - 1.0, atol=0)
        assert float(np.max(np.abs(np.diff(fbm_zeta(q, H), 2)))) < 1e-15

    def test_tau_root_is_one_over_hurst(self) -> None:
        H = 0.65
        assert abs(float(fbm_tau([1.0 / H], H)[0])) < 1e-15

    def test_falpha_is_a_spike(self) -> None:
        H = 0.4
        f = fbm_falpha([H, H - 1e-9, H + 1e-9], H, d=1.0)
        assert float(f[0]) == 1.0
        assert np.all(np.isneginf(f[1:]))

    def test_fenchel_of_a_line_is_a_degenerate_spike(self) -> None:
        H = 0.35
        q = np.linspace(-5.0, 5.0, 101)
        alpha, f = legendre_fenchel(q, fbm_zeta(q, H), d_support=1.0, n_alpha=401)
        peak = int(np.argmax(f))
        assert abs(float(alpha[peak]) - H) < 2e-3
        assert abs(float(f[peak]) - 1.0) < 1e-6
        # Away from H the transform collapses linearly at slope +-q_max.
        far = np.abs(alpha - H) > 0.02
        assert np.all(f[far] < 1.0 - 0.02 * 5.0 + 1e-6)

    def test_hurst_outside_the_unit_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            fbm_zeta([1.0], 1.5)


# --------------------------------------------------------------------------- #
# MMAR
# --------------------------------------------------------------------------- #


class TestMmar:
    @pytest.mark.parametrize("H", [0.4, 0.55, 0.7, 0.9])
    @pytest.mark.parametrize("lam", [1.02, 1.05, 1.2])
    def test_tau_vanishes_at_one_over_hurst(self, H: float, lam: float) -> None:
        s2 = lognormal_conservation_sigma2(lam, B)
        tau = mmar_tau_lognormal([1.0 / H], H, lam, s2, B)
        assert abs(float(tau[0])) < 1e-12, f"tau_X(1/H) = {float(tau[0]):.3e} for H={H}, lam={lam}"

    def test_tau_zero_is_minus_one(self) -> None:
        assert abs(float(mmar_tau_lognormal([0.0], 0.55, LAM, S2, B)[0]) + 1.0) < 1e-15

    def test_generic_mmar_matches_the_lognormal_specialisation(self) -> None:
        H, q = 0.6, np.linspace(-4.0, 4.0, 81)
        generic = mmar_tau(q, H, lambda p: lognormal_tau(p, LAM, S2, B))
        assert np.allclose(generic, mmar_tau_lognormal(q, H, LAM, S2, B), atol=0)

    def test_generic_mmar_over_a_binomial_trading_time(self) -> None:
        H = 0.5
        tau = mmar_tau([1.0 / H], H, lambda p: cascade_tau(p, M_BINOM))
        assert abs(float(tau[0])) < 1e-14

    def test_spectrum_is_the_trading_time_spectrum_stretched(self) -> None:
        H = 0.55
        a_theta = np.linspace(LAM - 0.3, LAM + 0.3, 101)
        assert np.allclose(
            mmar_falpha_lognormal(H * a_theta, H, LAM, S2, B),
            lognormal_falpha(a_theta, LAM, S2, B),
            atol=1e-14,
        )

    def test_alpha0_identity(self) -> None:
        H = 0.55
        a0 = mmar_alpha0(H, LAM)
        assert abs(a0 - H * LAM) < 1e-15
        assert abs(mmar_lambda_from_alpha0(a0, H) - LAM) < 1e-15
        # alpha_0 is the argmax of the MMAR spectrum.
        a = np.linspace(a0 - 0.3, a0 + 0.3, 601)
        f = mmar_falpha_lognormal(a, H, LAM, S2, B)
        assert abs(float(a[int(np.argmax(f))]) - a0) < 1e-3
        assert abs(float(f.max()) - 1.0) < 1e-9

    @pytest.mark.parametrize("H", [0.35, 0.55, 0.8])
    def test_hurst_recovered_from_tau(self, H: float) -> None:
        q = np.linspace(0.05, 6.0, 601)
        tau = mmar_tau_lognormal(q, H, LAM, S2, B)
        assert abs(mmar_hurst_from_tau(q, tau) - H) < 1e-10

    def test_hurst_from_tau_needs_a_sign_change(self) -> None:
        q = np.linspace(0.05, 1.0, 40)
        tau = mmar_tau_lognormal(q, 0.4, LAM, S2, B)
        with pytest.raises(ValueError, match="no sign change"):
            mmar_hurst_from_tau(q, tau)

    def test_hurst_from_tau_ignores_the_down_crossing(self) -> None:
        r"""tau_X has TWO positive roots; only the up-crossing is 1/H.

        For the lognormal trading time tau_theta(p) = p*lam - p^2*kappa/2 - 1
        vanishes at p = 1 and at p = (2*lam - kappa)/kappa = 20 for
        (lam, kappa) = (1.05, 0.1), i.e. at q = 1/H and q = 20/H. A window that
        contains only the second must refuse to answer, not return 1/(20/H).
        """
        H = 0.55
        q = np.linspace(2.5, 45.0, 2001)  # 1/H = 1.82 excluded, 20/H = 36.4 included
        tau = mmar_tau_lognormal(q, H, LAM, S2, B)
        assert tau[0] > 0.0 and tau[-1] < 0.0, "the window must contain a down-crossing"
        with pytest.raises(ValueError, match="no sign change"):
            mmar_hurst_from_tau(q, tau)

    def test_hurst_from_tau_with_both_roots_in_the_window(self) -> None:
        H = 0.55
        q = np.linspace(0.05, 60.0, 4001)
        tau = mmar_tau_lognormal(q, H, LAM, S2, B)
        assert np.any(tau > 0.0) and tau[-1] < 0.0
        assert abs(mmar_hurst_from_tau(q, tau) - H) < 1e-8

    def test_hurst_from_tau_when_the_root_is_a_grid_point(self) -> None:
        H = 0.55
        q = np.concatenate([[1.0 / H], np.linspace(2.0, 6.0, 21)])
        tau = mmar_tau_lognormal(q, H, LAM, S2, B)
        assert float(tau[0]) == 0.0
        assert abs(mmar_hurst_from_tau(q, tau) - H) < 1e-12

    def test_hurst_from_tau_on_a_binomial_trading_time(self) -> None:
        H = 0.5
        q = np.linspace(0.05, 6.0, 601)
        tau = mmar_tau(q, H, lambda p: cascade_tau(p, M_BINOM))
        assert abs(mmar_hurst_from_tau(q, tau) - H) < 1e-10


# --------------------------------------------------------------------------- #
# Result objects and provenance
# --------------------------------------------------------------------------- #


class TestResultObjects:
    def test_cascade_exponents_carry_both_conventions(self) -> None:
        q = np.linspace(-5.0, 5.0, 41)
        ex = cascade_exponents(q, M_BINOM)
        assert np.allclose(ex.tau, cascade_tau(q, M_BINOM), atol=0)
        assert np.allclose(ex.zeta, ex.tau + 1.0, atol=0)
        assert ex.provenance is not None
        assert len(ex.provenance.root) == 64
        assert not ex.is_monofractal()

    def test_provenance_is_reproducible_and_input_sensitive(self) -> None:
        q = np.linspace(-5.0, 5.0, 41)
        a = cascade_exponents(q, M_BINOM).provenance
        b = cascade_exponents(q, M_BINOM).provenance
        c = cascade_exponents(q, (0.7, 0.3)).provenance
        assert a is not None and b is not None and c is not None
        assert a.root == b.root
        assert a.root != c.root

    def test_cascade_spectrum_apex(self) -> None:
        sp = cascade_spectrum(np.linspace(-8.0, 8.0, 161), M_BINOM)
        d = sp.descriptors()
        a_min, a_max = cascade_alpha_range(M_BINOM)
        assert abs(d.f_max - 1.0) < 1e-12
        assert abs(d.alpha_0 - 0.5 * (a_min + a_max)) < 1e-12
        assert d.width < (a_max - a_min)

    def test_lognormal_objects(self) -> None:
        ex = lognormal_exponents(np.linspace(-5.0, 5.0, 41), lam=LAM, b=B)
        assert abs(ex.at(1.0) - 1.0) < 1e-12
        assert abs(ex.hurst_tau_root - 1.0) < 1e-6
        sp = lognormal_spectrum(lam=LAM, b=B)
        lo, hi = lognormal_alpha_range(LAM, S2, B)
        d = sp.descriptors()
        assert abs(d.alpha_min - lo) < 1e-12
        assert abs(d.alpha_max - hi) < 1e-12
        assert abs(d.f_max - 1.0) < 1e-12
        assert sp.provenance is not None

    def test_fbm_exponents_are_monofractal(self) -> None:
        ex = fbm_exponents(np.linspace(-5.0, 5.0, 41), H=0.6)
        assert ex.is_monofractal()
        assert abs(ex.h2 - 0.6) < 1e-15
        assert abs(ex.hurst_tau_root - 0.6) < 1e-12

    def test_mmar_objects(self) -> None:
        H = 0.55
        q = np.linspace(-5.0, 5.0, 401)
        ex = mmar_exponents_lognormal(q, H=H, b=B)
        assert np.allclose(ex.tau, mmar_tau_lognormal(q, H, LAM, S2, B), atol=0)
        # ScalingExponents.hurst_tau_root interpolates the root linearly, so its
        # error is O(h^2 tau''/tau'); the oracle's own root-finder is far tighter.
        assert abs(ex.hurst_tau_root - H) < 1e-4
        assert abs(mmar_hurst_from_tau(ex.q, ex.tau) - H) < 1e-9
        sp = mmar_spectrum_lognormal(H=H, b=B)
        assert abs(sp.descriptors().alpha_0 - mmar_alpha0(H, LAM)) < 1e-3
        assert abs(sp.descriptors().f_max - 1.0) < 1e-12
        assert sp.provenance is not None
