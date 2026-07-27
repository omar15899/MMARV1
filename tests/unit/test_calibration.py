"""Calibration must recover the MFC identities and diagnose their failure."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmar.calibration import (
    MMARCalibrationError,
    calibrate_from_spectrum,
    critical_moment,
    implied_spectrum,
    lambda_from_alpha0,
    sigma2_from_lambda,
)
from mmar.core.provenance import new_provenance
from mmar.core.types import ScalingExponents
from mmar.models import theory


def _mmar_exponents(H: float, lam: float, *, q_max: float = 5.0, n: int = 81) -> ScalingExponents:
    """Exact MMAR exponents, so calibration is tested against the truth, not an estimate."""
    sigma2 = theory.lognormal_conservation_sigma2(lam)
    q = np.linspace(-q_max, q_max, n)
    tau = theory.mmar_tau_lognormal(q, H, lam, sigma2)
    return ScalingExponents(
        q=q,
        zeta=np.asarray(tau) + 1.0,
        d_support=1.0,
        estimator="exact-mmar",
        provenance=new_provenance("exact-mmar", params={"H": H, "lam": lam}),
    )


@pytest.mark.parametrize(("H", "lam"), [(0.4, 1.1), (0.55, 1.25), (0.7, 1.15), (0.5, 1.4)])
def test_recovers_the_generating_parameters(H: float, lam: float) -> None:
    result = calibrate_from_spectrum(_mmar_exponents(H, lam))
    assert result.admissible, result.diagnosis
    assert result.hurst == pytest.approx(H, abs=2e-3)
    assert result.lam == pytest.approx(lam, abs=5e-3)
    assert result.sigma2 == pytest.approx(theory.lognormal_conservation_sigma2(lam), abs=5e-3)


def test_the_three_identities_are_mutually_consistent() -> None:
    H, lam = 0.55, 1.25
    result = calibrate_from_spectrum(_mmar_exponents(H, lam))
    assert result.alpha_0 == pytest.approx(result.hurst * result.lam, abs=1e-6)
    assert result.sigma2 == pytest.approx(sigma2_from_lambda(result.lam), abs=1e-12)
    assert result.lam == pytest.approx(lambda_from_alpha0(result.alpha_0, result.hurst), abs=1e-12)


def test_width_matches_the_closed_form_prediction() -> None:
    result = calibrate_from_spectrum(_mmar_exponents(0.55, 1.25))
    assert result.width_ratio == pytest.approx(1.0, abs=0.35), (
        f"observed {result.width_observed:.4f} vs implied {result.width_implied:.4f}; "
        "the observed width is limited by the finite q grid, not by the model"
    )


def test_monofractal_input_is_diagnosed_not_silently_calibrated() -> None:
    """alpha_0 = H exactly for an fBm, so lambda = 1 and sigma^2 = 0: no MMAR fits it.

    This is v1's bare ValueError("Negative variance"), now with a diagnosis.
    """
    q = np.linspace(-5.0, 5.0, 81)
    exps = ScalingExponents(q=q, zeta=q * 0.6, d_support=1.0, estimator="fbm")
    result = calibrate_from_spectrum(exps)
    assert not result.admissible
    assert any("lambda" in d and "<= 1" in d for d in result.diagnosis)
    assert any("alpha_0" in d for d in result.diagnosis)
    with pytest.raises(MMARCalibrationError, match="inadmissible"):
        result.to_spec_kwargs()


def test_inadmissible_result_still_reports_its_numbers() -> None:
    """A caller sweeping many series must be able to tabulate failures, not catch them."""
    q = np.linspace(-5.0, 5.0, 81)
    result = calibrate_from_spectrum(ScalingExponents(q=q, zeta=q * 0.6, estimator="fbm"))
    assert not result.admissible
    assert math.isfinite(result.hurst) and math.isfinite(result.alpha_0)
    assert "H" in result.summary() and result.summary()["admissible"] is False


def test_degenerate_cascade_is_rejected_by_kahane_peyriere() -> None:
    """lambda >= 2 makes the canonical cascade converge to the zero measure."""
    result = calibrate_from_spectrum(_mmar_exponents(0.5, 1.25), hurst=0.5, alpha_0=1.05)
    assert result.lam == pytest.approx(2.1, abs=1e-9)
    assert not result.admissible
    assert any("Kahane-Peyriere" in d for d in result.diagnosis)


def test_critical_moment_matches_the_cascade_root() -> None:
    for lam in (1.05, 1.2, 1.5):
        p = critical_moment(lam)
        sigma2 = theory.lognormal_conservation_sigma2(lam)
        assert float(np.asarray(theory.lognormal_tau(p, lam, sigma2)).ravel()[0]) == pytest.approx(
            0.0, abs=1e-10
        ), "p_crit must be the second root of tau_theta"
    assert critical_moment(1.0) == float("inf")


def test_fit_beyond_the_critical_moment_is_flagged() -> None:
    """lambda=1.5 gives p_crit=2, so q beyond p_crit/H=4 probes divergent moments."""
    result = calibrate_from_spectrum(_mmar_exponents(0.5, 1.5, q_max=8.0))
    assert result.q_critical == pytest.approx(2.0, abs=1e-9)
    assert any("q_crit" in d for d in result.diagnosis)


def test_no_tau_root_falls_back_and_says_so() -> None:
    """tau with no positive-q sign change cannot identify H via the MFC identity."""
    q = np.linspace(0.5, 3.0, 21)
    exps = ScalingExponents(q=q, zeta=q * 0.6 + 2.0, d_support=1.0, estimator="rootless")
    result = calibrate_from_spectrum(exps)
    assert any("no root" in d for d in result.diagnosis)
    assert any("falling back to h(2)" in d for d in result.diagnosis)


def test_implied_spectrum_matches_theory_and_is_nan_when_inadmissible() -> None:
    H, lam = 0.55, 1.25
    result = calibrate_from_spectrum(_mmar_exponents(H, lam))
    alpha = np.linspace(result.alpha_0 - 0.1, result.alpha_0 + 0.1, 21)
    expected = theory.mmar_falpha_lognormal(alpha, result.hurst, result.lam, result.sigma2)
    assert np.allclose(implied_spectrum(result, alpha), expected, atol=1e-12)

    q = np.linspace(-5.0, 5.0, 81)
    bad = calibrate_from_spectrum(ScalingExponents(q=q, zeta=q * 0.6, estimator="fbm"))
    assert np.all(np.isnan(implied_spectrum(bad, alpha)))


def test_calibration_provenance_chains_to_the_exponents() -> None:
    exps = _mmar_exponents(0.55, 1.25)
    result = calibrate_from_spectrum(exps)
    assert result.provenance is not None and exps.provenance is not None
    assert exps.provenance.root in {i.digest for i in result.provenance.inputs}
    assert result.provenance.root != exps.provenance.root


@pytest.mark.parametrize(("n_q", "tol"), [(21, 2e-4), (41, 5e-5), (81, 2e-6), (401, 1e-7)])
def test_hurst_tau_root_is_accurate_on_a_coarse_grid(n_q: int, tol: float) -> None:
    """Regression: the root was found by linear interpolation of a convex tau.

    tau is markedly convex near its root, so a chord between the bracketing grid
    points undershoots. At 41 points that put H at 0.5374 against a true 0.55 --
    an error of 1.3e-2, an order of magnitude larger than the sampling error it
    was meant to report. Brent on a PCHIP interpolant gives 1.3e-5 on the same
    grid on the agent's fixture; on this one the chord error is -7.0e-4 against
    -1.3e-5, a factor of 53. Either way the residual is PCHIP's own O(h^4)
    interpolation error rather than a systematic bias -- which is why the
    tolerances below tighten with the grid.
    """
    H, lam = 0.55, 1.25
    exps = _mmar_exponents(H, lam, n=n_q)
    assert exps.hurst_tau_root == pytest.approx(H, abs=tol), (
        f"n_q={n_q} gave H={exps.hurst_tau_root:.8f}"
    )


def test_hurst_tau_root_beats_the_chord_it_replaced() -> None:
    """The chord is not merely less accurate, it is biased low. Pin that."""
    H, lam = 0.55, 1.25
    exps = _mmar_exponents(H, lam, n=41)
    tau, q = exps.tau, exps.q
    pos = q > 0
    qq, tt = q[pos], tau[pos]
    i = int(np.nonzero(np.diff(np.sign(tt)) != 0)[0][0])
    chord_root = qq[i] - tt[i] * (qq[i + 1] - qq[i]) / (tt[i + 1] - tt[i])
    chord_H = 1.0 / chord_root
    assert abs(chord_H - H) > 20 * abs(exps.hurst_tau_root - H)
    assert chord_H < H, "the chord undershoots because tau is convex near its root"
