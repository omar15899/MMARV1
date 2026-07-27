"""The conventions table in core/types.py is documentation, so pin it as code.

Three independent verification agents converged on the same error in that
table: the lognormal row read ``zeta(q) = q*lam - (sigma^2/2) q^2 ln 2 + 1``,
whose ``zeta(0)`` is 1 and ``zeta(1)`` is 2, where they must be 0 and 1. The
file's own header calls itself "law", and it was wrong.

A docstring that states a formula and is checked by nobody will drift again.
These tests transcribe each row of the table independently and assert it against
:mod:`mmar.models.theory`, so the next drift fails CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmar.models import theory

Q = np.linspace(-4.0, 4.0, 33)
pytestmark = pytest.mark.theory


def _zeta_of(tau: np.ndarray, d: float = 1.0) -> np.ndarray:
    return np.asarray(tau) + d


def test_zeta_vanishes_at_zero_for_every_row() -> None:
    """S(0, delta) = 1 for every delta, so the slope is identically zero."""
    assert theory.fbm_zeta(0.0, 0.7) == pytest.approx(0.0, abs=1e-15)
    assert _zeta_of(theory.cascade_tau(0.0, [0.6, 0.4])) == pytest.approx(0.0, abs=1e-15)
    sigma2 = theory.lognormal_conservation_sigma2(1.2)
    assert _zeta_of(theory.lognormal_tau(0.0, 1.2, sigma2)) == pytest.approx(0.0, abs=1e-15)


def test_zeta_is_one_at_q_one_for_a_conservative_cascade() -> None:
    """Mass conservation is tau(1) = 0, hence zeta(1) = 1."""
    assert _zeta_of(theory.cascade_tau(1.0, [0.6, 0.4])) == pytest.approx(1.0, abs=1e-14)
    for lam in (1.05, 1.2, 1.6):
        sigma2 = theory.lognormal_conservation_sigma2(lam)
        assert _zeta_of(theory.lognormal_tau(1.0, lam, sigma2)) == pytest.approx(1.0, abs=1e-14)


def test_table_row_fbm() -> None:
    assert np.allclose(theory.fbm_zeta(Q, 0.7), Q * 0.7, atol=1e-15)


@pytest.mark.parametrize("m0", [0.5, 0.6, 0.7, 0.9])
def test_table_row_binomial(m0: float) -> None:
    """zeta(q) = 1 - log2(m0^q + m1^q), transcribed straight from the table."""
    expected = 1.0 - np.log2(m0**Q + (1.0 - m0) ** Q)
    assert np.allclose(_zeta_of(theory.cascade_tau(Q, [m0, 1 - m0])), expected, atol=1e-13)


@pytest.mark.parametrize("lam", [1.05, 1.2, 1.6])
def test_table_row_lognormal(lam: float) -> None:
    """zeta(q) = q*lam - (sigma^2 ln2 / 2) q^2, with NO constant term.

    Regression for the '+1' that three verifiers independently caught.
    """
    sigma2 = theory.lognormal_conservation_sigma2(lam)
    expected = Q * lam - (sigma2 * np.log(2) / 2.0) * Q**2
    assert np.allclose(_zeta_of(theory.lognormal_tau(Q, lam, sigma2)), expected, atol=1e-13)

    wrong = expected + 1.0
    assert not np.allclose(_zeta_of(theory.lognormal_tau(Q, lam, sigma2)), wrong, atol=1e-3)


def test_the_documented_table_text_matches_these_formulas() -> None:
    """Guard the prose itself: the stale '+ 1' must not come back."""
    from mmar.core import types

    doc = types.__doc__ or ""
    assert r"q\lambda - \tfrac{\sigma^2\ln 2}{2}q^2" in doc
    assert r"\tfrac{\sigma^2}{2}q^2 \ln 2 + 1" not in doc, "the incorrect lognormal row is back"


def test_measure_and_signal_conventions_differ_by_exactly_d() -> None:
    """zeta = tau + d is the only bridge between the two views."""
    for d in (1.0, 2.0):
        tau = np.asarray(theory.cascade_tau(Q, [0.6, 0.4]))
        assert np.allclose(theory.cascade_zeta(Q, [0.6, 0.4], d=d), tau + d, atol=0.0)
