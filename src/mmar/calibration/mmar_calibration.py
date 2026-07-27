r"""Calibrating the MMAR from an estimated spectrum.

The Mandelbrot--Fisher--Calvet model has three parameters: the Hurst exponent
:math:`H` of the subordinated fBm, and the trading-time cascade's
:math:`(\lambda, \sigma^2)`. Under the lognormal cascade they are pinned by
three identities, each of which is exact and each of which is a *testable*
constraint rather than a fitting convenience:

.. math::
    \tau_X(1/H) = 0, \qquad \alpha_0 = H\lambda, \qquad
    \sigma^2 = \frac{2(\lambda-1)}{\ln b}.

The first follows from :math:`\tau_X(q)=\tau_\theta(qH)` together with mass
conservation :math:`\tau_\theta(1)=0`; the second from
:math:`\alpha_0^X = H\,\alpha_0^\theta` and :math:`\alpha_0^\theta=\lambda`; the
third from :math:`\mathbb{E}[b^{-V}]=1/b`.

Because the system is over-determined --- three identities, three parameters,
but :math:`H` and :math:`\alpha_0` are estimated *independently* --- calibration
is also a **specification test**. If the estimated :math:`\alpha_0` and
:math:`H` imply :math:`\lambda \le 1`, then :math:`\sigma^2 \le 0` and no
lognormal MMAR can have produced the data. That is information, not an
inconvenience.

v1 responded to this case with ``ValueError("Negative variance: inconsistent
parameters.")`` and no diagnosis. :class:`CalibrationResult` reports which
identity failed, by how much, and what the data would have had to look like for
the model to fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mmar.core.provenance import InputRef, Provenance
from mmar.core.types import FloatArray, HolderSpectrum, ScalingExponents

__all__ = [
    "CalibrationResult",
    "MMARCalibrationError",
    "calibrate_from_spectrum",
    "critical_moment",
    "lambda_from_alpha0",
    "sigma2_from_lambda",
]


class MMARCalibrationError(ValueError):
    """The estimated spectrum is inconsistent with a lognormal MMAR."""


def sigma2_from_lambda(lam: float, b: float = 2.0) -> float:
    r""":math:`\sigma^2 = 2(\lambda-1)/\ln b`, from :math:`\mathbb{E}[b^{-V}]=1/b`."""
    if b <= 1:
        raise ValueError(f"cascade base must exceed 1, got b={b}")
    return 2.0 * (lam - 1.0) / math.log(b)


def lambda_from_alpha0(alpha_0: float, hurst: float) -> float:
    r""":math:`\lambda = \alpha_0 / H`, from :math:`\alpha_0^X = H\alpha_0^\theta`."""
    if not 0.0 < hurst < 1.0:
        raise ValueError(f"the Hurst exponent must lie in (0,1), got H={hurst}")
    return alpha_0 / hurst


def critical_moment(lam: float) -> float:
    r"""Largest :math:`p` with :math:`\mathbb{E}[(\Delta\theta)^p]<\infty`.

    For a conservative lognormal cascade :math:`\tau_\theta(p)>0` exactly on
    :math:`1<p<p_{\rm crit}=1/(\lambda-1)`, the second root of
    :math:`\tau_\theta`. Moments of the trading time beyond :math:`p_{\rm crit}`
    diverge, so estimated exponents at :math:`q>p_{\rm crit}/H` describe the
    sample maximum rather than the process (Kahane & Peyrière 1976).
    """
    return float("inf") if lam <= 1.0 else 1.0 / (lam - 1.0)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Calibrated MMAR parameters, with every consistency check reported.

    ``admissible`` is the verdict. When it is false, ``diagnosis`` says which
    identity failed and by how much, and the parameter fields are still
    populated so the failure can be inspected rather than guessed at.
    """

    hurst: float
    alpha_0: float
    lam: float
    sigma2: float
    base: float
    admissible: bool
    diagnosis: tuple[str, ...]
    q_critical: float
    """Moment order beyond which the estimated exponents are sample artefacts."""
    width_implied: float
    """Spectrum width the calibrated parameters predict, :math:`2H\\sqrt{2\\sigma^2\\ln b}`."""
    width_observed: float
    provenance: Provenance | None = None

    @property
    def width_ratio(self) -> float:
        """Observed over implied width. Far from 1 means the lognormal shape is wrong."""
        return self.width_observed / self.width_implied if self.width_implied > 0 else float("nan")

    def to_spec_kwargs(self) -> dict[str, Any]:
        """Arguments for :class:`mmar.models.mmar.MMARSpec`.

        Raises
        ------
        MMARCalibrationError
            If the calibration was not admissible. Simulating from inadmissible
            parameters produces a path with no relation to the data, so the
            refusal is deliberate.
        """
        if not self.admissible:
            raise MMARCalibrationError(
                "refusing to build an MMARSpec from an inadmissible calibration:\n  "
                + "\n  ".join(self.diagnosis)
            )
        return {"H": self.hurst, "lam": self.lam, "sigma2": self.sigma2, "b": int(self.base)}

    def summary(self) -> dict[str, Any]:
        return {
            "H": self.hurst,
            "alpha_0": self.alpha_0,
            "lambda": self.lam,
            "sigma2": self.sigma2,
            "admissible": self.admissible,
            "q_critical": self.q_critical,
            "width_observed": self.width_observed,
            "width_implied": self.width_implied,
            "width_ratio": self.width_ratio,
        }

    def __str__(self) -> str:
        verdict = "admissible" if self.admissible else "INADMISSIBLE"
        head = (
            f"MMAR calibration [{verdict}]  H={self.hurst:.4f}  alpha_0={self.alpha_0:.4f}  "
            f"lambda={self.lam:.4f}  sigma2={self.sigma2:.4f}"
        )
        return head + ("" if not self.diagnosis else "\n  " + "\n  ".join(self.diagnosis))


def calibrate_from_spectrum(
    exponents: ScalingExponents,
    spectrum: HolderSpectrum | None = None,
    *,
    base: float = 2.0,
    hurst: float | None = None,
    alpha_0: float | None = None,
) -> CalibrationResult:
    r"""Recover :math:`(H, \lambda, \sigma^2)` and test the specification.

    Parameters
    ----------
    exponents
        Estimated scaling exponents. :math:`H` is taken from the MFC root
        identity :math:`\tau(1/H)=0` unless overridden.
    spectrum
        Estimated Hölder spectrum, used for :math:`\alpha_0` and for the
        observed width. Omit it and :math:`\alpha_0` is taken from
        :math:`\tau'` at the spectrum apex via a Legendre transform.
    hurst, alpha_0
        Override the estimates, e.g. to calibrate at a value fixed by theory.

    Returns
    -------
    CalibrationResult
        Never raises on inadmissible input — the verdict is a field, so a caller
        sweeping many series can tabulate failures instead of catching them.

    Notes
    -----
    Four checks are performed and reported:

    1. :math:`H \in (0,1)` — outside it the subordinand is not an fBm.
    2. :math:`\lambda = \alpha_0/H > 1` — otherwise :math:`\sigma^2 \le 0` and
       the cascade is degenerate. This is v1's bare ``ValueError``.
    3. :math:`\tau_\theta'(1) = 2-\lambda > 0` — the Kahane--Peyrière
       non-degeneracy condition. At :math:`\lambda \ge 2` the canonical cascade
       converges to the zero measure and the model is vacuous.
    4. Width agreement. The lognormal MMAR predicts
       :math:`\Delta\alpha = 2H\sqrt{2\sigma^2\ln b}`; a large discrepancy
       against the observed width means the *shape* of the spectrum is not
       parabolic, so the lognormal cascade is the wrong family even where the
       apex matches.
    """
    if base <= 1:
        raise ValueError(f"cascade base must exceed 1, got b={base}")

    H = float(exponents.hurst_tau_root) if hurst is None else float(hurst)
    if spectrum is None:
        from mmar.core.legendre import spectrum_from_exponents  # noqa: PLC0415 - avoids a cycle

        spectrum = spectrum_from_exponents(exponents)
    desc = spectrum.descriptors()
    a0 = float(desc.alpha_0) if alpha_0 is None else float(alpha_0)

    problems: list[str] = []
    hurst_unidentified = not math.isfinite(H)

    if not math.isfinite(H):
        # The estimated and the supplied H fail in the same place but for opposite
        # reasons. Blaming tau when the caller passed hurst=nan names a defect the
        # exponents do not have and sends them to widen a q grid that is already fine.
        problems.append(
            f"the hurst= override is {hurst!r}, which is not a finite number, so no "
            "Hurst exponent entered the calibration. Pass a finite H in (0,1), or omit "
            "hurst= to take it from the MFC identity tau(1/H)=0."
            if hurst is not None
            else "tau(q) has no root on the positive-q branch, so the MFC identity "
            "tau(1/H)=0 identifies no Hurst exponent. Widen the q grid, or pass "
            "hurst= explicitly from an independent estimator (e.g. h(2))."
        )
        H = float(exponents.h2)
        if math.isfinite(H):
            problems.append(f"falling back to h(2) = {H:.4f} for the remaining checks.")
    if math.isfinite(H) and not 0.0 < H < 1.0:
        problems.append(
            f"H = {H:.4f} lies outside (0,1); the MMAR's subordinand is a fractional "
            "Brownian motion, which requires 0 < H < 1. Typically this means the "
            "scaling range is contaminated by a trend or a crossover."
        )

    lam = lambda_from_alpha0(a0, H) if math.isfinite(H) and 0.0 < H < 1.0 else float("nan")
    if math.isfinite(lam) and lam <= 1.0:
        needed = H  # alpha_0 must exceed H for lambda > 1
        problems.append(
            f"lambda = alpha_0/H = {a0:.4f}/{H:.4f} = {lam:.4f} <= 1, so "
            f"sigma^2 = 2(lambda-1)/ln b = {sigma2_from_lambda(lam, base):.4f} <= 0 and the "
            "lognormal cascade is degenerate. Mechanically: the estimated modal Hölder "
            f"exponent alpha_0 = {a0:.4f} does not exceed the estimated H = {H:.4f}. "
            f"For a lognormal MMAR to fit, alpha_0 would have to exceed {needed:.4f}. "
            "Subordination can only push the modal exponent UP (alpha_0 = H*lambda with "
            "lambda > 1), so alpha_0 <= H is evidence against the model itself, not a "
            "numerical accident."
        )
    if math.isfinite(lam) and lam >= 2.0:
        problems.append(
            f"lambda = {lam:.4f} >= 2 violates the Kahane-Peyriere non-degeneracy condition "
            "tau_theta'(1) = 2 - lambda > 0: the canonical cascade converges to the zero "
            "measure, so the calibrated model is vacuous."
        )

    sigma2 = sigma2_from_lambda(lam, base) if math.isfinite(lam) else float("nan")
    q_crit = critical_moment(lam) if math.isfinite(lam) else float("nan")

    if math.isfinite(q_crit) and math.isfinite(H) and H > 0:
        q_max_fitted = float(np.max(exponents.q))
        q_bound = q_crit / H
        if q_max_fitted > q_bound:
            problems.append(
                f"the fit used q up to {q_max_fitted:.2f}, beyond q_crit/H = {q_bound:.2f}. "
                "Trading-time moments diverge there, so those exponents describe the sample "
                "maximum rather than the process; the calibration inherits that bias."
            )

    width_implied = (
        2.0 * H * math.sqrt(2.0 * sigma2 * math.log(base))
        if math.isfinite(sigma2) and sigma2 > 0 and math.isfinite(H)
        else float("nan")
    )
    width_observed = float(desc.width)
    if math.isfinite(width_implied) and width_implied > 0:
        ratio = width_observed / width_implied
        if not 0.5 <= ratio <= 2.0:
            problems.append(
                f"observed spectrum width {width_observed:.4f} against the {width_implied:.4f} "
                f"implied by the calibrated parameters (ratio {ratio:.2f}). The apex may match "
                "while the shape does not: a lognormal cascade gives a parabolic f(alpha), and "
                "a strong mismatch points to a different multiplier law (log-Poisson, "
                "log-stable) or to a scaling range that is not clean."
            )

    fatal = (
        # H was reassigned to h(2) above, so without this term the fallback launders a
        # failed identification into `admissible=True`: the MFC identity tau(1/H)=0 --
        # the thing this module calibrates -- identified nothing, yet to_spec_kwargs()
        # would still hand back an MMARSpec and the CLI would still exit 0. The
        # fallback H remains reported, in the field, for inspection.
        hurst_unidentified
        or not math.isfinite(H)
        or not 0.0 < H < 1.0
        or not math.isfinite(lam)
        or lam <= 1.0
        or lam >= 2.0
    )

    prov = None
    if exponents.provenance is not None:
        # The spectrum is a second input, not a function of the first: alpha_0 and
        # the observed width come from it. Chaining only the exponents let two
        # calibrations off different spectra share one content address.
        spectrum_ref = (
            InputRef.from_provenance("spectrum", spectrum.provenance)
            if spectrum.provenance is not None
            else InputRef.from_array("spectrum.f_alpha", spectrum.f_alpha)
        )
        prov = exponents.provenance.derive(
            "mmar_calibration",
            params={"base": base, "hurst_override": hurst, "alpha0_override": alpha_0},
            extra_inputs=[spectrum_ref],
        )

    return CalibrationResult(
        hurst=H,
        alpha_0=a0,
        lam=lam,
        sigma2=sigma2,
        base=base,
        admissible=not fatal,
        diagnosis=tuple(problems),
        q_critical=q_crit,
        width_implied=width_implied,
        width_observed=width_observed,
        provenance=prov,
    )


def implied_spectrum(result: CalibrationResult, alpha: FloatArray) -> FloatArray:
    r"""The :math:`f(\alpha)` the calibrated MMAR predicts, for overlay on the estimate.

    Plotting this against the empirical spectrum is the sharpest available test
    of the lognormal specification: the two agree at the apex by construction,
    so any disagreement is pure shape.
    """
    from mmar.models.theory import mmar_falpha_lognormal  # noqa: PLC0415 - avoids a cycle

    if not result.admissible:
        return np.full_like(alpha, np.nan)
    return np.asarray(
        mmar_falpha_lognormal(alpha, result.hurst, result.lam, result.sigma2, result.base),
        dtype=np.float64,
    )
