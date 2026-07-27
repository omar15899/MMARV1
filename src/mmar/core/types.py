r"""Canonical types and, more importantly, the canonical *conventions*.

The multifractal literature carries at least four incompatible sign and offset
conventions for :math:`\tau(q)`. Silent convention drift is the single most
common source of wrong Hölder spectra, so this library fixes one carrier and
derives every other view from it.

Conventions
-----------
Let the support have topological dimension :math:`d` (:math:`d=1` for a time
series).

*Measure view.* For a positive measure :math:`\mu` partitioned into boxes of
side :math:`\delta`,

.. math::
    Z(q,\delta) = \sum_i \mu_i^q \sim \delta^{\tau(q)}, \qquad \delta\to 0 .

*Signal view.* For a signal :math:`X` with increments
:math:`\Delta_\delta X(t) = X(t+\delta)-X(t)`,

.. math::
    S(q,\delta) = \big\langle |\Delta_\delta X|^q \big\rangle \sim \delta^{\zeta(q)} .

Because a box partition of a :math:`d`-dimensional support has
:math:`N(\delta)=\delta^{-d}` boxes, the two are related by

.. math::
    \zeta(q) = \tau(q) + d .

:class:`ScalingExponents` stores :math:`\zeta` and derives
:math:`\tau=\zeta-d` and the generalised Hurst function :math:`h(q)=\zeta(q)/q`.

*Spectrum.* The singularity spectrum is the Legendre–Fenchel transform

.. math::
    f(\alpha) = \inf_q \{ q\alpha - \tau(q) \}
              = \inf_q \{ q\alpha - \zeta(q) \} + d ,

so ``f_alpha`` is always the *measure-view* spectrum: :math:`f(\alpha_0)=d` for
a spectrum supported on the whole domain, and :math:`\alpha` is the Hölder
exponent in both views.

Sanity anchors that every estimator in this library is tested against:

======================  ====================================  ==================
Object                  :math:`\zeta(q)`                      :math:`f(\alpha)`
======================  ====================================  ==================
fBm, Hurst :math:`H`    :math:`qH`                            :math:`\{(H,1)\}`
Binomial :math:`m_0`    :math:`1-\log_2(m_0^q+m_1^q)`         entropy formula
Lognormal cascade       :math:`q\lambda - \tfrac{\sigma^2}{2}q^2 \ln 2 + 1`   parabola
======================  ====================================  ==================
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mmar.core.provenance import Provenance

__all__ = [
    "Convention",
    "FitMethod",
    "FloatArray",
    "HolderSpectrum",
    "LogCumulants",
    "LogLogFit",
    "MultifractalReport",
    "PointwiseHolder",
    "ScalingExponents",
    "ScalingRange",
    "SpectrumDescriptors",
    "StructureFunctions",
    "dyadic_scales",
    "log_spaced_scales",
]

FloatArray = NDArray[np.float64]


class Convention(StrEnum):
    """Which scaling exponent a caller wants to see."""

    MEASURE = "measure"
    """:math:`\\tau(q)`: box-sum of a measure, :math:`Z\\sim\\delta^{\\tau}`."""

    SIGNAL = "signal"
    """:math:`\\zeta(q)`: moment of increments, :math:`S\\sim\\delta^{\\zeta}`."""


class FitMethod(StrEnum):
    """Estimator used for the log--log regression of a scaling law."""

    OLS = "ols"
    WLS = "wls"
    THEILSEN = "theilsen"
    HUBER = "huber"


# --------------------------------------------------------------------------- #
# Scale grids
# --------------------------------------------------------------------------- #


def log_spaced_scales(
    n: int,
    *,
    s_min: int = 8,
    s_max: int | None = None,
    n_scales: int = 24,
    max_fraction: float = 0.25,
) -> NDArray[np.int64]:
    """Integer scales geometrically spaced over ``[s_min, s_max]``, deduplicated.

    ``s_max`` defaults to ``max_fraction * n``: beyond roughly a quarter of the
    sample the number of non-overlapping windows falls below four and the
    fluctuation estimate stops being meaningful.
    """
    if n < 16:
        raise ValueError(f"series too short for multifractal analysis: n={n}")
    hi = int(s_max if s_max is not None else max(s_min + 1, math.floor(max_fraction * n)))
    if hi <= s_min:
        raise ValueError(f"empty scale range: s_min={s_min} >= s_max={hi}")
    raw = np.geomspace(s_min, hi, num=max(n_scales, 2))
    return np.unique(np.round(raw).astype(np.int64))


def dyadic_scales(n: int, *, j_min: int = 3, j_max: int | None = None) -> NDArray[np.int64]:
    """Scales :math:`2^j`, the natural grid for wavelet-based estimators."""
    top = int(math.floor(math.log2(n))) if j_max is None else j_max
    if top < j_min:
        raise ValueError(f"series too short: log2(n)={math.log2(n):.1f} < j_min={j_min}")
    return np.array([2**j for j in range(j_min, top + 1)], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Regression products
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ScalingRange:
    """The scale window actually used by a fit, and how it was chosen."""

    s_min: float
    s_max: float
    n_scales: int
    selection: str = "manual"
    r_squared: float = float("nan")
    mask: tuple[bool, ...] = ()

    @property
    def decades(self) -> float:
        """Extent of the fitted power law, in decades. Below ~1 be suspicious."""
        return math.log10(self.s_max / self.s_min) if self.s_min > 0 else float("nan")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "s_min": self.s_min,
            "s_max": self.s_max,
            "n_scales": self.n_scales,
            "selection": self.selection,
        }


@dataclass(frozen=True, slots=True)
class LogLogFit:
    r"""One power-law fit :math:`\log y = \text{slope}\cdot\log x + \text{intercept}`.

    Standard errors follow the textbook OLS formulas *computed in log space*
    (a classical error is to compute the leverage term from the raw scales
    instead of their logarithms).
    """

    slope: float
    intercept: float
    slope_stderr: float
    intercept_stderr: float
    slope_ci: tuple[float, float]
    r_squared: float
    n_points: int
    method: FitMethod = FitMethod.OLS
    residuals: FloatArray = field(default_factory=lambda: np.empty(0))
    log_x: FloatArray = field(default_factory=lambda: np.empty(0))
    log_y: FloatArray = field(default_factory=lambda: np.empty(0))
    scaling_range: ScalingRange | None = None
    weights: FloatArray | None = None

    @property
    def dof(self) -> int:
        return max(self.n_points - 2, 0)

    def predict(self, x: FloatArray) -> FloatArray:
        return np.asarray(self.slope * np.log(x) + self.intercept, dtype=np.float64)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "slope_stderr": self.slope_stderr,
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "method": str(self.method),
        }


# --------------------------------------------------------------------------- #
# Scaling exponents
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StructureFunctions:
    r"""The raw scaling surface :math:`S(q,\delta)` before any exponent is fitted.

    Keeping this is what makes an analysis auditable: the exponents can always
    be refitted from it over a different scaling range without recomputing.
    """

    q: FloatArray
    scales: FloatArray
    values: FloatArray
    """Shape ``(n_q, n_scales)``; ``values[i, j] = S(q_i, delta_j)``."""
    log_base: float = math.e
    counts: NDArray[np.int64] | None = None
    """Number of windows/coefficients averaged at each scale, for weighting."""
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if self.values.shape != (self.q.size, self.scales.size):
            raise ValueError(
                f"structure-function grid mismatch: values{self.values.shape} "
                f"vs (n_q={self.q.size}, n_scales={self.scales.size})"
            )

    def row(self, q: float) -> FloatArray:
        idx = int(np.argmin(np.abs(self.q - q)))
        return np.asarray(self.values[idx], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ScalingExponents:
    r"""Canonical carrier of the scaling exponents.

    Stores :math:`\zeta(q)` (signal view) and derives :math:`\tau=\zeta-d` and
    :math:`h(q)=\zeta(q)/q`. Never store one convention only.
    """

    q: FloatArray
    zeta: FloatArray
    d_support: float = 1.0
    zeta_stderr: FloatArray | None = None
    zeta_ci: tuple[FloatArray, FloatArray] | None = None
    fits: tuple[LogLogFit, ...] = ()
    scaling_range: ScalingRange | None = None
    structure: StructureFunctions | None = None
    estimator: str = "unknown"
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if self.q.shape != self.zeta.shape:
            raise ValueError(f"q{self.q.shape} and zeta{self.zeta.shape} must match")
        if self.q.size < 3:
            raise ValueError("at least three q values are required to build a spectrum")

    @property
    def tau(self) -> FloatArray:
        r""":math:`\tau(q) = \zeta(q) - d`, the measure-view exponent."""
        return np.asarray(self.zeta - self.d_support, dtype=np.float64)

    @property
    def hq(self) -> FloatArray:
        r"""Generalised Hurst function :math:`h(q)=\zeta(q)/q`; NaN at ``q=0``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(self.q != 0.0, self.zeta / self.q, np.nan)
        return np.asarray(out, dtype=np.float64)

    def exponent(self, convention: Convention = Convention.SIGNAL) -> FloatArray:
        return self.zeta if convention is Convention.SIGNAL else self.tau

    def at(self, q: float) -> float:
        """:math:`\\zeta` interpolated at an arbitrary ``q`` (monotone grid assumed)."""
        return float(np.interp(q, self.q, self.zeta))

    @property
    def h2(self) -> float:
        r""":math:`h(2)=\zeta(2)/2`, the classical (second-order) Hurst exponent."""
        return self.at(2.0) / 2.0

    @property
    def hurst_tau_root(self) -> float:
        r"""MMAR Hurst exponent: the :math:`H=1/q^\*` with :math:`\tau(q^\*)=0`.

        Mandelbrot--Fisher--Calvet show :math:`\tau(1/H)=0` for the MMAR, so the
        root of :math:`\tau` on the *positive* branch identifies :math:`H`.
        Returns NaN when :math:`\tau` has no sign change over the fitted grid,
        rather than silently substituting another estimator.
        """
        tau = self.tau
        pos = self.q > 0
        if pos.sum() < 2:
            return float("nan")
        qq, tt = self.q[pos], tau[pos]
        sign_change = np.nonzero(np.diff(np.sign(tt)) != 0)[0]
        if sign_change.size == 0:
            return float("nan")
        i = int(sign_change[0])
        q_star = qq[i] - tt[i] * (qq[i + 1] - qq[i]) / (tt[i + 1] - tt[i])
        return float(1.0 / q_star) if q_star != 0 else float("nan")

    def is_monofractal(self, tol: float = 0.05) -> bool:
        """Whether :math:`h(q)` is constant to within ``tol`` over the fitted grid."""
        h = self.hq[np.isfinite(self.hq)]
        return bool(h.size > 0 and (h.max() - h.min()) < tol)


# --------------------------------------------------------------------------- #
# Hölder spectrum
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SpectrumDescriptors:
    r"""Scalar summary of an :math:`f(\alpha)` curve.

    ``width`` (:math:`\Delta\alpha=\alpha_{\max}-\alpha_{\min}`) is the standard
    scalar measure of multifractality; it is exactly zero for a monofractal.
    ``asymmetry`` :math:`B=(\alpha_0-\alpha_{\min})/(\alpha_{\max}-\alpha_0)`
    exceeds 1 for a left-truncated (large-fluctuation dominated) spectrum.
    """

    alpha_min: float
    alpha_max: float
    alpha_0: float
    width: float
    f_max: float
    asymmetry: float
    curvature: float
    """Second derivative of the parabolic fit at :math:`\\alpha_0`."""

    def to_canonical(self) -> dict[str, Any]:
        return {
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "alpha_0": self.alpha_0,
            "width": self.width,
            "f_max": self.f_max,
            "asymmetry": self.asymmetry,
        }


@dataclass(frozen=True, slots=True)
class HolderSpectrum:
    r"""A singularity spectrum :math:`\alpha \mapsto f(\alpha)`.

    ``f_alpha`` is always in the measure convention, so a space-filling
    monofractal has :math:`\max_\alpha f(\alpha) = d`.
    """

    alpha: FloatArray
    f_alpha: FloatArray
    q: FloatArray | None = None
    alpha_stderr: FloatArray | None = None
    f_stderr: FloatArray | None = None
    d_support: float = 1.0
    method: str = "legendre"
    exponents: ScalingExponents | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if self.alpha.shape != self.f_alpha.shape:
            raise ValueError(f"alpha{self.alpha.shape} and f{self.f_alpha.shape} must match")

    @property
    def alpha_0(self) -> float:
        r"""The most probable Hölder exponent, :math:`\arg\max_\alpha f(\alpha)`."""
        finite = np.isfinite(self.f_alpha)
        if not finite.any():
            return float("nan")
        return float(self.alpha[finite][int(np.argmax(self.f_alpha[finite]))])

    def descriptors(self) -> SpectrumDescriptors:
        finite = np.isfinite(self.alpha) & np.isfinite(self.f_alpha)
        if finite.sum() < 3:
            nan = float("nan")
            return SpectrumDescriptors(nan, nan, nan, nan, nan, nan, nan)
        a, f = self.alpha[finite], self.f_alpha[finite]
        a_min, a_max = float(a.min()), float(a.max())
        a0 = float(a[int(np.argmax(f))])
        left, right = a0 - a_min, a_max - a0
        curvature = float("nan")
        # A degenerate (monofractal) spectrum has zero alpha-spread, for which the
        # quadratic design matrix is singular; report NaN rather than a fitted artefact.
        if a.size >= 3 and (a_max - a_min) > 1e-9 * max(1.0, abs(a0)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", np.exceptions.RankWarning)
                    curvature = float(2.0 * np.polyfit(a, f, 2)[0])
            except (np.linalg.LinAlgError, ValueError, np.exceptions.RankWarning):
                pass
        return SpectrumDescriptors(
            alpha_min=a_min,
            alpha_max=a_max,
            alpha_0=a0,
            width=a_max - a_min,
            f_max=float(f.max()),
            asymmetry=float(left / right) if right > 0 else float("inf"),
            curvature=curvature,
        )

    def evaluate(self, alpha: FloatArray) -> FloatArray:
        """Interpolate ``f`` at arbitrary ``alpha`` (outside the support: NaN)."""
        order = np.argsort(self.alpha)
        a, f = self.alpha[order], self.f_alpha[order]
        out = np.interp(alpha, a, f, left=np.nan, right=np.nan)
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LogCumulants:
    r"""Log-cumulant expansion :math:`\zeta(q)=\sum_{p\ge1} c_p q^p / p!`.

    :math:`c_1` is the most probable Hölder exponent, :math:`c_2\le 0` measures
    multifractality (:math:`c_2=0` exactly for a monofractal), and :math:`c_3`
    the asymmetry of the spectrum. Near the apex,
    :math:`f(\alpha)\approx d + \tfrac{(\alpha-c_1)^2}{2c_2}`.
    """

    c: FloatArray
    """``c[0] = c_1``, ``c[1] = c_2``, ..."""
    stderr: FloatArray | None = None
    order: int = 2
    d_support: float = 1.0
    provenance: Provenance | None = None

    @property
    def c1(self) -> float:
        return float(self.c[0])

    @property
    def c2(self) -> float:
        return float(self.c[1]) if self.c.size > 1 else float("nan")

    @property
    def c3(self) -> float:
        return float(self.c[2]) if self.c.size > 2 else float("nan")

    @property
    def intermittency(self) -> float:
        r""":math:`-c_2 \ge 0`; zero for a monofractal, ~0.03 for turbulence."""
        return -self.c2

    def parabolic_spectrum(self, alpha: FloatArray) -> FloatArray:
        r"""Second-order spectrum :math:`f(\alpha)=d+(\alpha-c_1)^2/(2c_2)`."""
        if self.c2 >= 0:
            return np.full_like(alpha, np.nan)
        return np.asarray(self.d_support + (alpha - self.c1) ** 2 / (2.0 * self.c2), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PointwiseHolder:
    r"""Local regularity :math:`t \mapsto h(t)`: a Hölder exponent per sample.

    Where :class:`HolderSpectrum` is the *distribution* of exponents, this is
    the *field* of them, so singular events can be located in time.
    """

    t: FloatArray
    h: FloatArray
    stderr: FloatArray | None = None
    r_squared: FloatArray | None = None
    method: str = "oscillation"
    scales: FloatArray = field(default_factory=lambda: np.empty(0))
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if self.t.shape != self.h.shape:
            raise ValueError(f"t{self.t.shape} and h{self.h.shape} must match")

    def singular_times(self, threshold: float) -> NDArray[np.int64]:
        """Indices whose local exponent falls below ``threshold`` (rough points)."""
        return np.asarray(np.nonzero(self.h < threshold)[0], dtype=np.int64)

    def to_spectrum(self, bins: int = 50) -> HolderSpectrum:
        r"""Coarse histogram spectrum from the pointwise field.

        This is a *descriptive* device, not the large-deviation spectrum: it
        reports :math:`-\log_2 N(h)/\log_2 \delta`-style box counts only when a
        scale is available, and otherwise a normalised density. Prefer
        :mod:`mmar.estimators.wavelet_leaders` for inference.
        """
        finite = np.isfinite(self.h)
        counts, edges = np.histogram(self.h[finite], bins=bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        with np.errstate(divide="ignore"):
            f = np.where(counts > 0, np.log(counts) / np.log(counts.max()), np.nan)
        return HolderSpectrum(
            alpha=centres,
            f_alpha=np.asarray(f, dtype=np.float64),
            method=f"histogram({self.method})",
            provenance=self.provenance,
        )


@dataclass(frozen=True, slots=True)
class MultifractalReport:
    """Everything a full analysis produced, with one auditable root."""

    exponents: ScalingExponents
    spectrum: HolderSpectrum
    cumulants: LogCumulants | None = None
    pointwise: PointwiseHolder | None = None
    tests: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None

    def summary(self) -> dict[str, Any]:
        d = self.spectrum.descriptors()
        out: dict[str, Any] = {
            "estimator": self.exponents.estimator,
            "h2": self.exponents.h2,
            "hurst_tau_root": self.exponents.hurst_tau_root,
            "alpha_0": d.alpha_0,
            "width": d.width,
            "asymmetry": d.asymmetry,
            "f_max": d.f_max,
        }
        if self.cumulants is not None:
            out["c1"] = self.cumulants.c1
            out["c2"] = self.cumulants.c2
        if self.provenance is not None:
            out["root"] = self.provenance.root
        if self.tests:
            out["tests"] = dict(self.tests)
        return out


def as_float_array(x: Sequence[float] | NDArray[Any], name: str = "array") -> FloatArray:
    """Coerce to a contiguous float64 array, rejecting non-numeric input."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    return a
