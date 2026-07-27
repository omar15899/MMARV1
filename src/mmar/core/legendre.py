r"""Legendre transforms: from scaling exponents to the Hölder spectrum.

Two routes from :math:`\zeta(q)` to :math:`f(\alpha)`, and they are not
equivalent:

**Parametric (tangent) construction.**
:math:`\alpha(q)=\zeta'(q)`, :math:`f(\alpha(q)) = q\alpha(q)-\zeta(q)+d`.
Cheap, gives an :math:`(\alpha,f)` pair per :math:`q`, and is what almost every
paper does --- but it silently assumes :math:`\zeta` is differentiable *and
concave*. On a finite sample :math:`\hat\zeta` is neither, and a small wiggle in
:math:`\hat\zeta` becomes a loop or a spurious backward fold in the spectrum.

**Legendre--Fenchel transform.**
:math:`f(\alpha)=\inf_q\{q\alpha-\zeta(q)\}+d`, evaluated on an :math:`\alpha`
grid. This is the *definition*, it is always concave, and it is insensitive to
exactly the non-concave part of :math:`\hat\zeta` that has no physical meaning.
It is the default here.

Both are exposed, and :func:`concavity_defect` quantifies how far the estimated
:math:`\zeta` is from concave --- a large defect means the sample does not
support a multifractal reading, however pretty the resulting curve looks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline

from mmar.core.types import FloatArray, HolderSpectrum, ScalingExponents

__all__ = [
    "concave_envelope",
    "concavity_defect",
    "legendre_fenchel",
    "legendre_parametric",
    "spectrum_from_exponents",
]


def _spline(q: FloatArray, zeta: FloatArray, smoothing: float | None) -> UnivariateSpline:
    k = 3 if q.size > 3 else max(1, q.size - 1)
    return UnivariateSpline(q, zeta, s=0.0 if smoothing is None else smoothing, k=k)


def concave_envelope(q: FloatArray, zeta: FloatArray) -> FloatArray:
    r"""Least concave majorant of :math:`\zeta` on the given grid.

    Computed as the upper convex hull of the points :math:`(q_i,\zeta_i)`. The
    Legendre transform of :math:`\zeta` equals that of its concave envelope, so
    this is the part of :math:`\hat\zeta` the spectrum can actually see.
    """
    order = np.argsort(q)
    qs, zs = q[order], zeta[order]
    hull: list[int] = []
    for i in range(qs.size):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (qs[b] - qs[a]) * (zs[i] - zs[a]) - (zs[b] - zs[a]) * (qs[i] - qs[a])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    idx = np.array(hull, dtype=int)
    env = np.interp(qs, qs[idx], zs[idx])
    out = np.empty_like(env)
    out[order] = env
    return np.asarray(out, dtype=np.float64)


def concavity_defect(q: FloatArray, zeta: FloatArray) -> float:
    r"""Maximum vertical gap between :math:`\zeta` and its concave envelope.

    Zero for a genuinely concave estimate. Compare against the typical standard
    error of :math:`\hat\zeta`: a defect much larger than that is evidence the
    scaling assumption has failed (crossover, insufficient sample, or moments
    that do not converge).
    """
    return float(np.max(concave_envelope(q, zeta) - zeta))


def legendre_parametric(
    q: FloatArray,
    zeta: FloatArray,
    *,
    d_support: float = 1.0,
    smoothing: float | None = None,
    n_fine: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Tangent construction: returns ``(q, alpha, f)``.

    ``smoothing`` is passed to :class:`~scipy.interpolate.UnivariateSpline`;
    ``None`` interpolates exactly, which amplifies noise in
    :math:`\alpha=\zeta'`. A sensible non-zero choice is
    :math:`s \approx \sum_i \hat\sigma_i^2`.

    ``n_fine`` resamples the curve on a denser :math:`q` grid.
    """
    order = np.argsort(q)
    qs, zs = q[order], zeta[order]
    spl = _spline(qs, zs, smoothing)
    q_out = np.linspace(qs[0], qs[-1], n_fine) if n_fine > 0 else qs
    alpha = np.asarray(spl.derivative()(q_out), dtype=np.float64)
    f = np.asarray(q_out * alpha - spl(q_out) + d_support, dtype=np.float64)
    return np.asarray(q_out, dtype=np.float64), alpha, f


def legendre_fenchel(
    q: FloatArray,
    zeta: FloatArray,
    *,
    d_support: float = 1.0,
    alpha_grid: FloatArray | None = None,
    n_alpha: int = 201,
    q_refine: int = 2001,
    smoothing: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    r"""True Legendre--Fenchel transform :math:`f(\alpha)=\inf_q\{q\alpha-\zeta(q)\}+d`.

    The infimum is taken over a refined :math:`q` grid (spline-interpolated from
    the estimated one), because an infimum over a coarse grid systematically
    *overestimates* :math:`f`.

    When ``alpha_grid`` is omitted the grid spans
    :math:`[\min\zeta', \max\zeta']`, i.e. exactly the range of Hölder exponents
    the fitted :math:`q` window can resolve. Exponents outside it are not
    "missing from the data" --- they are outside the moment range probed, which
    is why widening ``q`` widens the spectrum.
    """
    order = np.argsort(q)
    qs, zs = q[order], zeta[order]
    spl = _spline(qs, zs, smoothing)
    qf = np.linspace(qs[0], qs[-1], max(q_refine, qs.size))
    zf = np.asarray(spl(qf), dtype=np.float64)

    if alpha_grid is None:
        slopes = np.gradient(zf, qf)
        a_lo, a_hi = float(np.min(slopes)), float(np.max(slopes))
        if not np.isfinite(a_lo) or not np.isfinite(a_hi) or a_hi <= a_lo:
            a_lo, a_hi = a_lo - 0.05, a_lo + 0.05
        alpha_grid = np.linspace(a_lo, a_hi, n_alpha)

    # f(alpha) = min_q (q*alpha - zeta(q)) + d, vectorised over the alpha grid.
    obj = np.outer(alpha_grid, qf) - zf[None, :]
    f = np.min(obj, axis=1) + d_support
    return np.asarray(alpha_grid, dtype=np.float64), np.asarray(f, dtype=np.float64)


def spectrum_from_exponents(
    exponents: ScalingExponents,
    *,
    method: str = "fenchel",
    smoothing: float | None = None,
    n_alpha: int = 201,
    alpha_grid: FloatArray | None = None,
    clip_negative: bool = False,
) -> HolderSpectrum:
    r"""Build the Hölder spectrum from fitted exponents.

    Parameters
    ----------
    method
        ``"fenchel"`` (default, always concave, recommended) or
        ``"parametric"`` (tangent construction; keeps the :math:`q` labels and
        so is the one to use when you need :math:`\alpha(q)` itself).
    clip_negative
        Negative :math:`f(\alpha)` values are meaningful in the large-deviation
        reading (they describe events too rare to occur in a single sample) but
        are an artefact in the box-counting reading. Off by default: clipping is
        a *presentation* choice and should be explicit.

    The propagated ``alpha_stderr``/``f_stderr`` are a delta-method
    approximation from ``zeta_stderr``; for real error bars use
    :func:`mmar.inference.bootstrap.bootstrap_spectrum`.
    """
    q, zeta, d = exponents.q, exponents.zeta, exponents.d_support
    prov = (
        exponents.provenance.derive(
            "legendre",
            params={"method": method, "smoothing": smoothing, "n_alpha": n_alpha},
        )
        if exponents.provenance is not None
        else None
    )

    if method == "parametric":
        q_out, alpha, f = legendre_parametric(q, zeta, d_support=d, smoothing=smoothing)
        q_label: FloatArray | None = q_out
    elif method == "fenchel":
        alpha, f = legendre_fenchel(
            q,
            zeta,
            d_support=d,
            alpha_grid=alpha_grid,
            n_alpha=n_alpha,
            smoothing=smoothing,
        )
        # Label each alpha with the q attaining the infimum, for interpretability.
        spl = _spline(np.sort(q), zeta[np.argsort(q)], smoothing)
        qf = np.linspace(float(q.min()), float(q.max()), 2001)
        obj = np.outer(alpha, qf) - np.asarray(spl(qf))
        q_label = np.asarray(qf[np.argmin(obj, axis=1)], dtype=np.float64)
    else:
        raise ValueError(f"unknown method {method!r}; use 'fenchel' or 'parametric'")

    if clip_negative:
        f = np.where(f < 0.0, np.nan, f)

    alpha_se: FloatArray | None = None
    f_se: FloatArray | None = None
    if exponents.zeta_stderr is not None and method == "parametric":
        se = np.asarray(exponents.zeta_stderr, dtype=np.float64)
        dq = np.gradient(np.sort(q))
        alpha_se = np.asarray(np.sqrt(2.0) * se / np.maximum(dq, 1e-12), dtype=np.float64)
        f_se = np.asarray(np.abs(q_label) * alpha_se + se if q_label is not None else se)

    return HolderSpectrum(
        alpha=alpha,
        f_alpha=f,
        q=q_label,
        alpha_stderr=alpha_se,
        f_stderr=f_se,
        d_support=d,
        method=f"legendre-{method}",
        exponents=exponents,
        provenance=prov,
    )


def spectrum_residual(
    spectrum: HolderSpectrum, reference: NDArray[np.float64], alpha_ref: NDArray[np.float64]
) -> float:
    """Root-mean-square distance to a reference spectrum on its own support.

    Used by the theory tests: a cascade with a closed-form :math:`f(\\alpha)`
    must be recovered to within a stated tolerance.
    """
    f_hat = spectrum.evaluate(alpha_ref)
    ok = np.isfinite(f_hat) & np.isfinite(reference)
    if ok.sum() == 0:
        return float("inf")
    return float(np.sqrt(np.mean((f_hat[ok] - reference[ok]) ** 2)))
