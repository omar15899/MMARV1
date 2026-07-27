r"""The high-level entry point: one call from a series to a complete analysis.

:func:`analyze` runs an estimator, builds the Hölder spectrum, extracts the log-
cumulants and (optionally) the surrogate tests, and returns a single
:class:`~mmar.core.types.MultifractalReport` with one provenance root covering
the lot.

:func:`compare_estimators` runs several and reports where they agree. This is
not a convenience: the estimators carry *different biases*, so a spectrum that
survives all of them means something a single estimator's output does not.
Where they disagree, :func:`consensus` says by how much, and the honest reading
of a large disagreement is that the scaling assumption has failed rather than
that one estimator is right.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mmar.core.legendre import concavity_defect, spectrum_from_exponents
from mmar.core.provenance import InputRef, new_provenance
from mmar.core.rng import SeedLike, seed_info
from mmar.core.types import (
    FloatArray,
    HolderSpectrum,
    MultifractalReport,
    ScalingExponents,
)
from mmar.core.validate import validate_series

__all__ = [
    "ESTIMATORS",
    "ConsensusReport",
    "analyze",
    "compare_estimators",
    "consensus",
    "get_estimator",
]

Estimator = Callable[..., ScalingExponents]


def _lazy(module: str, name: str) -> Estimator:
    """Resolve an estimator on first use, so optional extras stay optional.

    ``wavelet_leaders`` and ``wtmm`` need PyWavelets; importing the registry
    must not require it.
    """

    def call(*args: Any, **kwargs: Any) -> ScalingExponents:
        import importlib  # noqa: PLC0415 - deferred by design

        fn: Estimator = getattr(importlib.import_module(module), name)
        return fn(*args, **kwargs)

    call.__name__ = name
    return call


ESTIMATORS: Mapping[str, Estimator] = {
    "wavelet_leaders": _lazy("mmar.estimators.wavelet_leaders", "wavelet_leaders"),
    "mfdfa": _lazy("mmar.estimators.mfdfa", "mfdfa"),
    "mfdma": _lazy("mmar.estimators.mfdma", "mfdma"),
    "wtmm": _lazy("mmar.estimators.wtmm", "wtmm"),
    "structure_function": _lazy("mmar.estimators.structure_function", "mfc_exponents"),
    "ghe": _lazy("mmar.estimators.ghe", "generalised_hurst"),
}
"""Estimators that map a *series* to :class:`ScalingExponents`.

Measure-domain estimators (:mod:`~mmar.estimators.chhabra_jensen`,
:mod:`~mmar.estimators.box_counting`) take a measure rather than a series and
are called directly; pointwise estimators return a field, not a spectrum, and
live in :mod:`~mmar.estimators.pointwise`.
"""

_DEFAULT = "wavelet_leaders"

SeriesKind = Literal["level", "increments"]

_LEVEL_KWARGS: Mapping[str, Mapping[str, Any]] = {
    # Every estimator is handed the LEVEL series (see `_to_level`), so the ones
    # that build their own profile must be told not to integrate again. Their
    # shipped defaults disagree with each other -- mfdfa integrate=True, mfdma
    # integrate=False -- which silently shifts h by exactly 1 between two
    # estimators of the same quantity. Pinning the flag here removes the trap.
    "mfdfa": {"integrate": False},
    "mfdma": {"integrate": False},
    "ghe": {"series_kind": "level"},
}


def _to_level(x: FloatArray, kind: SeriesKind) -> FloatArray:
    r"""Canonicalise the input to the *level* representation.

    Estimators in this library disagree about what they expect. MF-DFA and
    MF-DMA integrate to a profile; the structure functions, the wavelet families
    and WTMM difference (or take wavelet coefficients of) whatever they are
    given. Feeding a series of increments to the second group does not measure
    the process: it measures the regularity of its noise, giving
    :math:`h \approx 0` for fGn where the process has :math:`H = 0.7`.

    Handing every estimator the same level series makes them answer the same
    question, which is the precondition for comparing them at all.
    """
    if kind == "level":
        return x
    if kind == "increments":
        return np.asarray(np.cumsum(x - float(np.mean(x))), dtype=np.float64)
    raise ValueError(f"kind must be 'level' or 'increments', got {kind!r}")


def get_estimator(name: str) -> Estimator:
    """Look up an estimator by name, with a useful error when it is misspelt."""
    try:
        return ESTIMATORS[name]
    except KeyError:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(ESTIMATORS)}") from None


def analyze(
    x: Sequence[float] | NDArray[Any],
    *,
    estimator: str | Estimator = _DEFAULT,
    kind: SeriesKind = "level",
    q: Sequence[float] | NDArray[Any] | None = None,
    scales: Sequence[int] | NDArray[Any] | None = None,
    spectrum_method: str = "fenchel",
    cumulants: bool = True,
    surrogate_test: bool = False,
    n_surrogates: int = 200,
    seed: SeedLike = None,
    nan_policy: str = "raise",
    **estimator_kwargs: Any,
) -> MultifractalReport:
    r"""Estimate the full multifractal description of a series.

    Parameters
    ----------
    estimator
        A name from :data:`ESTIMATORS`, or any callable with the standard
        signature. Defaults to wavelet leaders, the only family in the library
        that is theoretically justified for :math:`q<0`.
    kind
        Whether ``x`` is a **level** series (a log-price, an fBm path: already
        integrated) or a series of **increments** (log-returns, fGn). Increments
        are integrated once so that every estimator sees the same
        representation. Getting this wrong shifts every exponent by exactly 1,
        and it is the commonest error in applied multifractal analysis: MF-DFA
        on an fBm path reports :math:`h(2)=1.7` for a process with
        :math:`H=0.7` if told the path is increments.
    q
        Moment grid. Defaults to the estimator's own, which spans **negative**
        :math:`q` — the right branch of :math:`f(\alpha)` exists only there.
    spectrum_method
        ``"fenchel"`` (the definition, always concave) or ``"parametric"``.
    cumulants
        Extract :math:`(c_1,c_2,c_3)`. :math:`c_2` is the single most reliable
        multifractality statistic available and costs nothing extra.
    surrogate_test
        Run the shuffle and IAAFT tests. Off by default because it costs
        ``2 * n_surrogates`` estimator evaluations — but a spectrum width
        reported without it is not evidence of multifractality.

    Returns
    -------
    MultifractalReport
        Carries the exponents, the spectrum, the cumulants, any tests, and a
        provenance root chaining all of them.

    Notes
    -----
    The report's ``tests`` mapping always includes ``concavity_defect``: the
    largest vertical gap between the estimated :math:`\zeta` and its concave
    envelope. Compared against the typical :math:`\hat\zeta` standard error, a
    large defect means the multifractal formalism does not apply to this sample,
    whatever the resulting curve looks like.
    """
    series = validate_series(x, nan_policy=nan_policy)  # type: ignore[arg-type]
    fn = get_estimator(estimator) if isinstance(estimator, str) else estimator
    name = estimator if isinstance(estimator, str) else getattr(fn, "__name__", "custom")

    series = _to_level(series, kind)

    kwargs: dict[str, Any] = dict(estimator_kwargs)
    if isinstance(estimator, str):
        # Caller-supplied kwargs win, so an explicit integrate= is still honoured.
        kwargs = {**_LEVEL_KWARGS.get(estimator, {}), **kwargs}
    if q is not None:
        kwargs["q"] = q
    if scales is not None:
        kwargs["scales"] = scales
    exponents = fn(series, seed=seed, **kwargs)
    spectrum = spectrum_from_exponents(exponents, method=spectrum_method)

    cumulant_result = None
    if cumulants:
        from mmar.inference.cumulants import cumulants_from_exponents  # noqa: PLC0415

        cumulant_result = cumulants_from_exponents(exponents, order=3)

    tests: dict[str, Any] = {}
    defect = concavity_defect(exponents.q, exponents.zeta)
    tests["concavity_defect"] = defect
    if exponents.zeta_stderr is not None:
        typical = float(np.nanmedian(exponents.zeta_stderr))
        tests["concavity_defect_in_stderr"] = defect / typical if typical > 0 else float("inf")

    if surrogate_test:
        from mmar.inference.tests import multifractality_report  # noqa: PLC0415

        # The surrogate ensemble must go through the *same* estimator configuration
        # as the reported exponents. Handing over the bare registry callable instead
        # would drop the caller's q grid and scales, and would drop _LEVEL_KWARGS --
        # so mfdfa would integrate the already-integrated series a second time and
        # the tested width would belong to a spectrum shifted by one whole exponent.
        bound = functools.partial(fn, seed=seed, **kwargs)
        tests["surrogates"] = multifractality_report(
            series, estimator=bound, n_surrogates=n_surrogates, seed=seed
        )

    inputs = [InputRef.from_array("x", series)]
    if exponents.provenance is not None:
        inputs.append(InputRef.from_provenance("exponents", exponents.provenance))
    provenance = new_provenance(
        "analyze",
        params={
            "estimator": name,
            "kind": kind,
            "spectrum_method": spectrum_method,
            "cumulants": cumulants,
            "surrogate_test": surrogate_test,
            "n_surrogates": n_surrogates if surrogate_test else None,
            "nan_policy": nan_policy,
            **{k: v for k, v in kwargs.items() if not isinstance(v, np.ndarray)},
        },
        inputs=[InputRef.from_array("x", series)]
        + (
            [InputRef.from_provenance("exponents", exponents.provenance)]
            if exponents.provenance
            else []
        ),
        seed=seed_info(seed),
    )

    return MultifractalReport(
        exponents=exponents,
        spectrum=spectrum,
        cumulants=cumulant_result,
        tests=tests,
        provenance=provenance,
    )


def compare_estimators(
    x: Sequence[float] | NDArray[Any],
    *,
    estimators: Sequence[str] = ("wavelet_leaders", "mfdfa", "mfdma", "structure_function"),
    kind: SeriesKind = "level",
    q: Sequence[float] | NDArray[Any] | None = None,
    seed: SeedLike = None,
    skip_unavailable: bool = True,
    **kwargs: Any,
) -> dict[str, MultifractalReport]:
    """Run several estimators on the same series.

    ``kind`` is applied to each estimator through :data:`_KIND_KWARGS`. Without
    it the comparison is meaningless: ``mfdfa`` and ``mfdma`` ship opposite
    defaults for the integration flag, so their exponents would differ by 1 for
    reasons that have nothing to do with the data.

    ``skip_unavailable`` tolerates a missing optional dependency (PyWavelets for
    the wavelet families) by omitting that estimator, rather than failing the
    whole comparison. Anything else — a genuine estimation failure — propagates.
    """
    if len(estimators) == 0:
        # Otherwise the empty result falls through to the "nothing could run"
        # branch below, which blames a missing optional dependency.
        raise ValueError(f"estimators is empty; pass at least one of {sorted(ESTIMATORS)}")
    out: dict[str, MultifractalReport] = {}
    for name in estimators:
        try:
            out[name] = analyze(x, estimator=name, kind=kind, q=q, seed=seed, **kwargs)
        except ImportError:
            if not skip_unavailable:
                raise
    if not out:
        raise RuntimeError(
            f"no estimator from {list(estimators)} could run; install the optional "
            "extras with `pip install mmar[full]`"
        )
    return out


@dataclass(frozen=True, slots=True)
class ConsensusReport:
    r"""How much the estimators agree, and therefore how much to believe them."""

    alpha_0: dict[str, float]
    width: dict[str, float]
    h2: dict[str, float]
    c2: dict[str, float]
    alpha_0_spread: float
    width_spread: float
    h2_spread: float
    verdict: str

    def summary(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "alpha_0": self.alpha_0,
            "alpha_0_spread": self.alpha_0_spread,
            "width": self.width,
            "width_spread": self.width_spread,
            "h2": self.h2,
            "h2_spread": self.h2_spread,
            "c2": self.c2,
        }

    def __str__(self) -> str:
        rows = [f"consensus: {self.verdict}"]
        for name in sorted(self.alpha_0):
            rows.append(
                f"  {name:<20} alpha_0={self.alpha_0[name]:+.4f}  width={self.width[name]:.4f}  "
                f"h(2)={self.h2[name]:+.4f}  c2={self.c2.get(name, float('nan')):+.4f}"
            )
        rows.append(
            f"  spread               alpha_0={self.alpha_0_spread:.4f}  "
            f"width={self.width_spread:.4f}  h(2)={self.h2_spread:.4f}"
        )
        return "\n".join(rows)


def consensus(reports: Mapping[str, MultifractalReport], *, tol: float = 0.10) -> ConsensusReport:
    r"""Summarise agreement across estimators.

    ``tol`` is the spread in :math:`\alpha_0` above which the verdict turns
    negative. The default of 0.10 is deliberately loose: estimator families
    differ by construction, and demanding tighter agreement than that on a
    finite sample would reject almost everything. What the verdict is really
    for is catching the case where one estimator reports a spectrum an order of
    magnitude wider than the others — which normally means a moment that does
    not exist, or a scaling range straddling a crossover.
    """
    alpha_0, width, h2, c2 = {}, {}, {}, {}
    for name, rep in reports.items():
        d = rep.spectrum.descriptors()
        alpha_0[name] = d.alpha_0
        width[name] = d.width
        h2[name] = rep.exponents.h2
        if rep.cumulants is not None:
            c2[name] = rep.cumulants.c2

    def spread(values: Mapping[str, float]) -> float:
        finite = [v for v in values.values() if math.isfinite(v)]
        return max(finite) - min(finite) if len(finite) >= 2 else float("nan")

    a_spread, w_spread, h_spread = spread(alpha_0), spread(width), spread(h2)

    if not math.isfinite(a_spread):
        verdict = "single estimator: no cross-check performed"
    elif a_spread <= tol and w_spread <= 2 * tol:
        verdict = f"estimators agree (alpha_0 spread {a_spread:.3f} <= {tol})"
    elif a_spread <= tol:
        verdict = (
            f"apex agrees (spread {a_spread:.3f}) but widths disagree ({w_spread:.3f}); "
            "the spectrum's shape is not robust, so quote alpha_0 and not the width"
        )
    else:
        verdict = (
            f"estimators DISAGREE: alpha_0 spread {a_spread:.3f} > {tol}. Treat the scaling "
            "assumption as unsupported until the scaling range and moment existence are checked"
        )

    return ConsensusReport(
        alpha_0=alpha_0,
        width=width,
        h2=h2,
        c2=c2,
        alpha_0_spread=a_spread,
        width_spread=w_spread,
        h2_spread=h_spread,
        verdict=verdict,
    )


def spectrum_table(reports: Mapping[str, MultifractalReport]) -> str:
    """A plain-text table of the headline numbers, for a log or a paper draft."""
    header = (
        f"{'estimator':<20}{'h(2)':>9}{'alpha_0':>10}{'width':>9}{'asym':>8}{'c2':>9}{'defect':>9}"
    )
    lines = [header, "-" * len(header)]
    for name in sorted(reports):
        rep = reports[name]
        d = rep.spectrum.descriptors()
        c2 = rep.cumulants.c2 if rep.cumulants is not None else float("nan")
        defect = float(rep.tests.get("concavity_defect", float("nan")))
        lines.append(
            f"{name:<20}{rep.exponents.h2:>9.4f}{d.alpha_0:>10.4f}{d.width:>9.4f}"
            f"{d.asymmetry:>8.2f}{c2:>9.4f}{defect:>9.4f}"
        )
    return "\n".join(lines)


def spectrum_agreement(a: HolderSpectrum, b: HolderSpectrum, *, n: int = 101) -> float:
    """RMS distance between two spectra on their common support.

    Returns NaN when the supports do not overlap, which is itself the finding.
    """
    lo = max(float(np.nanmin(a.alpha)), float(np.nanmin(b.alpha)))
    hi = min(float(np.nanmax(a.alpha)), float(np.nanmax(b.alpha)))
    if not (hi > lo):
        return float("nan")
    grid: FloatArray = np.linspace(lo, hi, n)
    fa, fb = a.evaluate(grid), b.evaluate(grid)
    ok = np.isfinite(fa) & np.isfinite(fb)
    return float(np.sqrt(np.mean((fa[ok] - fb[ok]) ** 2))) if ok.any() else float("nan")
