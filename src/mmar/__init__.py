r"""``mmar`` — a multifractal formalism engine with content-addressed provenance.

Hölder spectra by nine estimator families, cascade and MMAR models with
closed-form theory, surrogate-based inference, and a Merkle DAG over every
computation.

Quick start
-----------
>>> import numpy as np
>>> from mmar import analyze
>>> from mmar.models import fbm
>>> report = analyze(fbm(2**13, 0.7, seed=0), q=np.linspace(-5, 5, 33), seed=0)
>>> round(report.exponents.h2, 1)
0.7

Conventions
-----------
:math:`\zeta(q)` is the structure-function exponent,
:math:`\tau(q) = \zeta(q) - d` the measure exponent, and
:math:`f(\alpha) = \inf_q\{q\alpha - \tau(q)\}`. Only :math:`\zeta` is stored;
the rest are derived, so the convention cannot drift. See
:mod:`mmar.core.types` for the full statement.

Layout
------
``mmar.core``
    Conventions, provenance, log--log regression, Legendre transforms.
``mmar.estimators``
    The nine Hölder-spectrum estimators, plus pointwise :math:`h(t)`.
``mmar.inference``
    Surrogates, bootstrap, hypothesis tests, log-cumulants.
``mmar.models``
    Theory oracle, exact fBm, cascades, the MMAR.
``mmar.calibration``
    Fitting MMAR parameters, and testing the specification while doing it.
``mmar.tracing``
    Session-level traceability.
``mmar.viz``, ``mmar.io``
    Figures and loaders; both need optional extras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "2.0.0"

if TYPE_CHECKING:  # pragma: no cover - the runtime path is the lazy one below
    from mmar.analysis import analyze, compare_estimators, consensus
    from mmar.core import (
        ArtifactStore,
        HolderSpectrum,
        LogCumulants,
        MultifractalReport,
        PointwiseHolder,
        Provenance,
        ScalingExponents,
        spectrum_from_exponents,
    )
    from mmar.tracing import Run

_LAZY: dict[str, str] = {
    "analyze": "mmar.analysis",
    "compare_estimators": "mmar.analysis",
    "consensus": "mmar.analysis",
    "ArtifactStore": "mmar.core",
    "HolderSpectrum": "mmar.core",
    "LogCumulants": "mmar.core",
    "MultifractalReport": "mmar.core",
    "PointwiseHolder": "mmar.core",
    "Provenance": "mmar.core",
    "ScalingExponents": "mmar.core",
    "spectrum_from_exponents": "mmar.core",
    "Run": "mmar.tracing",
}


def __getattr__(name: str) -> Any:
    """Resolve the top-level API on first access.

    Importing ``mmar`` must not pull in scipy, PyWavelets, pandas and
    matplotlib. These names are re-exported lazily so that ``import mmar``
    stays cheap and the optional extras stay genuinely optional.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'mmar' has no attribute {name!r}")
    import importlib  # noqa: PLC0415 - deferred by design

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted([*_LAZY, "__version__"])


__all__ = [
    "ArtifactStore",
    "HolderSpectrum",
    "LogCumulants",
    "MultifractalReport",
    "PointwiseHolder",
    "Provenance",
    "Run",
    "ScalingExponents",
    "__version__",
    "analyze",
    "compare_estimators",
    "consensus",
    "spectrum_from_exponents",
]
