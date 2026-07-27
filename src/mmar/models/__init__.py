"""Generative models and their closed-form theory.

:mod:`~mmar.models.theory` is the oracle: exact :math:`\\tau(q)` and
:math:`f(\\alpha)` for every family here, against which every estimator in the
library is tested. The generators are exact where exactness is available ---
fBm by circulant embedding, deterministic cascades combinatorially --- so a
recovery failure is an estimator's fault, never the fixture's.
"""

from mmar.models import theory
from mmar.models.cascade import (
    CascadeResult,
    CascadeSpec,
    cascade_measure,
    cascade_moments,
    deterministic_cascade,
    generate,
    log_gamma_cascade,
    log_poisson_cascade,
    log_stable_cascade,
    lognormal_cascade,
    normalise_cascade,
    random_binomial_cascade,
    trading_time,
)
from mmar.models.fbm import (
    brownian_bridge,
    fbm,
    fbm_with_provenance,
    fgn,
    fgn_acf,
    fgn_hosking,
    levy_stable_increments,
    mbm,
)
from mmar.models.mmar import (
    MMARForecast,
    MMARPath,
    MMARSpec,
    calibrate_mmar,
    mmar_forecast_distribution,
    simulate_ensemble,
    simulate_mmar,
)

__all__ = [
    "CascadeResult",
    "CascadeSpec",
    "MMARForecast",
    "MMARPath",
    "MMARSpec",
    "brownian_bridge",
    "calibrate_mmar",
    "cascade_measure",
    "cascade_moments",
    "deterministic_cascade",
    "fbm",
    "fbm_with_provenance",
    "fgn",
    "fgn_acf",
    "fgn_hosking",
    "generate",
    "levy_stable_increments",
    "log_gamma_cascade",
    "log_poisson_cascade",
    "log_stable_cascade",
    "lognormal_cascade",
    "mbm",
    "mmar_forecast_distribution",
    "normalise_cascade",
    "random_binomial_cascade",
    "simulate_ensemble",
    "simulate_mmar",
    "theory",
    "trading_time",
]
