"""Statistical inference: deciding whether a spectrum means anything.

Every estimator returns a smooth :math:`f(\\alpha)` for any input, so the width
of a curve is not by itself evidence of multifractality. This subpackage
supplies the evidence:

- :mod:`~mmar.inference.surrogates` destroys one candidate cause at a time
  (temporal correlation, nonlinear structure) while preserving the others.
- :mod:`~mmar.inference.tests` turns that into p-values, plus moment-existence,
  scaling-range and concavity diagnostics.
- :mod:`~mmar.inference.bootstrap` gives block-bootstrap intervals for the
  exponents, the spectrum and its scalar descriptors.
- :mod:`~mmar.inference.cumulants` gives :math:`(c_1,c_2,c_3)`, where
  :math:`c_2` is the best-conditioned multifractality statistic available.
"""

from mmar.inference.bootstrap import (
    BootstrapResult,
    DescriptorBootstrap,
    SpectrumBootstrap,
    bootstrap_descriptors,
    bootstrap_exponents,
    bootstrap_spectrum,
    circular_block_bootstrap,
    moving_block_bootstrap,
    optimal_block_length,
    residual_bootstrap_fit,
)
from mmar.inference.cumulants import (
    compare_cumulants,
    cumulant_spectrum,
    cumulants_from_exponents,
    cumulants_from_increments,
    cumulants_from_moments,
    cumulants_from_series,
    multifractality_index,
)
from mmar.inference.surrogates import (
    aaft_surrogate,
    block_shuffle_surrogate,
    iaaft_surrogate,
    phase_randomised_surrogate,
    shuffle_surrogate,
    surrogate_ensemble,
)
from mmar.inference.tests import (
    MomentDiagnostic,
    MultifractalityReport,
    TestResult,
    concavity_test,
    moment_existence,
    monofractal_test,
    multifractality_report,
    multifractality_test,
    scaling_range_test,
)

__all__ = [
    "BootstrapResult",
    "DescriptorBootstrap",
    "MomentDiagnostic",
    "MultifractalityReport",
    "SpectrumBootstrap",
    "TestResult",
    "aaft_surrogate",
    "block_shuffle_surrogate",
    "bootstrap_descriptors",
    "bootstrap_exponents",
    "bootstrap_spectrum",
    "circular_block_bootstrap",
    "compare_cumulants",
    "concavity_test",
    "cumulant_spectrum",
    "cumulants_from_exponents",
    "cumulants_from_increments",
    "cumulants_from_moments",
    "cumulants_from_series",
    "iaaft_surrogate",
    "moment_existence",
    "monofractal_test",
    "moving_block_bootstrap",
    "multifractality_index",
    "multifractality_report",
    "multifractality_test",
    "optimal_block_length",
    "phase_randomised_surrogate",
    "residual_bootstrap_fit",
    "scaling_range_test",
    "shuffle_surrogate",
    "surrogate_ensemble",
]
