"""Hölder-spectrum estimators.

Series-domain estimators map a signal to :class:`~mmar.core.types.ScalingExponents`
and share one signature. Measure-domain estimators take a discrete positive
measure. Pointwise estimators return a :class:`~mmar.core.types.PointwiseHolder`
field: an exponent per sample rather than their distribution.
"""

from mmar.estimators.box_counting import (
    box_partition,
    coarse_holder,
    dq,
    generalised_dimensions,
    histogram_spectrum,
    information_dimension,
    multifractal_measure_spectrum,
)
from mmar.estimators.chhabra_jensen import (
    chhabra_jensen,
    chhabra_jensen_series,
    coarse_grain,
)
from mmar.estimators.ghe import (
    aggregated_variance,
    detrended_fluctuation,
    generalised_hurst,
    hurst_consensus,
    periodogram_hurst,
    rescaled_range,
)
from mmar.estimators.mfdfa import dfa, mfdfa, mfdfa_structure
from mmar.estimators.mfdma import mfdma, mfdma_structure
from mmar.estimators.pointwise import (
    holder_at,
    leader_holder,
    local_regularity_map,
    oscillation_holder,
    wavelet_holder,
)
from mmar.estimators.structure_function import (
    critical_moment_order,
    hurst_from_tau_root,
    increment_counts,
    mfc_exponents,
    partition_function,
    structure_functions,
)
from mmar.estimators.wavelet_leaders import (
    leader_cumulants,
    wavelet_coefficients_exponents,
    wavelet_leaders,
)
from mmar.estimators.wtmm import maxima_lines, wtmm, wtmm_structure

__all__ = [
    "aggregated_variance",
    "box_partition",
    "chhabra_jensen",
    "chhabra_jensen_series",
    "coarse_grain",
    "coarse_holder",
    "critical_moment_order",
    "detrended_fluctuation",
    "dfa",
    "dq",
    "generalised_dimensions",
    "generalised_hurst",
    "histogram_spectrum",
    "holder_at",
    "hurst_consensus",
    "hurst_from_tau_root",
    "increment_counts",
    "information_dimension",
    "leader_cumulants",
    "leader_holder",
    "local_regularity_map",
    "maxima_lines",
    "mfc_exponents",
    "mfdfa",
    "mfdfa_structure",
    "mfdma",
    "mfdma_structure",
    "multifractal_measure_spectrum",
    "oscillation_holder",
    "partition_function",
    "periodogram_hurst",
    "rescaled_range",
    "structure_functions",
    "wavelet_coefficients_exponents",
    "wavelet_holder",
    "wavelet_leaders",
    "wtmm",
    "wtmm_structure",
]
