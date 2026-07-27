"""The window floor must count what the estimator actually partitions."""

from __future__ import annotations

import numpy as np
import pytest

from mmar.core.validate import DataQualityError, check_scales


def test_increment_basis_is_one_window_stricter_when_the_scale_divides_n() -> None:
    """Regression: the sample basis admitted a scale yielding three increments.

    A series of n samples has n-1 unit increments, so an estimator built from
    differences gets (n-1)//s windows, not n//s. The two differ by exactly one
    whenever s divides n -- the coarse end of a dyadic grid on a power-of-two
    series, which is exactly where the fit is most fragile.
    """
    n, s = 2**16, 2**14
    assert n // s == 4 and (n - 1) // s == 3
    assert s in check_scales([s], n, min_windows=4, basis="samples")
    assert s not in check_scales([s, 2**10], n, min_windows=4, basis="increments")


@pytest.mark.parametrize("basis", ["samples", "increments"])
def test_all_surviving_scales_meet_the_floor(basis: str) -> None:
    n, floor = 4096, 4
    kept = check_scales(np.arange(2, 3000), n, min_windows=floor, basis=basis)  # type: ignore[arg-type]
    available = n if basis == "samples" else n - 1
    assert np.all(available // kept >= floor)
    assert kept.max() == available // floor


def test_the_two_bases_agree_when_the_scale_does_not_divide_n() -> None:
    n = 1000
    for s in (3, 7, 13, 111):
        assert (n // s >= 4) == ((n - 1) // s >= 4)


def test_output_is_sorted_deduplicated_and_int64() -> None:
    kept = check_scales([64, 8, 8, 32, 16], 4096)
    assert kept.dtype == np.int64
    assert kept.tolist() == [8, 16, 32, 64]


@pytest.mark.parametrize(("bad", "match"), [([], "empty"), ([1, 4], ">= 2"), ([0], ">= 2")])
def test_invalid_scale_lists_are_rejected(bad: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_scales(bad, 4096)


def test_a_series_too_short_for_any_scale_says_the_largest_admissible_one() -> None:
    with pytest.raises(DataQualityError, match="largest admissible scale is 16"):
        check_scales([100, 200], 64, min_windows=4)
