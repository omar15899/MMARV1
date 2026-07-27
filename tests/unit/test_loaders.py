"""Tests for :mod:`mmar.io.loaders`.

The load path is where traceability either starts or is lost, so most of what is
asserted here is about *identity*: that the root is a function of the bytes, of
nothing else, and that it moves when the bytes move.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from mmar.core.validate import DataQualityError
from mmar.io import (
    Series,
    align_series,
    datetime64_to_days,
    days_to_datetime64,
    file_digest,
    load_array,
    load_csv,
    load_lockheed,
    resample_business_days,
)
from mmar.io.loaders import (
    LOCKHEED_FILENAME,
    LOCKHEED_FIRST_DATE,
    LOCKHEED_LAST_DATE,
    LOCKHEED_ROWS,
    _lockheed_path,
)
from mmar.tracing import Run

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

pytest.importorskip("pandas", reason="load_csv needs the optional 'data' extra")

_HAS_FIXTURE = _lockheed_path().is_file()
needs_fixture = pytest.mark.skipif(
    not _HAS_FIXTURE, reason=f"Lockheed fixture absent at {_lockheed_path()}"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_csv(path: Path, rows: list[str], *, sep: str = ",", header: str = "Date,Close") -> Path:
    text = "\n".join([header.replace(",", sep), *[r.replace(",", sep) for r in rows]]) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _toy(tmp_path: Path, *, sep: str = ",") -> Path:
    rows = [
        "2020-01-02,100.0",
        "2020-01-03,101.0",
        "2020-01-06,102.5",
        "2020-01-07,101.5",
        "2020-01-08,103.0",
    ]
    return _write_csv(tmp_path / "toy.csv", rows, sep=sep)


# --------------------------------------------------------------------------- #
# file_digest
# --------------------------------------------------------------------------- #


def test_file_digest_is_sha256_of_the_bytes(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    payload = b"multifractal\n" * 100
    p.write_bytes(payload)
    assert file_digest(p) == hashlib.sha256(payload).hexdigest()
    assert len(file_digest(p)) == 64


def test_file_digest_changes_on_one_byte(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abcdef")
    before = file_digest(p)
    p.write_bytes(b"abcdeg")
    assert file_digest(p) != before


# --------------------------------------------------------------------------- #
# The Lockheed fixture: the documented facts
# --------------------------------------------------------------------------- #


@needs_fixture
def test_lockheed_shape_and_endpoints() -> None:
    s = load_lockheed()
    assert s.n == LOCKHEED_ROWS == 7049
    assert s.t is not None and s.t.size == 7049
    stamps = s.times_as_datetime64().astype("datetime64[D]")
    assert str(stamps[0]) == LOCKHEED_FIRST_DATE == "1995-03-17"
    assert str(stamps[-1]) == LOCKHEED_LAST_DATE == "2023-03-17"
    assert s.name == "LMT.Close"
    assert np.all(s.values > 0.0)


@needs_fixture
def test_lockheed_root_is_stable_across_repeated_loads() -> None:
    a, b = load_lockheed(), load_lockheed()
    assert a.provenance.root == b.provenance.root
    assert a.provenance.created_utc  # recorded, but not part of the root


@needs_fixture
def test_lockheed_root_is_content_addressed_not_path_addressed(tmp_path: Path) -> None:
    """The same bytes in a different directory are the same data."""
    original = _lockheed_path()
    copy = tmp_path / original.name
    shutil.copy2(original, copy)
    assert load_lockheed().provenance.root == load_lockheed(path=copy).provenance.root


@needs_fixture
def test_lockheed_root_is_invariant_to_the_filename_too(tmp_path: Path) -> None:
    """Not just the directory: the basename is not part of the data's identity.

    ``load_lockheed`` passes ``name=`` explicitly, so nothing path-shaped is left
    in the identifying parameters. (A bare ``load_csv`` without ``name=`` derives
    its label from the stem, and that label *is* identifying -- asserted below.)
    """
    copy = tmp_path / "some_other_name.tsv"
    shutil.copy2(_lockheed_path(), copy)
    assert load_lockheed().provenance.root == load_lockheed(path=copy).provenance.root
    assert "filename" not in load_lockheed().provenance.params
    assert load_lockheed().provenance.diagnostics["filename"] == LOCKHEED_FILENAME


def test_load_csv_label_is_identifying_when_derived_from_the_stem(tmp_path: Path) -> None:
    a = _write_csv(tmp_path / "a.csv", ["2020-01-02,1.0", "2020-01-03,2.0"])
    b = _write_csv(tmp_path / "b.csv", ["2020-01-02,1.0", "2020-01-03,2.0"])
    assert file_digest(a) == file_digest(b)
    # Different derived labels -> different roots ...
    assert load_csv(a, value_column="Close").provenance.root != (
        load_csv(b, value_column="Close").provenance.root
    )
    # ... but pin the label and the identity is the bytes alone.
    assert load_csv(a, value_column="Close", name="x").provenance.root == (
        load_csv(b, value_column="Close", name="x").provenance.root
    )


@needs_fixture
def test_lockheed_root_changes_when_one_byte_changes(tmp_path: Path) -> None:
    """A single edited digit in an *unread* column must still move the root."""
    original = _lockheed_path()
    copy = tmp_path / original.name
    shutil.copy2(original, copy)
    baseline = load_lockheed(path=copy)

    raw = copy.read_bytes()
    # Flip one byte inside the Volume field of the last row: not the Close column,
    # so the values array is bit-identical and only the file digest can catch it.
    last_line_start = raw.rfind(b"\n", 0, len(raw) - 1) + 1
    line = bytearray(raw[last_line_start:])
    assert line.endswith(b"6525176\n")
    line[-2:-1] = b"7"
    mutated = raw[:last_line_start] + bytes(line)
    assert len(mutated) == len(raw)
    assert mutated != raw
    copy.write_bytes(mutated)

    after = load_lockheed(path=copy)
    assert np.array_equal(after.values, baseline.values), "the Close column must be untouched"
    assert after.provenance.root != baseline.provenance.root


@needs_fixture
def test_lockheed_provenance_records_the_file_facts() -> None:
    s = load_lockheed()
    prov = s.provenance
    assert prov.estimator == "io.load_csv"
    assert prov.params["n_rows"] == 7049
    assert prov.params["columns"] == ["Date", "Close", "Open", "High", "Low", "Volume"]
    assert prov.params["nan_policy"] == "raise"
    assert prov.params["sep"] == "\t"
    file_refs = [i for i in prov.inputs if i.dtype == "text/sha256"]
    assert len(file_refs) == 1
    assert file_refs[0].digest == file_digest(_lockheed_path())
    assert prov.diagnostics["n_nonfinite"] == 0
    assert prov.diagnostics["file_sha256"] == file_refs[0].digest


@needs_fixture
def test_lockheed_summary_reports_fat_tails() -> None:
    """Excess kurtosis of the daily log returns is the legacy audit's 8.8."""
    summary = load_lockheed().summary()
    assert summary["n"] == 7049
    assert summary["returns_basis"] == "log"
    assert summary["n_nonfinite"] == 0
    assert summary["root"] == load_lockheed().provenance.root
    assert 8.0 < summary["kurtosis"] < 10.0
    assert summary["span"] == pytest.approx(10227.0)


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #


@needs_fixture
def test_log_price_starts_at_exactly_zero() -> None:
    lp = load_lockheed().log_price
    assert lp[0] == 0.0
    assert lp.dtype == np.float64


def test_log_price_starts_at_exactly_zero_synthetic() -> None:
    s = load_array(np.array([3.7, 4.1, 3.9, 5.0]), name="p")
    assert s.log_price[0] == 0.0
    np.testing.assert_allclose(s.log_price, np.log(s.values / s.values[0]))


def test_log_returns_are_the_difference_of_log_price() -> None:
    s = load_array(np.array([100.0, 101.0, 99.5, 103.0]), name="p")
    np.testing.assert_allclose(s.log_returns, np.diff(np.log(s.values)))
    assert s.log_returns.size == s.n - 1


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_log_price_rejects_non_positive_values(bad: float) -> None:
    s = load_array(np.array([100.0, bad, 102.0]), name="p")
    with pytest.raises(DataQualityError, match="non-positive"):
        _ = s.log_price
    with pytest.raises(DataQualityError):
        _ = s.log_returns


def test_summary_falls_back_to_differences_for_a_non_price_series() -> None:
    rng = np.random.Generator(np.random.PCG64(11))
    s = load_array(rng.standard_normal(500), name="returns")
    summary = s.summary()
    assert summary["returns_basis"] == "difference"
    assert np.isfinite(summary["kurtosis"])


def test_increments_non_overlapping_count_is_exact() -> None:
    """The non-overlapping count is exactly ``floor((n-1)/delta)``, brute-forced."""
    n = 101
    x = np.arange(n, dtype=np.float64)
    s = load_array(x, name="ramp")
    for delta in (1, 2, 3, 7, 13, 50):
        inc = s.increments(delta, overlapping=False)
        expected = [x[(i + 1) * delta] - x[i * delta] for i in range((n - 1) // delta)]
        assert inc.size == (n - 1) // delta
        np.testing.assert_array_equal(inc, expected)
        assert s.increments(delta, overlapping=True).size == n - delta


def test_increments_agree_with_strided_slicing_and_do_not_fix_d6() -> None:
    """The docstring must not claim a fix it does not deliver.

    ``x[d::d]`` and ``x[:-d:d]`` always have the *same* length
    ``floor((n-1)/d)``, so the legacy strided subtraction never truncated
    anything and explicit indexing returns bit-identical output. Legacy defect
    D6 is the floor itself: the count slope at ``q=0`` is -1.0179, not -1, and
    no indexing convention in this module can move it.
    """
    for n in (2, 7, 10, 101, 257):
        x = np.random.default_rng(n).normal(size=n)
        s = load_array(x, name="x")
        for d in range(1, n):
            np.testing.assert_array_equal(s.increments(d, overlapping=False), x[d::d] - x[:-d:d])

    n = LOCKHEED_ROWS
    scales = np.arange(2, 1000, dtype=np.int64)
    counts = np.array(
        [
            load_array(np.zeros(n), name="z").increments(int(d), overlapping=False).size
            for d in scales
        ]
    )
    np.testing.assert_array_equal(counts, (n - 1) // scales)
    slope = float(np.polyfit(np.log(scales), np.log(counts), 1)[0])
    assert slope == pytest.approx(-1.0179, abs=5e-4), "D6 is the floor and is still here"
    assert abs(slope + 1.0) > 1e-3, "increments() must not be advertised as fixing D6"


def test_series_arrays_are_write_protected() -> None:
    """A content address that can be falsified in place is not a content address."""
    s = load_array(np.arange(5.0), name="x", t=np.arange(5.0))
    root = s.provenance.root
    with pytest.raises(ValueError, match="read-only"):
        s.values[0] = 999.0
    assert s.t is not None
    with pytest.raises(ValueError, match="read-only"):
        s.t[0] = -1.0
    assert s.provenance.root == root
    np.testing.assert_array_equal(s.values, np.arange(5.0))
    # The lock must not be a view over a writable base.
    assert s.values.base is None and s.t.base is None


def test_constructing_a_series_does_not_freeze_the_callers_array() -> None:
    caller = np.arange(4.0)
    load_array(caller, name="x")
    caller[0] = 7.0  # must not raise
    assert caller[0] == 7.0


def test_increments_reject_out_of_range_delta() -> None:
    s = load_array(np.arange(10, dtype=np.float64), name="ramp")
    with pytest.raises(ValueError, match="delta"):
        s.increments(0)
    with pytest.raises(ValueError, match="delta"):
        s.increments(10)


# --------------------------------------------------------------------------- #
# load_array
# --------------------------------------------------------------------------- #


def test_load_array_root_is_the_array_content_address() -> None:
    x = np.linspace(1.0, 2.0, 64)
    a = load_array(x, name="x")
    b = load_array(x.copy(), name="x")
    c = load_array(x + 1e-15, name="x")
    assert a.provenance.root == b.provenance.root
    assert a.provenance.root != c.provenance.root
    assert load_array(x, name="y").provenance.root != a.provenance.root


def test_load_array_rejects_mismatched_time_index() -> None:
    with pytest.raises(ValueError, match="same length"):
        load_array(np.arange(5.0), name="x", t=np.arange(4.0))


def test_series_requires_a_strictly_increasing_time_index() -> None:
    with pytest.raises(ValueError, match="non-increasing"):
        load_array(np.arange(4.0), name="x", t=np.array([0.0, 2.0, 1.0, 3.0]))


def test_series_rejects_empty_input() -> None:
    with pytest.raises(DataQualityError, match="empty"):
        load_array(np.empty(0), name="x")


def test_load_array_drop_policy_filters_the_time_index_too() -> None:
    v = np.array([1.0, np.nan, 3.0, 4.0])
    t = np.array([10.0, 11.0, 12.0, 13.0])
    s = load_array(v, name="x", t=t, nan_policy="drop")
    assert s.n == 3
    np.testing.assert_array_equal(s.t, np.array([10.0, 12.0, 13.0]))
    assert s.provenance.diagnostics["n_nonfinite"] == 1


# --------------------------------------------------------------------------- #
# load_csv
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sep", [",", "\t", ";", "|"])
def test_load_csv_sniffs_the_separator(tmp_path: Path, sep: str) -> None:
    p = _toy(tmp_path, sep=sep)
    sniffed = load_csv(p, value_column="Close", time_column="Date")
    explicit = load_csv(p, value_column="Close", time_column="Date", sep=sep)
    assert sniffed.provenance.params["sep"] == sep
    assert sniffed.provenance.root == explicit.provenance.root
    assert sniffed.n == 5


def test_load_csv_records_columns_rows_and_digest(tmp_path: Path) -> None:
    p = _toy(tmp_path)
    s = load_csv(p, value_column="Close", time_column="Date")
    prov = s.provenance
    assert prov.params["columns"] == ["Date", "Close"]
    assert prov.params["n_rows"] == 5
    assert prov.params["value_column"] == "Close"
    assert prov.params["time_column"] == "Date"
    assert prov.diagnostics["path"] == str(p.resolve())
    assert prov.diagnostics["file_sha256"] == file_digest(p)
    assert s.name == "toy.Close"


def test_load_csv_root_changes_with_one_mutated_byte(tmp_path: Path) -> None:
    p = _toy(tmp_path)
    before = load_csv(p, value_column="Close", time_column="Date").provenance.root
    raw = p.read_bytes()
    mutated = raw.replace(b"102.5", b"102.6", 1)
    assert len(mutated) == len(raw) and mutated != raw
    p.write_bytes(mutated)
    assert load_csv(p, value_column="Close", time_column="Date").provenance.root != before


def test_load_csv_times_round_trip_to_the_calendar(tmp_path: Path) -> None:
    s = load_csv(_toy(tmp_path), value_column="Close", time_column="Date")
    days = s.times_as_datetime64().astype("datetime64[D]")
    assert [str(d) for d in days] == [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
    ]


def test_load_csv_rejects_a_missing_column(tmp_path: Path) -> None:
    with pytest.raises(DataQualityError, match="not in"):
        load_csv(_toy(tmp_path), value_column="Adjusted")


def test_load_csv_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/does_not_exist.csv", value_column="Close")


def test_unsniffable_file_says_so(tmp_path: Path) -> None:
    p = tmp_path / "plain.txt"
    p.write_text("Close\n1.0\n2.0\n", encoding="utf-8")
    with pytest.raises(DataQualityError, match="could not infer a separator"):
        load_csv(p, value_column="Close")


# --------------------------------------------------------------------------- #
# NaN policy
# --------------------------------------------------------------------------- #


def _csv_with_a_hole(tmp_path: Path) -> Path:
    rows = [
        "2020-01-02,100.0",
        "2020-01-03,101.0",
        "2020-01-06,",
        "2020-01-07,103.0",
        "2020-01-08,104.0",
    ]
    return _write_csv(tmp_path / "hole.csv", rows)


def test_nan_raises_under_the_default_policy(tmp_path: Path) -> None:
    p = _csv_with_a_hole(tmp_path)
    with pytest.raises(DataQualityError) as excinfo:
        load_csv(p, value_column="Close", time_column="Date")
    message = str(excinfo.value)
    assert "1 non-finite" in message
    assert "[2]" in message, "the offending row must be named"
    assert "interpolate" in message and "drop" in message


def test_nan_interpolated_and_counted_in_diagnostics(tmp_path: Path) -> None:
    p = _csv_with_a_hole(tmp_path)
    s = load_csv(p, value_column="Close", time_column="Date", nan_policy="interpolate")
    assert s.n == 5
    assert np.all(np.isfinite(s.values))
    # Linear in index between 101.0 and 103.0.
    assert s.values[2] == pytest.approx(102.0)
    assert s.provenance.diagnostics["n_nonfinite"] == 1
    assert s.provenance.diagnostics["n_missing_in_file"] == 1
    assert s.provenance.diagnostics["n_unparsed"] == 0
    assert s.provenance.diagnostics["n_values"] == 5
    assert s.provenance.params["nan_policy"] == "interpolate"


def test_nan_dropped_shortens_both_columns(tmp_path: Path) -> None:
    s = load_csv(
        _csv_with_a_hole(tmp_path), value_column="Close", time_column="Date", nan_policy="drop"
    )
    assert s.n == 4
    assert s.t is not None and s.t.size == 4
    assert s.provenance.diagnostics["n_nonfinite"] == 1
    assert s.provenance.diagnostics["n_values"] == 4


def test_nan_policies_have_distinct_roots(tmp_path: Path) -> None:
    p = _csv_with_a_hole(tmp_path)
    a = load_csv(p, value_column="Close", time_column="Date", nan_policy="drop")
    b = load_csv(p, value_column="Close", time_column="Date", nan_policy="interpolate")
    assert a.provenance.root != b.provenance.root


def test_unparseable_entry_is_counted_separately(tmp_path: Path) -> None:
    # "closed" is not in pandas' default NA vocabulary, so it reaches to_numeric as a
    # string and is a schema error, not a gap. "n/a" would be counted as a gap instead.
    rows = ["2020-01-02,100.0", "2020-01-03,closed", "2020-01-06,102.0"]
    p = _write_csv(tmp_path / "junk.csv", rows)
    s = load_csv(p, value_column="Close", time_column="Date", nan_policy="interpolate")
    assert s.provenance.diagnostics["n_unparsed"] == 1
    assert s.provenance.diagnostics["n_missing_in_file"] == 0
    assert s.provenance.diagnostics["n_nonfinite"] == 1


def test_nonfinite_diagnostics_reconcile(tmp_path: Path) -> None:
    """n_nonfinite must equal missing + unparsed + infinite, or it is not an audit."""
    rows = [
        "2020-01-02,100.0",
        "2020-01-03,",
        "2020-01-06,closed",
        "2020-01-07,inf",
        "2020-01-08,104.0",
    ]
    p = _write_csv(tmp_path / "mixed.csv", rows)
    s = load_csv(p, value_column="Close", time_column="Date", nan_policy="drop")
    d = s.provenance.diagnostics
    assert (d["n_missing_in_file"], d["n_unparsed"], d["n_infinite_in_file"]) == (1, 1, 1)
    assert d["n_nonfinite"] == d["n_missing_in_file"] + d["n_unparsed"] + d["n_infinite_in_file"]
    assert s.n == 2


def test_interpolation_refuses_to_extrapolate_at_the_edges(tmp_path: Path) -> None:
    rows = ["2020-01-02,", "2020-01-03,101.0", "2020-01-06,102.0"]
    p = _write_csv(tmp_path / "edge.csv", rows)
    with pytest.raises(DataQualityError, match="first or last row"):
        load_csv(p, value_column="Close", time_column="Date", nan_policy="interpolate")


def test_unknown_nan_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown nan_policy"):
        load_array(np.array([1.0, np.nan, 3.0]), name="x", nan_policy="fill")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Time conversion
# --------------------------------------------------------------------------- #


def test_days_round_trip() -> None:
    stamps = np.array(["1995-03-17", "2023-03-17", "1970-01-01"], dtype="datetime64[ns]")
    days = datetime64_to_days(stamps)
    np.testing.assert_array_equal(days_to_datetime64(days), stamps)
    assert days[2] == 0.0


def test_nat_is_rejected() -> None:
    stamps = np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="NaT"):
        datetime64_to_days(stamps)


# --------------------------------------------------------------------------- #
# Resampling and alignment
# --------------------------------------------------------------------------- #


def test_resample_business_days_fills_the_weekday_grid(tmp_path: Path) -> None:
    # 2020-01-02 Thu, 03 Fri, 06 Mon, 07 Tue, 08 Wed: one weekend, no weekday gap.
    s = load_csv(_toy(tmp_path), value_column="Close", time_column="Date")
    r = resample_business_days(s)
    assert r.n == 5
    np.testing.assert_allclose(r.values, s.values)
    assert r.provenance.diagnostics["n_gaps_filled"] == 0


def test_resample_business_days_forward_fills_a_holiday() -> None:
    """A missing weekday becomes a repeated price, i.e. an exact zero return."""
    days = datetime64_to_days(
        np.array(["2020-01-02", "2020-01-03", "2020-01-07"], dtype="datetime64[ns]")
    )
    s = load_array(np.array([100.0, 101.0, 103.0]), name="p", t=days)
    r = resample_business_days(s)
    assert r.n == 4  # Thu, Fri, Mon (filled), Tue
    np.testing.assert_allclose(r.values, [100.0, 101.0, 101.0, 103.0])
    # r = [ln(101/100), ln(101/101), ln(103/101)]: the filled Monday is an exact zero.
    assert r.log_returns[1] == 0.0
    assert r.provenance.diagnostics["n_gaps_filled"] == 1
    assert r.provenance.root != s.provenance.root


def test_resample_business_days_can_leave_gaps_missing() -> None:
    days = datetime64_to_days(
        np.array(["2020-01-02", "2020-01-03", "2020-01-07"], dtype="datetime64[ns]")
    )
    s = load_array(np.array([100.0, 101.0, 103.0]), name="p", t=days)
    r = resample_business_days(s, method="nan")
    assert np.isnan(r.values[2])
    assert r.provenance.diagnostics["n_gaps_left_missing"] == 1
    assert r.summary()["n_nonfinite"] == 1


def test_resample_counts_intraday_observations_it_collapses() -> None:
    """Two prints on one calendar day: one of them is deleted, and it is counted."""
    days = datetime64_to_days(
        np.array(
            ["2020-01-02T09:30", "2020-01-02T16:00", "2020-01-03T16:00"],
            dtype="datetime64[ns]",
        )
    )
    s = load_array(np.array([10.0, 20.0, 30.0]), name="p", t=days)
    r = resample_business_days(s)
    np.testing.assert_allclose(r.values, [20.0, 30.0])  # last print of the day wins
    d = r.provenance.diagnostics
    assert d["n_observations_collapsed"] == 1
    assert d["n_in"] - d["n_out"] == (
        d["n_observations_collapsed"] + d["n_observations_off_grid"] - d["n_gaps_filled"]
    )


def test_resample_reports_zero_collapses_on_a_clean_daily_grid(tmp_path: Path) -> None:
    s = load_csv(_toy(tmp_path), value_column="Close", time_column="Date")
    assert resample_business_days(s).provenance.diagnostics["n_observations_collapsed"] == 0


def test_resample_requires_a_time_index() -> None:
    with pytest.raises(ValueError, match="no time index"):
        resample_business_days(load_array(np.arange(5.0), name="x"))


def test_align_series_intersects_exactly() -> None:
    a_days = datetime64_to_days(
        np.array(["2020-01-02", "2020-01-03", "2020-01-06"], dtype="datetime64[ns]")
    )
    b_days = datetime64_to_days(
        np.array(["2020-01-03", "2020-01-06", "2020-01-07"], dtype="datetime64[ns]")
    )
    a = load_array(np.array([1.0, 2.0, 3.0]), name="a", t=a_days)
    b = load_array(np.array([20.0, 30.0, 40.0]), name="b", t=b_days)
    ax, bx = align_series(a, b)
    np.testing.assert_allclose(ax.values, [2.0, 3.0])
    np.testing.assert_allclose(bx.values, [20.0, 30.0])
    np.testing.assert_array_equal(ax.t, bx.t)
    assert ax.provenance.diagnostics["n_out"] == 2
    parents = {i.digest for i in ax.provenance.inputs}
    assert b.provenance.root in parents, "the alignment partner must be in the DAG"


def test_align_series_disjoint_grids_raise() -> None:
    a = load_array(np.arange(3.0), name="a", t=np.array([0.0, 1.0, 2.0]))
    b = load_array(np.arange(3.0), name="b", t=np.array([10.0, 11.0, 12.0]))
    with pytest.raises(DataQualityError, match="share no timestamp"):
        align_series(a, b)


def test_align_series_needs_two() -> None:
    a = load_array(np.arange(3.0), name="a", t=np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="at least two"):
        align_series(a)


# --------------------------------------------------------------------------- #
# The traceability contract
# --------------------------------------------------------------------------- #


def test_series_is_recordable_in_a_run(tmp_path: Path) -> None:
    s = load_csv(_toy(tmp_path), value_column="Close", time_column="Date")
    with Run("io-smoke") as run:
        run.record(s, name="prices")
    node = run.manifest.by_name("prices")
    assert node.root == s.provenance.root
    assert node.estimator == "io.load_csv"
    assert node.summary["n"] == 5


def test_repr_is_informative(tmp_path: Path) -> None:
    s = load_csv(_toy(tmp_path), value_column="Close")
    assert "toy.Close" in str(s)
    assert s.provenance.short_root in str(s)


def test_series_is_frozen() -> None:
    s = load_array(np.arange(5.0), name="x")
    with pytest.raises(AttributeError):
        s.name = "y"  # type: ignore[misc]
    assert isinstance(s, Series)
