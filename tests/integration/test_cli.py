"""The v1 README documented a CLI that did not exist. This one must."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mmar.cli import _q_grid, build_parser, main


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    code = main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else {}


def test_every_advertised_subcommand_parses() -> None:
    parser = build_parser()
    for cmd in ("analyze", "calibrate", "simulate", "verify", "replay"):
        argv = [cmd, "x.json"] if cmd in ("verify", "replay") else [cmd]
        assert parser.parse_args(argv).command == cmd


def test_help_is_available_for_every_subcommand() -> None:
    for cmd in ("analyze", "calibrate", "simulate", "verify", "replay"):
        res = subprocess.run(
            [sys.executable, "-m", "mmar.cli", cmd, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, res.stderr
        assert "usage:" in res.stdout


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("-5:5:11", 11), ("0:3:5", 5), ("-2,-1,0,1,2", 5)],
)
def test_q_grid_parsing(spec: str, expected: int) -> None:
    assert _q_grid(spec).size == expected


@pytest.mark.parametrize("spec", ["1:2", "1", "1,2", "a:b:c"])
def test_bad_q_grids_are_rejected(spec: str) -> None:
    # ArgumentTypeError derives from Exception, not ValueError.
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _q_grid(spec)


def test_analyze_emits_a_summary(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(["analyze", "--estimator", "mfdfa", "--q=-3:3:9"], capsys)
    assert code == 0
    assert payload["series"] == "LMT.Close"
    assert payload["n"] == 7049
    assert {"h2", "alpha_0", "width", "root"} <= set(payload)


def test_analyze_with_several_estimators_reports_consensus(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, payload = _run(["analyze", "--estimator", "mfdfa", "mfdma", "--q=-3:3:9"], capsys)
    assert code == 0
    assert set(payload["reports"]) == {"mfdfa", "mfdma"}
    assert "verdict" in payload["consensus"]


def test_store_writes_a_manifest_that_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "artifacts"
    code, payload = _run(
        ["analyze", "--estimator", "mfdfa", "--q=-3:3:9", "--store", str(store), "--label", "t"],
        capsys,
    )
    assert code == 0
    manifest = store / "run-t.json"
    assert manifest.exists()

    code, verified = _run(["verify", str(manifest)], capsys)
    assert code == 0
    assert verified["verified"] is True
    assert verified["manifest_root"] == payload["manifest_root"]


def test_verify_fails_on_a_tampered_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "artifacts"
    _run(
        ["analyze", "--estimator", "mfdfa", "--q=-3:3:9", "--store", str(store), "--label", "t"],
        capsys,
    )
    manifest = store / "run-t.json"
    manifest.write_text(manifest.read_text().replace('"mfdfa"', '"mfdfa_edited"', 1))
    # 1, not 5: load_manifest refuses the document outright because its recorded
    # root no longer matches its contents, which is a different failure from
    # "the manifest is intact but the stored arrays do not match it".
    assert main(["verify", str(manifest)]) == 1


def test_verify_reports_corrupted_arrays_separately_from_a_forged_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An intact manifest whose payload was swapped must exit 5, not 1."""
    store = tmp_path / "artifacts"
    _run(
        ["analyze", "--estimator", "mfdfa", "--q=-3:3:9", "--store", str(store), "--label", "t"],
        capsys,
    )
    manifest = store / "run-t.json"
    assert main(["verify", str(manifest)]) == 0

    npz = next((store / "arrays").glob("*.npz"))
    np.savez_compressed(npz, **{k: np.zeros(4) for k in np.load(npz).files})
    assert main(["verify", str(manifest)]) == 5


def test_replay_lists_the_nodes_in_dependency_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "artifacts"
    _run(
        ["analyze", "--estimator", "mfdfa", "--q=-3:3:9", "--store", str(store), "--label", "t"],
        capsys,
    )
    assert main(["replay", str(store / "run-t.json")]) == 0
    plan = capsys.readouterr().out
    assert plan.index("mfdfa:exponents =") < plan.index("mfdfa:spectrum =")


def test_simulate_writes_a_path_and_reports_fat_tails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "path.txt"
    code, payload = _run(
        ["simulate", "--kmax", "11", "--hurst", "0.55", "--lam", "1.25", "--output", str(out)],
        capsys,
    )
    assert code == 0
    assert np.loadtxt(out).size == payload["n"]
    assert payload["kurtosis_log_returns"] > 3.0, (
        "a composed MMAR must be leptokurtic; v1's uncomposed simulator gave ~0"
    )


def test_simulate_is_reproducible(capsys: pytest.CaptureFixture[str]) -> None:
    roots = []
    for _ in range(2):
        _, payload = _run(["simulate", "--kmax", "10", "--seed", "5"], capsys)
        roots.append(payload["root"])
    assert roots[0] == roots[1]


def test_calibrate_signals_inadmissibility_through_the_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An inadmissible fit is a finding, but must not look like success to a shell."""
    code, payload = _run(["calibrate", "--q=-3:3:9"], capsys)
    assert code in (0, 4), "4, not 2: argparse owns 2 for usage errors"
    assert (code == 0) is payload["admissible"]
    assert {"H", "lambda", "sigma2", "diagnosis"} <= set(payload)


def test_missing_input_file_is_a_clean_error() -> None:
    assert main(["analyze", "--input", "/nonexistent/nope.csv"]) == 1


def test_exit_codes_are_distinct_and_documented() -> None:
    """A shell script must be able to tell the failure modes apart.

    Exit 2 belongs to argparse (any usage error), so an inadmissible
    calibration cannot also be 2 without making `$? -eq 2` ambiguous between
    "the MMAR specification was rejected by the data" and "you misspelled a
    flag". The epilog documents the full set.
    """
    epilog = build_parser().epilog or ""
    for code in ("0 success", "1 error", "2 argparse", "3 missing", "4 calibration", "5 stored"):
        assert code in epilog, f"exit code {code!r} is undocumented"

    res = subprocess.run(
        [sys.executable, "-m", "mmar.cli", "analyze", "--nope"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 2, "argparse usage errors must stay 2"
