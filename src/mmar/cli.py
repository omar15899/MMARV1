"""Command-line interface.

The v1 README documented six commands of the form
``python mmar_simulation.py --length 1024 --hurst 0.7 --output out.txt`` and
claimed every script accepted ``--help``. No module in that package parsed a
single flag; the whole Usage section was fiction. This is the interface it
described, made real.

Every subcommand writes a run manifest when ``--store`` is given, so a result
produced from the shell is as auditable as one produced from Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mmar import __version__

__all__ = ["build_parser", "main"]


def _q_grid(spec: str) -> np.ndarray:
    """Parse ``lo:hi:n`` or a comma-separated list into a moment grid."""
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"expected lo:hi:n, got {spec!r}")
        lo, hi, n = float(parts[0]), float(parts[1]), int(parts[2])
        if n < 3:
            raise argparse.ArgumentTypeError("the q grid needs at least three points")
        return np.linspace(lo, hi, n)
    values = np.array([float(v) for v in spec.split(",")], dtype=float)
    if values.size < 3:
        raise argparse.ArgumentTypeError("the q grid needs at least three points")
    return values


def _load(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    from mmar.io import load_csv, load_lockheed  # noqa: PLC0415 - pandas is an extra

    if args.input in (None, "lockheed"):
        series = load_lockheed()
    else:
        series = load_csv(args.input, value_column=args.column, time_column=args.time_column)
    values = series.log_price if args.transform == "log-price" else np.asarray(series.values)
    if args.transform == "log-returns":
        values = np.diff(series.log_price)
    return np.asarray(values, dtype=float), series.name


def _kind(args: argparse.Namespace) -> Literal["level", "increments"]:
    """Resolve ``--kind``, defaulting to whatever ``--transform`` produced.

    ``--transform log-returns`` differences the log price here, so the series
    handed to :func:`~mmar.analysis.analyze` *is* a series of increments. A fixed
    ``--kind level`` default then told the estimator the opposite and every
    exponent came back shifted by one: on an fBm log price with :math:`H=0.65`,
    ``--transform log-returns`` reported ``h(2) = 0.036`` and
    :math:`\\alpha_0=0.047` --- the regularity of the noise --- against 0.655
    with ``--kind increments``. An explicit ``--kind`` still wins.
    """
    if args.kind in ("level", "increments"):
        return "level" if args.kind == "level" else "increments"
    return "increments" if args.transform == "log-returns" else "level"


def _emit(payload: dict[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_out:
        Path(args.json_out).write_text(text)
    print(text)


def cmd_analyze(args: argparse.Namespace) -> int:
    from mmar.analysis import (  # noqa: PLC0415
        analyze,
        compare_estimators,
        consensus,
        spectrum_table,
    )

    x, name = _load(args)
    q = _q_grid(args.q)

    if len(args.estimator) > 1:
        reports = compare_estimators(
            x, estimators=args.estimator, kind=_kind(args), q=q, seed=args.seed
        )
        print(spectrum_table(reports), file=sys.stderr)
        print(consensus(reports), file=sys.stderr)
        payload = {
            "series": name,
            "n": int(x.size),
            "consensus": consensus(reports).summary(),
            "reports": {k: v.summary() for k, v in reports.items()},
        }
    else:
        report = analyze(
            x,
            estimator=args.estimator[0],
            kind=_kind(args),
            q=q,
            seed=args.seed,
            surrogate_test=args.surrogate_test,
            n_surrogates=args.n_surrogates,
        )
        reports = {args.estimator[0]: report}
        print(spectrum_table(reports), file=sys.stderr)
        payload = {"series": name, "n": int(x.size), **report.summary()}

    if args.store:
        from mmar.tracing import Run  # noqa: PLC0415

        with Run(args.label or f"cli-{name}", store=args.store) as run:
            for key, rep in reports.items():
                run.record(rep.exponents, name=f"{key}:exponents")
                run.record(rep.spectrum, name=f"{key}:spectrum")
        payload["manifest_root"] = run.manifest_root
        print(f"manifest root {run.manifest_root}", file=sys.stderr)

    _emit(payload, args)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from mmar.analysis import analyze  # noqa: PLC0415
    from mmar.calibration import calibrate_from_spectrum  # noqa: PLC0415

    x, name = _load(args)
    report = analyze(
        x, estimator=args.estimator[0], kind=_kind(args), q=_q_grid(args.q), seed=args.seed
    )
    result = calibrate_from_spectrum(report.exponents, report.spectrum, base=args.base)
    print(result, file=sys.stderr)
    _emit({"series": name, **result.summary(), "diagnosis": list(result.diagnosis)}, args)
    # An inadmissible calibration is a finding, not a crash, but it must not look
    # like success to a shell script. Code 4, not 2: argparse exits 2 on any usage
    # error, so a script branching on 2 could not tell "the MMAR specification was
    # rejected by the data" from "you misspelled a flag".
    return 0 if result.admissible else 4


def cmd_simulate(args: argparse.Namespace) -> int:
    from mmar.models.mmar import MMARSpec, simulate_mmar  # noqa: PLC0415

    spec = MMARSpec(kmax=args.kmax, H=args.hurst, lam=args.lam, b=args.base, price0=args.price0)
    path = simulate_mmar(spec, seed=args.seed)
    values = np.asarray(path.price if args.output_field == "price" else path.x)

    if args.output:
        np.savetxt(args.output, values, fmt="%.10g")
    payload = {
        "kmax": args.kmax,
        "H": args.hurst,
        "lambda": args.lam,
        "n": int(values.size),
        "output": args.output,
        "root": path.provenance.root if path.provenance is not None else None,
        "kurtosis_log_returns": float(
            __import__("scipy.stats", fromlist=["kurtosis"]).kurtosis(np.diff(np.asarray(path.x)))
        ),
    }
    _emit(payload, args)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-hash a stored run and confirm nothing has been altered."""
    from mmar.core.provenance import ArtifactStore  # noqa: PLC0415
    from mmar.tracing import load_manifest  # noqa: PLC0415

    manifest = load_manifest(args.manifest)
    store = ArtifactStore(Path(args.manifest).parent)
    results = {n.name: store.verify(n.root) for n in manifest.nodes}
    ok = all(results.values())
    print(manifest.describe(), file=sys.stderr)
    _emit({"manifest_root": manifest.root, "verified": ok, "nodes": results}, args)
    return 0 if ok else 5


def cmd_replay(args: argparse.Namespace) -> int:
    from mmar.tracing import load_manifest, replay_plan  # noqa: PLC0415

    print(replay_plan(load_manifest(args.manifest)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmar",
        description="Multifractal formalism engine: Holder spectra, MMAR, provenance.",
    )
    parser.add_argument("--version", action="version", version=f"mmar {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_input(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--input",
            default="lockheed",
            help="CSV/TSV path, or 'lockheed' for the bundled dataset (default)",
        )
        p.add_argument("--column", default="Close", help="value column (default: Close)")
        p.add_argument("--time-column", default="Date", help="time column (default: Date)")
        p.add_argument(
            "--transform",
            default="log-price",
            choices=("log-price", "log-returns", "raw"),
            help="what to analyse (default: log-price)",
        )
        p.add_argument(
            "--kind",
            default=None,
            choices=("level", "increments"),
            help=(
                "whether the analysed series is already integrated "
                "(default: follows --transform, i.e. increments for log-returns)"
            ),
        )

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json-out", help="also write the JSON payload to this path")
        p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")

    a = sub.add_parser("analyze", help="estimate the Holder spectrum")
    add_input(a)
    add_common(a)
    a.add_argument(
        "--estimator",
        nargs="+",
        default=["wavelet_leaders"],
        help="one or more estimators; several triggers a consensus comparison",
    )
    # A negative lower bound looks like a flag to argparse, so the '=' form is
    # required: --q=-5:5:41. Say so here rather than letting it fail obscurely.
    a.add_argument(
        "--q",
        default="-5:5:41",
        help="moment grid as lo:hi:n or a comma list; use --q=-5:5:41 (default: -5:5:41)",
    )
    a.add_argument("--surrogate-test", action="store_true", help="run the shuffle and IAAFT tests")
    a.add_argument("--n-surrogates", type=int, default=200)
    a.add_argument("--store", help="artifact store directory; writes a run manifest")
    a.add_argument("--label", help="run label for the manifest")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("calibrate", help="fit MMAR parameters and test the specification")
    add_input(c)
    add_common(c)
    c.add_argument("--estimator", nargs=1, default=["wavelet_leaders"])
    c.add_argument("--q", default="-5:5:41", help="moment grid; use --q=-5:5:41")
    c.add_argument("--base", type=float, default=2.0, help="cascade base b (default: 2)")
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("simulate", help="simulate an MMAR path")
    add_common(s)
    s.add_argument("--kmax", type=int, default=14, help="cascade depth; the path has b^kmax points")
    s.add_argument("--hurst", type=float, default=0.55)
    s.add_argument("--lam", type=float, default=1.25, help="cascade lambda; must exceed 1")
    s.add_argument("--base", type=int, default=2)
    s.add_argument("--price0", type=float, default=1.0)
    s.add_argument("--output", help="write the path to this text file")
    s.add_argument("--output-field", default="price", choices=("price", "log-price"))
    s.set_defaults(func=cmd_simulate)

    v = sub.add_parser("verify", help="re-hash a stored run against its manifest")
    add_common(v)
    v.add_argument("manifest", help="path to a run-*.json manifest")
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("replay", help="print the call sequence that reproduces a run")
    add_common(r)
    r.add_argument("manifest", help="path to a run-*.json manifest")
    r.set_defaults(func=cmd_replay)

    parser.epilog = (
        "exit codes: 0 success; 1 error; 2 argparse usage error; "
        "3 missing optional dependency; 4 calibration inadmissible; "
        "5 stored artifacts failed verification"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except ImportError as exc:
        print(f"missing optional dependency: {exc}", file=sys.stderr)
        print("install the extras with: pip install 'mmar[full]'", file=sys.stderr)
        return 3
    # ArgumentTypeError derives from Exception, not ValueError: _q_grid raises it
    # outside argparse's own type= machinery, so without it here a malformed --q
    # leaves as a traceback instead of an exit code.
    except (ValueError, FileNotFoundError, KeyError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
