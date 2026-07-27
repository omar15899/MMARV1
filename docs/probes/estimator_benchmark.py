"""Measured bias and dispersion of every estimator, against known ground truth.

Produces the numbers in ``docs/ESTIMATOR_GUIDE.md``. Nothing here is tuned to
flatter the library: the fixtures have closed-form answers, and whatever the
estimators report is what gets tabulated.

Run:  uv run python docs/probes/estimator_benchmark.py [--replicas 20]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mmar.analysis import ESTIMATORS, analyze
from mmar.core.rng import spawn_rngs
from mmar.models import theory
from mmar.models.cascade import lognormal_cascade, trading_time
from mmar.models.fbm import fbm, levy_stable_increments

Q = np.linspace(-5.0, 5.0, 33)
N = 2**14
SEED = 20260727


def _mean_sd(values: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return (float(a.mean()), float(a.std(ddof=1))) if a.size > 1 else (float("nan"), float("nan"))


def monofractal_bias(replicas: int) -> None:
    """h(2) against a known H. Bias here contaminates everything downstream."""
    print("\n### Monofractal recovery: exact fBm, n = 2^14")
    print(f"{'estimator':<20}" + "".join(f"{f'H={h}':>16}" for h in (0.3, 0.5, 0.7, 0.9)))
    print("-" * (20 + 16 * 4))
    for name in sorted(ESTIMATORS):
        cells = []
        for H in (0.3, 0.5, 0.7, 0.9):
            vals = []
            for i, rng in enumerate(spawn_rngs(SEED + int(H * 100), replicas)):
                x = fbm(N, H, seed=int(rng.integers(1 << 31)))
                try:
                    vals.append(analyze(x, estimator=name, q=Q, seed=i).exponents.h2 - H)
                except (ValueError, RuntimeError):
                    vals.append(float("nan"))
            m, s = _mean_sd(vals)
            cells.append(f"{m:+.4f}+-{s:.3f}")
        print(f"{name:<20}" + "".join(f"{c:>16}" for c in cells))
    print("  cells are (mean bias in h(2)) +- (sd across replicas)")


def monofractal_false_width(replicas: int) -> None:
    """A monofractal has zero spectrum width. Whatever an estimator reports is noise."""
    print("\n### False multifractality on a monofractal (fBm H=0.7): the detection floor")
    print(f"{'estimator':<20}{'width':>18}{'|c2|':>18}")
    print("-" * 56)
    for name in sorted(ESTIMATORS):
        widths, c2s = [], []
        for i in range(replicas):
            try:
                rep = analyze(fbm(N, 0.7, seed=1000 + i), estimator=name, q=Q, seed=i)
                widths.append(rep.spectrum.descriptors().width)
                if rep.cumulants is not None:
                    c2s.append(abs(rep.cumulants.c2))
            except (ValueError, RuntimeError):
                pass
        wm, ws = _mean_sd(widths)
        cm, cs = _mean_sd(c2s)
        print(f"{name:<20}{f'{wm:.4f}+-{ws:.3f}':>18}{f'{cm:.4f}+-{cs:.3f}':>18}")
    print("  anything below these values is indistinguishable from a monofractal")


def multifractal_recovery(replicas: int) -> None:
    """Recovery of a lognormal cascade whose (c1, c2) are known exactly."""
    lam = 1.3
    sigma2 = theory.lognormal_conservation_sigma2(lam)
    c1_true, c2_true = theory.cumulants_from_tau_lognormal(lam, sigma2)
    print(f"\n### Lognormal cascade, lambda={lam}: true c1={c1_true:.4f}, c2={c2_true:.4f}")
    print(f"{'estimator':<20}{'c1 bias':>18}{'c2 bias':>18}{'width':>16}")
    print("-" * 72)
    for name in sorted(ESTIMATORS):
        d1, d2, ws = [], [], []
        for i in range(replicas):
            theta = trading_time(lognormal_cascade(14, lam, seed=2000 + i))
            try:
                rep = analyze(theta, estimator=name, q=Q, seed=i)
            except (ValueError, RuntimeError):
                continue
            ws.append(rep.spectrum.descriptors().width)
            if rep.cumulants is not None:
                d1.append(rep.cumulants.c1 - float(c1_true))
                d2.append(rep.cumulants.c2 - float(c2_true))
        m1, s1 = _mean_sd(d1)
        m2, s2 = _mean_sd(d2)
        wm, _ = _mean_sd(ws)
        print(f"{name:<20}{f'{m1:+.4f}+-{s1:.3f}':>18}{f'{m2:+.4f}+-{s2:.3f}':>18}{wm:>16.4f}")


def false_positive_on_fat_tails(replicas: int) -> None:
    """The impostor: iid heavy tails, no temporal structure, wide spectrum anyway."""
    print("\n### iid Levy-stable noise (alpha=1.6): NO temporal structure at all")
    print(f"{'estimator':<20}{'width':>16}{'c2':>16}")
    print("-" * 52)
    for name in sorted(ESTIMATORS):
        ws, cs = [], []
        for i in range(replicas):
            x = np.cumsum(levy_stable_increments(N, 1.6, seed=3000 + i))
            try:
                rep = analyze(x, estimator=name, q=Q, seed=i)
            except (ValueError, RuntimeError):
                continue
            ws.append(rep.spectrum.descriptors().width)
            if rep.cumulants is not None:
                cs.append(rep.cumulants.c2)
        wm, _ = _mean_sd(ws)
        cm, _ = _mean_sd(cs)
        print(f"{name:<20}{wm:>16.4f}{cm:>16.4f}")
    print("  a non-zero width here is a FALSE POSITIVE; only the surrogate tests catch it")


def timings() -> None:
    print(f"\n### Runtime, n = 2^14, {Q.size} moments (single call, seconds)")
    x = fbm(N, 0.7, seed=7)
    for name in sorted(ESTIMATORS):
        t0 = time.perf_counter()
        try:
            analyze(x, estimator=name, q=Q, seed=0)
            print(f"  {name:<20}{time.perf_counter() - t0:>8.2f}")
        except (ValueError, RuntimeError) as exc:
            print(f"  {name:<20}   failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    print(f"mmar estimator benchmark | n={N}, q in [{Q.min():g},{Q.max():g}], "
          f"{args.replicas} replicas, seed {SEED}")
    for name, fn in (
        ("mono", monofractal_bias),
        ("floor", monofractal_false_width),
        ("multi", multifractal_recovery),
        ("falsepos", false_positive_on_fat_tails),
    ):
        if name not in args.skip:
            fn(args.replicas)
    if "timings" not in args.skip:
        timings()


if __name__ == "__main__":
    main()
