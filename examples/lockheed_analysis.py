"""Full multifractal analysis of Lockheed Martin, 1995-2023.

The v2 replacement for ``Scripts/pruebaMMAR1.py``, which read:

    Lockheed = MMAR(df, "Date", "Close", a=0, b=3, npoints=10)
    Lockheed.partition_functions()
    Lockheed.compute_tau_and_alpha()
    Lockheed.multifractal_measure_rand(graf1=True)
    x, y, z = Lockheed.simulacion(n=10000, result=True)
    Lockheed.analizadorprobabilidades(4000, x, y)

That script produced a half-spectrum (q >= 0 only), confidence intervals with
the wrong sign, and 10 000 simulated paths of a plain fractional Brownian
motion. This one does the same job and reports what it can and cannot support.

Run:  uv run python examples/lockheed_analysis.py [--figures]
"""

from __future__ import annotations

import argparse

import numpy as np

from mmar.analysis import analyze, compare_estimators, consensus, spectrum_table
from mmar.calibration import calibrate_from_spectrum
from mmar.estimators import critical_moment_order, oscillation_holder
from mmar.inference import moment_existence, multifractality_report
from mmar.io import load_lockheed
from mmar.tracing import Run

SEED = 20260727
Q = np.linspace(-5.0, 5.0, 41)


def main(show_figures: bool = False) -> None:
    series = load_lockheed()
    x = series.log_price
    print(f"{series.name}: n={x.size}, file digest {series.provenance.short_root}\n")

    # ------------------------------------------------------------------ #
    # 1. Do the moments we intend to fit actually exist?
    #    Above q_crit the log-log plot is still straight, but its slope is
    #    determined by the single largest increment rather than by the process.
    # ------------------------------------------------------------------ #
    diag = moment_existence(np.diff(x), q=Q)
    print(f"tail index of |returns|: {diag.tail_index:.2f} -> q_crit = {diag.q_critical:.2f}")
    print(f"hill-based crossover from structure functions: {critical_moment_order(x):.2f}")
    if diag.q_critical < float(Q.max()):
        print(
            f"  NOTE: the grid reaches q={Q.max():.0f} > q_crit; exponents beyond "
            f"{diag.q_critical:.2f} describe the sample maximum, not the process.\n"
        )

    # ------------------------------------------------------------------ #
    # 2. Four estimators, one series. Their disagreement is information.
    # ------------------------------------------------------------------ #
    with Run("lockheed", store="artifacts", notes=["examples/lockheed_analysis.py"]) as run:
        reports = compare_estimators(
            x,
            estimators=("wavelet_leaders", "mfdfa", "mfdma", "structure_function"),
            kind="level",
            q=Q,
            seed=SEED,
        )
        for name, rep in reports.items():
            run.record(rep.exponents, name=f"{name}:exponents")
            run.record(rep.spectrum, name=f"{name}:spectrum")

        print(spectrum_table(reports))
        print()
        print(consensus(reports))
        print()

        # -------------------------------------------------------------- #
        # 3. Is the width real, or is it the fat tails?
        #    Shuffling keeps the marginal exactly and destroys every
        #    correlation. IAAFT keeps the linear autocorrelation too. A width
        #    that survives both is the only kind worth reporting.
        # -------------------------------------------------------------- #
        surrogates = multifractality_report(
            x, estimator="wavelet_leaders", n_surrogates=200, seed=SEED
        )
        print(surrogates)
        print()

        # -------------------------------------------------------------- #
        # 4. Calibrate the MMAR, and let the calibration test the model.
        # -------------------------------------------------------------- #
        primary = reports["wavelet_leaders"]
        calibration = calibrate_from_spectrum(primary.exponents, primary.spectrum)
        print(calibration)
        print()

        # -------------------------------------------------------------- #
        # 5. Where in time is the roughness? This is what v1 could not do at
        #    all: an exponent per instant rather than their distribution.
        # -------------------------------------------------------------- #
        pointwise = oscillation_holder(x, seed=SEED)
        run.record(pointwise, name="pointwise")
        rough = pointwise.singular_times(threshold=float(np.nanpercentile(pointwise.h, 1)))
        dates = np.asarray(series.t)[rough] if series.t is not None else rough
        print(f"pointwise h(t): median {np.nanmedian(pointwise.h):.3f}, "
              f"1st percentile {np.nanpercentile(pointwise.h, 1):.3f}")
        print(f"  {rough.size} roughest instants; first few at index {rough[:5].tolist()}")
        if series.t is not None:
            print(f"  e.g. {np.asarray(dates)[:3]}")
        print()

    print(f"run manifest root {run.manifest_root}")
    print(f"artifacts verified: {all(run.verify().values())}")
    print("  mmar verify artifacts/run-lockheed.json")

    if show_figures:
        import matplotlib.pyplot as plt  # noqa: PLC0415 - optional extra

        from mmar.viz import multifractal_dashboard, plot_comparison  # noqa: PLC0415

        multifractal_dashboard(primary)
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_comparison({k: v.spectrum for k, v in reports.items()}, ax=ax)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="show the diagnostic figures")
    main(parser.parse_args().figures)
