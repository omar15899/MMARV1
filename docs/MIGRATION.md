# Migrating from v1

v1 is preserved verbatim under `legacy/` and still imports, so nothing here is
urgent and no thesis result becomes unreproducible. This is the map from the old
API to the new one, and — more importantly — a note against each entry saying
whether the numbers will change and why.

```python
import sys; sys.path.insert(0, "legacy")
from MMAR import MMAR, Binomial_Multifractal_Det   # still works
```

---

## The one-line translation

```python
# v1
Lockheed = MMAR(df, "Date", "Close", a=0, b=3, npoints=10)
Lockheed.partition_functions()
Lockheed.compute_tau_and_alpha()

# v2
from mmar.io import load_csv
from mmar.analysis import analyze
report = analyze(load_csv("Lockheed.txt", value_column="Close").log_price,
                 q=np.linspace(-5, 5, 41))
```

The v2 call differs in three ways that matter, all deliberate:

- **`q` spans negative values.** v1's `a=0, b=3` could not reach the right
  branch of `f(α)`; see audit D2.
- **Nothing is computed in a constructor.** `analyze` returns a report; the
  object holding your data does not silently run an analysis.
- **The result carries a provenance root.** `report.provenance.root`.

---

## Class and function map

| v1 | v2 | Numbers change? |
|---|---|---|
| `Graphs(df, time, price)` | `mmar.io.load_csv(...) -> Series` | no; `Series.log_price` is the same transform |
| `Graphs.grafPrice()` | `mmar.viz.plot_*` | plotting is no longer a base class of the estimator |
| `Graphs.graf_Price_change(deltat)` | `Series.increments(delta)` | **yes** for `deltat > 1` — v1 divided a `Δt`-step difference by a 1-step midpoint (D9) |
| `MultifractalCharacteristics.partition_functions()` | `estimators.structure_functions` / `partition_function` | **yes** — geometric scales, count weighting, exact increment counts (D5, D6) |
| `.compute_tau_and_alpha()` | `analyze(...)` → `.exponents`, `.spectrum` | **yes** — Legendre–Fenchel by default, negative `q` reachable |
| `.hurst_exponent()` | `estimators.hurst_from_tau_root(exponents)` | **yes** — bracketed `brentq` on the positive-`q` branch, not `UnivariateSpline.roots()` |
| `.multifractal_measure_rand(...)` | `models.cascade.lognormal_cascade(...)` | **yes** — vectorised, seeded, single-Ω normalisation (§5 of the audit) |
| `MMAR.path_simulation()` | `models.simulate_mmar(spec, seed=...)` | **yes, completely** — v1 never composed `B_H(θ(t))` (D1) |
| `MMAR.simulacion(n=...)` | `models.simulate_ensemble(spec, n, seed=...)` | as above, plus reproducible independent streams |
| `MMAR.analizadorprobabilidades(day, ...)` | `models.mmar_forecast_distribution(...)` | as above |
| `Binomial_Multifractal_Det(kmax, m0, m1)` | `models.cascade.deterministic_cascade(kmax, [m0, m1])` | no — the measure is identical |
| `.f1(alpha)` | `models.theory.cascade_falpha(alpha, [m0, m1])` | no — same entropy formula, now continuous at the endpoints |
| `.Test0()` (histogram) | `estimators.histogram_spectrum` | **yes** — v1 normalised by `2^-(kmax-1)` instead of `2^-kmax`, inflating `f(α)` by `k/(k-1)` (D8) |
| `.Test1()`, `.Test2()` (moments) | `estimators.generalised_dimensions`, `chhabra_jensen` | **yes** — v1 fitted `τ` from a two-point difference between the two coarsest levels (D7) |
| `.Test3()` (convergence) | `inference.bootstrap` | v1's convergence study measured the histogram method's own bias; bootstrap measures sampling error |
| *(nothing)* | `inference.multifractality_report` | new: the surrogate tests that decide whether a width means anything |
| *(nothing)* | `estimators.oscillation_holder` etc. | new: pointwise `h(t)` |
| *(nothing)* | `mmar.tracing.Run` | new: session-level traceability |
| the README's CLI | `mmar analyze / calibrate / simulate / verify / replay` | v1's CLI never existed (D14) |

---

## Numbers that will move, and by how much

Re-running an old analysis in v2 will not reproduce the old figures. That is the
point, but the differences are predictable:

**`f(α)` gains a right branch.** With `q ∈ [0,3]` v1 recovered `α ∈ [0.878,
1.020]` on the binomial `m₀=0.6` whose true support is `[0.737, 1.322]`. v2 at
`q ∈ [-5,5]` recovers essentially all of it.

**`Δα` is measured differently.** v1 reported the extent of the sampled `α`
grid; v2 reports the distance between the crossings `f(α)=0`. The grid extent
grows with the `q` window and is not comparable across analyses. Where the curve
never reaches zero, `descriptors().support_truncated` is set and the width is a
lower bound.

**Confidence intervals change by ~5 orders of magnitude.** v1's slope standard
error used `var(Δt)` where the algebra needs `Σ(log Δt − mean)²`; the
denominators differ by a factor of 1.9 × 10⁵ on the default grid (D3). The
interval also had the wrong sign (D4).

**`τ(0)` becomes exactly −1.** v1 read −1.017 on Lockheed because
`X[dt::dt] - X[:-dt:dt]` truncates when `dt ∤ N` (D6). v2 counts increments
exactly and tests the identity.

**The simulated MMAR stops being a Gaussian process.** v1: kurtosis 0.064.
v2 at `kmax=14, H=0.55, λ=1.25`: kurtosis ≈ 115, lag-1 autocorrelation of
`|ΔX|` ≈ 0.46 against ≈ −0.01 for the uncomposed fBm.

**MF-DFA on a log-price no longer double-integrates.** `analyze(..., kind=...)`
canonicalises the representation before dispatch. Passing a log-price as
`kind="level"` gives `h(2) ≈ H`; the same series treated as increments gives
`H + 1`.

---

## Reproducing a v1 result deliberately

Sometimes you want the old number, to check a figure in the thesis. The
components are all still reachable:

```python
from mmar.estimators import partition_function, mfc_exponents

# v1's summed partition function over its linear scale grid
S = partition_function(x, q=np.linspace(0, 3, 10), scales=np.arange(1, 1000))

# v1's unweighted OLS fit
exps = mfc_exponents(x, q=np.linspace(0, 3, 10), scales=np.arange(1, 1000),
                     fit_method=FitMethod.OLS)
```

This reproduces v1's estimator up to the increment-count fix. It is not
recommended for new work, and the docstrings say why at each step.

---

## Things v1 did that v2 refuses to do

- **Return a different type depending on a boolean.**
  `multifractal_measure_rand2(coef=…, cumsum=…, graf1=…)` returned a 3-tuple, an
  array or `None`. v2 functions return one type.
- **Compute in a constructor.** `MultifractalCharacteristics.__init__` ran the
  whole analysis before returning.
- **Use the global RNG.** Every stochastic routine now takes a seed.
- **Plot from an estimator.** No v2 function calls `plt.show()`; the viz
  functions return an `Axes` and the caller decides.
- **Report a fit without its scaling range.** Every `LogLogFit` carries the
  window it was fitted over and how that window was chosen.
