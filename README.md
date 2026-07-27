# mmar

A multifractal formalism engine: **Hölder spectra**, cascade models, and the
Mandelbrot–Fisher–Calvet Multifractal Model of Asset Returns — with
content-addressed provenance on every number it produces.

```python
import numpy as np
from mmar.io import load_lockheed
from mmar.estimators import wavelet_leaders
from mmar.core import spectrum_from_exponents

series = load_lockheed()
exps = wavelet_leaders(series.log_price, q=np.linspace(-5, 5, 41))
spec = spectrum_from_exponents(exps)

spec.descriptors()      # alpha_0, width, asymmetry, f_max
exps.provenance.root    # sha256 content address of this exact computation
```

---

## What this is

Estimating a Hölder spectrum is easy to do and hard to do correctly. Every
estimator returns a smooth, plausible `f(α)` curve for any input whatsoever —
including white noise, a trend, and a series with fat tails and no temporal
structure at all. The failure mode is not a crash; it is a publishable figure.

This library is built around that problem. Three commitments:

**One convention, enforced by the types.** `ScalingExponents` stores `ζ(q)` and
derives `τ = ζ − d` and `h(q) = ζ(q)/q`. A dropped `±1` between the measure and
signal conventions is the most common error in this field, and it cannot happen
here because there is only one carrier.

**Nine estimators, one interface.** They disagree, and their disagreement is
information. `plot_comparison` puts them on one axis.

**Every number is attributable.** Each result carries a `Provenance` whose
`root` is a SHA-256 over the input digests, every parameter, the RNG seed and
the environment — never the wall clock, so reproducibility is *testable*.
`Run` records a whole session as a Merkle DAG; `diff_manifests` tells you which
parameter changed when two analyses disagree.

---

## Install

```sh
git clone https://github.com/TheVaccinator/MMARV1 && cd MMARV1
uv sync --all-extras     # or: pip install -e ".[full]"
```

Python ≥ 3.12. `numpy` and `scipy` are required; `PyWavelets`, `pandas` and
`matplotlib` are extras (`wavelets`, `data`, `viz`, or `full`).

---

## Estimators

| Estimator | Function | Notes |
|---|---|---|
| Wavelet leaders | `wavelet_leaders` | **recommended default.** Theoretically valid for `q<0`; yields log-cumulants `c₁,c₂,c₃` |
| MF-DFA | `mfdfa` | Kantelhardt 2002. Robust to polynomial trends |
| MF-DMA | `mfdma` | Gu & Zhou 2010. Independent cross-check on MF-DFA |
| WTMM | `wtmm` | Muzy–Bacry–Arneodo. Best resolution on isolated singularities |
| Structure functions | `mfc_exponents` | The classical MFC method of moments |
| Chhabra–Jensen | `chhabra_jensen` | `α` and `f` **directly**, no Legendre transform |
| Box counting | `generalised_dimensions`, `histogram_spectrum` | Coarse Hölder exponents, `D_q` |
| Generalised Hurst | `generalised_hurst` | Di Matteo 2005, plus R/S, DFA and GPH baselines |
| **Pointwise** | `oscillation_holder`, `wavelet_holder`, `leader_holder` | `h(t)` per sample — the *field*, not the distribution |

The first eight estimate the **distribution** of Hölder exponents. The ninth
estimates the exponent **at each instant**, so singular events can be located
rather than merely counted.

---

## Is it actually multifractal?

Usually the honest answer is no, and the library is built to find that out.

```python
from mmar.inference import multifractality_report

multifractality_report(x, estimator=wavelet_leaders, seed=0)
```

A wide spectrum has three possible causes, and an estimator cannot tell them
apart:

1. genuine multifractal correlation structure,
2. a **fat-tailed marginal** with no temporal structure at all,
3. non-stationarity.

Surrogates can. Shuffling destroys all temporal correlation while preserving
the marginal exactly: if the width survives, the cause is (2). IAAFT preserves
the linear autocorrelation while destroying nonlinear structure: if the width
survives that too, (1) is not established either.

The `inference` package also provides moment-existence diagnostics (fitting
`q > q_crit` gives a straight log–log line determined by the single largest
increment), scaling-range linearity tests, concavity tests, and block-bootstrap
confidence intervals.

`c₂` from the wavelet leaders is the most reliable single multifractality
statistic here: exactly zero for a monofractal, and far better conditioned than
the width of a numerically Legendre-transformed curve.

### What that gives on the library's own dataset

`examples/lockheed_analysis.py`, on the Lockheed Martin series the original
thesis used:

```
estimator                h(2)   alpha_0    width    asym       c2   defect
mfdfa                  0.5104    0.5216   0.1697    3.00  -0.0103   0.0000
mfdma                  0.5479    0.6049   0.4822    1.70  -0.0496   0.0000
structure_function     0.5254    0.5828   1.1002    0.38  -0.1263   0.0002
wavelet_leaders        0.4572    0.4831   0.1931    2.39  -0.0193   0.0000

consensus: estimators DISAGREE: alpha_0 spread 0.122 > 0.1
observed width = 0.1931   vs shuffle p = 0.567   vs iaaft p = 0.557
verdict: NO EVIDENCE -- a series with the same marginal distribution and no
time ordering at all already reproduces the statistic. Heavy tails, not
multifractality.
```

Also flagged, before any of that: 26 of the 41 requested moments lie beyond
`q_crit = 2.62`, so their exponents describe the sample extremes rather than
the process. v1 reported a spectrum for this series and ran no such check.

---

## Models

```python
from mmar.models import MMARSpec, simulate_mmar

path = simulate_mmar(MMARSpec(kmax=14, H=0.55, lam=1.25), seed=7)
```

`X(t) = B_H(θ(t))`, actually composed. Also: exact fBm/fGn by circulant
embedding, deterministic and canonical random cascades (lognormal, log-Poisson,
log-gamma, log-stable), and closed-form `τ(q)` and `f(α)` for each in
`models.theory` — the oracle every estimator is tested against, exact to machine
precision on `τ(0) = −1`, `τ(1) = 0`, `f(α₀) = 1` and `τ_X(1/H) = 0`.

---

## Traceability

```python
from mmar.tracing import Run, replay_plan

with Run("lockheed", store="artifacts") as run:
    exps = run.record(wavelet_leaders(x, q=q), name="exponents")
    spec = run.record(spectrum_from_exponents(exps), name="spectrum")

run.manifest_root          # one hash covering the whole session
run.verify()               # re-hash stored arrays against recorded digests
print(replay_plan(run.manifest))
```

Re-running the same pipeline on the same input reproduces the same session
root. Editing a stored manifest by hand makes `load_manifest` refuse it.
`diff_manifests(a, b)` names the parameter that changed.

---

## Relationship to v1

This is a ground-up rebuild. The previous package is preserved verbatim and
importable under `legacy/`, so the thesis results remain auditable.

`docs/LEGACY_AUDIT.md` records what the audit found, with numbers observed by
running v1 against its own reference script. The headline: **the v1 MMAR
simulator generated a multifractal trading time and then discarded it**, so the
simulated price was a plain fractional Brownian motion — kurtosis 0.06 against
8.8 for the Lockheed returns it was calibrated on. Its default moment grid
`q ∈ [0,3]` also made the right branch of `f(α)` unreachable in principle,
recovering 24 % of the true support on the binomial `m₀ = 0.6`.

The audit also records two hypotheses that did **not** survive testing, so they
are not repeated as folklore.

Most of v1's mathematics was right, and is kept. `H = 1/q*` where `τ(q*) = 0` is
exact MMAR theory, `σ² = 2(λ−1)/ln 2` is the correct base-2 conservation
constraint, and `λ = α₀/H` is the correct calibration identity. Each is now a
test against a closed-form oracle rather than a comment saying "use with
caution".

`docs/BRANCHES.md` reconstructs the branch topology and identifies the commit
that removed the dataset, the notebook and the bibliography from `main`.

---

## Repository layout

```
src/mmar/
  core/          conventions, provenance, regression, Legendre transforms
  estimators/    the nine Hölder-spectrum estimators
  inference/     surrogates, bootstrap, hypothesis tests, log-cumulants
  models/        theory oracle, exact fBm, cascades, MMAR
  viz/           spectra and diagnostics
  io/            loaders with provenance from the first byte
  tracing.py     session-level Merkle DAG
tests/
  theory/        estimators against closed-form results
  unit/          module behaviour
docs/            audit, branch analysis, bibliography
legacy/          v1, verbatim and importable
data/            Lockheed Martin daily, 1995-03-17 to 2023-03-17
```

## References

The bibliography assembled for the original thesis is preserved at
`docs/BIBLIOGRAPHY_legacy.md`. The results implemented here come principally
from Mandelbrot, Fisher & Calvet (1997); Calvet & Fisher (2002); Evertsz &
Mandelbrot (1992); Kantelhardt et al. (2002); Muzy, Bacry & Arneodo (1993);
Jaffard (2004); and Wendt, Abry & Jaffard (2007). Each estimator's docstring
cites its own source.

## Licence

MIT.
