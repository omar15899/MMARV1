# Which estimator, and what it will get wrong

Every number below was **measured**, not quoted from a paper: each is produced
by the test suite against exact fBm or against a cascade with a closed-form
spectrum. Where an estimator is biased, the bias is stated with its sign, its
size, its cause, and what reduces it. Nothing here is tuned to flatter the
library.

The short version:

| You have | Use | Because |
|---|---|---|
| A financial or geophysical **series** | `wavelet_leaders` | the only family theoretically valid for `q<0`; gives `c₂` |
| A series with **trends** | `mfdfa` (order ≥ 2) | polynomial detrending is built in |
| A **measure** (cascade, trading time, mass) | `chhabra_jensen` | `α` and `f` directly, with error bars, no Legendre |
| An **isolated singularity** to locate | `wtmm`, then `pointwise` | maxima lines converge on the singular point |
| A quick, robust **`H`** | `generalised_hurst` | near-unbiased, cheap, and has baselines to disagree with |
| To know **where in time** the roughness is | `pointwise` | an exponent per instant, not a distribution |

And one rule that overrides all of it: **run `multifractality_report` before
believing any width you compute.**

---

## 1. The single most important number

`c₂`, the second log-cumulant, from `wavelet_leaders`. It is **exactly zero for
a monofractal** and is far better conditioned than the width of a numerically
Legendre-transformed curve. Measured detection floor on exact fBm, `n=2^14`:

| Estimator | Legendre width on a monofractal | `\|c₂\|` on the same data |
|---|---|---|
| `wavelet_leaders` | 0.028 | 0.005 |
| `mfdfa` | 0.043 | 0.001 |
| `mfdma` | 0.074 | 0.002 |
| `structure_function` | **0.842** | 0.014 |

The structure-function row is the lesson. On a *monofractal* it reports a width
of 0.84 — pure estimation noise amplified by the Legendre transform — while its
`c₂` correctly stays near zero. A width below roughly these values is
indistinguishable from a monofractal, whatever the curve looks like.

---

## 2. Per-estimator, measured

### `wavelet_leaders` — the recommended default
Jaffard 2004; Wendt, Abry & Jaffard 2007.

- **Bias**: `h(q)` low, worsening as `H` falls: −0.055 at `H=0.3`, −0.012 at
  `H=0.7`. Cause: count-weighted WLS is the correct inverse-variance weighting
  but puts ~¾ of the weight on the finest retained octave, where the leader is a
  supremum over rapidly decaying fine coefficients. **Remedy**: raise `j_min`.
- `|ζ(q)|` contracts at large `|q|` (0.33 at `|q|=5` vs 0.09 at `|q|=2` on a
  depth-18 binomial). Sample extremes are interior to population extremes; the
  bias shrinks like `1/log n`, **not** `1/√n`.
- **Deterministic** cascades are the worst case and are strongly
  wavelet-dependent: at `kmax=16, j_min=3`, max `|Δζ|` is 0.05 for `db4`, 1.43
  for `db6`. The db filter's phase correlates with the cascade's dyadic digit
  structure. A *randomised* cascade with identical `τ` gives 0.13–0.15.
- Octaves `j=1,2` are unusable and **contaminate every coarser octave** when
  included, because the leader recursion propagates suprema upward: including
  `j=1` blows the `h(q)` spread from 0.021 to 0.756. Hence `j_min=3`.
- Requires `N_ψ` vanishing moments exceeding the largest Hölder exponent
  present, or the exponents saturate at `N_ψ`.

### `mfdfa` — when there are trends
Kantelhardt et al. 2002.

- **The integration trap.** MF-DFA builds a profile. Handing it an
  already-integrated series (a log-price, an fBm path) with `integrate=True`
  returns `h = H + 1`. `analyze(..., kind="level")` handles this; calling the
  estimator directly does not. This is the commonest MF-DFA error in the wild.
- Small-scale detrending bias: DFA-*m* removes `m+1` degrees of freedom per
  window, deflating `F²` by `~(1 − (m+1)/s)`, which dies as `1/s`, tilts the
  log–log plot and **overestimates** `h`. Default `s_min = 10(m+1)`.
- `q<0` overestimates `h`: +0.016/+0.010/+0.010 at `H=0.3/0.5/0.7` — a spurious
  right branch of `f(α)` even for a monofractal.
- Saturation: DFA-*m* cannot resolve `h > m+1`.

### `mfdma` — the independent cross-check
Gu & Zhou 2010.

- Backward (`theta=0`, default) carries a **level offset** of +0.030/+0.036/
  +0.041 at `H=0.3/0.5/0.7`, essentially independent of `q`.
- That offset is a **small-scale transient, not a finite-size effect**: on the
  cascade `τ̂(1)` is +0.0340/+0.0348/+0.0364/+0.0374 at `kmax=12/14/16/18` —
  flat in `n`. Raising the scale floor is what fixes it: `τ̂(1)` = +0.0348
  (`s_min=8`), +0.0139 (16), +0.0034 (32).
- **`theta ≠ ½` does not remove a linear trend.** For `Y = a + bi` the residual
  is a constant proportional to `n`, so a drift injects an `n¹` term and pulls
  `h` toward 1. DMA is a phase-shifted high-pass filter, not a polynomial
  detrender. Use MF-DFA if there are trends.

### `structure_function` — the classical MFC method
Mandelbrot, Fisher & Calvet 1997.

- **Negative `q` on a signal is not an estimate.** For Gaussian increments
  `E|Z|^q = ∞` for `q ≤ −1`, so `S(q,δ)` is a function of the sample *minimum*.
  Measured on fBm `H=0.7`: `h(−5) = 1.43` against a true 0.7, and the `h(q)`
  spread is 0.63 where the truth is 0. On a **measure** (a monotone trading time)
  the negative branch is well posed — which is why the recovery tests use it.
- `h(2)` biased low, growing with `H`: −0.015/−0.018/−0.024 at `H=0.3/0.5/0.7`.
  Cause: `log` of a sample mean is downward-biased (Jensen) by `O(1/N(δ))`,
  worst at coarse scales. Capping `δ` at 256 instead of `n/4` halves it.
- **Use dyadic scales on a dyadic cascade.** Non-dyadic lags straddle branches
  of the tree and inject log-periodic oscillations: max `|τ̂ − τ|` is 0.157 on a
  geometric grid versus **1.3e-8** on a dyadic one. Seven orders of magnitude.

### `wtmm` — for isolated singularities
Muzy, Bacry & Arneodo 1993.

- `h(2)` biased **up**, worsening as the path gets rougher: +0.063/+0.032/+0.012
  at `H=0.3/0.5/0.7`. WTMM over-smooths; maxima are densest and chaining most
  ambiguous at small `H`.
- Negative `q` on a monofractal **diverges by construction**, predictably:
  writing line suprema as `s_l = C_l a^H`, the density of `C` is positive at 0,
  so `E[C^q] = ∞` for `q ≤ −1` and `Z(q,a)` collapses onto its smallest term,
  giving `ĥ(q) ~ H + 1 + 1/q`. Observed `ĥ(−5) = 1.34` at `H=0.5`; predicted
  1.30.
- Wavelet choice **dominates** accuracy on measures: `n_vanishing=1` analyses
  smoothed mass; the Mexican hat (`=2`) analyses a smoothed left–right mass
  *difference*, with cancellations.
- Coarse-scale line starvation is the dominant large-`|q|` bias, hence the
  coarsest scale defaults to `n/16` rather than the library-wide `n/4`.

### `chhabra_jensen` — for measures, with error bars
Chhabra & Jensen 1989.

- **It consumes a measure, not a signal.** Applied to a series via
  `μ_i = |ΔX_i|/Σ|ΔX|`, the box mass is a sum of `s` absolute increments, so
  `μ_i(s) = s·E|ΔX|(1 + O(s^{H−1}))`: the measure is asymptotically **uniform**
  and recovers `α ≡ 1, f(1) = 1` for *every* `H`. That is not a bug; it is what
  the total-variation measure of a monofractal is.
- Spurious width at fine scales on fBm: the recovered `α` extent is 1.539 at
  `s_min=2`, collapsing monotonically to 0.012 by `s_min=256`.
- Empty boxes are excluded and counted in provenance. Exclusion is **not
  neutral**: it removes exactly the cells the negative-`q` tilt would weight
  most, so `α(q)` for `q ≪ 0` is biased downward.
- Finite `|q|` truncates: on `q ∈ [−5,5]` only **76.7 %** of the binomial support
  is reached and the endpoints sit at `f = 0.519`, not 0. Reaching `f ≈ 1e-4`
  needs `|q| ≈ 30`.

### `generalised_hurst` and the baselines
Di Matteo 2005.

Measured on exact fBm, mean bias in `H`:

| Estimator | `H=0.2` | `H=0.5` | `H=0.95` |
|---|---|---|---|
| GHE | −0.010 | ≈0 | ≈0 (sd 0.023) |
| R/S uncorrected | **+0.150** | **+0.074** | — |
| R/S Anis–Lloyd corrected | +0.079 | −0.013 | −0.182 |
| Aggregated variance | — | — | −0.075 (sd 0.029) |

R/S corrected is the right statistic for **testing** `H = ½` and the wrong one
for **measuring** an `H` already far from ½ — it over-corrects away from the
null. GHE is essentially unbiased but its variance grows with `H`, because the
default `τ ≤ 19` window spans only 0.98 decades.

### `pointwise` — `h(t)`, not its distribution

Measured on an isolated cusp `|t − t₀|^a`, error in the recovered `a`:

| Method | `a=0.3` | `a=0.5` | `a=0.7` | Note |
|---|---|---|---|---|
| `oscillation_holder` | +0.115 | +0.074 | +0.049 | discrete-range deficit; ceiling at `h=1` |
| `wavelet_holder` | −0.040 | −0.016 | −0.007 | −0.008 at `n=2^16`; best of the three |
| `leader_holder` | −0.185 | −0.150 | −0.165 | structural, see below |

`leader_holder`'s large negative bias is **not** a coding defect: `log₂ L(j,·)`
is *convex* in `j` because the supremum needs several octaves to escape the
discretisation floor, and a straight line through a convex curve undershoots.
Raising `j_min` to 5 halves it, to 6 more than halves it again.

`oscillation_holder` annihilates only constants, so any `C¹` signal returns 1
(measured 1.038). Exponents above 1 are not measurable that way.

---

## 3. The rules that follow from all of this

1. **Check moment existence before widening `q`.** Beyond `q_crit` the log–log
   plot stays straight and its slope is set by the single largest increment
   (Lashermes, Abry & Chainais 2004). `moment_existence` and
   `critical_moment_order` are there for this.
2. **Match the scale grid to the object.** Dyadic scales for a dyadic cascade —
   seven orders of magnitude of accuracy hang on it.
3. **Negative `q` on a Gaussian signal is a diagnostic, not a measurement.** Use
   wavelet leaders, or work on the induced measure.
4. **Report `c₂` alongside any width.** They disagree exactly when the width is
   noise.
5. **Run more than one estimator.** `consensus()` exists because they fail
   differently; agreement is evidence and disagreement is a warning.
6. **Then run the surrogates.** On the Lockheed series this library was built
   around, four estimators produce a spectrum of width 0.13–1.10 — and the
   shuffle and IAAFT tests both return `p ≈ 0.56`. There is no evidence of
   multifractality in it beyond what its marginal distribution already explains.
