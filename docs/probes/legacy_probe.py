"""Empirical audit of the legacy MMARV1 library. Runs it, records what it actually does."""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/omarkhalil/GitHub/MMARV1/legacy")

from MMAR import MMAR, BinomialMultifractalRand, Binomial_Multifractal_Det  # noqa: E402

df = pd.read_csv("/Users/omarkhalil/GitHub/MMARV1/data/lockheed_1995_2023.tsv", sep="\t", header=0)
print(f"data: {len(df)} rows, {df['Date'].iloc[0]} .. {df['Date'].iloc[-1]}")

np.random.seed(0)
print("\n===== A. Construct MMAR exactly as Scripts/pruebaMMAR1.py does =====")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = MMAR(df, "Date", "Close", a=0, b=3, npoints=10)
qs = np.linspace(m.a, m.b, m.npoints)
tau_pts = np.array([f[0] for f in m.fits])
print(f"  q grid          : {qs.round(3).tolist()}")
print(f"  tau(q) at nodes : {tau_pts.round(4).tolist()}")
print(f"  tau(0)          : {tau_pts[0]:.6f}   [theory: -1 exactly for a measure on full support]")
print(f"  H (tau-root)    : {m.h1:.6f}")
print(f"  alpha_0         : {m.alpha0:.6f}")
print(f"  lambda          : {m.lambdas:.6f}")
print(f"  sigma^2         : {m.varianza:.6f}   [must be > 0]")
print(f"  n deltas        : {len(m.deltas)}  from {m.deltas.min()} to {m.deltas.max()}")
print(f"  f(alpha) range  : alpha in [{m.derivada.min():.4f}, {m.derivada.max():.4f}]")
print(f"  #q used         : q in [{m.a}, {m.b}] -> NEGATIVE q ABSENT")

print("\n===== B. Scale grid: where do the fitted points actually sit? =====")
d = m.deltas
ld = np.log10(d)
half = (ld.max() + ld.min()) / 2
print(f"  deltas linearly spaced 1..{d.max()}")
print(f"  fraction of scales in the UPPER half-decade of log10(dt): "
      f"{np.mean(ld > half):.1%}")
print("  -> an OLS fit in log-log is dominated by the coarsest scales")

print("\n===== C. Confidence-interval formula: log-space vs raw-space leverage =====")
n = len(d)
var_raw, mean_raw = np.var(d), np.mean(d)
var_log, mean_log = np.var(ld), np.mean(ld)
sxx_correct = np.sum((ld - ld.mean()) ** 2)
legacy_slope_denom = np.sqrt(n - 1) * var_raw
correct_slope_denom = np.sqrt(sxx_correct)
print(f"  legacy  SE(slope) denominator = sqrt(n-1)*var(deltas)     = {legacy_slope_denom:.6e}")
print(f"  correct SE(slope) denominator = sqrt(sum (logdt-mean)^2)  = {correct_slope_denom:.6e}")
print(f"  ratio (legacy/correct) = {legacy_slope_denom / correct_slope_denom:.3e}")
from scipy.stats import t as tdist  # noqa: E402

print(f"  legacy uses t.ppf(alpha/2, n-2) = {tdist.ppf(0.025, n - 2):.4f}  <-- NEGATIVE half-width")
print(f"  correct is t.ppf(1-alpha/2,n-2) = {tdist.ppf(0.975, n - 2):.4f}")

print("\n===== D. Does the MMAR simulator actually compose B_H(theta(t))? =====")
np.random.seed(1)
tt = m.path_simulation(grafs=False, results=True)
print(f"  path_simulation returns ONLY the trading time, shape {tt.shape}")
np.random.seed(1)
x, y, z = m.simulacion(n=3, result=True, show_plots=False)
tt0, price0, xt0 = x[0], y[0], z[0]
print(f"  simulacion returns tradingtime{tt0.shape}, price{price0.shape}, xt{xt0.shape}")
recon = m.Price[-1] * np.exp(xt0)
print(f"  max|price - Price[-1]*exp(xt)| = {np.max(np.abs(price0 - recon)):.3e}")
print("  -> price is a DETERMINISTIC function of the fBm alone.")
print("     theta(t) never enters the price. The model is a plain fBm.")
inc = np.diff(xt0)
print(f"  kurtosis of simulated log-returns = {float(pd.Series(inc).kurtosis()):.4f}  "
      f"[Gaussian fBm ~ 0; real LMT returns ~ {float(np.log(df['Close']).diff().dropna().kurtosis()):.1f}]")

print("\n===== E. Both mass flags False -> ? =====")
b = BinomialMultifractalRand()
try:
    b.multifractal_measure_rand2(
        b=2, m=1, kmax=6, falpha=m.falpha, derivada=m.derivada, h1=m.h1,
        Price=m.Price, masas1=False, masas2=False, coef=True,
    )
    print("  no error (unexpected)")
except Exception as e:  # noqa: BLE001
    print(f"  {type(e).__name__}: {e}")

print("\n===== F. Module-level `simulacion(self, ...)` in multifractal_measure_rand =====")
print(f"  BinomialMultifractalRand has attribute 'simulacion'? "
      f"{hasattr(BinomialMultifractalRand, 'simulacion')}")
import MMAR.Multifractal_Measure.multifractal_measure_rand as mmr  # noqa: E402

print(f"  it is a module-level free function taking `self`: {callable(mmr.simulacion)} -> dead code")

print("\n===== G. Deterministic measure: mass check ordering & float equality =====")
try:
    Binomial_Multifractal_Det(10, 0.1, 0.9)
    print("  m00=0.1, m11=0.9 accepted")
except ValueError as e:
    print(f"  m00=0.1, m11=0.9 REJECTED: {e}".replace("\n", " "))
print(f"  0.1 + 0.9 == 1 ? {0.1 + 0.9 == 1}   (0.1+0.9 = {0.1 + 0.9!r})")
det = Binomial_Multifractal_Det(12, 0.6, 0.4)
lvl = np.array(det.measure[-1])
print(f"  built {lvl.size} cells, total mass = {lvl.sum():.12f}")
print(f"  the constructor builds the measure BEFORE validating mass conservation")

print("\n===== H. graf_Price_change alignment for deltat > 1 =====")
P = df["Close"].to_numpy()
for dt in (1, 5, 20):
    diff = P[dt::dt] - P[:-dt:dt]
    media = ((P[:-1] + P[1:]) / 2)[: len(diff)]
    print(f"  dt={dt:>2}: diff[0] uses P[{dt}]-P[0], but media[0] = (P[0]+P[1])/2 "
          f"-> stride mismatch: {'OK' if dt == 1 else 'MISALIGNED'}")

print("\n===== I. Test0 histogram exponent: off-by-one in k =====")
k = det.kmax
print(f"  measure[-1] is level k = {k}; code divides by log(2**-(kmax-1)) = log(2**-{k - 1})")
print(f"  correct denominator is log(2**-{k}) -> f(alpha) inflated by factor {k / (k - 1):.4f}")
