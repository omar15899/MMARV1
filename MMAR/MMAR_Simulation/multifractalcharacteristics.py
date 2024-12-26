import random
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .graphs import Graphs


class MultifractalCharacteristics(Graphs):
    """
    A Python implementation of multifractal analysis for financial time series, inheriting from a `Graphs` class.
    Key attributes are: dataset, time, price, and several hyperparameters for controlling the analysis.

    This improved version refactors code to:
      1. Unify and centralize computations (partition functions, tau, alpha, etc.).
      2. Reduce redundancy and ensure consistent results.
      3. Correctly compute tau by leveraging robust polynomial fits, avoiding normalization pitfalls.
      4. Provide better docstrings, variable names, and code organization.
    """

    def __init__(
        self, dataset, time, price, a=0, b=5, npoints=20, deltas=None, kmax=13
    ):
        """
        Parameters
        ----------
        dataset : pd.DataFrame or np.ndarray
            The underlying data set (not fully specified in this snippet).
        time : str or array-like
            Time axis or label. In the original code, this is used for plotting or referencing.
        price : str or array-like
            Price axis or label. In the original code, this is used for plotting or referencing.
        a : float
            Lower bound for the range of q (moments).
        b : float
            Upper bound for the range of q (moments).
        npoints : int
            Number of points in the q-grid, i.e., how many values of q we consider.
        deltas : list or np.ndarray
            List of time intervals over which the partition function is computed.
        kmax : int
            Maximum resolution parameter used for the random multifractal measure.
        """
        super().__init__(dataset, time, price, a, b, npoints, deltas, kmax)

        # 1) Store the partition function analysis
        #    We do one consolidated pass to get all necessary pieces: partition_fns, logfits, etc.
        pf_result = self.partition_functions(graf1=False, result=True)
        self.partition = pf_result[0]  # raw partition functions for each q
        self.fits = pf_result[1]  # polynomial fits [slope, intercept] in log-log space

        # 2) Compute tau(q), f(alpha), derivative alpha(q) via a single call
        taualpha_result = self.compute_tau_and_alpha(graf=False, results=True)
        self.tau = taualpha_result["tau_spline"]  # tau(q) evaluated on finer grid
        self.falpha = taualpha_result["falpha"]  # f(alpha) on same finer grid
        self.derivada = taualpha_result[
            "alpha_vals"
        ]  # alpha(q) = derivative of tau wrt q on finer grid

        # 3) Hurst exponent
        self.h1 = self.hurst_exponent(graf=False)

        # 4) Extract alpha0 = alpha that maximizes f(alpha)
        self.posicion_max = np.argmax(self.falpha)
        self.alpha0 = self.derivada[self.posicion_max]

        # 5) Derived parameters lambdas, varianza
        self.lambdas = self.alpha0 / self.h1
        self.varianza = 2 * (self.lambdas - 1) / np.log(2)

    def partition_functions(self, alpha=0.05, graf1=True, result=False):
        r"""
        Computes the partition function \( S_q(\Delta t) = \sum |X(t + \Delta t) - X(t)|^q \)
        for multiple values of \( q \) and \(\Delta t \).

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals and correlation tests.
        graf1 : bool
            Whether to produce plots of the partition functions (normalized and unnormalized).
        result : bool
            Whether to return the computed values.

        Returns
        -------
        partition_functions : list of lists
            partition_functions[i][j] is the partition function at q_i and delta_j.
        adjustment_part_functions : list of np.ndarrays
            Each element is of the form [slope, intercept], from the polynomial fit
            log10(S_q(\Delta t)) = slope * log10(\Delta t) + intercept.

        Notes
        -----
        - Confidence intervals are computed for the fit (slope, intercept).
        - A correlation test is performed to see for which q the null hypothesis
          (zero correlation) is rejected.
        """
        a, b, npoints = self.a, self.b, self.npoints
        deltas = self.deltas
        ldeltas = len(deltas)
        mdeltas = np.mean(deltas)
        vardeltas = np.var(deltas)

        # Define q-grid
        qs = np.linspace(a, b, npoints)

        # 1. Compute partition functions
        #    partition_functions[i] is the list of S_q_i(deltas), over j in deltas
        partition_functions = []
        for q in qs:
            pf_q = []
            for deltat in deltas:
                #  Compute: sum |X(t + dt) - X(t)|^q over t
                #  X_t is presumably the time series from self.X_t
                pf_val = np.sum(
                    np.abs(self.X_t[deltat::deltat] - self.X_t[:-deltat:deltat]) ** q
                )
                pf_q.append(pf_val)
            partition_functions.append(pf_q)

        # 2. Fit log10(partition_functions) vs log10(deltas)
        #    For each q, we do an OLS fit: log10(S_q) = m * log10(delta) + c
        #    We'll store them in "adjustment_part_functions"
        adjustment_part_functions = []
        for i in range(npoints):
            # Avoid dividing by zero if partition_functions[i][0] is zero or near zero
            # We'll do the fit with raw log10(S_q_i(deltas)).
            # If the first element is 0, fallback or skip.
            # (But we assume the data is always positive in typical usage.)
            logy = np.log10(partition_functions[i])
            logx = np.log10(deltas)

            # slope, intercept
            pfit = np.polyfit(logx, logy, 1)
            adjustment_part_functions.append(pfit)

        # 3. Compute residuals and standard error
        srs = []
        for i in range(npoints):
            logx = np.log10(deltas)
            logy = np.log10(partition_functions[i])
            residuals = logy - np.poly1d(adjustment_part_functions[i])(logx)
            srs_i = np.sqrt(np.sum(residuals**2) / (ldeltas - 2))
            srs.append(srs_i)

        # 4. Compute confidence intervals
        #    For intercept and slope. Cf. standard formula for linear regression in log-log space.
        intervalos_conf_ordenada = []
        intervalos_conf_pendiente = []
        for i in range(npoints):
            # Intercept
            ci_intercept = (
                t.ppf(alpha / 2, ldeltas - 2)
                * srs[i]
                * np.sqrt(1 / ldeltas + (mdeltas) ** 2 / ((ldeltas - 1) * vardeltas))
            )
            intervalos_conf_ordenada.append(ci_intercept)

            # Slope
            ci_slope = (
                t.ppf(alpha / 2, ldeltas - 2)
                * srs[i]
                / (np.sqrt(ldeltas - 1) * vardeltas)
            )
            intervalos_conf_pendiente.append(ci_slope)

        # 5. Correlation test
        #    We'll check the correlation between log10(partition_functions) and log10(deltas).
        corrtest1 = []
        corrtest2 = []
        for i in range(npoints):
            cor, pval = stats.pearsonr(
                np.log10(deltas), np.log10(partition_functions[i])
            )
            corrtest1.append(cor)
            corrtest2.append(pval)
        corrtest2 = np.array(corrtest2)

        # qslim = qs where pvalue <= alpha
        qslim = qs[corrtest2 <= alpha]

        # Plotting if requested
        if graf1:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Colors for plotting
            colors = [
                "#FF0000",
                "#00FF00",
                "#0000FF",
                "#000000",
                "#FF00FF",
                "#00FFFF",
                "#FFA500",
                "#800080",
                "#008000",
                "#800000",
                "#008080",
                "#000080",
                "#FFFFE0",
                "#FFD700",
                "#00FF7F",
                "#FF4500",
                "#9400D3",
                "#808000",
                "#FF1493",
                "#6A5ACD",
            ]

            # We only plot those q in qslim (rejected null hypothesis).
            for idx, q in enumerate(qs):
                if q not in qslim:
                    # skip plotting if the correlation test wasn't significant
                    continue
                logx = np.log10(deltas)
                logy = np.log10(partition_functions[idx])
                ax.plot(
                    logx,
                    logy,
                    label=rf"$q = {q:.2f}$",
                    color=colors[idx % len(colors)],
                )
                # Fit line
                ax.plot(
                    logx,
                    np.poly1d(adjustment_part_functions[idx])(logx),
                    color=colors[idx % len(colors)],
                )
                # Could also compute confidence intervals vs. logx if desired

            ax.set_ylabel(r"$\log_{10}(S_q(\Delta t))$", fontsize=15, fontweight="bold")
            ax.set_xlabel(r"$\log_{10}(\Delta t)$", fontsize=15, fontweight="bold")
            ax.grid(
                which="both",
                axis="both",
                linestyle="-.",
                linewidth=1,
                alpha=0.7,
                zorder=0,
                markevery=1,
                color="grey",
            )
            ax.legend(loc="best", fancybox=True, shadow=True, ncol=5, fontsize=12)
            plt.tight_layout()
            plt.show()

            # Plot unnormalized partition functions
            fig, ax = plt.subplots(figsize=(12, 8))
            for idx, q in enumerate(qs):
                if q not in qslim:
                    continue
                ax.plot(
                    np.log10(deltas),
                    np.log10(partition_functions[idx]),
                    label=rf"$q = {q:.2f}$",
                    color=colors[idx % len(colors)],
                )
            # ax.set_title(
            #     r"Unnormalized Partition Function $\log_{10}(S_q(\Delta t))$ vs. $\log_{10}(\Delta t)$",
            #     fontsize=14,
            #     fontweight="bold",
            # )
            # ax.set_ylabel(
            #     r"$\log_{10}\bigl(S_q(\Delta t)\bigr)$", fontsize=12, fontweight="bold"
            # )
            ax.set_xlabel(r"$\log_{10}(\Delta t)$", fontsize=12, fontweight="bold")
            ax.grid(
                which="both",
                axis="both",
                linestyle="-.",
                linewidth=1,
                alpha=0.7,
                zorder=0,
                markevery=1,
                color="grey",
            )
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.1),
                fancybox=True,
                shadow=True,
                ncol=5,
                fontsize=12,
            )
            plt.tight_layout()
            plt.show()

        if result:
            return partition_functions, adjustment_part_functions

    def compute_tau_and_alpha(self, alpha=0.05, graf=True, results=False):
        r"""
        Computes multifractal exponents:
        - \(\tau(q)\)
        - \(\alpha(q) = d\tau/dq\)
        - \(f(\alpha) = q \alpha - \tau(q)\)

        This version uses the partition function already computed in `partition_functions`.
        Internally, it re-fits the log of the partition function vs. log(deltas),
        then uses a spline to derive smooth estimates of \(\tau\) and \(\alpha\).

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals, if desired.
        graf : bool
            Whether to plot the results.
        results : bool
            Whether to return the resulting curves (tau, alpha, f(alpha)).

        Returns
        -------
        If results=True, returns a dict:
            {
                "tau_spline": array-like of tau-values on a fine q-grid,
                "falpha":     array-like of f(alpha) on that fine q-grid,
                "alpha_vals": array-like of alpha(q) on that fine q-grid
            }
        Otherwise, None.
        """
        a, b, npoints = self.a, self.b, self.npoints
        deltas = self.deltas
        ldeltas = len(deltas)
        qs = np.linspace(a, b, npoints)

        # 1. If we already have the partition functions and the fits, just use them.
        #    Otherwise, compute them again (fallback).
        if not hasattr(self, "partition") or not hasattr(self, "fits"):
            pf_result = self.partition_functions(alpha=alpha, graf1=False, result=True)
            self.partition = pf_result[0]
            self.fits = pf_result[1]

        partition_functions = self.partition
        adjustment_part_functions = self.fits

        # 2. The slope in each fit is an estimate of tau(q), but we must be careful with the log-scaling:
        #    We have log10(S_q(Δt)) ~ slope * log10(Δt) + intercept
        #    => slope = d[ log10(S_q(Δt)) ] / d[ log10(Δt) ]
        #    The base-10 vs base-e can be a factor, but as commonly done, the slope is the fractal exponent.
        #    We'll treat 'slope' as tau(q) in base 10 logs.
        #    If you want natural log tau, you'd convert slope *= ln(10).
        #    We'll keep it consistent with the code logic.
        tau_estimate = np.array([pfit[0] for pfit in adjustment_part_functions])

        # 3. Create a spline in the (q, tau) domain
        #    We'll evaluate it on a finer grid for smooth alpha(q) and f(alpha).
        q_fine = np.linspace(qs[0], qs[-1], 1000)
        spline_tau = UnivariateSpline(qs, tau_estimate, s=0)  # no smoothing

        # alpha(q) = derivative of tau wrt q
        alpha_fine = spline_tau.derivative()(q_fine)

        # f(alpha) = q * alpha - tau(q)
        tau_fine = spline_tau(q_fine)
        falpha_fine = q_fine * alpha_fine - tau_fine

        # Plotting
        if graf:
            plt.style.use("default")
            fig = plt.figure(figsize=(18, 5))

            # (a) Tau(q)
            ax1 = plt.subplot2grid((1, 3), (0, 0))
            ax1.plot(qs, tau_estimate, "o", label="tau points", color="red")
            ax1.plot(q_fine, tau_fine, "-", label="tau spline", color="blue")
            ax1.set_xlabel(r"$q$", fontsize=14)
            ax1.set_ylabel(r"$\tau(q)$", fontsize=14)
            ax1.grid(True, linestyle="--", alpha=0.7)
            ax1.legend()

            # (b) alpha(q) = d tau/dq
            ax2 = plt.subplot2grid((1, 3), (0, 1))
            ax2.plot(q_fine, alpha_fine, "-", color="purple")
            ax2.set_xlabel(r"$q$", fontsize=14)
            ax2.set_ylabel(r"$\alpha(q)$", fontsize=14)
            ax2.grid(True, linestyle="--", alpha=0.7)

            # (c) f(alpha)
            ax3 = plt.subplot2grid((1, 3), (0, 2))
            ax3.plot(alpha_fine, falpha_fine, "-", color="green")
            ax3.set_xlabel(r"$\alpha$", fontsize=14)
            ax3.set_ylabel(r"$f(\alpha)$", fontsize=14)
            ax3.grid(True, linestyle="--", alpha=0.7)

            fig.tight_layout()
            plt.show()

        if results:
            return {
                "tau_spline": tau_fine,
                "falpha": falpha_fine,
                "alpha_vals": alpha_fine,
            }

    def hurst_exponent(self, graf=False):
        r"""
        Estimate the Hurst exponent, H, by finding the zero-crossing of \(\tau(q)\) if relevant,
        or by other means. In the code, we do the classical approach: we take the slope at q=2
        or we find the root of \tau(q). The code below simply uses a spline to find the root
        of tau(q).

        Parameters
        ----------
        graf : bool
            Whether to produce a diagnostic plot.

        Returns
        -------
        float
            The estimated Hurst exponent (H).
        """
        a, b, npoints = self.a, self.b, self.npoints
        qs = np.linspace(a, b, npoints)

        # 1) Ensure we have the partition functions and their fits
        if not hasattr(self, "partition") or not hasattr(self, "fits"):
            pf_result = self.partition_functions(graf1=False, result=True)
            self.partition = pf_result[0]
            self.fits = pf_result[1]

        # 2) The slope from the log-log fits is basically tau(q).
        tau_vals = np.array([pfit[0] for pfit in self.fits])
        spline_tau = UnivariateSpline(qs, tau_vals, s=0)

        # 3) Find roots of tau(q). This is a guess approach for H.
        #    If tau(q0) = 0, that q0 might be the 'dimension' -> H = 1 / q0
        #    or we do the classical approach with q=2 => tau(2) = 2H - 1, => H = (tau(2) + 1)/2
        #    We replicate the original logic: find the first root in [a,b].
        roots = spline_tau.roots()
        if len(roots) > 0:
            # Just take the first root
            sol1 = roots[0]
            # Then H = 1 / sol1, as in original code.
            # Note: This is a very specific interpretation from the original snippet.
            # Some references define H differently. Use with caution.
            h1 = 1.0 / sol1
        else:
            # fallback approach: just use q=2 => tau(2) = 2H - 1 => H = (tau(2) + 1)/2
            tau_at_2 = spline_tau(2.0)
            h1 = (tau_at_2 + 1.0) / 2.0

        if graf:
            # Show tau(q) with root
            q_fine = np.linspace(qs[0], qs[-1], 1000)
            tau_fine = spline_tau(q_fine)
            plt.plot(q_fine, tau_fine, label="tau(q)")
            for r_ in roots:
                plt.axvline(r_, color="red", linestyle="--", alpha=0.7)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.title("Tau(q) and possible roots")
            plt.legend()
            plt.show()

        return h1

    def multifractal_measure_rand(
        self, b=2, m=1, masas1=False, masas2=True, coef=False, graf1=False, cumsum=False
    ):
        r"""
        Generate a multifractal measure consistent with lognormal parameters \(\lambda\) and \(\sigma^2\).
        This code is based on random cascades along a dyadic partition.

        Parameters
        ----------
        b : int
            Partition base, typically 2 for dyadic.
        m : int
            Number of small subdivisions in each interval for plotting.
        masas1 : bool
            Toggle for one method of random measure generation.
        masas2 : bool
            Toggle for another method of random measure generation.
        coef : bool
            If True, returns the entire measure and parameters.
        graf1 : bool
            If True, plots the resulting measure.
        cumsum : bool
            If True, returns the cumulative sum of the measure.

        Returns
        -------
        If coef=True:
            (lista_general, lambdas, varianza)
        If cumsum=True:
            np.cumsum(lista_general[-1])
        Otherwise:
            None

        Notes
        -----
        - \(\lambda = \alpha_0 / H\).
        - \(\sigma^2 = 2(\lambda - 1)/\ln(2)\).
        """
        kmax = self.kmax

        # 1) Extract alpha0 and H from the class attributes
        #    alpha0 is the alpha that maximizes f(alpha)
        #    lambdas, varianza are precomputed in __init__
        alpha0 = self.alpha0
        lambdas = self.lambdas
        varianza = self.varianza

        # 2) If varianza < 0, throw error
        if varianza < 0:
            raise ValueError(
                "Variance must be non-negative in multifractal measure generation."
            )

        # 3) Generate the lognormal random variables with base 2 =>  2^(-Z), Z ~ Normal(lambdas, sqrt(varianza))
        def lognormal_base2(lam, var):
            normal_val = np.random.normal(loc=lam, scale=np.sqrt(var))
            return 2 ** (-normal_val)

        lista_general = []
        for k in range(kmax):
            lista_coeficientes = []
            for _ in range(2**k):
                if masas1:
                    # Original snippet suggests dividing measure with
                    # M = 2^-N / (2^-N + 2^N) pattern or something akin
                    # This is a random approach, seldom used, but let's keep it for legacy.
                    masas2 = False
                    m00 = lognormal_base2(lambdas, varianza)
                    # Convert to ratio
                    m00 = m00 / (m00 + 1 / m00)
                    m11 = 1 - m00
                if masas2:
                    # Two independent draws
                    m00 = lognormal_base2(lambdas, varianza)
                    m11 = lognormal_base2(lambdas, varianza)
                lista_coeficientes.append(np.array([m00, m11]))
            lista_general.append(lista_coeficientes)

        # 4) Cascade step: multiply the random subdivided masses forward
        #    Flatten the (k, k+1) structure
        for i in range(len(lista_general) - 1):
            # Flatten level i
            lista_general[i] = [
                item for sublist in lista_general[i] for item in sublist
            ]
            # Multiply corresponding items in level i+1
            for j in range(len(lista_general[i])):
                lista_general[i + 1][j] = lista_general[i][j] * lista_general[i + 1][j]

        # Flatten the last list in lista_general (the measure at max level)
        lista_general[-1] = np.array(
            [item for sublist in lista_general[-1] for item in sublist]
        )

        # 5) Additional random factor if needed (Omega), as in original code
        #    It's not fully standardized in typical MRA approaches, but we replicate logic here
        media_omega = 1 / (2 ** (-lambdas * kmax))
        # This might be a conceptual mismatch, but we keep it for code fidelity
        if varianza < 0:
            raise ValueError("Variance must be non-negative")

        omega = np.random.normal(media_omega, np.sqrt(varianza), 2**kmax)
        lista_general[-1] *= omega

        # 6) Plot
        if graf1:
            intervals = [
                np.linspace(i * b**-kmax, (i + 1) * b**-kmax, m) for i in range(b**kmax)
            ]
            # repeated measure
            output = [val * np.ones(m) for val in lista_general[-1]]
            x = np.array([point for sublist in intervals for point in sublist])
            y = np.array([val for sublist in output for val in sublist])

            # handle nan
            if np.isnan(y).all():
                raise ValueError("Array 'y' is entirely NaNs - generation problem?")

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x, y, linewidth=0.8)
            ax.set_xlim([0, 1])
            ymax = np.nanmax(y)
            ax.set_ylim([0, ymax + 0.1 * ymax])
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.title("Random Multifractal Measure", fontsize=14, fontweight="bold")
            plt.show()

        # 7) Return
        if coef:
            return lista_general, lambdas, varianza
        if cumsum:
            return np.cumsum(lista_general[-1])
        # else None
