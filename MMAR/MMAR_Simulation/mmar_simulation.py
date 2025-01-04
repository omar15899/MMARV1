import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from stochastic.processes.continuous import FractionalBrownianMotion
from .multifractalcharacteristics import MultifractalCharacteristics
from ..Multifractal_Measure.multifractal_measure_rand import BinomialMultifractalRand


class MMAR(MultifractalCharacteristics):
    """
    MMAR (Multifractal Model of Asset Returns) class that inherits from
    MultifractalCharacteristics. It uses fractional Brownian motion to
    simulate price paths under time deformation dictated by a multifractal
    measure.

    Parameters
    ----------
    dataset : pd.DataFrame or np.ndarray
        The dataset with time and price columns.
    time : str
        Column name or key that identifies the time axis in 'dataset'.
    price : str
        Column name or key that identifies the price column in 'dataset'.
    a : float, optional
        Lower bound for q-range in the multifractal analysis (default=0).
    b : float, optional
        Upper bound for q-range in the multifractal analysis (default=5).
    npoints : int, optional
        Number of q-values for the multifractal analysis (default=20).
    deltas : array-like or None, optional
        Array of time intervals used in the partition function computations.
        If None, defaults to np.arange(1, 1000).
    kmax : int, optional
        Maximum resolution (2^kmax data points) for multifractal measure
        generation and FBM simulation (default=13).
    """

    def __init__(
        self, dataset, time, price, a=0, b=5, npoints=20, deltas=None, kmax=13
    ):
        """
        Initializes the MMAR object, leveraging the base initialization of
        the MultifractalCharacteristics class.
        """
        super().__init__(dataset, time, price, a, b, npoints, deltas, kmax)

    def path_simulation(self, grafs=False, results=False):
        """
        Simulate a single price path using:
        1) A multifractal time deformation from BinomialMultifractalRand (lognormal base 2).
        2) Fractional Brownian motion increments in 'physical' time.

        Parameters
        ----------
        grafs : bool
            Whether to produce diagnostic plots.
        results : bool
            Whether to return the computed trading time array.

        Returns
        -------
        tradingtime : np.ndarray, optional
            The (normalized) multifractal trading time if results=True.
        """
        # 1) Generate the multifractal measure
        binom = BinomialMultifractalRand()
        npts = 2**self.kmax
        tradingtime = binom.multifractal_measure_rand2(
            b=2,
            m=1,
            kmax=self.kmax,
            falpha=self.falpha,
            derivada=self.derivada,
            h1=self.h1,
            Price=self.Price,
            masas1=False,
            masas2=True,
            coef=False,
            graf1=False,
            cumsum=True,
        )

        # Normalize trading time to range [0, 2^kmax]
        tradingtime = npts * (tradingtime / np.max(tradingtime))

        #  Fractional Brownian Motion increments
        fbm = FractionalBrownianMotion(hurst=self.h1)
        simulacion_fbm = fbm._sample_fractional_brownian_motion(npts - 1)

        # 3) Construct simulated log-prices
        xt_simulados = simulacion_fbm
        # For the initial price, we take self.Price[0]
        precio_final = self.Price[0] * np.exp(xt_simulados)

        # 4) Optionally plot
        if grafs:
            plt.style.use("default")

            # (a) Trading time vs integer index
            plt.figure(figsize=(10, 4))
            plt.plot(range(npts), tradingtime, label="Multifractal Trading Time")
            plt.xlabel("Integer Index")
            plt.ylabel("Trading Time")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # (b) Price vs trading time
            plt.figure(figsize=(10, 4))
            plt.plot(tradingtime, precio_final, label="Price (Deformed Time)")
            plt.title("Price under Multifractal Time Deformation")
            plt.xlabel("Multifractal Trading Time")
            plt.ylabel("Price")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # (c) Price vs integer steps
            plt.figure(figsize=(10, 4))
            plt.plot(range(npts), precio_final, label="Price (Physical Time)")
            plt.title("Price in Physical (Integer) Time")
            plt.xlabel("Integer Time Steps")
            plt.ylabel("Price")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # (d) Price increments
            plt.figure(figsize=(10, 4))
            plt.plot(
                range(npts - 1),
                precio_final[1:] - precio_final[:-1],
                lw=0.5,
                label="Price Increments",
            )
            plt.title("Price Increments in Physical Time")
            plt.xlabel("Integer Time Steps")
            plt.ylabel("Δ Price")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if results:
            return tradingtime

    def simulacion(self, n=1000, result=False, show_plots=True):
        r"""
        Monte Carlo simulation of n price paths, each constructed by:
        1) Generating a multifractal trading time (lognormal base 2).
        2) Generating fractional Brownian motion increments.
        3) Composing them to obtain final prices.

        Parameters
        ----------
        n : int
            Number of simulated paths (default=1000).
        result : bool
            If True, return the arrays with the simulated paths.
        show_plots : bool
            If True, plot the ensemble of paths.

        Returns
        -------
        (almacen_tradingtime, almacen_precio_final, almacen_xt) : tuple of lists
            - almacen_tradingtime : list of multifractal time arrays for each of n simulations
            - almacen_precio_final: list of final price arrays for each of n simulations
            - almacen_xt : list of FBM increments arrays for each of n simulations
        """
        binom = BinomialMultifractalRand()
        npts = 2**self.kmax

        almacen_tradingtime = []
        almacen_xt = []
        almacen_precio_final = []

        for _ in range(n):
            # 1) Generate multifractal time
            tradingtime = binom.multifractal_measure_rand2(
                b=2,
                m=1,
                kmax=self.kmax,
                falpha=self.falpha,
                derivada=self.derivada,
                h1=self.h1,
                Price=self.Price,
                masas1=False,
                masas2=True,
                coef=False,
                graf1=False,
                cumsum=True,
            )
            # Scale
            tradingtime = npts * (tradingtime / np.max(tradingtime))

            #  Generate FBM increments
            fbm = FractionalBrownianMotion(hurst=self.h1)
            simulacion_fbm = fbm._sample_fractional_brownian_motion(npts - 1)

            # 3) Compute final prices
            xt_sim = simulacion_fbm
            # We use the last known price as reference
            precio_final = self.Price[-1] * np.exp(xt_sim)

            almacen_tradingtime.append(tradingtime)
            almacen_xt.append(xt_sim)
            almacen_precio_final.append(precio_final)

        if show_plots:
            self._plot_simulation_ensemble(
                almacen_tradingtime, almacen_xt, "Real Time (days)", "X(t)", n
            )
            self._plot_simulation_ensemble(
                [range(npts)] * n,
                almacen_tradingtime,
                "Integer Steps",
                "Trading Time",
                n,
            )

        if result:
            return almacen_tradingtime, almacen_precio_final, almacen_xt

    def analizadorprobabilidades(self, day, almacen_tradingtime, almacen_precio_final):
        r"""
        Analyzes and plots the distribution of simulated prices at a specific
        'day' (in real/physical time) across all n simulations. This effectively
        shows how the random 'time deformation' changes the distribution of
        prices at the chosen real day.

        Additionally, checks for 'fat tails' by comparing the empirical
        distribution to a Gaussian, plotting the difference in log-scale.

        Parameters
        ----------
        day : float
            The real (physical) time at which we want to evaluate the price distribution.
        almacen_tradingtime : list of np.ndarrays
            Each entry is the multifractal trading time array for one simulation.
        almacen_precio_final : list of np.ndarrays
            Each entry is the array of simulated prices for one simulation.

        Returns
        -------
        None. Displays a histogram with a Gaussian overlay and a log-plot for tail comparison.
        """
        n = len(almacen_tradingtime)

        # Interpolate each path at the chosen real day
        precios_en_dia = []
        for t_array, p_array in zip(almacen_tradingtime, almacen_precio_final):
            val = np.interp(day, t_array, p_array)
            precios_en_dia.append(val)

        # Basic stats
        media = np.mean(precios_en_dia)
        stdd = np.std(precios_en_dia)

        # (1) Plot histogram + Gaussian
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(12, 8))
        hist_vals, bins, _ = ax.hist(
            precios_en_dia,
            bins=50,
            alpha=0.5,
            density=True,
            edgecolor="black",
            label="Empirical Distribution",
        )

        xvals = np.linspace(bins[0] - 1, bins[-1] + 1, 1000)
        yvals = stats.norm.pdf(xvals, media, stdd)
        ax.plot(xvals, yvals, color="red", linewidth=2, label="Gaussian Reference")

        # Mean and std annotation
        textbox_text = f"Mean: {media:.2f}\nStd Dev: {stdd:.2f}"
        ax.text(
            0.95,
            0.95,
            textbox_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
        )

        ax.set_xlabel(f"P({day}) ($)", fontsize=16)
        ax.set_ylabel("Probability Density", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(loc="upper right", fontsize=14)
        plt.title(f"Distribution of Prices at Real Time = {day}", fontsize=14)
        plt.tight_layout()
        plt.show()

        # (2) Check for fat tails:
        # One approach is to compare the log of the empirical survival function with
        # the log of the Gaussian survival function.
        # We can do a simple tail index check, or just do a Q-Q plot as well.
        self._check_fat_tails(precios_en_dia, media, stdd)

    def _plot_simulation_ensemble(
        self, x_values_list, y_values_list, x_label, y_label, n
    ):
        """
        Utility function to plot multiple simulation paths in a single figure.
        """
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(12, 8))
        for x_vals, y_vals in zip(x_values_list, y_values_list):
            ax.plot(x_vals, y_vals, lw=0.5, alpha=0.1, color="black")

        # Highlight one path in blue for better visibility
        mid_index = n // 2
        ax.plot(
            x_values_list[mid_index],
            y_values_list[mid_index],
            lw=1.0,
            alpha=1,
            color="blue",
            label="One Example Path",
        )

        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def _check_fat_tails(self, data, mean_val, std_val):
        """
        Check for fat tails by comparing the empirical tail distribution
        to that of a normal distribution. We'll produce a Q-Q plot and
        a log-survival function plot.

        data : array-like
            The sample of prices at a given time.
        mean_val : float
            Mean of the data.
        std_val : float
            Std dev of the data.
        """
        sorted_data = np.sort(data)
        n = len(sorted_data)
        # Q-Q plot vs normal
        theoretical_q = stats.norm.ppf(
            (np.arange(n) + 0.5) / n, loc=mean_val, scale=std_val
        )

        # (A) Q-Q plot
        plt.figure(figsize=(8, 6))
        plt.plot(
            theoretical_q, sorted_data, "o", markersize=3, alpha=0.5, label="Data Q-Q"
        )
        min_q, max_q = min(theoretical_q), max(theoretical_q)
        plt.plot(
            [min_q, max_q],
            [min_q, max_q],
            color="red",
            linestyle="--",
            label="Perfect Normal",
        )
        plt.title("Q-Q Plot (Data vs. Normal)")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Empirical Quantiles")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # (B) Survival function (SF): P(X > x). We'll compare log(SF_data) vs. log(SF_normal).
        # Empirical survival
        sf_data = 1.0 - np.arange(1, n + 1) / n
        # Normal SF for sorted_data
        sf_norm = 1.0 - stats.norm.cdf(sorted_data, loc=mean_val, scale=std_val)

        # Plot in log scale
        plt.figure(figsize=(8, 6))
        plt.plot(
            sorted_data, sf_data, "o", markersize=3, alpha=0.5, label="Empirical SF"
        )
        plt.plot(sorted_data, sf_norm, color="red", alpha=0.7, label="Normal SF")
        plt.yscale("log")  # log scale on y
        plt.title("Survival Function Comparison (Log Scale)")
        plt.xlabel("Price")
        plt.ylabel("P(X > x) (log scale)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
