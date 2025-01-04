import numpy as np
import matplotlib.pyplot as plt
from stochastic.processes.continuous import FractionalBrownianMotion
from .multifractalcharacteristics import MultifractalCharacteristics
import scipy.stats as stats


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
        1) A multifractal time deformation via multifractal_measure_rand().
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
        # Number of data points is 2^kmax
        kmax = self.kmax
        npts = 2**kmax

        # 1) Compute multifractal trading time via cumsum
        #    This is the time deformation function
        tradingtime = self.multifractal_measure_rand(cumsum=True)
        # Normalize and scale to the range [0, 2^kmax]
        tradingtime = npts * (tradingtime / np.max(tradingtime))

        # 2) Fractional Brownian Motion (FBM)
        fbm = FractionalBrownianMotion(hurst=self.h1)
        # The 'stochastic' package uses _sample_fractional_brownian_motion(...)
        # internally for generating the FBM increments
        simulacion_fbm = fbm._sample_fractional_brownian_motion(npts - 1)

        # 3) Construct simulated log-prices: X(t) = increments of FBM
        xt_simulados = simulacion_fbm
        # Use self.Price[0] as initial price
        precio_final = self.Price[0] * np.exp(xt_simulados)

        # 4) Optionally plot
        if grafs:
            plt.style.use("default")

            # Plot the multifractal trading time vs. integer steps
            plt.figure(figsize=(10, 4))
            plt.plot(range(npts), tradingtime, label="Multifractal Trading Time")
            plt.xlabel("Integer Index")
            plt.ylabel("Trading Time")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Plot price vs. multifractal time
            plt.figure(figsize=(10, 4))
            plt.plot(tradingtime, precio_final, label="Price (Deformed Time)")
            plt.title("Price under Multifractal Time Deformation")
            plt.xlabel("Multifractal Trading Time")
            plt.ylabel("Price")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Plot price vs. integer steps
            plt.figure(figsize=(10, 4))
            plt.plot(range(npts), precio_final, label="Price (Physical Time)")
            plt.title("Price in Physical (Integer) Time")
            plt.xlabel("Integer Time Steps")
            plt.ylabel("Price")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Plot increments of price
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

    def simulacion(self, n=1000, result=False):
        """
        Monte Carlo simulation of n price paths, each constructed by:
        1) Generating a multifractal trading time.
        2) Generating fractional Brownian motion increments.
        3) Composing them to obtain final prices.

        Parameters
        ----------
        n : int
            Number of simulated paths (default=1000).
        result : bool
            If True, return the arrays with the simulated paths.

        Returns
        -------
        (almacen_tradingtime, almacen_precio_final, almacen_xt) : tuple of lists
            - almacen_tradingtime : list of multifractal time arrays for each of n simulations
            - almacen_precio_final: list of final price arrays for each of n simulations
            - almacen_xt : list of FBM increments arrays for each of n simulations
        """
        kmax = self.kmax
        npts = 2**kmax

        almacen_tradingtime = []
        almacen_xt = []
        almacen_precio_final = []

        for _ in range(n):
            # 1) Generate multifractal time (unnormalized) + normalization
            tradingtime = self.multifractal_measure_rand(cumsum=True)
            tradingtime = npts * (tradingtime / np.max(tradingtime))

            # 2) Fractional Brownian Motion
            fbm = FractionalBrownianMotion(hurst=self.h1)
            simulacion_fbm = fbm._sample_fractional_brownian_motion(npts - 1)

            # 3) Construct log-prices
            xt_simulados = simulacion_fbm
            # Scale from final known price in the original dataset, if desired
            # or from an initial price (the user logic may differ).
            # Here we do from last known price:
            precio_final = self.Price[-1] * np.exp(xt_simulados)

            # Store
            almacen_tradingtime.append(tradingtime)
            almacen_xt.append(xt_simulados)
            almacen_precio_final.append(precio_final)

        # 4) Plot: arrays of all simulations
        def plot_simulation(x_values_list, y_values_list, x_label, y_label):
            """
            Utility function to plot multiple simulation paths.
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

        # Plot X(t) vs Real (integer) time for each simulation
        plot_simulation(almacen_tradingtime, almacen_xt, "Real Time (days)", "X(t)")

        # Alternatively, we can also plot the tradingtime vs. index
        # but the user logic may differ.
        plot_simulation(
            [range(npts)] * n, almacen_tradingtime, "Integer Steps", "Trading Time"
        )

        if result:
            return almacen_tradingtime, almacen_precio_final, almacen_xt

    def analizadorprobabilidades(self, day, almacen_tradingtime, almacen_precio_final):
        """
        Analyzes and plots the distribution of simulated prices at a specific
        'day' (in real/physical time) across all n simulations. This effectively
        shows how the random 'time deformation' changes the distribution of
        prices at the chosen real day.

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
        None. Displays a histogram with a Gaussian overlay.
        """
        # Number of simulations
        n = len(almacen_tradingtime)

        # For each simulation i, we do a linear interpolation to find
        # the price at "day" in real time. This is effectively
        # P_i( day ), given the time deformation stored in almacen_tradingtime[i].
        precios_en_dia = []
        for t_array, p_array in zip(almacen_tradingtime, almacen_precio_final):
            # Interpolation can fail if 'day' < t_array[0] or 'day' > t_array[-1].
            # One might want to clamp or skip. We assume day is within [0, 2^kmax].
            # np.interp returns a float, so we append it directly.
            val = np.interp(day, t_array, p_array)
            precios_en_dia.append(val)

        # Build the histogram
        hist, bins = np.histogram(precios_en_dia, bins=50, density=True)
        media = np.mean(precios_en_dia)
        stdd = np.std(precios_en_dia)

        # Gaussian for reference
        xvals = np.linspace(bins[0] - 1, bins[-1] + 1, 1000)
        yvals = stats.norm.pdf(xvals, media, stdd)

        # Plot the histogram + Gaussian overlay
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(
            precios_en_dia,
            bins=50,
            alpha=0.5,
            density=True,
            edgecolor="black",
            label="Actual Distribution",
        )
        ax.plot(xvals, yvals, color="red", linewidth=2, label="Gaussian (Reference)")

        # Display mean and standard deviation in a text box
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

        plt.tight_layout()
        plt.show()
