import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Graphs:
    r"""
    This class provides basic data handling and plotting functionality
    for a financial time series, including:
    - Price plot
    - Relative price change plot
    - Log-price transform X(t) plot

    Parameters
    ----------
    dataset : pd.DataFrame or np.ndarray
        The entire dataset containing time and price information.
    time : str
        Column name or key representing the time index in 'dataset'.
    price : str
        Column name or key representing the asset's price in 'dataset'.
    a : float, optional
        Lower bound for q (used by inheriting classes), default is 0.
    b : float, optional
        Upper bound for q (used by inheriting classes), default is 5.
    npoints : int, optional
        Number of q-values to sample (used by inheriting classes), default is 20.
    deltas : list or np.ndarray, optional
        Array of time intervals for partition function computations.
        If None, uses np.arange(1, 1000).
    kmax : int, optional
        Maximum resolution parameter for further multifractal analysis, default is 13.
    """

    def __init__(
        self, dataset, time, price, a=0, b=5, npoints=20, deltas=None, kmax=13
    ):
        # Validate or set deltas
        if deltas is None:
            deltas = np.arange(1, 1000)

        # Store parameters
        self.dataset = dataset
        self.time = time
        self.price = price
        self.a = a
        self.b = b
        self.npoints = npoints
        self.deltas = deltas
        self.kmax = kmax

        # Convert columns to numpy arrays
        # For the time column
        self.date = self.dataset[self.time].to_numpy()
        # Numeric index of days
        self.days = np.arange(len(self.date))
        # Price array
        self.Price = self.dataset[self.price].to_numpy()
        # Log-price transform
        self.X_t = np.log(self.Price) - np.log(self.Price[0])

        # Compute and store the relative price variation as a default
        # (using deltat=1, but not plotting)
        self.variacionprices = self.graf_Price_change(deltat=1, result=True, graf=False)

    def grafPrice(self):
        """
        Plot the asset price over time.
        """
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot data
        ax.plot(self.date, self.Price, color="black", linewidth=1.5, label="Price")

        # X-axis formatting
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        date_fmt = mdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_fmt)

        # Grid and labels
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
        ax.set_ylabel("Closing Price ($)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Time (ET)", fontsize=22, fontweight="bold")
        ax.legend(loc="upper left", fontsize=12)

        # Additional text info
        closing_time = "16:00:00"
        ax.tick_params(axis="both", labelsize=22)
        ax.text(
            self.date[-1],
            self.Price.max(),
            f"Closing Time: {closing_time} ET",
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
            color="gray",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
        )

        # Fill the area below the graph
        ax.fill_between(
            self.date, self.Price, where=self.Price >= 0, color="green", alpha=0.4
        )

        plt.tight_layout()
        plt.show()

    def graf_Price_change(self, deltat=1, result=False, graf=True):
        r"""
        Calculate and optionally plot the relative change in asset prices.

        Parameters
        ----------
        deltat : int
            Step size (in indices) for computing price differences.
        result : bool
            If True, returns the raw price-difference array.
        graf : bool
            If True, produces the plot.

        Returns
        -------
        variacion_prices1 : np.ndarray (if result=True)
            Array of absolute price differences over the chosen deltat.
        """
        # 1) Compute price differences
        variacion_prices1 = self.Price[deltat::deltat] - self.Price[:-deltat:deltat]

        # 2) Compute the average of consecutive prices to scale the difference
        #    so that the result is a relative difference
        media = (self.Price[:-1] + self.Price[1:]) / 2.0
        # Ensure we only use the portion that matches variacion_prices1 length
        media = media[: len(variacion_prices1)]

        # 3) Relative price variation
        variacion_prices = variacion_prices1 / media

        # 4) Plot if requested
        if graf:
            fig, ax = plt.subplots(figsize=(24, 5))
            # The x-axis: days up to len(variacion_prices)
            ax.plot(self.days[:-1], variacion_prices, linewidth=0.5)

            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            date_fmt = mdates.DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(date_fmt)

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

            ax.set_title(f"{self.price} Price History", fontsize=16, fontweight="bold")
            ax.set_ylabel("Relative Price Change ($)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (ET)", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left", fontsize=12)

            plt.tight_layout()
            plt.show()

        if result:
            return variacion_prices1

    def grafX_t(self):
        """
        Plot the function X(t) for the asset price: X(t) = ln(Price(t)) - ln(Price(0)).
        """
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.date, self.X_t, color="black", linewidth=1.5, label="X(t)")

        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        date_fmt = mdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_fmt)

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

        ax.set_title(
            f"{self.price} Price History: X(t) = ln(Price) - ln(Price[0])",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_ylabel("X(t)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ET)", fontsize=12, fontweight="bold")

        closing_time = "16:00:00"
        ax.text(
            self.date[-1],
            self.X_t.max(),
            f"Closing Time: {closing_time} ET",
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
            color="gray",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
        )

        fig.set_facecolor("#EAEAEA")

        ax.fill_between(
            self.date, self.X_t, where=self.X_t >= 0, color="green", alpha=0.4
        )

        plt.tight_layout()
        plt.show()
