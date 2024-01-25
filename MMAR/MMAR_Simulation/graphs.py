# Revisar los paquetes que no se usan. En este caso, Pandas.
# Esto debería decírtelo automáticamente tu IDE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Graphs:
    # 2 problemas veo en la siguiente línea:
    #     1. deltas=np.array([x for x in range(1, 1000)] puede reescribirse como np.array(range(1, 1000)).
    #        Mejor aún, se puede usar np.arange
    #     2. Mucho ojo con poner objetos mutables como parámetros por defecto. Malísima práctica:
    #     https://florimond.dev/en/posts/2018/08/python-mutable-defaults-are-the-source-of-all-evil/
    def __init__(
        self, dataset, time, price, a=0, b=5, npoints=20, deltas=None, kmax=13
    ):
        """
        Initialize the Graphs class with the given parameters.

        time: String indicating the column name representing the time interval
        price: String indicating the column name representing the asset price
        deltas: nd.array indicating which are going to be the intervals from where
                is going to be stracted the multifractal characteristics.

        The constructor allows for importing multiple prices from a single Excel file,
        allowing flexibility in data manipulation.

        The instance attributes:
            - dataset: The entire DataFrame
            - date: Numpy array of temporal values as dates
            - days: Numpy array of temporal values in days
            - Price: Numpy array of asset prices
            - X_t: Numpy array representing X(t)
            - variacionprices: Relative price variation
        """
        # Falta homogeneidad en los nombres. O todo en español, o todo en inglés.

        if a is None:
            a = np.arange(1, 1000)
        # Initialize attributes
        # Llamar dataset a una variable no significa nada. Podría llamarse 'datos' o cualquier otro
        # nombre más descriptivo
        self.dataset = dataset
        self.time = time
        self.price = price
        self.a = a
        self.b = b
        self.npoints = npoints
        self.deltas = deltas
        self.kmax = kmax

        # Convert columns to numpy arrays
        self.date = dataset[time].to_numpy()
        # Se puede reescribir (como en la línea 8)
        self.days = np.array([x for x in range(len(self.date))])
        # Nombres de los atributos siempre en minúscula. Intenta seguir la guía de estilo PEP8
        self.Price = dataset[price].to_numpy()

        # Calculate X(t) values
        self.X_t = np.log(self.Price) - np.log(self.Price[0])

        # Calculate price variation
        # Renombrar (guía PEP8)
        self.variacionprices = self.graf_Price_change(deltat=1, result=True, graf=False)

    def grafPrice(self):
        """
        Plot the asset price.
        """

        # Styling and figure setup
        plt.style.use("default")
        # Cuando una variable no se usa, a veces es mejor enfatizarlo.
        # En este caso, por ejemplo, fig no se usa. Yo lo reescribiría como:
        # _, ax = plt.subplots(figsize=(12, 8))
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

        # Show the plot
        plt.tight_layout()
        plt.show()

    def graf_Price_change(self, deltat=1, result=False, graf=True):
        """
        Plot the relative change in asset prices.
        """

        # Calculate price variation
        variacion_prices1 = self.Price[deltat::deltat] - self.Price[:-deltat:deltat]

        # Calculate the average between two prices
        # Más elegante: media = (price[:-1] + price[1:])/2
        media = [
            (self.Price[i] + self.Price[i + 1]) / 2 for i in range(len(self.Price) - 1)
        ]

        # Calculate relative price variation
        # Más elegante: variacion_prices = variacion_prices1/media[:len(variacion_prices1)]
        variacion_prices = [
            variacion_prices1[i] / media[i] for i in range(len(variacion_prices1))
        ]

        # Plotting if graf is True
        if graf:
            fig, ax = plt.subplots(figsize=(24, 5))
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

        # Return the calculated values if result is True
        if result:
            return variacion_prices1

    def grafX_t(self):
        """
        Plot the function X(t) for the asset price.
        """

        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.date, self.X_t, color="black", linewidth=1.5, label="Price")

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
        ax.set_ylabel("X_t ($)", fontsize=12, fontweight="bold")
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
