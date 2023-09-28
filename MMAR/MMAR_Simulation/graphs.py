import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Graphs():

    def __init__(self, df, tiempo, precio, a=0, b=5, npuntos=20, deltas=np.array([x for x in range(1, 1000)]), kmax=13):
        """
        Initialize the Graphs class with the given parameters.

        tiempo: String indicating the column name representing the time interval
        precio: String indicating the column name representing the asset price
        deltas: nd.array indicating which are going to be the intervals from where
                is going to be stracted the multifractal characteristics. 

        The constructor allows for importing multiple prices from a single Excel file,
        allowing flexibility in data manipulation.

        The instance attributes:
            - df: The entire DataFrame
            - date: Numpy array of temporal values as dates
            - days: Numpy array of temporal values in days
            - Price: Numpy array of asset prices
            - X_t: Numpy array representing X(t)
            - variacionprecios: Relative price variation
        """

        # Initialize attributes
        self.df = df
        self.tiempo = tiempo
        self.precio = precio
        self.a = a
        self.b = b
        self.npuntos = npuntos
        self.deltas = deltas
        self.kmax = kmax

        # Convert columns to numpy arrays
        self.date = df[tiempo].to_numpy()
        self.days = np.array([x for x in range(len(self.date))])
        self.Price = df[precio].to_numpy()

        # Calculate X(t) values
        self.X_t = np.log(self.Price) - np.log(self.Price[0])

        # Calculate price variation
        self.variacionprecios = self.graf_Price_change(deltat=1, result=True, graf=False)

    def grafPrice(self):
        """
        Plot the asset price.
        """

        # Styling and figure setup
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot data
        ax.plot(self.date, self.Price, color='black', linewidth=1.5, label='Price')

        # X-axis formatting
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        date_fmt = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_fmt)

        # Grid and labels
        ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')
        ax.set_ylabel('Closing Price ($)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (ET)', fontsize=22, fontweight='bold')
        ax.legend(loc='upper left', fontsize=12)
        
        # Additional text info
        closing_time = '16:00:00'
        ax.tick_params(axis='both', labelsize=22)
        ax.text(self.date[-1], self.Price.max(), f'Closing Time: {closing_time} ET',
                fontsize=12, fontweight='bold', va='bottom', ha='right', color='gray',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

        # Fill the area below the graph
        ax.fill_between(self.date, self.Price, where=self.Price >= 0, color="green", alpha=0.4)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def graf_Price_change(self, deltat=1, result=False, graf=True):
        """
        Plot the relative change in asset prices.
        """
        
        # Calculate price variation
        variacion_precios1 = self.Price[deltat::deltat] - self.Price[:-deltat:deltat]
        
        # Calculate the average between two prices
        media = [(self.Price[i] + self.Price[i + 1]) / 2 for i in range(len(self.Price) - 1)]
        
        # Calculate relative price variation
        variacion_precios = [variacion_precios1[i] / media[i] for i in range(len(variacion_precios1))]

        # Plotting if graf is True
        if graf:
            fig, ax = plt.subplots(figsize=(24, 5))
            ax.plot(self.days[:-1], variacion_precios, linewidth=0.5)

            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            date_fmt = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(date_fmt)

            ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')
            
            ax.set_title(f'{self.precio} Price History', fontsize=16, fontweight='bold')
            ax.set_ylabel('Relative Price Change ($)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (ET)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=12)

            plt.tight_layout()
            plt.show()

        # Return the calculated values if result is True
        if result:
            return variacion_precios1

    def grafX_t(self):
        """
        Plot the function X(t) for the asset price.
        """

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.date, self.X_t, color='black', linewidth=1.5, label='Price')

        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        date_fmt = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_fmt)

        ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')

        ax.set_title(f'{self.precio} Price History', fontsize=16, fontweight='bold')
        ax.set_ylabel('X_t ($)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ET)', fontsize=12, fontweight='bold')

        #ax.legend(loc='upper left', fontsize=12)

        closing_time = '16:00:00'
        ax.text(self.date[-1], self.X_t.max(), f'Closing Time: {closing_time} ET',
                fontsize=12, fontweight='bold', va='bottom', ha='right', color='gray',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

        fig.set_facecolor('#EAEAEA')

        ax.fill_between(self.date, self.X_t, where=self.X_t >= 0, color="green", alpha=0.4)

        plt.tight_layout()
        plt.show()