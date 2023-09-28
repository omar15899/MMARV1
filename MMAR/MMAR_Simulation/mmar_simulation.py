# Asumo que has leído antes los comentarios de Graph.py
# Guía de estilos, eliminar paquetes no usados, objetos mutables 
# como parámetros...
import random
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Modulos de librerias externas
from stochastic.processes.continuous import FractionalBrownianMotion 
from .multifractalcharacteristics import MultifractalCharacteristics

# MMAR class that inherits from MultifractalCharacteristics
class MMAR(MultifractalCharacteristics):
    
    def __init__(self, df, tiempo, precio, a=0, b=5, npuntos=20, 
                 deltas=np.array([x for x in range(1, 1000)]), kmax=13):
        """
        Initialize the MMAR object, which inherits attributes and methods 
        from the MultifractalCharacteristics class.
        """
        # Esto no es correcto. Si el parámetro a fuese igual a 5,
        # siempre se pasaría el valor 0 a la superclase. Lo correcto sería:
        # super().__init__(df, tiempo, precio, a, b, npuntos, deltas, kmax)
        super().__init__(df, tiempo, precio, a=0, b=5, npuntos=20, 
                         deltas=np.array([x for x in range(1, 1000)]), kmax=13)
        
    def path_simulation(self, grafs=False, results=False):
        """
        Simulate a path using fractional Brownian motion.
        grafs: Boolean to indicate if graphs should be plotted.
        results: Boolean to indicate if results should be returned.
        """
        
        # kmax is 2^kmax, which represents the number of days we want to simulate
        kmax = self.kmax
        
        # Calculate normalized trading time
        # No está mal, pero no diría que está bien. Una clase
        # que hereda a otra, hereda todos sus métodos. Por tanto,
        # esto podría escribirse así:
        # tradingtime = self.multifractal_measure_rand(cumsum=True)
        tradingtime = super().multifractal_measure_rand(cumsum=True)
        
        # Normalize and multiply to obtain the number of days
        tradingtime = 2**kmax * (tradingtime / np.amax(tradingtime))
        
        # Calculate the fractional Brownian motion (fbm)
        
        # 1. Create an instance of the class with the fbm object
        fbm = FractionalBrownianMotion(hurst=self.h1)
        
        # 2. Use the _sample_fractional_brownian_motion method to create the list of increments
        # relative to real time.
        simulacionfbm = fbm._sample_fractional_brownian_motion(2**kmax - 1)
        
        # Calculate the simulated Xt values relative to real time
        xtsimulados = simulacionfbm
        precio_final = self.Price[0] * np.exp(xtsimulados)
        
        # Plot graphs if grafs flag is True
        if grafs:
            # Por qué no plt.plot(range(2**kmax), tradingtime)?
            plt.plot(np.array([x for x in range(2**kmax)]), tradingtime)
            plt.show()
            
            # In the graph, function composition is performed
            plt.plot(tradingtime, precio_final)
            plt.title("Graph with Deformation")
            plt.xlabel("Real Time")
            plt.show()
            
            # Por qué no plt.plot(range(2**kmax), precio_final)?
            plt.plot([x for x in range(2**kmax)], precio_final)
            plt.title("Graph Without Deformation")
            plt.xlabel("Real Time")
            plt.show()
            
            # Igual que antes, simplifícalo
            plt.plot(np.array([x for x in range(2**kmax - 1)]), precio_final[1:] - precio_final[:-1], lw=0.5)
            plt.show()
            
            plt.plot(tradingtime, precio_final)
            plt.title("Graph Without Deformation")
            plt.xlabel("TradingTime")
            plt.show()
        
        # Return results if results flag is True
        # Veo un poco absurdo el parámetro results. 
        # Yo devolvería siempre el resultado, quien use
        # la función que use el resultado si quiere o no
        if results:
            return tradingtime


    # El parámetro resultado no se usa. No sé qué IDE usas, pero debería avisarte
    def simulacion(self, n=1000, resultado=False):
        # Monte Carlo simulation for n curves compared to the simulation of 1 curve in the previous function
        almacen_tradingtime = []
        almacen_xt = []
        almacen_precio_final = []

        kmax = self.kmax

        for _ in range(n):
            # Calculate unnormalized trading time
            # Usar self en vez de super()
            tradingtime = super().multifractal_measure_rand(cumsum=True)
            # Normalized trading time
            tradingtime = 2**kmax * (tradingtime / np.amax(tradingtime))
            fbm = FractionalBrownianMotion(hurst=self.h1)
            simulacionfbm = fbm._sample_fractional_brownian_motion(2**kmax - 1)
            xtsimulados = simulacionfbm
            precio_final = self.Price[-1] * np.exp(xtsimulados)

            almacen_tradingtime.append(tradingtime)
            almacen_xt.append(xtsimulados)
            almacen_precio_final.append(precio_final)

        # Data Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        # Más elegante:
        # for trading_time, almacen in zip(almacen_tradingtime, almacen_xt):
        #   ax.plot(trading_time, almacen, lw=0.5, alpha=0.1)
        for i in range(len(almacen_precio_final)):
            ax.plot(almacen_tradingtime[i], almacen_xt[i], lw=0.5, alpha=0.1)
        ax.plot(almacen_tradingtime[n // 2], almacen_xt[n // 2], lw=0.5, alpha=1, color='b')
        ax.set_xlabel("Real Time (days)", fontsize=22)
        ax.set_ylabel("X(t)", fontsize=22)
        ax.tick_params(axis='both', labelsize=22)
        ax.grid(True, linestyle='-.')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 8))
        for i in range(len(almacen_tradingtime)):
            ax.plot([x for x in range(2**kmax)], almacen_tradingtime[i], lw=0.5, alpha=0.1)
        ax.plot([x for x in range(2**kmax)], almacen_tradingtime[n // 2], lw=0.5, alpha=1, color='b')
        ax.set_xlabel("Real Time (days)", fontsize=22)
        ax.set_ylabel("Trading Time (days)", fontsize=22)
        ax.tick_params(axis='both', labelsize=22)
        ax.grid(True, linestyle='-.')
        plt.tight_layout()
        plt.show()

        # ¡Hay código repetido! Si hay código repetido, entonces toca crear una función.
        # Por ejemplo, las líneas 126 - 149 podrían reescribirse: 
        # def plot_simulation(x_values, y_values, x_label, y_label):
        #     _, ax = plt.subplots(figsize=(12, 8))
        #     for x, y in zip(x_values, y_values):
        #         ax.plot(x, y, lw=0.5, alpha=0.1)
        #     ax.plot(x_values[n // 2], y_values[n // 2], lw=0.5, alpha=1, color='b')
        #     ax.set_xlabel(x_label, fontsize=22)
        #     ax.set_ylabel(y_values, fontsize=22)
        #     ax.tick_params(axis='both', labelsize=22)
        #     ax.grid(True, linestyle='-.')
        #     plt.tight_layout()
        #     plt.show()
        #
        # plot_simulation(almacen_tradingtime, almacen_xt, "Real Time (days)", "X(t)")
        # plot_simulation(range(2**kmax), almacen_tradingtime, "Real Time (days)", "X(t)")

        return almacen_tradingtime, almacen_precio_final, almacen_xt


    def analizadorprobabilidades(self, dia, almacen_tradingtime, almacen_precio_final):
        """
        This function analyzes the probabilities for each output of the simulacion2 method.
        n represents the number of simulations carried out in simulacion2.
        """
      
        # Calculate probability density functions for each time unit. We make use of Laplace's rule.
        # Specifically, we compute histograms for each time point.
      
        # 1. Generate lists for histograms:
        # The key challenge here is that we are composing functions, and the time assigned to
        # each price with the same index i of each sublist isn't the same. This depends on each one's
        # transition function. Therefore, we perform linear interpolation for each pair of 
        # almacen_precio_final[i] and almacen_tradingtime[i] and proceed accordingly.

        # Obtain the prices for the specific day (real) while accommodating time deformation due to trading time.
        # Linear interpolation is employed (although other interpolation methods could be better).
        # The interpolation is computationally the most efficient way to compose functions:
        n = len(almacen_tradingtime)
        # Parte las líneas muy largas en varias líneas y utiliza el zip que te he dicho antes:
        # precios_serios = [np.interp(dia, trading_time, precio_final) 
        #                   for trading_time, precio_final in zip(almacen_tradingtime, almacen_precio_final)
        #                   if np.interp(dia, trading_time, precio_final)]
        precios_serios = [np.interp(dia, almacen_tradingtime[i], almacen_precio_final[i]) for i in range(n) if np.interp(dia, almacen_tradingtime[i], almacen_precio_final[i])]
        hist, bins = np.histogram(precios_serios, bins=50, density=True)
        media = np.mean(precios_serios)
        stdd = np.std(precios_serios)
      
        # Generate a Gaussian distribution for comparison:
        x = np.linspace(bins[0]-1, bins[-1]+1, 10000)
        y = stats.norm.pdf(x, media, stdd)
      
        # Plotting the histogram and Gaussian distribution:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(precios_serios, bins=50, alpha=0.5, density=True, edgecolor='black', label="Actual Distribution")
        ax.plot(x, y, color='red', linewidth=2, label="Gaussian Distribution (CLT)")
      
        # Display mean and standard deviation:
        textbox_text = f'Mean: {media:.2f}\nStandard Deviation: {stdd:.2f}'
        ax.text(0.95, 0.95, textbox_text, transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
      
        # Label x and y axis:
        ax.set_xlabel(f'P({dia}) ($)', fontsize=22)
        ax.set_ylabel('Probability Density', fontsize=22)
      
        # Set tick size:
        ax.tick_params(axis='both', which='major', labelsize=22)
      
        # Adding horizontal grid lines:
        ax.grid(axis='y')
      
        # Add legend:
        plt.legend(loc='center right', fontsize=18)
      
        # Ensure a clean layout:
        plt.tight_layout()
      
        # Display the plot:
        plt.show()
