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
'''
 Por el archivo init del directorio se ve que el autor poermite importar directamente 
 la clase desde el directorio sin llamar al módulo que la contiene.
'''

# Modulos de mi libreria
#from .graphs import Graphs \\Esto no es necesario porque ya estamos haciendo una herencia de la previa.
from .multifractalcharacteristics import MultifractalCharacteristics

class MMAR(MultifractalCharacteristics):


    def __init__(self,df, tiempo, precio, a = 0, b = 5, npuntos = 20, deltas = np.array([x for x in range(1, 1000)]), kmax = 13):
        super().__init__(df, tiempo, precio, a = 0, b = 5, npuntos = 20, deltas = np.array([x for x in range(1, 1000)]), kmax = 13)
        """
        Otra forma más anticuada de hacerlo es mediante:

        Graphs.__init__(self)
        MultifractalCharacteristics.__init__(self)
        """




    def path_simulation(self, grafs = False, results = False):
        
        # kmax es 2^kmax número de días que queremos simular
        kmax = self.kmax

        # Calculamos el trading time normalizado
        # tradingtime = 2**kmax*super().multifractal_measure_rand(kmax, cumsum = True)
        tradingtime = super().multifractal_measure_rand(cumsum = True)
        # Normalizamos y multiplicamos para obtener el número de días.
        tradingtime =  2**kmax*(tradingtime/np.amax(tradingtime))


        # Calculamos el fractional brownian motion fbm:

        # 1. Instanciamos la clase con el objeto fbm
        fbm = FractionalBrownianMotion(hurst= self.h1)
        # 2. Usamos el método _sample_fractional_brownian_motion para crear la lista con los incrementos
        # con respecto al tiempo real.
        simulacionfbm = fbm._sample_fractional_brownian_motion(2**kmax-1)

        # Calculamos los valores Xt con respecto al tiempo real:
        xtsimulados = simulacionfbm
        precio_final = self.Price[0]*np.exp(xtsimulados)

        



        if grafs:
            plt.plot(np.array([x for x in range(2**kmax)]), tradingtime)
            plt.show()

            # En la gráfica es donde se hace la composición de funciones! Mirar reflexión 
            # en simulacion_sucio1:

            plt.plot(tradingtime, precio_final )
            # Tenemos que etiquetar con los valores del tiempo real, así es como se produce
            # la deformación que buscamos!
            plt.title("Grafica con deformación")
            plt.xlabel("Tiempo real")
            plt.show()

            plt.plot([x for x in range(2**kmax)], precio_final)
            plt.title("Grafica sin deformación")
            plt.xlabel("Tiempo real")
            plt.show()

            # plt.plot(np.array([x for x in range(2**kmax - 1)]), [precio_final[i+1] - precio_final[i] for i in range(len(precio_final)-1)] )
            plt.plot(np.array([x for x in range(2**kmax - 1)]), precio_final[1:] - precio_final[:-1], lw = 0.5)
            plt.show()


            plt.plot(tradingtime, precio_final)
            # Tenemos que etiquetar con los valores del tiempo real, así es como se produce
            # la deformación que buscamos!
            plt.title("Grafica sin deformación")
            plt.xlabel("TradingTime")
            plt.show()

            
        if results:
            return tradingtime







    def simulacion(self, n = 1000, resultado = False):
        """
        Esta es la simulación montecarlo de n curvas en contraste con la simulación de 1 curva en la 
        función anterior. 
        """
        almacen_tradingtime = []
        almacen_xt = []
        almacen_precio_final = []

        kmax = self.kmax

        for _ in range(n):
            # Hallamos el trading time sin normalizar.
            tradingtime = super().multifractal_measure_rand(cumsum = True)
            # Normalizado:
            tradingtime =  2**kmax*(tradingtime/np.amax(tradingtime))
            fbm = FractionalBrownianMotion(hurst= self.h1)
            simulacionfbm = fbm._sample_fractional_brownian_motion(2**kmax-1)
            xtsimulados = simulacionfbm
            precio_final = self.Price[-1]*np.exp(xtsimulados)

            almacen_tradingtime.append(tradingtime)
            almacen_xt.append(xtsimulados)
            almacen_precio_final.append(precio_final)


        fig, ax = plt.subplots(figsize=(12, 8))
        

        for i in range(len(almacen_precio_final)):
            ax.plot(almacen_tradingtime[i], almacen_xt[i], lw=0.5, alpha=0.1)

        ax.plot(almacen_tradingtime[n//2], almacen_xt[n//2], lw=0.5, alpha=1, color='b')

        ax.set_xlabel("Tiempo real (días)", fontsize=22)
        ax.set_ylabel(r"X(t)", fontsize=22)
        # ax.set_xlim(0, 8000)
        ax.tick_params(axis='both', labelsize=22)
        ax.grid(True, linestyle='-.')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 8))

        for i in range(len(almacen_tradingtime)):
            ax.plot([x for x in range(2**kmax)], almacen_tradingtime[i], lw=0.5, alpha=0.1)
        ax.plot([x for x in range(2**kmax)], almacen_tradingtime[n//2], lw=0.5, alpha=1, color='b')


        ax.set_xlabel("Tiempo real (días)", fontsize=22)
        ax.set_ylabel("Trading time (días)", fontsize=22)
        # ax.set_ylim(0, 8000)
        ax.tick_params(axis='both', labelsize=22)
        ax.grid(True, linestyle='-.')
        plt.tight_layout()
        plt.show()

        return almacen_tradingtime, almacen_precio_final, almacen_xt











    
    def analizadorprobabilidades(self, dia, almacen_tradingtime, almacen_precio_final):
        # Esta función nos analizará las probabilidades para cada output del método simulacion2.
        # n es el número de simulaciones hechas en simulacion2



        # Calculamos las funciones de densidad de probabilidad para cada tiempo, para ello calculamos la regla de laplace,
        # es decir, vamos a calcular los histogramas para cada tiempo y listo:

        # 1. Calculamos las listas para los histogramas:
        # El principal problema es que como hacemos una composición de funciones, el tiempo asignado a cada precio con un mismo
        # indice i de cada sublista no es el mismo. Dependerá de la función de transición de cada uno de ellos. Por tnato, hacemos
        # una interporlación lieal de cada uno de los pares almacen_precio_final[i] - almacen_tradingtime[i] y trabajamos siendo consecuentes
        # con la composición de funciones:

        # Hallamos los precios para el día concreto (real) siendo consecuentes con la deformación del tiempo con el trading time
        # mediante la interpolación lineal (no se si hay interpolaciones mejores). La interpolación es la mejor forma de hacer 
        # composiciones de funcinoes a nivel computacional:
        n = len(almacen_tradingtime)
        precios_serios = [np.interp(dia, almacen_tradingtime[i], almacen_precio_final[i]) for i in range(n) if np.interp(dia, almacen_tradingtime[i], almacen_precio_final[i])]
        hist, bins = np.histogram(precios_serios, bins=50, density=True)
        media = np.mean(precios_serios)
        stdd = np.std(precios_serios)


        # Generamos una gaussiana con la que comparar:
        x = np.linspace(bins[0]-1, bins[-1]+1, 10000)
        y = stats.norm.pdf(x, media, stdd)


        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(precios_serios, bins=50, alpha=0.5, density=True, edgecolor='black', label = "Distribución real")
        ax.plot(x, y, color = 'red', linewidth = 2, label = "Distribución gaussiana (TLC)")
        textbox_text = f'Media: {media:.2f}\nDesviación estándar: {stdd:.2f}'
        ax.text(0.95, 0.95, textbox_text, transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))



        #ax.set_title(f'Precio en el día {dia} con media {media:.2f} y desviación {stdd:.2f}', fontsize=16)
        ax.set_xlabel(f'P({dia}) ($)', fontsize=22)
        ax.set_ylabel('Densidad de probabilidad', fontsize=22)

        ax.tick_params(axis='both', which='major', labelsize=22)

        # Agregando lineas horizontales
        ax.grid(axis='y')
        plt.legend(loc='center right', fontsize=18)
        plt.tight_layout()
        plt.show()