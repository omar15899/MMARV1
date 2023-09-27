import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates




class Graphs():

    def __init__(self, df, tiempo, precio, a = 0, b = 5, npuntos = 20, deltas = np.array([x for x in range(1, 1000)]), kmax = 13):
        """
        tiempo = String. Aquí tengo que meter el nombre de la columna que mide el intervalo temporal
        precio = String. Aquí tengo que meter el nombre de la columa que mide el precio del activo
        
        Esto está hecho para que de un mismo archivo Excel se puedan importar varios precios a la vez
        indicando en todo momento cómo se va a hacer mediante el excel. 

        Vamos a trabajar con objetos nd.array porque estoy más familiarizado con ellos. 

        self.df = nos muestra el dataframe entero del que se ha instanciado el objeto
        self.date = nd.array con los valores temporales en fecha
        self.days = nd.array con los valores temporales en días
        self.Price = nd.array con los valores del precio
        self.X_t = nd.array con los valores de X(t)
        self.intervalo = nd.array con los valores en los que podemos dividir el intervalo en subintervalos exactos. 
        """




        self.df = df
        self.tiempo = tiempo # Nombre de la columna temporal
        self.precio = precio # Nombre de la columna de precios, es interesante en excell llamarla "Lockheed Martin Closing Price"
        self.a = a
        self.b = b
        self.npuntos = npuntos
        self.deltas = deltas
        self.kmax = kmax
        self.date = df[tiempo].to_numpy()
        self.days = np.array([x for x in range(len(self.date))])
        self.Price = df[precio].to_numpy()
        self.X_t = np.log(self.Price) - np.log(self.Price[0])
        self.variacionprecios = self.graf_Price_change(deltat = 1, result = True, graf = False)



    def grafPrice(self):

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.date, self.Price, color='black', linewidth=1.5, label='Precio')

        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        date_fmt = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_fmt)

        ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')
        
        #ax.set_title(f'{self.precio} Histórico de Precios', fontsize=16, fontweight='bold')
        ax.set_ylabel('Precio de Cierre ($)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Tiempo (Horario de Nueva York)', fontsize=22, fontweight='bold')

        ax.legend(loc='upper left', fontsize=12)

        closing_time = '16:00:00'
        ax.tick_params(axis='both', labelsize=22)

        ax.text(self.date[-1], self.Price.max(), f'Closing Time: {closing_time} ET',
                fontsize=12, fontweight='bold', va='bottom', ha='right', color='gray',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

        
        ax.fill_between(self.date, self.Price, where=self.Price >= 0, color="green", alpha=0.4)
        

        plt.tight_layout()
        plt.show()

    def graf_Price_change(self, deltat = 1, result = False, graf = True):

        # Calculamos la variación de precios relativa:

        # 1. Calculamos la variación de precios:
        variacion_precios1 = self.Price[deltat::deltat] - self.Price[:-deltat:deltat]
        
        # 2. Calculamos la media entre los dos precios:
        media = [(self.Price[i] + self.Price[i+1])/2 for i in range(len(self.Price) - 1)]

        # 3. Calculamos la variación de precios relativa:
        variacion_precios = [variacion_precios1[i]/media[i] for i in range(len(variacion_precios1))]

        if graf:
            fig, ax = plt.subplots(figsize=(24,5))
            ax.plot(self.days[:-1] , variacion_precios, linewidth = 0.5)

            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            date_fmt = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(date_fmt)


            ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')
            

            ax.set_title(f'{self.precio} Histórico de Precios', fontsize=16, fontweight='bold')
            ax.set_ylabel('Variación de precios (relativa) ($)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Tiempo (Horario de Nueva York)', fontsize=12, fontweight='bold')

            ax.legend(loc='upper left', fontsize=12)

            
            

            plt.tight_layout()
            plt.show()

        if result:
            return variacion_precios1

    
    def grafX_t(self):
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.date, self.X_t, color='black', linewidth=1.5, label='Price')

        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        date_fmt = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_fmt)


        ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, 
                zorder=0, markevery=1, color='grey')
        

        ax.set_title(f'{self.precio} Histórico de Precios', fontsize=16, fontweight='bold')
        ax.set_ylabel('X_t ($)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tiempo (Horario de Nueva York)', fontsize=12, fontweight='bold')

        ax.legend(loc='upper left', fontsize=12)

        closing_time = '16:00:00'
        ax.text(self.date[-1], self.X_t.max(), f'Closing Time: {closing_time} ET',
                fontsize=12, fontweight='bold', va='bottom', ha='right', color='gray',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))


        fig.set_facecolor('#EAEAEA')

        ax.fill_between(self.date, self.X_t, where=self.X_t >= 0, color="green", alpha=0.4)
        ax.fill_between(self.date, self.X_t, where=self.X_t < 0, color="red", alpha=0.4)

        plt.tight_layout()
        plt.show()
