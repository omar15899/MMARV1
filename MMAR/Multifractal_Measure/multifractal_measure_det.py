import numpy as np
import pandas as pd
import scipy.optimize
from scipy.interpolate import UnivariateSpline
import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import statsmodels.api as sm


"""
En este modulo se definen la medida multifractal determinista por antonomasia. La medida
binomial determinista. Esta medida se puede encontrar correctamente definida en el paper:

Carl J.G.Eversz, B. B. Mandelbrot. (1992). Multifractal Measures. Chaos and Fractals - Springer-Verlag, New York, Appendix B, pp. 849-881.

Además se presentan una serie de tests acerca del estudio de esta medida, entre los cuales se encuentran los dos métodos más importantes 
para hallar el espectro multifractal f(alpha) de esta medida. Todos estos métodos vienen explicados y justificados en el paper mostrado. 

"""

class Binomial_Multifractal_Det:
        '''
        Parameters
        ----------
        kmax : int
            Maximum iteration number.
        m00 : float
            Coefficient m0.
        m11 : float
            Coefficient m1.
        b : int, optional
            Base of the cascade, default is 2.
        m : int, optional
            Number of points in each interval, default is 5.

        Methods
        -------
        f1() : Theorical multifractal spectrum of the binomial measure. 

        multifractal_measure_det() : Binomial multifractal measure.

        Test0 : Obtendción y convergencia de su espectro multifractal f(alpha) mediante:
            a) Método del histograma: Convergencia más lenta a la función teórica.

        Test1 : Numerical method for obtaining the multifractal spectrum from the method of moments throught
               the partition function (reason of this name can be found in the study of all the states of the
               measure at some coarse-grained state of it).

        Test2: Finds the partition function of the binomial measure. This partition function is crucial
            to obtain the multifractal spectrum f(alpha) by the method of moments. 

        Test3: Finds the Lagrange's transform of the multifractal spectrum and then finds the multifractal 
            spectrum itself. This method shows much better convergence to the theoretical value.

        Test4: Compares the convergence of both methods presented (histogram and moments method) and compares 
            them to the Monte Carlo convergence value. 

        '''

        def __init__(self, kmax, m00, m11, b = 2, m = 5):

            self.kmax = kmax
            self.m00 = m00
            self.m11 = m11
            self.b = b
            self.m = m
            self.measure = self.multifractal_measure_det(coef = True)
            # self.measure_graph = self.multifractal_measure_det(graf = True)

            if self.m00 + self.m11 != 1:
                raise ValueError('WARNING. This multifractal measure does not preserve its mass.')





        def f1(self, alpha):
            a = max(self.m00, self.m11)
            alpha_min = -np.log2(a)
            alpha_max = -np.log2(1 - a)
            return -(alpha_max - alpha)/(alpha_max - alpha_min)*np.log2((alpha_max - alpha)/(alpha_max - alpha_min))-(alpha - alpha_min)/(alpha_max - alpha_min)*np.log2((alpha - alpha_min)/(alpha_max - alpha_min))

        def multifractal_measure_det(self, coef = False, graf = False):
            """
            Parameters
            ----------
            coef : bool, optional
                If True, return coefficients, default is False.
            graf1 : bool, optional
                If True, plot the deterministic multifractal measure, default is False.

            """
            
            # Generación de todos los pesos
            lista_general = []
            for k in range(self.kmax):
                lista_coeficientes = [np.array([self.m00, self.m11])] * (2 ** k)
                lista_general.append(lista_coeficientes)

            # Multiplicación en cascada
            '''
            for i in range(self.kmax - 1):
                lista_general[i] = np.concatenate(lista_general[i])
                for j in range(len(lista_general[i])):
                    lista_general[i + 1][j] *= lista_general[i][j]
            lista_general[-1] = np.concatenate(lista_general[-1])
            '''
            for i in range(self.kmax- 1):
                # Hacemos la lista lista i plana
                lista_general[i] = [item for sublist in lista_general[i] for item in sublist]
                # Multiplicamos a cada uno de sus elementos por una sublista entera de la 
                # siguiente sublista:
                for j in range(len(lista_general[i])):
                    a = lista_general[i][j]* lista_general[i + 1][j]
                    lista_general[i + 1][j] = a
            lista_general[-1] = [item for sublist in lista_general[-1] for item in sublist]
            

            if coef:
                return lista_general

            if graf:
                intervalos = [np.linspace(i*self.b**(-self.kmax), (i+1)*self.b**(-self.kmax), self.m) for i in range(self.b**self.kmax)]
                # Multiplicamos el output por 2**k para conservar la masa y no el valor real de la medida: 
                output = [2**(self.kmax)*coef* np.ones(self.m) for coef in lista_general[-1]]
                x = np.array([item for sublist in intervalos for item in sublist])
                y = np.array([item for sublist in output for item in sublist])
                plt.plot(x, y, linewidth = 0.5)
                plt.title(rf"Medida multifractal determinista renormalizada en un factor $2^k$" + f"\n en la iteración k = {self.kmax}" 
                    + "\ncon " + fr"pesos $m_0$ = {self.m00}, $m_1$ = {self.m11}")
                plt.xlim(0,1)
                plt.ylim(0, np.amax(y)+ 0.1)
                plt.grid(True, linestyle="dashdot")
                plt.savefig('Medida_multifractal_det.png', dpi=300)
                plt.show()


        def Test0(self):
            '''
            Multifractal spectrum with the Histograms method.
            '''
            # Aprovechamos que tenemos todos los valores de las medidas
            # en las iteraciones k previas en lista_general. Por tanto, lo único que se
            # debe hacer es hallar los coeficientes gruesos de Holder de cada elemento 
            # de cada sublista y calcular finalmente el histograma:
            # 1. Creamos la lista con los coeficientes Holder de cada iteración:
            bins1 = self.kmax - 1
            holder = [np.log(np.array(self.measure[i]))/np.log(2**(-(i+1))) for i in range(self.kmax)]
            # 2. Creamos el histograma solo de la última sublista (es la que nos interesa):
            hist, bin_edges = np.histogram(holder[-1], bins = bins1)
            # Calculamos f(alha) como nos muestra en el algoritmo:
            # x = [x + (bin_edges.tolist()[1]- bin_edges.tolist()[0])/2 for x in bin_edges.tolist()] # Realmente es innecesario, pero por hacerlo más bonito. 
            # Probamos analizando alpha en el punto de más de la izquierda:
            x = [x - (bin_edges.tolist()[1]- bin_edges.tolist()[0])/2 for x in bin_edges.tolist()] # Realmente es innecesario, pero por hacerlo más bonito. 
            y = -np.log(hist)/np.log(2**(-(self.kmax-1)))
            # plt.plot(x[1:], y)
            # Creamos el dominio de la función f(alpha) teórica para ser representada. 
            a = max(self.m00, self.m11)
            alpha_min = -np.log2(a)
            alpha_max = -np.log2(1 - a)
            x2 = np.linspace(alpha_min+0.01, alpha_max-0.01, 1000)
            # plt.plot(x2, f1(x2, m00, m11))
            # plt.xlabel('alpha')
            # plt.ylabel('f(alpha)')


            # Fijamos el tamaño de la letra del plot:
            plt.rcParams.update({'font.size': 12})

            #plt.style.use('dark_background')
            #plt.style.use('default')

            # Fijamos el tamaño de la caja del plot:
            plt.rcParams['figure.figsize'] = [8, 8]
            figure, axis = plt.subplots(1)


            axis.plot(x2, self.f1(x2), label = "funcion teórica")
            axis.plot(x[1:], y, label = "funcion numérica")
            axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)
            axis.set_title('f(alpha) experimental vs teórica \npor el método del histograma')
            axis.set_xlabel("alpha")
            axis.set_ylabel("f(alpha)")
            axis.grid(True, linestyle = 'dashdot')
            figure.set_facecolor('#EAEAEA')
            plt.savefig('f_alpha-hist.png', dpi=300)
            plt.show()

            
            print(f'Los valores alpha min y alpha max son respectivamente: {-np.log2(self.m00)}, {-np.log2(self.m11)}. Que como vemos coinciden con los valores experimentales. \n El máximo es {np.amax(y)} y los alphas experimentales son {x[0]} y {x[-1]}')
        
        
        def Test1(self):
            # Vamos a comprobar el apartado (c) del método de los momentos. Vamos a ver si se generan lineas rectas. 

            # Pedimos un intervalo:
            # a = input("Por favor introduzca límite inferior:")
            # b = input("Por favor introduzca límite superior:")
            a = -20
            b = 20
            npuntos = 10
            h = (b - a)/npuntos # Tamaño del intervalo entre dos puntos. Nos servirá para la derivada (su definición normal como el límite), h no puede ser 
            # arbitrariamente pequeño, tiene que ser del tamaño de f(x + h).
            # Estudiamos para un conjunto de q's determinados:
            interval = np.linspace(a, b, npuntos)
            # Creamos una nueva variable con los datos de la lista_general pero con arrays en vez de listas. 
            medidas = [np.array(sublist) for sublist in self.measure]
            # Creammos una lista cuyos elementos son la función de partición para cada q en cada uno de los k
            # hasta llegar a kmax. Esta lista tendrá por tanto muchísimas funciones de partición:
            funciones_particion = [[np.sum(submedidas**q) for submedidas in medidas] for q in interval]
            # funciones_particion = [[np.sum(submedidas**q) for submedidas in medidas] for q in interval]
            # Hallamos la función de partición, que se define (en el límite eps --> 0), como la pendiente de la log/log que se muesta aquí abajo. 
            # En los apuntes se escribe con \sim porque es similar hasta el límite, donde se cumple la igualdad:


            # Fijamos el tamaño de la letra del plot:
            plt.rcParams.update({'font.size': 12})

            #plt.style.use('dark_background')
            #plt.style.use('default')

            # Fijamos el tamaño de la caja del plot:
            plt.rcParams['figure.figsize'] = [8, 8]
            """
            figure, axis = plt.subplots(1)
            for k in range(kmax):
                axis.scatter([np.log(2**(-(k + 1)))]*npuntos, np.log(funciones_particion[k:k+npuntos]), s = 0.5, label = "Test")
            """
            fig, ax = plt.subplots(figsize=(12, 8))
            for i, q in enumerate(interval):
                ax.plot([np.log10(2**((i+1))) for i in range(len(funciones_particion[i]))], np.log10(funciones_particion[i]/funciones_particion[i][0]))


            #axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)
            ax.set_title(r"Función de partición normalizada $log_{10}(S_q(\Delta t))$ vs. $log(\Delta t)$ para diferentes valores q.", fontsize=16, fontweight='bold')
            ax.set_ylabel(r"$log_{10}(S_q(\Delta t))$", fontsize=12, fontweight='bold')
            ax.set_xlabel(r"$log_{10}(\Delta t)$", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle = 'dashdot')

            fig.set_facecolor('#EAEAEA')
            plt.savefig('funciones_parcicion.png', dpi=300)

            plt.show()
        

        def Test2(self):
            # Pedimos un intervalo:
            # a = input("Por favor introduzca límite inferior:")
            # b = input("Por favor introduzca límite superior:")
            a = -5
            b = 5
            npuntos = 10000
            h = (b - a)/npuntos # Tamaño del intervalo entre dos puntos. Nos servirá para la derivada (su definición normal como el límite), h no puede ser 
            # arbitrariamente pequeño, tiene que ser del tamaño de f(x + h).
            # Estudiamos para un conjunto de q's determinados:
            qs = np.linspace(a, b, npuntos)
            deltas = [np.log(2**(-(k+1))) for k in range(self.kmax)]
            # Creamos una nueva variable con los datos de la lista_general pero con arrays en vez de listas. 
            #medidas = [np.array(sublist) for sublist in lista_general]
            medidas = [np.array(sublist) for sublist in self.measure]
            # Creammos una lista cuyos elementos son la función de partición para cada q en el kmax (sumamos las medidas elevadas a la q para la última lista)
            partition_functions = [np.array([np.sum(submedidas**q) for submedidas in medidas]) for q in qs]
            # Ahora normalizadas respecto al primer valor de cada una de las funciones:
            partition_functions_norm = [np.log(funcion/funcion[0]) for funcion in partition_functions]
            # Hallamos la función de partición, que se define (en el límite eps --> 0), como la pendiente de la log/log que se muesta aquí abajo. 
            # En los apuntes se escribe con \sim porque es similar hasta el límite, donde se cumple la igualdad:

            tau = [(funcion[1] - funcion[0])/(deltas[1] - deltas[0]) for funcion in partition_functions_norm]


            # En caso de que nos interesara hacer un análisis más exhasutivo a diferentes eps (en este caso solo nos hemos quedado con el último,
            # deberemos reformular el código, pero no es nada complicado)


            # Vamos a calcular la derivada mediante splines cúbicos por una cuestión de precisión frente a la derivada numérica:
            intervalo1 = np.linspace(qs[0], qs[-1], 1000)
            spline = UnivariateSpline(qs, tau)
            derivada1 = spline.derivative()(intervalo1)
            legendre1 = intervalo1*derivada1 - spline(intervalo1)

            # Calculamos los intervalos de confianza de la forma en la que me ha explicado Juan José:

            # 1. Hallamos todas las taus posibles (todas las lineas verticales y no el promedio de ellas), 
            # para ello hallamos la pendiente para cada una de las tau presuponiendo que nacen desde 0 
            # (esta presuposición es correcta porque se cumple la hipótesis nula y los intervalos de confianza
            # de las curvas ajustadas por OLS contienen a 0), (f(x) - f(0))/(x - 0), que se manifiesta como:




                
            # Ploteamos:

            # plt.style.use('dark_background')
            figure = plt.figure(figsize=(17, 8))

            # Los intervalos de confianza de la pendiente son tan pequeños que no se visualizan.
            ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
            ax1.plot(intervalo1, spline(intervalo1), lw=2, color='blue')
            ax1.set_title(r"Momento $\tau$")
            ax1.set_xlabel("q, que son las pendientes de f(alpha)")
            ax1.set_ylabel(r"$\tau$")
            ax1.grid(True, linestyle='dashdot')
            ax1.set_ylim(-1,3.5)
            ax1.set_xlim(0,4)


            ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
            ax2.plot(intervalo1, derivada1, lw=2, color = "blue")
            ax2.set_title(r"f($\alpha$(q)) obtenida mediante la transformada de legendre\n en función de q.")
            ax2.set_xlabel("q")
            ax2.set_ylabel(r'f($\alpha$(q))')
            ax2.grid(True, linestyle='dashdot')

            ax3 = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
            a = max(self.m00, self.m11)
            alpha_min = -np.log2(a)
            alpha_max = -np.log2(1 - a)
            x2 = np.linspace(alpha_min+0.01, alpha_max-0.01, 1000)
            ax3.plot(x2, self.f1(x2), color = 'red', label = "función teórica", linestyle = "dashdot")
            ax3.plot(derivada1, legendre1, lw=2, color='blue', label = "función experimental")
            ax3.set_ylim(0,1.1)
            ax3.set_title(r"f($\alpha$)")
            ax3.set_xlabel(r"$\alpha$")
            ax3.set_ylabel(r"$\tau$")
            ax3.grid(True, linestyle='dashdot')

            figure.set_facecolor('#EAEAEA')

            custom_lines = [Line2D([0], [0], color='blue', lw=2),
                            Line2D([0], [0], color='green', markerfacecolor='green', markersize=5), # marker='o',
                            Line2D([0], [0], color='red', lw=2)]
            custom_labels = ['Tau', r'f($\alpha$(q))', r'f($\alpha$)']

            legend = plt.legend(custom_lines, custom_labels, loc='lower center', bbox_to_anchor=(-0.1, -0.1), fancybox=True, shadow=True, ncol=3)

            plt.subplots_adjust(wspace=0.5, hspace=0.9)
            plt.show()




        def Test3(self):
            # Definimos la convergencia teórica:
            def theoretical_convergence(x):
                return 1/np.sqrt(2**(x))
            
            # Hacemos un ajuste a ver si los datos encajan en una convergencia del tipo montecarlo (con constante))
            def adjustment(x, c, a):
                return (c/np.sqrt(x))**a
            
            z = self.kmax + 1
            a = 5
            y = []

            for k in range(a, z):
                bins1 = k - 1
                holder = [np.log(np.array(self.measure[i]))/np.log(2**(-(i+1))) for i in range(k)]
                hist, bin_edges = np.histogram(holder[-1], bins = bins1)
                y.append(1 - np.amax(-np.log(hist)/np.log(2**(-(k+1)))))
        
            
            # Hacemos el ajuste:
            xdata = np.array([x for x in range(a, z)])
            popt, pcov = scipy.optimize.curve_fit(adjustment, xdata , y)
            perr = np.sqrt(np.diag(pcov))

            print(f"Los valores para el ajuste a una función (K/np.sqrt(x))**W son K = {popt[0]} y W = {popt[1]}, con los respectivos errores del ajuste: {perr}")
            
            # Ploteamos:
            x = np.linspace(a, z, 1000)
            plt.rcParams.update({'font.size': 12})
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = [8, 8]
            figure, axis = plt.subplots(2)
            axis[0].plot(xdata, adjustment(xdata, *popt), label = "Ajuste al método histograma")
            axis[0].plot(xdata, y, label = "Error método histograma")
            axis[0].plot(xdata, theoretical_convergence(xdata), label = "Error método montecarlo teórico")
            axis[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)
            axis[0].set_title("Diferencia entre el valor experimental y el valor real\n(error relativo del valor máximo experimental frente al máximo real = 1)\n en comparación con el error de la simulación montecarlo", y=1.05)
            axis[0].set_xlabel("Número de iteración K")
            axis[0].set_ylabel("Error relativo")
            axis[0].grid(True, linestyle = 'dashdot')
            #Ploteamos incluso la diferencia:
            axis[1].scatter(xdata, abs(theoretical_convergence(xdata) - y), c = "#39ff14")
            axis[1].set_title("Diferencia entre la evolución de los errores de montecarlo y\n el error del histográma")
            axis[1].set_xlabel("Número de iteración K")
            axis[1].set_ylabel("Diferencia entre las dos funciones ") 
            axis[1].grid(True, linestyle = 'dashdot')
            figure.set_facecolor('#EAEAEA')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace =  2.5)
            plt.subplots_adjust(hspace=0.8)
            plt.show()


