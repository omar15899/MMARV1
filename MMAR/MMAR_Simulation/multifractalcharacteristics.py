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

    def __init__(self, df, tiempo, precio, a=0, b=5, npuntos=20, deltas=np.array([x for x in range(1, 1000)]), kmax=13):
        # Heredamos los atributos de Graphs. 
        super().__init__(df, tiempo, precio, a=0, b=5, npuntos=20, deltas=np.array([x for x in range(1, 1000)]), kmax=13)
        self.partition = self.Partition_functions(graf1=False, resultado=True)
        self.tau = self.tauandalpha(graf=False, resultados=True)[0]
        self.falpha = self.tauandalpha(graf=False, resultados=True)[1]
        self.derivada = self.tauandalpha(graf=False, resultados=True)[2]
        self.h1 = self.Hurst(graf=False)
        self.posicion_max = np.argmax(self.falpha)
        self.alpha0 = self.derivada[self.posicion_max]
        self.lambdas = self.alpha0 / self.h1
        self.varianza = 2 * (self.lambdas - 1) / np.log(2)




    def Partition_functions(self, alpha = 0.05, graf1 = True, resultado = False):

        """
        alpha = nivel de significancia
        qs = list. Son los momentos q que queremos calcular
        deltas = list. son los intervalos temporales que queremos analizar.
        a,b,c = limites y paso de las qs que queremos calcular. 
        """
        

        a = self.a
        b = self.b 
        npuntos = self.npuntos
        deltas = self.deltas
        ldeltas = len(deltas) # Serán los números de puntos que tenga cada "recta" de partición para distintos momentos. 
        mdeltas = np.mean(deltas)
        vardeltas = np.var(deltas)
        qs = np.linspace(a, b, npuntos)

        # Calculamos las funciones de partición, calculamos una para cada deltat y cada q, así, cada punto de la gráfica que representemos
        # será una función de partición concreta, una recta serán ldeltas funciones de paricion distintas asociadas a un mismo 1
        # acto seguido llevamos a cabo la regresión OLS.

        # Otra forma sería generar una matriz donde cada fila es para una q y cada columna es un deltat. Si queremos hallar esto pues transponemos. 

        # La lista partition_functions tiene subsilistas, cada elemento de una de esas sublistas es una función de partición para un delta t determinado
        # y un q determinado, por lo que estamos ploteando muchas funciones de partición y las agrupamos por q's. Una vez agrupados estudiamos la
        # media mediante OLS, porque es la condición de la esperanza matemática.

        partition_functions = [[np.sum((abs(self.X_t[deltat::deltat] - self.X_t[:-deltat:deltat]))**q) for deltat in deltas] for q in qs]
        adjustment_part_functions = [np.polyfit(np.log10(deltas), np.log10(partition_functions[i]/partition_functions[i][0]), 1) for i in range(npuntos)]
        coeficiente_normalizador = [adjustment_part_functions[i][1] for i in range(npuntos)]

        # Para calcular el intervalo de confianza (que para nosotros va a ser el error) se hará utilziando el método de la pg 55
        # del T7 de estadística y análisis de datos. Hallamos por tanto los intervalos de confianza para el valor medio.
        # 1. Calculamos las varianzas residuales (cuidado que hay que trabajar con los logaritmos):

        srs = [(np.sum((np.log10(partition_functions[i]/partition_functions[i][0]) - np.poly1d(adjustment_part_functions[i])(np.log10(deltas)))**2)/(ldeltas - 2))**(0.5) for i in range(npuntos)]

        # 2. Calculamos los intervalos de confianza sobre la estimación de la media, en este caso el conjunto dominio es deltas.

        intervalos_conf = [[t.ppf(alpha/2, ldeltas - 2)*srs[i]*np.sqrt(1/ldeltas + (deltas[j] -  mdeltas)**2/((ldeltas - 1)*vardeltas)) for j in range(ldeltas)] for i in range(npuntos)]

        # 3. Con estos datos podemos calcular los intervalos de confianza de la ordenada en el origen y la pendiente. Nos interesa
        # conocer este intervalo porque tauq es la pendiente de la regresión, por lo que el error asociado a tau es justamente ese:

        intervalos_conf_ordenada = [t.ppf(alpha/2, ldeltas - 2)*srs[i]*np.sqrt(1/ldeltas + (mdeltas)**2/((ldeltas - 1)*vardeltas)) for i in range(npuntos)]
        intervalos_conf_pendiente = [t.ppf(alpha/2, ldeltas - 2)*srs[i]/(np.sqrt(ldeltas - 1)*vardeltas) for i in range(npuntos)]

        # 4. Hacemos una prueba de correlación (equivalente al cor.test) en R (toda la teoría en los apuntes de estadística). La primera lista
        # tendrá los valores de las correlaciones y la segunda los p-value:

        corrtest1 = [stats.pearsonr(np.log10(deltas), np.log10(partition_functions[i]/partition_functions[i][0]))[0] for i in range(npuntos)]
        corrtest2 = np.array([stats.pearsonr(np.log10(deltas), np.log10(partition_functions[i]/partition_functions[i][0]))[1] for i in range(npuntos)])
        qslim = qs[corrtest2 <= alpha]

        
        if graf1:
            # Los valores de corrtest2 son el p-value. El p-value no es más que la probabilidad de que la variable aleatoria (estimador estadístico)
            # obtenga un valor más extremal que el obtenido en el estimador en caso de confirmarse la hipótesis nula (en este caso la hipótesis nula
            # es que la correlación sea 0). Fijamos el intervalo de confinaza al 95% (viene especificado por la variable alpha de este método).
            # 1. Calculamos la lista de los q tal que su p-value es mayor que el nivel de significancia:
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            
            colors = ["#FF0000", "#00FF00", "#0000FF", "#000000", "#FF00FF", "#00FFFF", "#FFA500", "#800080", "#008000", "#800000", "#008080", "#000080", "#FFFFE0", "#FFD700", "#00FF7F", "#FF4500", "#9400D3", "#808000", "#FF1493", "#6A5ACD"]

            for i, q in enumerate(qslim):
                ax.plot(np.log10(deltas), np.poly1d([q/2-1,0])(np.log10(deltas)), linestyle="-.", color = "black")
                ax.plot(np.log10(deltas), np.log10(partition_functions[i]/partition_functions[i][0]), 
                        label=f"q = {q:.2f}", color=colors[i % len(colors)])
                ax.plot(np.log10(deltas), np.poly1d(adjustment_part_functions[i])(np.log10(deltas)), color=colors[i % len(colors)])    
                ax.fill_between(np.log10(deltas), np.poly1d(adjustment_part_functions[i])(np.log10(deltas)) - intervalos_conf[i],  
                                np.poly1d(adjustment_part_functions[i])(np.log10(deltas)) + intervalos_conf[i], 
                                color=colors[i % len(colors)], alpha = 0.4)
        
            ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')

            # Añadimos títulos 
            #ax.set_title(r"Función de partición normalizada $log_{10}(S_q(\Delta t))$ vs. $log(\Delta t)$ para diferentes valores q." +"\n Curvas ajustadas por OLS:", fontsize=16, fontweight='bold')
            ax.set_ylabel(r"$log_{10}(\hat{S}_q(\Delta t))$", fontsize=15, fontweight='bold')
            ax.set_xlabel(r"$log_{10}(\Delta t)$", fontsize=15, fontweight='bold')
            #ax.set_ylim(-3.5,2.5)

            # Añadimos leyenda:
            ax.legend(loc='best', fancybox=True, shadow=True, ncol=5, fontsize=12)
            #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
            #fancybox=True, shadow=True, ncol=5, fontsize=12)
            

            plt.tight_layout()

            plt.subplots_adjust(bottom=0.2)
            
            plt.show()




            # Gragicamos las funciones de partición sin normalizar. 

            fig, ax = plt.subplots(figsize=(12, 8))
            
            
            colors = ["#FF0000", "#00FF00", "#0000FF", "#000000", "#FF00FF", "#00FFFF", "#FFA500", "#800080", "#008000", "#800000", "#008080", "#000080", "#FFFFE0", "#FFD700", "#00FF7F", "#FF4500", "#9400D3", "#808000", "#FF1493", "#6A5ACD"]

            for i, q in enumerate(qslim):
                ax.plot(np.log10(deltas), np.log10(partition_functions[i]), label=f"q = {q:.2f}", color=colors[i % len(colors)])        
                 # En las gráficas se observan mayor concentración de puntos cuanto mayor valor de x, es normal porque la distribución es homogénea y
                # al estar en escala logaritmica se aglomeran los puntos en los valores de más a la derecha:
                 
        
            ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')

            # Añadimos títulos 
            ax.set_title(r"Función de partición normalizada $log_{10}(S_q(\Delta t))$ vs. $log(\Delta t)$ para diferentes valores q." +"\n Curvas ajustadas por OLS:", fontsize=16, fontweight='bold')
            ax.set_ylabel(r"$log_{10}(S_q(\Delta t))$", fontsize=12, fontweight='bold')
            ax.set_xlabel(r"$log_{10}(\Delta t)$", fontsize=12, fontweight='bold')
            ax.set_ylim(-3.5,5)

            # Añadimos leyenda:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=5, fontsize=12)
            
            fig.set_facecolor('#EAEAEA')

            plt.tight_layout()

            plt.subplots_adjust(bottom=0.2)
            
            plt.show()

        
            ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')

            # Añadimos títulos 

            ax.set_ylabel(r"$\frac{log_{10}(S_q(\Delta t))}{log_{10}(\Delta t)}$", fontsize=18, fontweight='bold')
            ax.set_xlabel(r"$log_{10}(\Delta t)$", fontsize=18, fontweight='bold')
            ax.set_ylim(-3.5,5)

            # Añadimos leyenda:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=5, fontsize=12)
            
            fig.set_facecolor('#EAEAEA')

            plt.tight_layout()

            plt.subplots_adjust(bottom=0.2)
            
            plt.show()

        if resultado: 
            print(f"Los valores de los ajustes (a,m respectivame) son {adjustment_part_functions}" +
                  f"\nLos errores asociados a la ordenada (término independiente) vienen dados por {intervalos_conf_ordenada}")
            
            # print(f"Las magnitudes de los intervalos de confianza son: {intervalos_conf}")

            print(f"La hipótesis nula se descarta hasta el valor q = {qslim[-1]}.")
            print(f"El estimador coeficiente de correlación para cada q tiene como valores: {corrtest1}" +
                  f"\nMientras que los p-values asociados al estimador son: {corrtest2}")
            # Instanciamos los objetos fig, ax de la clase subplots.
            return partition_functions, adjustment_part_functions 
            






    def tauandalpha(self, alpha = 0.05, graf = True, resultados = False):
        plt.style.use('default')
        a = self.a
        b = self.b 
        npuntos = self.npuntos
        deltas = self.deltas
        ldeltas = len(deltas) # Serán los números de puntos que tenga cada "recta" de partición para distintos momentos. 
        mdeltas = np.mean(deltas)
        vardeltas = np.var(deltas)
        qs = np.linspace(a, b, npuntos)

        # Calculamos las funciones de partición:

        partition_functions = [[np.sum((abs(self.X_t[deltat::deltat] - self.X_t[:-deltat:deltat]))**q) for deltat in deltas] for q in qs]
        adjustment_part_functions = [np.polyfit(np.log10(deltas), np.log10(partition_functions[i]/partition_functions[i][0]), 1) for i in range(npuntos)]
        coeficiente_normalizador = [adjustment_part_functions[i][1] for i in range(npuntos)]


        # De nuevo, los intervalos de confianza:
        srs = [(np.sum((np.log10(partition_functions[i]) - np.poly1d(adjustment_part_functions[i])(np.log10(deltas)))**2)/(ldeltas - 2))**(0.5) for i in range(npuntos)]
        intervalos_conf = [[t.ppf(alpha/2, ldeltas - 2)*srs[i]*np.sqrt(1/ldeltas + (deltas[j] -  mdeltas)**2/((ldeltas - 1)*vardeltas)) for j in range(ldeltas)] for i in range(npuntos)]
        intervalos_conf_ordenada = [t.ppf(alpha/2, ldeltas - 2)*srs[i]*np.sqrt(1/ldeltas + (mdeltas)**2/((ldeltas - 1)*vardeltas)) for i in range(npuntos)]
        intervalos_conf_pendiente = [t.ppf(alpha/2, ldeltas - 2)*srs[i]/(np.sqrt(ldeltas - 1)*vardeltas) for i in range(npuntos)]

        
        # Calculamos tau como en el caso de la medida binomial (mirar Simulacion_sucio1 para entender mejor todo, que lo tengo bien redactado)
        tau = [sublistasajustes[0] for sublistasajustes in adjustment_part_functions]
        h = (b-a)/npuntos # distancia de puntos que tenemos en qs, como en el caso binomial
        # Vamos a calcular la deerivada mediante splines:
        intervalo1 = np.linspace(qs[0], qs[-1], 1000)
        spline = UnivariateSpline(qs, tau)
        derivada1 = spline.derivative()(intervalo1)
        legendre1 = intervalo1*derivada1 - spline(intervalo1)

        # Calculamos los intervalos de confianza para la función de escala \tau:

        # 1. Hallamos todas las taus posibles (todas las lineas verticales y no el promedio de ellas), 
        # para ello hallamos la pendiente para cada una de las tau presuponiendo que nacen desde 0 
        # (esta presuposición es correcta porque se cumple la hipótesis nula y los intervalos de confianza
        # de las curvas ajustadas por OLS contienen a 0), (f(x) - f(0))/(x - 0), que se manifiesta como:

        tausposibles = [[np.log10(partition_functions[i][j]/partition_functions[i][0])/np.log10(deltas[j]) for i in range(npuntos)] for j in range(1, ldeltas)] # Hacemos que vaya desde 1 porque ldeltas[0] = 1 y log(1) = 0 y si dividimos tenemos problemas.
        splinesposibles = [UnivariateSpline(qs, taus) for taus in tausposibles]
        lengendreposbiles = [intervalo1*splines.derivative()(intervalo1) - splines(intervalo1) for splines in splinesposibles]


        if graf:
            
    
            # Ploteamos:
            
            # plt.style.use('dark_background')
            figure = plt.figure(figsize=(17, 8))

            # Los intervalos de confianza de la pendiente son tan pequeños que no se visualizan.
            ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
            
            for i in range(ldeltas - 1):
                ax1.plot(intervalo1, splinesposibles[i](intervalo1), alpha = 0.1, color = "grey")
            
            ax1.plot(intervalo1, spline(intervalo1), lw=2, color='blue')
            ax1.fill_between(qs, np.array(tau) - np.array(intervalos_conf_pendiente), np.array(tau) + np.array(intervalos_conf_pendiente), color = "red")
            # ax1.fill_between(qs, np.array(tau) - np.array(intervalos_conf_pendiente), np.array(tau) + np.array(intervalos_conf_pendiente), color = "red")
            ax1.set_title(r"a)", fontsize=18)
            ax1.set_xlabel("q", fontsize=18)
            ax1.set_ylabel(r"$\hat{\tau}$", fontsize=18)
            ax1.grid(True, linestyle='dashdot')
            ax1.set_ylim(-1,1)
            ax1.set_xlim(0,qs[-1])


            ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
            
            for i in range(ldeltas - 1):
                ax2.plot(intervalo1, splinesposibles[i].derivative()(intervalo1), alpha = 0.1, color = "grey")
            
            ax2.plot(intervalo1, derivada1, lw=2, color = "blue")
            ax2.set_title(r"c)", fontsize=18)
            ax2.set_xlabel("q", fontsize=18)
            ax2.set_ylabel(r'$\hat{f}$($\alpha$(q))', fontsize=18)

            #ax2.set_xlim(0,qs[-1])
            ax2.grid(True, linestyle='dashdot')

            ax3 = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
            
            for i in range(ldeltas - 1):
                ax3.plot(splinesposibles[i].derivative()(intervalo1), lengendreposbiles[i], alpha = 0.1, color = "grey")
            
            ax3.plot(derivada1, legendre1, lw=2, color='red')
            ax3.set_ylim(0.5,1.05)
            ax3.set_title(r"b)", fontsize=18)
            ax3.set_xlabel(r"$\alpha$", fontsize=18)
            ax3.set_ylabel(r'$\hat{f}(\alpha)$', fontsize=18)
            ax3.grid(True, linestyle='dashdot')


            custom_lines = [Line2D([0], [0], color='blue', lw=2),
                            Line2D([0], [0], color='green', markerfacecolor='green', markersize=5), # marker='o',
                            Line2D([0], [0], color='red', lw=2)]
            custom_labels = ['Tau', r'f($\alpha$(q))', r'f($\alpha$)']

            #legend = plt.legend(custom_lines, custom_labels, loc='lower center', bbox_to_anchor=(-0.1, -0.1), fancybox=True, shadow=True, ncol=3)
            ax1.tick_params(axis='both', which='major', labelsize=17)
            ax2.tick_params(axis='both', which='major', labelsize=17)
            ax3.tick_params(axis='both', which='major', labelsize=17)
            plt.subplots_adjust(wspace=0.5, hspace=0.9)
            plt.show()
            

        if resultados:
            print(f"Los intervalos de confinaza son $\pm${intervalos_conf_pendiente}")
            return spline(intervalo1), legendre1, derivada1




    def Hurst(self, graf = False):

        # Procedemos a calcular el coeficiente Hurst. 

        a = self.a
        b = self.b 
        npuntos = self.npuntos
        deltas = self.deltas
        qs = np.linspace(a, b, npuntos)
        # Calculamos las funciones de particion y sus regresiones lineales OLS:
        partition_functions = [[np.sum((abs(self.X_t[deltat::deltat] - self.X_t[:-deltat:deltat]))**q) for deltat in deltas] for q in qs]
        adjustment_part_functions = [np.polyfit(np.log10(deltas), np.log10(partition_functions[i]), 1) for i in range(len(qs))]

        # Calculamos tau como en el caso de la medida binomial determinista:
        tau = [sublistasajustes[0] for sublistasajustes in adjustment_part_functions]
        h = (b-a)/npuntos 
        intervalo1 = np.linspace(qs[0], qs[-1], 1000)
        spline = UnivariateSpline(qs, tau)
        derivada = spline.derivative()(intervalo1)
        legendre = intervalo1*derivada - spline(intervalo1)



        if graf:
            plt.style.use('default')
            # Comparamos con el polinomio original visualmente:
            figure = plt.figure(figsize=(12, 6))

            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(qs, tau, lw=2, color='r',ls = "-.")
            ax1.plot(qs, p(qs), lw=2, color='green')
            ax1.set_title("Momento tau_p")
            ax1.set_xlabel("q, que son las pendientes de f(alpha)")
            ax1.set_ylabel("Tau")
            ax1.grid(True, linestyle='dashdot')

            plt.show()


        # Hallamos la unica solución definida en el intervalo en el que 
        # hemos interpolado el spline (se ve en la gráfica que tau solo
        # tiene una raiz en ese intervalo):
        sol1 = spline.roots()[0]

        # Inveritmos los coeficiente Hurst será:
        h1 = 1/sol1

        return h1






    def multifractal_measure_rand(self, b = 2, m = 1, masas1 = False, masas2 = True, coef = False, graf1 = False, cumsum = False):
        """
        Medida multifractal que cumple con los parámetros lambda y sigma de la distribución lognormal
        de la pagina 22 del DM/Dollar. Para mas info mirar Simulacion_sucio1:
        masas: nos muestran la forma de generar las variables aleatorias M. 
        """
        kmax = self.kmax
        # Puesto que ya hemos calculado previamente H, solo debemos calcular alpha_0, esta 
        # se define como el valor del dominio alpha donde f(alpha) alcanza el máximo, 
        # esto lo podemos encontrar mediante la función np.argmax(), que nos da la posición
        # en la lista del máximo indice:
        posicion_max = np.argmax(self.falpha)
        # Buscamos en el dominio que, debido a que estamos trabajando con la transformada de
        # legendre, se trata de las alphas, que son las derivadas de tau, es decir, sus pendientes:
        alpha0 = self.derivada[posicion_max]

        lambdas = alpha0/self.h1
        varianza = 2*(lambdas - 1)/np.log(2)

        # Generamos la medida multifractal que conserva la masa en promedio, por lo que necesitamos la omega. La omega tendrá
        # en nuestro caso una distribución normal con media centrada en aquel valor x tal que x*lambda*lambda = 1, en este caso:

        # media_omega = 1/(lambdas**2)

        # Generamos la variable aleatoria lognormal:
        def lognormal_base2(lambdas, varianza):
            # Generamos la variable aleatoria normal:
            normal_random_variable = np.random.normal(loc=lambdas, scale=np.sqrt(varianza))

            # Creamos la variable aleatoria lognormal:
            lognormal_base2var = 2 ** -normal_random_variable

            return lognormal_base2var


        # Generamos la lista de todas las variables aleatorias M_b a lo largo de la cascada:

        lista_general = [] 
        for k in range(kmax):
            lista_coeficientes = []
            for _ in range(2**k):
                if masas1:
                    masas2 = False
                    m00 = (lambda x: x / (x + (1 / x)))(2 ** -np.random.normal(lambdas, np.sqrt(varianza)))
                    m11 = 1 - m00

                if masas2:
                    m00 = lognormal_base2(lambdas, varianza)
                    m11 = lognormal_base2(lambdas, varianza)
                    

                lista_coeficientes.append(np.array([m00,m11]))
            lista_general.append(lista_coeficientes)

        # Multiplicamos en cascada y obtenemos la imagen para cada intervalo en la iteración k-ésima.
        # Además, en el proceso convertimos la lista en una lista bidimensional.  

        for i in range(len(lista_general) - 1):
            lista_general[i] = [item for sublist in lista_general[i] for item in sublist]
            for j in range(len(lista_general[i])):
                a = lista_general[i][j]* lista_general[i + 1][j]
                lista_general[i + 1][j] = a
        lista_general[-1] = np.array([item for sublist in lista_general[-1] for item in sublist])

        """
        # Normalizamos calulando la integral de riemann. Como estamos en un caso discreto lo único que se 
        # debe hacer es el sumatorio de riemann, sabiendo que los intervalos tienen l = 2^{-k +1} y aprovechando
        # las propiedades de los objetos ndarrays:

        constante_norm = np.sum(lista_general[-1]*2**(-(kmax + 1)))

        # Multiplicamos todos los valores imagen de la última iteración de la medida por la constante:
        lista_general[-1]/= constante_norm
        lista_general[-1].tolist()
        """

        # Procedemos a normalizar con el uso de la variable aleatoria Omega con la desviación típica de M para simplificar
        # (mirar mi cuadernito de apuntes para entender cómo derivo esta ecuación):

        media_omega = 1/(2**(-lambdas*kmax))
        omega = np.random.normal(media_omega, np.sqrt(varianza), 2**kmax)

        lista_general[-1] *= omega
        

        # Graficamos los resultados:

        if graf1:
            coef = False
            intervalos = [np.linspace(i*b**(-kmax), (i+1)*b**(-kmax), m) for i in range(b**kmax)]
            # output = [2**(kmax + 1)*coef* np.ones(m) for coef in lista_general[-1]] (para el caso sin normalizar con omega)
            output = [coef* np.ones(m) for coef in lista_general[-1]]
            x = np.array([item for sublist in intervalos for item in sublist])
            y = np.array([item for sublist in output for item in sublist])
            
            fig, ax = plt.subplots(figsize=(60,5))
            ax.plot(x, y, linewidth=0.8)
            #ax.set_title(f'Iteración {kmax}')  # use 'set_title' instead of 'title'
            ax.set_xlim([0, 1])  # use square brackets to set limits
            ax.set_ylim([0, np.amax(y) + 0.1*np.amax(y)])
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.grid(True)  # add gridlines
            plt.show()

        # Mostramos el valor de salida de cada uno de los intervalos:

        if coef:
            return lista_general, lambdas, varianza
        
        if cumsum:
            return np.cumsum(lista_general[-1])
