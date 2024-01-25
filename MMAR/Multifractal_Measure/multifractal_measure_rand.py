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





class Binomnial_Multifractal_Rand:





def multifractal_measure_rand1(kmax, b = 2, m = 5, coef = True, graf1 = False):
    
    """
    Generates a multiplicative cascade for a random measure. This is the most
    general method constructed until now and it is not based in taking advantage
    of the properties of the dyadic intervals. For the dyadic intervals we must
    follow the instructions given in Mandelbrot's paper " Multifractal Measures ".
    
    Parameters:

    kmax = Number of iterations for the multiplicative cascade

    b = number of splits in every iteration (= 2 binomial, = 3 trinomial...)
    ADVERTENCIA: El código todavía no está preparado todavía para b > 2. 

    m = number of points in every interval.

    coef =  Returns a 2-dimensional list with the outputs of the multrifactal
            measure for each k-iterative process and for each interval in that
            k-iterative process. Basically were are defining the image set of the
            function \mu. Notice that this function is a random variable 
            because we don´t now how the weights m0,m1 are distributed along [0,1].

    graf1 = Returns the renormalized (*2^-k) graf of the multifractal measure at kmax. 
            Notice that renormalized graf is the actual density measure. 
    """

    """
    Empezamos definiendo la lista general de variables aleatorias que nos van a ir 
    saliendo en forma de cascada hasta la iteración k. Si queremos trabajar con medidas
    con b > 2 y distribuciones probabilísticas concretas (yo aquí he sugerido la 
    distribución de la función random.random()) lo tenemos que modificar aquí.

    Esta modificación puede ser fundamental para el resultado. Depende de cómo se defina
    la variable aleatoria M_beta (y por tanto de las dstribuciones de probabilidad de 
    m00 y m11) pueden generar significativos cambios en la función densidad de medida. 

    """
    if b == 2:
        lista_general = [] 
        for k in range(kmax):
            lista_coeficientes = []
            for _ in range(b**k):
                m00 = random.random()
                m11 = 1 - m00
                lista_coeficientes.append(np.array([m00,m11]))
            lista_general.append(lista_coeficientes)





    for i in range(len(lista_general) - 1):
        lista_general[i] = [item for sublist in lista_general[i] for item in sublist]
        for j in range(len(lista_general[i])):
            a = lista_general[i][j]* lista_general[i + 1][j]
            lista_general[i + 1][j] = a
    lista_general[-1] = [item for sublist in lista_general[-1] for item in sublist]


    # Por lo que para representar gráficamente las medidas de dichos 
    # intervalos representamos de forma análoga a los casos anteriores:
    if graf1:
        coef = False
        intervalos = [np.linspace(i*b**(-kmax), (i+1)*b**(-kmax), m) for i in range(b**kmax)]
        output = [2**(kmax + 1)*coef* np.ones(m) for coef in lista_general[-1]]
        x = np.array([item for sublist in intervalos for item in sublist])
        y = np.array([item for sublist in output for item in sublist])
        # Creamos un spline a ver cómo se comporta:
        #spl = UnivariateSpline(x, y)
        plt.plot(x, y, linewidth = 0.5)
        plt.title(f'Iteración {kmax}')
        plt.xlim(0,1)
        plt.ylim(0, np.amax(y)+ 0.1)
        plt.show()


    if coef:
        return lista_general















def multifractal_measure_rand2(self, b = 2, m = 1, masas1 = False, masas2 = True, coef = False, graf1 = False, cumsum = False):
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
    

def simulacion(self, grafs = False, results = False):
    
    # kmax es 2^kmax número de días que queremos simular
    kmax = self.kmax

    # Calculamos el trading time normalizado
    # tradingtime = 2**kmax*self.multifractal_measure_rand(kmax, cumsum = True)
    tradingtime = self.multifractal_measure_rand(cumsum = True)
    # Normalizamos y multiplicamos para obtener el número de días.
    tradingtime =  2**kmax*(tradingtime/np.amax(tradingtime))


    # Calculamos el fractional brownian motion fbm:

    # 1. Instanciamos la clase con el objeto fbm
    fbm = fractional_brownian_motion.FractionalBrownianMotion(hurst= self.h1)
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

