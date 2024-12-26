import numpy as np
import matplotlib.pyplot as plt
import stochastic

# Nota: Se han eliminado importaciones no utilizadas (por ejemplo, scipy.optimize, statsmodels.api, etc.)
#       Si se requieren para otras partes del proyecto, vuélvelas a introducir.
#       También se ha sustituido 'import random' por 'np.random' para unificar la forma de generar aleatoriedad.

# Nota 2: Hay referencias a una clase o módulo fractional_brownian_motion (p.ej. fractional_brownian_motion.FractionalBrownianMotion)
#         que no está incluido en este snippet. Asumimos que se trata de un módulo propio o de un paquete externo.
#         Aquí mantenemos la invocación, pero ten en cuenta que deberás importar o definir esa clase por tu cuenta.


class BinomialMultifractalRand:
    """
    Clase que contiene métodos para la generación y simulación de medidas multifractales
    basadas en cascadas multiplicativas binomiales (y eventualmente lognormales).
    Los métodos principales son:

    1) multifractal_measure_rand1
    2) multifractal_measure_rand2
    3) simulacion

    Atributos relevantes (ejemplo):
    ------------------------------
    - kmax  : int
        Profundidad máxima de la cascada (número de iteraciones).
    - h1    : float
        Parámetro de Hurst para un posible movimiento Browniano fraccionario.
    - falpha: np.ndarray
        Valores de la función f(α) (exponente de singularidad) para la medida multifractal.
    - derivada : np.ndarray
        Derivada de τ(α) o array de exponentes α en un método de transformada de Legendre.
    - Price : np.ndarray
        Una serie (por ejemplo, de precios) para componer con la simulación.
    """

    @staticmethod
    def multifractal_measure_rand1(
        kmax: int, b: int = 2, m: int = 5, coef: bool = True, graf1: bool = False
    ):
        """
        Genera una medida multifractal (cascada multiplicativa) de tipo binomial
        de profundidad kmax y partición b. Esta versión asume b=2 (binomial).

        Parámetros
        ----------
        kmax : int
            Número de iteraciones para la cascada multiplicativa.
        b : int, opcional (default=2)
            Número de divisiones en cada iteración (b=2 → binomial, b>2 pendiente de implementar).
        m : int, opcional (default=5)
            Número de puntos que se toman en cada subintervalo para visualización.
        coef : bool, opcional (default=True)
            Si es True, la función retorna la lista de coeficientes (la imagen de la medida).
        graf1 : bool, opcional (default=False)
            Si es True, se grafica la medida multifractal renormalizada (densidad) para kmax.

        Retorno
        -------
        - list[list[np.ndarray]] o None
            Si coef es True, se devuelve la lista de coeficientes en cascada.
            Si coef es False o graf1 es True, no devuelve nada (solo grafica).

        Notas
        -----
        - Actualmente solo implementado correctamente para b=2.
        - Si se desea graficar, se muestra la densidad renormalizada 2^(kmax+1)*coef_i
          en cada subintervalo.
        """

        # Por ahora, la implementación solo asume b=2
        if b != 2:
            raise NotImplementedError("Esta función solo está lista para b=2.")

        # Paso 1: Generar la cascada de coeficientes aleatorios
        lista_general = []
        for k in range(kmax):
            lista_coeficientes = []
            for _ in range(b**k):
                m00 = np.random.random()  # cambia a np.random
                m11 = 1.0 - m00
                lista_coeficientes.append(np.array([m00, m11]))
            lista_general.append(lista_coeficientes)

        # Paso 2: Multiplicar en cascada
        for i in range(len(lista_general) - 1):
            # Aplanamos la lista en la iteración i
            lista_general[i] = [x for sublist in lista_general[i] for x in sublist]
            # Multiplicamos los coeficientes correspondientes
            for j in range(len(lista_general[i])):
                producto = lista_general[i][j] * lista_general[i + 1][j]
                lista_general[i + 1][j] = producto

        # Aplanamos la última lista de coeficientes
        lista_general[-1] = [x for sublist in lista_general[-1] for x in sublist]

        # Paso 3: Graficar si se solicita
        if graf1:
            # Para la gráfica no retornamos coef
            coef = False

            # Construimos los subintervalos
            intervalos = [
                np.linspace(i * b ** (-kmax), (i + 1) * b ** (-kmax), m)
                for i in range(b**kmax)
            ]
            # Construimos la densidad medida
            # 2^(kmax+1) * valor => densidad renormalizada
            salida = [2 ** (kmax + 1) * c * np.ones(m) for c in lista_general[-1]]

            x = np.array([p for sublist in intervalos for p in sublist])
            y = np.array([p for sublist in salida for p in sublist])

            plt.plot(x, y, linewidth=0.5)
            plt.title(f"Iteración {kmax}")
            plt.xlim(0, 1)
            plt.ylim(0, np.amax(y) + 0.1)
            plt.show()

        # Paso 4: Retornar los coeficientes (si se desea)
        if coef:
            return lista_general

    def multifractal_measure_rand2(
        self,
        b: int = 2,
        m: int = 1,
        masas1: bool = False,
        masas2: bool = True,
        coef: bool = False,
        graf1: bool = False,
        cumsum: bool = False,
    ):
        """
        Genera una medida multifractal con distribución lognormal base 2, en concordancia
        con ciertos parámetros λ y σ (varianza) derivados de la página 22 de un documento
        (DM/Dollar). Emplea la aproximación de cascada multiplicativa.

        Parámetros
        ----------
        b : int, opcional (default=2)
            Número de divisiones en cada iteración (normalmente binomial).
        m : int, opcional (default=1)
            Número de puntos en cada subintervalo para fines de graficación.
        masas1 : bool, opcional (default=False)
            Si True, usa una forma alternativa de asignar las masas (m00,m11).
        masas2 : bool, opcional (default=True)
            Si True, usa la versión lognormal base 2 para asignar las masas (m00,m11).
        coef : bool, opcional (default=False)
            Si es True, retorna la lista de coeficientes finales, así como λ y la varianza.
        graf1 : bool, opcional (default=False)
            Si es True, grafica la medida multifractal final (no retorna coeficientes).
        cumsum : bool, opcional (default=False)
            Si es True, retorna la suma acumulada de la última iteración (la medida).

        Retorno
        -------
        - (lista_general, lambdas, varianza) si coef es True.
        - np.ndarray (suma acumulada) si cumsum es True (y coef es False).
        - None en caso de solo graficar o no solicitar coef ni cumsum.
        """

        kmax = self.kmax
        # Buscamos la posición donde f(alpha) es máximo
        posicion_max = np.argmax(self.falpha)
        alpha0 = self.derivada[posicion_max]

        # Definimos lambdas y varianza según la teoría lognormal base 2
        lambdas = alpha0 / self.h1
        varianza = 2.0 * (lambdas - 1.0) / np.log(2.0)

        # Función interna para generar la variable lognormal base 2
        def lognormal_base2(lmbda, var):
            normal_rv = np.random.normal(loc=lmbda, scale=np.sqrt(var))
            return 2.0 ** (-normal_rv)

        # Lista principal de coeficientes en cascada
        lista_general = []
        for k in range(kmax):
            lista_coeficientes = []
            for _ in range(2**k):
                if masas1:
                    # Caso alternativo (masas1)
                    # Forma normalizada a partir de lognormal base 2
                    m00 = (lambda x: x / (x + 1.0 / x))(
                        2.0 ** -np.random.normal(lambdas, np.sqrt(varianza))
                    )
                    m11 = 1.0 - m00
                elif masas2:
                    # Caso lognormal base 2
                    m00 = lognormal_base2(lambdas, varianza)
                    m11 = lognormal_base2(lambdas, varianza)
                lista_coeficientes.append(np.array([m00, m11]))
            lista_general.append(lista_coeficientes)

        # Multiplicación en cascada y flatten
        for i in range(len(lista_general) - 1):
            lista_general[i] = [x for sublist in lista_general[i] for x in sublist]
            for j in range(len(lista_general[i])):
                producto = lista_general[i][j] * lista_general[i + 1][j]
                lista_general[i + 1][j] = producto

        # Convertimos a array para la última capa
        lista_general[-1] = np.array(
            [x for sublist in lista_general[-1] for x in sublist]
        )

        # Normalización adicional con una variable aleatoria 'omega'
        # (Explicada en los apuntes del autor)
        media_omega = 1.0 / (2.0 ** (-lambdas * kmax))
        omega = np.random.normal(media_omega, np.sqrt(varianza), 2**kmax)
        lista_general[-1] *= omega

        # Graficar la medida multifractal final
        if graf1:
            coef = False
            intervalos = [
                np.linspace(i * b ** (-kmax), (i + 1) * b ** (-kmax), m)
                for i in range(b**kmax)
            ]
            salida = [c * np.ones(m) for c in lista_general[-1]]
            x = np.array([p for sublist in intervalos for p in sublist])
            y = np.array([p for sublist in salida for p in sublist])

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x, y, linewidth=0.8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, np.amax(y) + 0.1 * np.amax(y)])
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(True)
            ax.set_title("Medida multifractal (lognormal base 2)")
            plt.show()

        # Retorno de coeficientes o cumsum
        if coef:
            return lista_general, lambdas, varianza
        if cumsum:
            return np.cumsum(lista_general[-1])

    def simulacion(self, grafs: bool = False, results: bool = False):
        """
        Simulación de un "trading time" deformado a partir de la medida multifractal
        y combinación con un movimiento Browniano fraccionario.

        Parámetros
        ----------
        grafs : bool, opcional (default=False)
            Si True, muestra diversas gráficas de la simulación.
        results : bool, opcional (default=False)
            Si True, retorna la serie 'tradingtime' resultante.

        Retorno
        -------
        - np.ndarray si results=True, con la deformación del tiempo simulada.
        - None si results=False (solo realiza las gráficas si grafs=True).
        """
        kmax = self.kmax

        # Obtenemos el 'trading time' normalizado a partir de la medida multifractal
        tradingtime = self.multifractal_measure_rand2(cumsum=True)
        tradingtime = 2**kmax * (tradingtime / np.amax(tradingtime))

        # Movimiento Browniano fraccionario
        fbm = stochastic.FractionalBrownianMotion(hurst=self.h1)
        simulacionfbm = fbm._sample_fractional_brownian_motion(2**kmax - 1)
        xtsimulados = simulacionfbm

        precio_final = self.Price[0] * np.exp(xtsimulados)

        # Gráficas
        if grafs:
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(2**kmax), tradingtime)
            plt.title("Tiempo de 'trading' (multifractal) normalizado")
            plt.xlabel("Días (índice)")
            plt.ylabel("TradingTime")
            plt.grid(True)
            plt.show()

            # Gráfica composición: tradingtime vs precio_final
            plt.figure(figsize=(10, 4))
            plt.plot(tradingtime, precio_final)
            plt.title("Precio con deformación multifractal del tiempo")
            plt.xlabel("TradingTime")
            plt.ylabel("Precio")
            plt.grid(True)
            plt.show()

            # Gráfica sin deformación (eje x = tiempo 'real')
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(2**kmax), precio_final)
            plt.title("Precio sin deformación (tiempo real)")
            plt.xlabel("Días")
            plt.ylabel("Precio")
            plt.grid(True)
            plt.show()

            # Diferencias del precio
            plt.figure(figsize=(10, 4))
            plt.plot(
                np.arange(2**kmax - 1), precio_final[1:] - precio_final[:-1], lw=0.5
            )
            plt.title("Incrementos de precio")
            plt.xlabel("Días")
            plt.ylabel("Incremento")
            plt.grid(True)
            plt.show()

            # Repetimos la gráfica de tradingtime vs precio_final, con título distinto
            plt.figure(figsize=(10, 4))
            plt.plot(tradingtime, precio_final)
            plt.title("Precio vs TradingTime (repetición)")
            plt.xlabel("TradingTime")
            plt.ylabel("Precio")
            plt.grid(True)
            plt.show()

        if results:
            return tradingtime
