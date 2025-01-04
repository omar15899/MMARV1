import numpy as np
import matplotlib.pyplot as plt
import stochastic


class BinomialMultifractalRand:
    """
    This class contains methods to generate and simulate multifractal measures
    based on binomial (or lognormal) multiplicative cascades.

    Basic attributes (used in larger projects):
    --------------------------------------------
    - kmax  : int
        Maximum depth of the cascade (number of iterations).
    - h1    : float
        Hurst exponent for potential fractional Brownian motion.
    - falpha: np.ndarray
        Array with values of the function f(α) (singularity exponent).
    - derivada : np.ndarray
        Array with the derivative of τ(q), which usually gives α(q).
    - Price : np.ndarray
        Price series used to combine with the simulation.
    """

    @staticmethod
    def multifractal_measure_rand1(
        kmax: int, b: int = 2, m: int = 5, coef: bool = True, graf1: bool = False
    ):
        """
        Generates a random binomial multiplicative cascade of depth kmax.

        Parameters
        ----------
        kmax : int
            Number of iterations for the cascade.
        b : int, optional (default=2)
            Number of divisions (b=2 => binomial cascade).
        m : int, optional (default=5)
            Points per subinterval for plotting.
        coef : bool, optional (default=True)
            If True, returns the final list of coefficients.
        graf1 : bool, optional (default=False)
            If True, plots the final cascade result.

        Returns
        -------
        list[list[np.ndarray]] or None
            If coef=True, returns the list with the final cascade.
            If coef=False, nothing is returned (only plotted if graf1 is True).
        """
        if b != 2:
            raise NotImplementedError("Only implemented for b=2 (binomial).")

        # 1) Build the cascade
        lista_general = []  # This will hold all levels of the cascade
        for k in range(kmax):
            fila_coefs = []  # Coefficients at level k
            for _ in range(b**k):  # b^k intervals at level k
                m00 = np.random.random()  # Random weight for first part
                m11 = 1.0 - m00  # Complement weight for second part
                fila_coefs.append(np.array([m00, m11]))  # Add both weights
            lista_general.append(fila_coefs)  # Add this level to the cascade

        # 2) Multiply the coefficients across levels
        for i in range(len(lista_general) - 1):
            lista_general[i] = [
                item for sublist in lista_general[i] for item in sublist
            ]
            for j in range(len(lista_general[i])):
                # Multiply coefficients from this level with the next one
                lista_general[i + 1][j] = lista_general[i][j] * lista_general[i + 1][j]

        # Flatten the final level
        lista_general[-1] = [x for sublist in lista_general[-1] for x in sublist]

        # 3) Plot if requested
        if graf1:
            coef = False  # Do not return coefficients if plotting
            intervalos = [
                np.linspace(i * b**-kmax, (i + 1) * b**-kmax, m) for i in range(b**kmax)
            ]
            # Scale the output to show the density
            salida = [2.0 ** (kmax + 1) * c * np.ones(m) for c in lista_general[-1]]

            x = np.array([p for sublist in intervalos for p in sublist])
            y = np.array([p for sublist in salida for p in sublist])

            # Plot the cascade
            plt.plot(x, y, linewidth=0.5, color="blue")
            plt.title(f"Random Binomial Cascade: kmax={kmax}")
            plt.xlim(0, 1)
            plt.ylim(0, np.amax(y) + 0.1 * np.amax(y))  # Add some margin above
            plt.grid(True)
            plt.show()

        if coef:
            return lista_general  # Return the coefficients if requested

    def multifractal_measure_rand2(
        self,
        b: int,
        m: int,
        kmax: int,
        falpha: np.ndarray,
        derivada: np.ndarray,
        h1: float,
        Price: np.ndarray,
        masas1: bool = False,
        masas2: bool = True,
        coef: bool = False,
        graf1: bool = False,
        cumsum: bool = False,
    ):
        """
        Generates a multifractal measure using a base-2 lognormal distribution
        with parameters derived from α₀ = derivada[posicion_max] and
        λ=α₀/h₁, var=2*(λ-1)/ln(2).

        Parameters
        ----------
        b : int
            Number of divisions at each iteration.
        m : int
            Points in each subinterval for plotting.
        kmax : int
            Depth of the cascade.
        falpha : np.ndarray
            Array with the function f(α).
        derivada : np.ndarray
            Array with α(q).
        h1 : float
            Hurst exponent.
        Price : np.ndarray
            Price series for combining with FBM simulation later.
        masas1 : bool
            Alternative method for assigning weights.
        masas2 : bool
            Lognormal base-2 method for assigning weights.
        coef : bool
            If True, returns the final measure and parameters λ, var.
        graf1 : bool
            If True, plots the final measure.
        cumsum : bool
            If True, returns the cumulative sum (np.cumsum).

        Returns
        -------
        - (lista_general, lambdas, varianza) if coef=True
        - np.cumsum(lista_general[-1]) if cumsum=True and coef=False
        - None otherwise.
        """

        # 1) Find the position where f(alpha) is maximum => α₀
        pos_max = np.argmax(falpha)
        alpha0 = derivada[pos_max]  # Value of α at the max position

        # 2) Calculate multifractal parameters
        lambdas = alpha0 / h1  # λ parameter
        varianza = 2.0 * (lambdas - 1.0) / np.log(2.0)  # Variance

        if varianza < 0:
            raise ValueError("Negative variance: inconsistent parameters.")

        # 3) Define a base-2 lognormal generator
        def lognormal_base2(lmbda, var):
            z = np.random.normal(lmbda, np.sqrt(var))
            return 2.0 ** (-z)

        # 4) Build the cascade
        lista_general = []  # This will store all levels of the cascade
        for k in range(kmax):
            fila_coefs = []  # Coefficients at level k
            for _ in range(2**k):  # Number of intervals at level k
                if masas1:
                    # Use the ratio method for masses
                    rnum = np.random.normal(lambdas, np.sqrt(varianza))
                    val = 2.0 ** (-rnum)
                    m00 = val / (val + 1.0 / val)
                    m11 = 1.0 - m00
                elif masas2:
                    # Use lognormal base-2
                    m00 = lognormal_base2(lambdas, varianza)
                    m11 = lognormal_base2(lambdas, varianza)
                fila_coefs.append(np.array([m00, m11]))
            lista_general.append(fila_coefs)

        # 5) Multiply coefficients across levels
        for i in range(kmax - 1):
            lista_general[i] = [x for sublist in lista_general[i] for x in sublist]
            for j in range(len(lista_general[i])):
                lista_general[i + 1][j] = lista_general[i][j] * lista_general[i + 1][j]

        lista_general[-1] = np.array(
            [x for sublist in lista_general[-1] for x in sublist]
        )

        # 6) Normalize using a random variable
        media_omega = 1.0 / (2.0 ** (-lambdas * kmax))
        omega = np.random.normal(media_omega, np.sqrt(varianza), 2**kmax)
        lista_general[-1] *= omega

        # 7) Plot the measure if requested
        if graf1:
            # Turn off coefficient return
            coef = False

            intervalos = [
                np.linspace(i * b**-kmax, (i + 1) * b**-kmax, m) for i in range(b**kmax)
            ]
            salida = [val * np.ones(m) for val in lista_general[-1]]
            x = np.array([p for sublist in intervalos for p in sublist])
            y = np.array([p for sublist in salida for p in sublist])

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x, y, linewidth=0.8, color="black")
            ax.set_xlim([0, 1])
            ymax = np.amax(y)
            ax.set_ylim([0, ymax + 0.1 * ymax])
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(True)
            ax.set_title("Base-2 Lognormal Multifractal Measure")
            plt.show()

        # 8) Return the appropriate result
        if coef:
            return lista_general, lambdas, varianza

        if cumsum:
            return np.cumsum(lista_general[-1])


def simulacion(self, grafs: bool = False, results: bool = False):
    """
    Simulates a price trajectory using a multifractal cascade
    combined with Fractional Brownian Motion (FBM).

    Parameters
    ----------
    grafs : bool (default=False)
        If True, plots the results (trading time and prices).
    results : bool (default=False)
        If True, returns the normalized multifractal trading time.

    Returns
    -------
    tradingtime : np.ndarray, optional
        The normalized trading time, if results=True.
    """
    kmax = self.kmax  # Depth of the cascade

    # Generate trading time using a multifractal cascade
    tradingtime = self.multifractal_measure_rand2(
        b=2,
        m=1,
        kmax=self.kmax,
        falpha=self.falpha,
        derivada=self.derivada,
        h1=self.h1,
        Price=self.Price,
        masas1=False,
        masas2=True,
        coef=False,
        graf1=False,
        cumsum=True,
    )
    tradingtime = 2**kmax * (tradingtime / np.amax(tradingtime))  # Normalize

    # Simulate FBM for log-price increments
    fbm = stochastic.FractionalBrownianMotion(hurst=self.h1)
    simulacionfbm = fbm._sample_fractional_brownian_motion(2**kmax - 1)
    xtsim = simulacionfbm

    # Compute the final price: P(t) = P(0) * exp(X(t))
    precio_final = self.Price[0] * np.exp(xtsim)

    # Plot results if requested
    if grafs:
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(2**kmax), tradingtime, color="blue")
        plt.title("Normalized Multifractal Trading Time")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(tradingtime, precio_final, color="green")
        plt.title("Price with Multifractal Time Deformation")
        plt.grid(True)
        plt.show()

    # Return trading time if requested
    if results:
        return tradingtime
