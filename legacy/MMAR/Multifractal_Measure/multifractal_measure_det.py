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
In this module, the quintessential deterministic multifractal measure is defined—the deterministic binomial measure. This measure is rigorously defined in the paper:

Carl J.G.Eversz, B. B. Mandelbrot. (1992). Multifractal Measures. Chaos and Fractals - Springer-Verlag, New York, Appendix B, pp. 849-881.

Additionally, a series of tests on the study of this measure are presented. These tests include the two most crucial methods for obtaining the multifractal spectrum 
 f(\alpha) of this measure. All these methods are explained and justified in the cited paper.
"""


class Binomial_Multifractal_Det:
    """
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
    f1() : Theoretical multifractal spectrum of the binomial measure.

    multifractal_measure_det() : Binomial multifractal measure.

    Test0 : Obtains and studies the convergence of its multifractal spectrum \( f(\alpha) \) using:
        a) Histogram Method: Slower convergence to the theoretical function.

    Test1 : Numerical method for obtaining the multifractal spectrum from the method of moments through
            the partition function (the reason for this name can be found in the study of all the states of the
            measure at some coarse-grained state of it).

    Test2 : Finds the partition function of the binomial measure. This partition function is crucial
            for obtaining the multifractal spectrum \( f(\alpha) \) by the method of moments.

    Test3 : Finds the Lagrange's transform of the multifractal spectrum and then computes the multifractal
            spectrum itself. This method shows much better convergence to the theoretical value.

    Test4 : Compares the convergence of both methods presented (histogram and moments method) and contrasts
            them with the Monte Carlo convergence value.
    """

    def __init__(self, kmax, m00, m11, b=2, m=5):
        self.kmax = kmax
        self.m00 = m00
        self.m11 = m11
        self.b = b
        self.m = m
        self.measure = self.multifractal_measure_det(coef=True)
        # self.measure_graph = self.multifractal_measure_det(graf = True)

        if self.m00 + self.m11 != 1:
            raise ValueError(
                "WARNING. This multifractal measure does not preserve its mass."
                "\nPlease make sure m00 + m11 = 1"
            )

    def f1(self, alpha):
        # Calculate the maximum between m00 and m11
        a = max(self.m00, self.m11)
        # Calculate alpha_min and alpha_max
        alpha_min = -np.log2(a)
        alpha_max = -np.log2(1 - a)
        # Compute and return the result using the formula
        return -(alpha_max - alpha) / (alpha_max - alpha_min) * np.log2(
            (alpha_max - alpha) / (alpha_max - alpha_min)
        ) - (alpha - alpha_min) / (alpha_max - alpha_min) * np.log2(
            (alpha - alpha_min) / (alpha_max - alpha_min)
        )

    def multifractal_measure_det(self, coef=False, graf=False):
        """
        Parameters
        ----------
        coef : bool, optional
            If True, return coefficients, default is False.
        graf : bool, optional
            If True, plot the deterministic multifractal measure, default is False.
        """

        # Generate all the weights
        lista_general = []
        for k in range(self.kmax):
            lista_coeficientes = [np.array([self.m00, self.m11])] * (2**k)
            lista_general.append(lista_coeficientes)

        # Cascade multiplication
        for i in range(self.kmax - 1):
            # Flatten list at index i
            lista_general[i] = [
                item for sublist in lista_general[i] for item in sublist
            ]
            # Multiply each element by the entire sublist of the next sublist
            for j in range(len(lista_general[i])):
                a = lista_general[i][j] * lista_general[i + 1][j]
                lista_general[i + 1][j] = a
        lista_general[-1] = [item for sublist in lista_general[-1] for item in sublist]

        """
        for i in range(self.kmax - 1):
            lista_general[i] = np.concatenate(lista_general[i])
            for j in range(len(lista_general[i])):
                lista_general[i + 1][j] *= lista_general[i][j]
        lista_general[-1] = np.concatenate(lista_general[-1])
        """

        # Return coefficients if specified
        if coef:
            return lista_general

        # Plot if specified
        if graf:
            intervals = [
                np.linspace(
                    i * self.b ** (-self.kmax), (i + 1) * self.b ** (-self.kmax), self.m
                )
                for i in range(self.b**self.kmax)
            ]
            # Multiply the output by 2**k to preserve mass, not the actual measure value
            output = [
                2 ** (self.kmax) * coef * np.ones(self.m) for coef in lista_general[-1]
            ]
            x = np.array([item for sublist in intervals for item in sublist])
            y = np.array([item for sublist in output for item in sublist])
            plt.plot(x, y, linewidth=0.5)
            plt.title(
                rf"Deterministic multifractal measure normalized by a factor $2^k$"
                + f"\n at iteration k = {self.kmax}"
                + "\nwith "
                + rf"weights $m_0$ = {self.m00}, $m_1$ = {self.m11}"
            )
            plt.xlim(0, 1)
            plt.ylim(0, np.amax(y) + 0.1)
            plt.grid(True, linestyle="dashdot")
            # plt.savefig('Deterministic_multifractal_measure.png', dpi=300)
            plt.show()

    def Test0(self):
        """
        Multifractal spectrum using the Histograms method.
        """
        # Leveraging the fact that we have all measure values from previous k iterations
        # in the 'lista_general'. The only task here is to find the coarse Holder coefficients
        # for each element in each sublist, and ultimately, to create the histogram.

        # Step 1: Create a list with Holder coefficients for each iteration
        bins1 = self.kmax - 1
        holder = [
            np.log(np.array(self.measure[i])) / np.log(2 ** (-(i + 1)))
            for i in range(self.kmax)
        ]

        # Step 2: Create a histogram only for the last sublist (the one of interest)
        hist, bin_edges = np.histogram(holder[-1], bins=bins1)

        # Calculate f(alpha) as shown in the algorithm
        x = [
            x - (bin_edges.tolist()[1] - bin_edges.tolist()[0]) / 2
            for x in bin_edges.tolist()
        ]
        y = -np.log(hist) / np.log(2 ** (-(self.kmax - 1)))

        # Define the domain of the theoretical f(alpha) function for plotting
        a = max(self.m00, self.m11)
        alpha_min = -np.log2(a)
        alpha_max = -np.log2(1 - a)
        x2 = np.linspace(alpha_min + 0.01, alpha_max - 0.01, 1000)

        # Set the font size for the plot
        plt.rcParams.update({"font.size": 12})

        # Set the figure size
        plt.rcParams["figure.figsize"] = [8, 8]
        figure, axis = plt.subplots(1)

        # Plotting
        axis.plot(x2, self.f1(x2), label="Theoretical function")
        axis.plot(x[1:], y, label="Numerical function")
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            shadow=True,
            ncol=1,
        )
        axis.set_title(
            "Experimental vs Theoretical f(alpha) \nvia the Histogram Method"
        )
        axis.set_xlabel("alpha")
        axis.set_ylabel("f(alpha)")
        axis.grid(True, linestyle="dashdot")
        figure.set_facecolor("#EAEAEA")
        # plt.savefig('f_alpha-hist.png', dpi=300)
        plt.show()

        print(
            f"""
            The minimum and maximum alpha values are respectively: 
            {-np.log2(self.m00)}, {-np.log2(self.m11)}. 
            As we can see, they match the experimental values. 
            The maximum is {np.amax(y)} and the experimental alphas are {x[0]} and {x[-1]}
        """
        )

    def Test1(self):
        # Check section (c) of the moments method. Investigate if straight lines are generated.

        # Define interval bounds:
        a, b = -20, 20
        npuntos = 10  # Number of points
        h = (b - a) / npuntos  # Step size, needed for derivative calculation

        # Study for a set of predefined 'q's:
        interval = np.linspace(a, b, npuntos)  # Define the range of 'q' values

        # Convert self.measure lists to numpy arrays:
        medidas = [np.array(sublist) for sublist in self.measure]

        # Create a list of partition functions, one for each 'q' and each 'k', up to 'kmax':
        funciones_particion = [
            [np.sum(submedidas**q) for submedidas in medidas] for q in interval
        ]

        # Plotting configurations
        plt.rcParams.update({"font.size": 12})
        plt.rcParams["figure.figsize"] = [8, 8]

        fig, ax = plt.subplots(figsize=(12, 8))
        for i, q in enumerate(interval):
            ax.plot(
                [np.log10(2 ** ((i + 1))) for i in range(len(funciones_particion[i]))],
                np.log10(np.array(funciones_particion[i]) / funciones_particion[i][0]),
            )

        # Set plot labels, title, and grid:
        ax.set_title(
            r"Normalized Partition Function $log_{10}(S_q(\Delta t))$ vs. $log(\Delta t)$ for different q values.",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_ylabel(r"$log_{10}(S_q(\Delta t))$", fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$log_{10}(\Delta t)$", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="dashdot")

        # Set the background color of the figure:
        fig.set_facecolor("#EAEAEA")

        # Uncomment to save the plot as a PNG file:
        # plt.savefig('funciones_parcicion.png', dpi=300)

        plt.show()

    def Test2(self):
        # Define interval bounds and number of points
        a, b = -5, 5
        npoints = 10000
        h = (
            b - a
        ) / npoints  # Size of the interval between points; relevant for derivative calculations

        # Create an array of q values within the defined interval
        qs = np.linspace(a, b, npoints)
        deltas = [np.log(2 ** (-(k + 1))) for k in range(self.kmax)]

        # Convert lists in 'self.measure' to numpy arrays
        measures = [np.array(sublist) for sublist in self.measure]

        # Calculate the partition function for each q in kmax
        partition_functions = [
            np.array([np.sum(submeasure**q) for submeasure in measures]) for q in qs
        ]

        # Normalize the partition functions relative to their first value
        partition_functions_norm = [
            np.log(func / func[0]) for func in partition_functions
        ]

        # Calculate tau, which is defined as the slope of the log/log graph
        tau = [
            (func[1] - func[0]) / (deltas[1] - deltas[0])
            for func in partition_functions_norm
        ]

        # Use cubic splines for more accurate derivative calculations
        interval1 = np.linspace(qs[0], qs[-1], 1000)
        spline = UnivariateSpline(qs, tau)
        derivative1 = spline.derivative()(interval1)
        legendre1 = interval1 * derivative1 - spline(interval1)

        # Create the plot
        figure = plt.figure(figsize=(17, 8))

        # Plot tau values
        ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
        ax1.plot(interval1, spline(interval1), lw=2, color="blue")
        ax1.set_title(r"Moment $\tau$")
        ax1.set_xlabel("q (slopes of f(alpha))")
        ax1.set_ylabel(r"$\tau$")
        ax1.grid(True, linestyle="dashdot")
        ax1.set_ylim(-1, 3.5)
        ax1.set_xlim(0, 4)

        # Plot the derivative
        ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
        ax2.plot(interval1, derivative1, lw=2, color="blue")
        ax2.set_title(r"f($\alpha$(q)) obtained via Legendre transform")
        ax2.set_xlabel("q")
        ax2.set_ylabel(r"f($\alpha$(q))")
        ax2.grid(True, linestyle="dashdot")

        # Plot the Legendre transform
        ax3 = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
        a = max(self.m00, self.m11)
        alpha_min = -np.log2(a)
        alpha_max = -np.log2(1 - a)
        x2 = np.linspace(alpha_min + 0.01, alpha_max - 0.01, 1000)
        ax3.plot(
            x2,
            self.f1(x2),
            color="red",
            label="Theoretical function",
            linestyle="dashdot",
        )
        ax3.plot(
            derivative1, legendre1, lw=2, color="blue", label="Experimental function"
        )
        ax3.set_ylim(0, 1.1)
        ax3.set_title(r"f($\alpha$)")
        ax3.set_xlabel(r"$\alpha$")
        ax3.set_ylabel(r"$\tau$")
        ax3.grid(True, linestyle="dashdot")

        # Set plot styling
        figure.set_facecolor("#EAEAEA")
        custom_lines = [
            Line2D([0], [0], color="blue", lw=2),
            Line2D([0], [0], color="red", lw=2),
        ]
        custom_labels = ["Tau", r"f($\alpha$)"]

        legend = plt.legend(
            custom_lines,
            custom_labels,
            loc="lower center",
            bbox_to_anchor=(-0.1, -0.1),
            fancybox=True,
            shadow=True,
            ncol=3,
        )
        plt.subplots_adjust(wspace=0.5, hspace=0.9)
        plt.show()

    def Test3(self):
        # Define the theoretical convergence
        def theoretical_convergence(x):
            return 1 / np.sqrt(2**x)

        # Fit data to check if it aligns with Monte Carlo type convergence (with constant)
        def adjustment(x, c, a):
            return (c / np.sqrt(x)) ** a

        z = self.kmax + 1  # Maximum index for iteration
        a = 5  # Starting index for iteration
        y = []  # Placeholder for storing results

        # Loop through range and calculate holder and histogram
        for k in range(a, z):
            bins1 = k - 1
            holder = [
                np.log(np.array(self.measure[i])) / np.log(2 ** (-(i + 1)))
                for i in range(k)
            ]
            hist, bin_edges = np.histogram(holder[-1], bins=bins1)
            y.append(1 - np.amax(-np.log(hist) / np.log(2 ** (-(k + 1)))))

        # Perform curve fitting
        xdata = np.array([x for x in range(a, z)])
        popt, pcov = scipy.optimize.curve_fit(adjustment, xdata, y)
        perr = np.sqrt(np.diag(pcov))

        print(
            f"Fit values for a function (K/np.sqrt(x))**W are K = {popt[0]} and W = {popt[1]}, with the respective fit errors: {perr}"
        )

        # Generate plots
        x = np.linspace(a, z, 1000)
        plt.rcParams.update({"font.size": 12})
        plt.style.use("default")
        plt.rcParams["figure.figsize"] = [8, 8]
        figure, axis = plt.subplots(2)
        axis[0].plot(xdata, adjustment(xdata, *popt), label="Histogram Method Fit")
        axis[0].plot(xdata, y, label="Histogram Method Error")
        axis[0].plot(
            xdata, theoretical_convergence(xdata), label="Theoretical Monte Carlo Error"
        )
        axis[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            shadow=True,
            ncol=1,
        )
        axis[0].set_title(
            "Difference between experimental and actual value\n(relative error of the experimental maximum vs. the actual maximum = 1)\ncompared to the Monte Carlo simulation error",
            y=1.05,
        )
        axis[0].set_xlabel("Iteration Number K")
        axis[0].set_ylabel("Relative Error")
        axis[0].grid(True, linestyle="dashdot")

        # Plot even the difference
        axis[1].scatter(xdata, abs(theoretical_convergence(xdata) - y), c="#39ff14")
        axis[1].set_title(
            "Difference between the Monte Carlo error evolution and\nthe histogram error"
        )
        axis[1].set_xlabel("Iteration Number K")
        axis[1].set_ylabel("Difference between the two functions")
        axis[1].grid(True, linestyle="dashdot")
        figure.set_facecolor("#EAEAEA")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=2.5)
        plt.subplots_adjust(hspace=0.8)
        plt.show()
