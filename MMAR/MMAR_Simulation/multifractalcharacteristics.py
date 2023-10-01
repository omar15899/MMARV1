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

    def __init__(self, dataset, time, price, a=0, b=5, npoints=20, deltas=None, kmax=13):
        # Heredamos los atributos de Graphs. 
        super().__init__(dataset, time, price, a, b, npoints, deltas, kmax)
        self.partition = self.Partition_functions(graf1=False, result=True)
        self.tau = self.tauandalpha(graf=False, results=True)[0]
        self.falpha = self.tauandalpha(graf=False, results=True)[1]
        self.derivada = self.tauandalpha(graf=False, results=True)[2]
        self.h1 = self.Hurst(graf=False)
        self.posicion_max = np.argmax(self.falpha)
        self.alpha0 = self.derivada[self.posicion_max]
        self.lambdas = self.alpha0 / self.h1
        self.varianza = 2 * (self.lambdas - 1) / np.log(2)




    def Partition_functions(self, alpha=0.05, graf1=True, result=False):
        """
        alpha: Significance level
        qs: List of moments 'q' we want to compute
        deltas: Time intervals we want to analyze
        a, b, c: Bounds and step for the qs to be computed
        """
        
        a = self.a
        b = self.b
        npoints = self.npoints
        deltas = self.deltas
        ldeltas = len(deltas)  # Number of points for each partition "line" for different moments.
        mdeltas = np.mean(deltas)
        vardeltas = np.var(deltas)
        qs = np.linspace(a, b, npoints)

        # Compute partition functions for each delta and q; then perform OLS regression.
        # Each element in the list partition_functions corresponds to a partition function for a given delta and q.
        partition_functions = [[np.sum((abs(self.X_t[deltat::deltat] - self.X_t[:-deltat:deltat])) ** q) for deltat in deltas] for q in qs]
        adjustment_part_functions = [np.polyfit(np.log10(deltas), np.log10(partition_functions[i] / partition_functions[i][0]), 1) for i in range(npoints)]
        coeficiente_normalizador = [adjustment_part_functions[i][1] for i in range(npoints)]

        # Compute residual variances; work with logarithms.
        srs = [(np.sum((np.log10(partition_functions[i] / partition_functions[i][0]) - np.poly1d(adjustment_part_functions[i])(np.log10(deltas))) ** 2) / (ldeltas - 2)) ** 0.5 for i in range(npoints)]

        # Compute confidence intervals for mean estimation; the domain set is deltas.
        intervalos_conf = [[t.ppf(alpha / 2, ldeltas - 2) * srs[i] * np.sqrt(1 / ldeltas + (deltas[j] - mdeltas) ** 2 / ((ldeltas - 1) * vardeltas)) for j in range(ldeltas)] for i in range(npoints)]

        # Compute confidence intervals for the intercept and slope, which is what interests us.
        intervalos_conf_ordenada = [t.ppf(alpha / 2, ldeltas - 2) * srs[i] * np.sqrt(1 / ldeltas + (mdeltas) ** 2 / ((ldeltas - 1) * vardeltas)) for i in range(npoints)]
        intervalos_conf_pendiente = [t.ppf(alpha / 2, ldeltas - 2) * srs[i] / (np.sqrt(ldeltas - 1) * vardeltas) for i in range(npoints)]

        # Perform correlation test (equivalent to cor.test in R). The first list will have correlation values and the second will have p-values.
        corrtest1 = [stats.pearsonr(np.log10(deltas), np.log10(partition_functions[i] / partition_functions[i][0]))[0] for i in range(npoints)]
        corrtest2 = np.array([stats.pearsonr(np.log10(deltas), np.log10(partition_functions[i] / partition_functions[i][0]))[1] for i in range(npoints)])
        qslim = qs[corrtest2 <= alpha]

        
        if graf1:
            # The values of corrtest2 are p-values, which represent the probability that a random variable (statistical estimator)
            # would obtain a value more extreme than the observed estimator under the null hypothesis (in this case, a null correlation).
            # We set the confidence interval to 95% (specified by the 'alpha' variable in this method).
            # 1. Compute the list of q values such that their p-value is greater than the significance level:
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ["#FF0000", "#00FF00", "#0000FF", "#000000", "#FF00FF", "#00FFFF", "#FFA500", "#800080", "#008000", "#800000", "#008080", "#000080", "#FFFFE0", "#FFD700", "#00FF7F", "#FF4500", "#9400D3", "#808000", "#FF1493", "#6A5ACD"]

            for i, q in enumerate(qslim):
                ax.plot(np.log10(deltas), np.poly1d([q/2-1,0])(np.log10(deltas)), linestyle="-.", color = "black")
                ax.plot(np.log10(deltas), np.log10(partition_functions[i]/partition_functions[i][0]), label=f"q = {q:.2f}", color=colors[i % len(colors)])
                ax.plot(np.log10(deltas), np.poly1d(adjustment_part_functions[i])(np.log10(deltas)), color=colors[i % len(colors)])    
                ax.fill_between(np.log10(deltas), np.poly1d(adjustment_part_functions[i])(np.log10(deltas)) - intervalos_conf[i], np.poly1d(adjustment_part_functions[i])(np.log10(deltas)) + intervalos_conf[i], color=colors[i % len(colors)], alpha = 0.4)

            # Grid, labels, and legend
            ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')
            ax.set_ylabel(r"$log_{10}(\hat{S}_q(\Delta t))$", fontsize=15, fontweight='bold')
            ax.set_xlabel(r"$log_{10}(\Delta t)$", fontsize=15, fontweight='bold')
            ax.legend(loc='best', fancybox=True, shadow=True, ncol=5, fontsize=12)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            plt.show()

            # Plotting unnormalized partition functions
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, q in enumerate(qslim):
                ax.plot(np.log10(deltas), np.log10(partition_functions[i]), label=f"q = {q:.2f}", color=colors[i % len(colors)])        
            
            # Grid, labels, and legend
            ax.grid(which='both', axis='both', linestyle='-.', linewidth=1, alpha=0.7, zorder=0, markevery=1, color='grey')
            ax.set_title(r"Unnormalized Partition Function $log_{10}(S_q(\Delta t))$ vs. $log(\Delta t)$ for different q values." + "\n Curves adjusted by OLS:", fontsize=16, fontweight='bold')
            ax.set_ylabel(r"$log_{10}(S_q(\Delta t))$", fontsize=12, fontweight='bold')
            ax.set_xlabel(r"$log_{10}(\Delta t)$", fontsize=12, fontweight='bold')
            ax.set_ylim(-3.5,5)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5, fontsize=12)

            fig.set_facecolor('#EAEAEA')

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            plt.show()

        if result:
            #print(f"The adjustment values (a, m respectively) are {adjustment_part_functions}" + 
            #      f"\nErrors associated with the ordinate (constant term) are given by {intervalos_conf_ordenada}")
            #print(f"The null hypothesis is rejected up to the value q = {qslim[-1]}.")
            #print(f"The correlation coefficient estimator for each q is: {corrtest1}" +
            #      f"\nThe p-values associated with the estimator are: {corrtest2}")

            return partition_functions, adjustment_part_functions

            






    def tauandalpha(self, alpha=0.05, graf=True, results=False):
        plt.style.use('default')  # Set the plot style
        a = self.a
        b = self.b
        npoints = self.npoints
        deltas = self.deltas
        ldeltas = len(deltas)  # Number of partition points for different moments
        mdeltas = np.mean(deltas)
        vardeltas = np.var(deltas)
        qs = np.linspace(a, b, npoints)

        # Compute partition functions
        partition_functions = [[np.sum((abs(self.X_t[deltat::deltat] - self.X_t[:-deltat:deltat])) ** q) for deltat in deltas] for q in qs]
        adjustment_part_functions = [np.polyfit(np.log10(deltas), np.log10(partition_functions[i] / partition_functions[i][0]), 1) for i in range(npoints)]
        coeficiente_normalizador = [adjustment_part_functions[i][1] for i in range(npoints)]

        # Compute confidence intervals again
        srs = [(np.sum((np.log10(partition_functions[i]) - np.poly1d(adjustment_part_functions[i])(np.log10(deltas))) ** 2) / (ldeltas - 2)) ** 0.5 for i in range(npoints)]
        intervalos_conf = [[t.ppf(alpha / 2, ldeltas - 2) * srs[i] * np.sqrt(1 / ldeltas + (deltas[j] - mdeltas) ** 2 / ((ldeltas - 1) * vardeltas)) for j in range(ldeltas)] for i in range(npoints)]
        intervalos_conf_ordenada = [t.ppf(alpha / 2, ldeltas - 2) * srs[i] * np.sqrt(1 / ldeltas + (mdeltas) ** 2 / ((ldeltas - 1) * vardeltas)) for i in range(npoints)]
        intervalos_conf_pendiente = [t.ppf(alpha / 2, ldeltas - 2) * srs[i] / (np.sqrt(ldeltas - 1) * vardeltas) for i in range(npoints)]

        # Compute tau as in the binomial case
        tau = [sublistasajustes[0] for sublistasajustes in adjustment_part_functions]
        h = (b - a) / npoints  # Distance between points in qs, as in the binomial case

        # Compute the derivative using splines
        intervalo1 = np.linspace(qs[0], qs[-1], 1000)
        spline = UnivariateSpline(qs, tau)
        derivada1 = spline.derivative()(intervalo1)
        legendre1 = intervalo1 * derivada1 - spline(intervalo1)

        # Compute confidence intervals for the scaling function tau
        tausposibles = [[np.log10(partition_functions[i][j] / partition_functions[i][0]) / np.log10(deltas[j]) for i in range(npoints)] for j in range(1, ldeltas)]
        splinesposibles = [UnivariateSpline(qs, taus) for taus in tausposibles]
        lengendreposbiles = [intervalo1 * splines.derivative()(intervalo1) - splines(intervalo1) for splines in splinesposibles]

        if graf:
            # Plotting code
            figure = plt.figure(figsize=(17, 8))

            ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
            for i in range(ldeltas - 1):
                ax1.plot(intervalo1, splinesposibles[i](intervalo1), alpha=0.1, color="grey")
            ax1.plot(intervalo1, spline(intervalo1), lw=2, color='blue')
            ax1.fill_between(qs, np.array(tau) - np.array(intervalos_conf_pendiente), np.array(tau) + np.array(intervalos_conf_pendiente), color="red")
            ax1.set_title("a)", fontsize=18)
            ax1.set_xlabel("q", fontsize=18)
            ax1.set_ylabel(r"$\hat{\tau}$", fontsize=18)
            ax1.grid(True, linestyle='dashdot')
            ax1.set_ylim(-1, 1)
            ax1.set_xlim(0, qs[-1])

            ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
            for i in range(ldeltas - 1):
                ax2.plot(intervalo1, splinesposibles[i].derivative()(intervalo1), alpha=0.1, color="grey")
            ax2.plot(intervalo1, derivada1, lw=2, color="blue")
            ax2.set_title("c)", fontsize=18)
            ax2.set_xlabel("q", fontsize=18)
            ax2.set_ylabel(r"$\hat{f}(\alpha(q))$", fontsize=18)
            ax2.grid(True, linestyle='dashdot')

            ax3 = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
            for i in range(ldeltas - 1):
                ax3.plot(splinesposibles[i].derivative()(intervalo1), lengendreposbiles[i], alpha=0.1, color="grey")
            ax3.plot(derivada1, legendre1, lw=2, color='red')
            ax3.set_ylim(0.5, 1.05)
            ax3.set_title("b)", fontsize=18)
            ax3.set_xlabel(r"$\alpha$", fontsize=18)
            ax3.set_ylabel(r"$f(\alpha)$", fontsize=18)
            ax3.grid(True, linestyle='dashdot')

            plt.tight_layout()
            plt.show()

        if results:
            #return (tau, coeficiente_normalizador, intervalos_conf_ordenada, intervalos_conf_pendiente)
            return spline(intervalo1), legendre1, derivada1






    def Hurst(self, graf=False):
        """
        Calculate the Hurst coefficient using the given object's attributes.

        Parameters:
        - graf (bool): Determines whether to plot graphs or not.

        Returns:
        - h1 (float): The inverted coefficient of the only root of tau (Hurst exponent).
        """

        # Extract relevant attributes from the object
        a, b = self.a, self.b
        npoints = self.npoints
        deltas = self.deltas
        qs = np.linspace(a, b, npoints)

        # Compute partition functions and their Ordinary Least Squares (OLS) linear adjustments
        partition_functions = [[np.sum((abs(self.X_t[deltat::deltat] - self.X_t[:-deltat:deltat])) ** q) 
                                for deltat in deltas] for q in qs]
        adjustment_part_functions = [np.polyfit(np.log10(deltas), np.log10(partition_functions[i]), 1) 
                                     for i in range(len(qs))]

        # Compute tau values similar to the deterministic binomial measure case
        tau = [adjustment[0] for adjustment in adjustment_part_functions]
        h = (b - a) / npoints

        # Perform spline interpolation and derivative calculations
        intervalo1 = np.linspace(qs[0], qs[-1], 1000)
        spline = UnivariateSpline(qs, tau)
        derivada = spline.derivative()(intervalo1)
        legendre = intervalo1 * derivada - spline(intervalo1)

        # If the graf flag is set, plot the graphs for visualization
        if graf:
            plt.style.use('default')
            figure = plt.figure(figsize=(12, 6))

            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(qs, tau, lw=2, color='r', ls="-.")
            ax1.plot(qs, p(qs), lw=2, color='green')
            ax1.set_title("Tau moment")
            ax1.set_xlabel("q, slopes of f(alpha)")
            ax1.set_ylabel("Tau")
            ax1.grid(True, linestyle='dashdot')

            plt.show()

        # Find the unique root within the interval where the spline has been interpolated
        sol1 = spline.roots()[0]

        # Calculate and return the inverted Hurst coefficient (Hurst exponent)
        h1 = 1 / sol1

        return h1





    def multifractal_measure_rand(self, b = 2, m = 1, masas1 = False, masas2 = True, coef = False, graf1 = False, cumsum = False):
        """
        Multifractal measure that complies with the lambda and sigma parameters
        of the lognormal distribution. Refer to page 22 of DM/Dollar for more information.
        """
        
        kmax = self.kmax
        # Since we have previously calculated H, we only need to calculate alpha_0,
        # which is the domain value where f(alpha) reaches its maximum.
        posicion_max = np.argmax(self.falpha)
        alpha0 = self.derivada[posicion_max]
        
        lambdas = alpha0 / self.h1
        varianza = 2 * (lambdas - 1) / np.log(2)
        
        # Handle potential negative variance
        if varianza < 0:
            raise ValueError("Variance must be non-negative")
        
        # Generate the lognormal random variable
        def lognormal_base2(lambdas, varianza):
            normal_random_variable = np.random.normal(loc=lambdas, scale=np.sqrt(varianza))
            return 2 ** -normal_random_variable
        
        lista_general = []
        for k in range(kmax):
            lista_coeficientes = []
            for _ in range(2 ** k):
                if masas1:
                    masas2 = False
                    m00 = (lambda x: x / (x + (1 / x)))(2 ** -np.random.normal(lambdas, np.sqrt(varianza)))
                    m11 = 1 - m00
                if masas2:
                    m00 = lognormal_base2(lambdas, varianza)
                    m11 = lognormal_base2(lambdas, varianza)
                lista_coeficientes.append(np.array([m00, m11]))
            lista_general.append(lista_coeficientes)
        
        # Multiply and obtain the image for each interval in the k-th iteration
        for i in range(len(lista_general) - 1):
            lista_general[i] = [item for sublist in lista_general[i] for item in sublist]
            for j in range(len(lista_general[i])):
                a = lista_general[i][j] * lista_general[i + 1][j]
                lista_general[i + 1][j] = a
        
        # Flatten the last list in lista_general
        lista_general[-1] = np.array([item for sublist in lista_general[-1] for item in sublist])

        media_omega = 1 / (2 ** (-lambdas * kmax))
        
        # Handle potential negative variance again
        if varianza < 0:
            raise ValueError("Variance must be non-negative")
        
        omega = np.random.normal(media_omega, np.sqrt(varianza), 2 ** kmax)
        
        lista_general[-1] *= omega

        # Plot the results if graf1 is True
        if graf1:
            coef = False
            intervals = [np.linspace(i * b ** -kmax, (i + 1) * b ** -kmax, m) for i in range(b ** kmax)]
            output = [coef * np.ones(m) for coef in lista_general[-1]]
            x = np.array([item for sublist in intervals for item in sublist])
            y = np.array([item for sublist in output for item in sublist])
            
            # Handle potential empty or NaN array
            if np.isnan(y).all():
                raise ValueError("Array 'y' contains only NaN values")
            
            fig, ax = plt.subplots(figsize=(60, 5))
            ax.plot(x, y, linewidth=0.8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, np.nanmax(y) + 0.1 * np.nanmax(y)])
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.grid(True)
            plt.show()
        
        if coef:
            return lista_general, lambdas, varianza
        
        if cumsum:
            return np.cumsum(lista_general[-1])
