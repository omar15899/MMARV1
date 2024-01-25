import sys
sys.path.append("/Users/omarkhalil/Documents/GitHub/MMARV1")
import pandas as pd
from MMAR import MMAR


if __name__ == "__main__":
	df = pd.read_csv("/Users/omarkhalil/Documents/GitHub/MMARV1/Scripts/Lockheed.txt", delimiter='\t', header=0)

	Lockheed = MMAR(df, "Date", "Close", a = 0, b = 3, npoints=10)
	Lockheed.grafPrice()
	Lockheed.grafX_t()
	Lockheed.graf_Price_change()
	Lockheed.Partition_functions()
	Lockheed.tauandalpha()
	Lockheed.multifractal_measure_rand(graf1 = True)


	# For initialiting the simulation:

	x, y, z = Lockheed.simulacion(n = 100)
	Lockheed.analizadorprobabilidades(4000, x, y)


