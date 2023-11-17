# Import the class:
import sys
sys.path.append("/Users/omarkhalil/Desktop/Universidad/TFG/Libreria_GitHub/limpiov1")
'''
Necesitamos importar el path para accerder a los directorios donde se encuentran los 
módulos. Esto tiene hacerse porque de forma predeterminada porque cuando instalamos con pip
un directorio, de forma automatica lo guarda en una de las rutas que se encuentran en la variable
sys.path. Pero en ese caso no lo tenemos en el path de ninguno de ellos. Ese es el motivo
por el cual no lo podemos ejecutar amenos que lo annadamos manualmente. En cambio para 
añadir ficheros, como añadimos la ruta completa del archivo neste no debe encontrarse en una
ruta contenida en sys.path.

'''
from MMAR import Binomial_Multifractal_Det

r1 = Binomial_Multifractal_Det(12, 0.6, 0.4)
print(r1.measure)
r1.Test0()
r1.Test1()
r1.Test2()
r1.Test3()