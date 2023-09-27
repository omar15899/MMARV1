from .Multifractal_Measure import *
from .MMAR_Simulation import * 
'''
Otra forma es directamente desde referenciar el camino absoluto, 
esto se logra iniciando siempre en el directorio que se encuentra
previo al archivo __init__.py.
from .Multifractal_Measure.multifractal_measure_det import *
from .Multifractal_Measure.multifractal_measure_rand import *

En el MMAR_classes, como tenemos además otro archivo __init__.py 
que importa todo de cada uno de los modulos.py, solo debemos importar
el directorio MMAR_classes y este automáticametne se anidará con el otro
archivo init. Podríamos hacer exactamente lo mismo con la carpeta ]
Multifractal_Measure, y para la versión final de la librería es lo que haré. 


'''