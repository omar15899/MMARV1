""" 
Con __all__ cuando importemos de los submódulos graphs y multifractalcharacteristics 
solo importaremos las funciones y clases que tengan los nombres que se encuentran 
en esa lista. 

__all__ = ["Graphs", "MultifractalCharacteristics"]
"""

from .graphs import *
from .multifractalcharacteristics import *
from .mmar_simulation import *
