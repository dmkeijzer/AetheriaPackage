
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.structures.pylon_design import PylonSizing
from input.data_structures import *


engine = Engine()
engine.load()
L = 2.2
x0 = (0.095,0.012)




def test_pylon_design():
    Pylon = PylonSizing(engine, L)
    print(Pylon.weight_func(x0))
    print(Pylon.weight_func(x0)*2)
    print(Pylon.eigenfreq_constraint(x0))
    print(Pylon.von_mises_constraint(x0))
    print(Pylon.column_buckling_constraint(x0))
    pass


test_pylon_design()