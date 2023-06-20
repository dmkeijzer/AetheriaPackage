
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.structures.pylon_design import PylonSizing
from input.data_structures import *


engine = Engine()
wing = Wing()


wing.load()
engine.load()

toc = (-1*(wing.chord_root - wing.chord_tip)/wing.span/2*2.3  + wing.chord_root)*0.12
print(f"toc = {toc/2*1.1}")


L = 2.6
x0 = (toc/2*1.1,0.014)
print(f"x0 = {x0}")




def test_pylon_design():
    Pylon = PylonSizing(engine, L)
    print(Pylon.weight_func(x0))
    print(Pylon.weight_func(x0)*2)
    print(Pylon.eigenfreq_constraint(x0))
    print(Pylon.von_mises_constraint(x0))
    print(Pylon.column_buckling_constraint(x0))
    print(Pylon.I_xx(x0))
    pass


test_pylon_design()