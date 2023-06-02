import sys
import os
import pathlib as pl
from scipy.optimize import minimize
import numpy as np
import pickle

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from modules.structures.wingbox_georgina  import *
import input.data_structures.GeneralConstants as const
from input.data_structures.material import Material
from input.data_structures.wing import Wing
from input.data_structures.engine import Engine
from input.data_structures.ISA_tool import ISA

#------------ Instantiate classes and load values from JSON -----------------------
WingClass = Wing()
MatClass = Material()
EngClass = Engine()
ISAClass = ISA(const.h_cruise)

WingClass.load()
MatClass.load()
EngClass.load()
#-----------------------------------------------------------------------------------------


#------------------------------- Run script  ----------------------------------------------


x0=np.array([7, 1.5, 0.003, 0.003, 0.12, 0.07, 0.003,0.003,0.004,0.0022])


res = wingbox_optimization(x0, MatClass, WingClass, EngClass)

with open(r"output/structures/wingbox_output.pkl", "wb") as f:
    pickle.dump(res, f)
    print("Succesfully loaded data structure into wingbox_output.pkl")


print(res)


