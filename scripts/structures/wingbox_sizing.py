import sys
import os
import pathlib as pl
from scipy.optimize import minimize
import numpy as np
import pickle

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import  modules.structures.wingbox_georgina as wb
import input.data_structures.GeneralConstants as const
from input.data_structures.material import Material
from input.data_structures.wing import Wing
from input.data_structures.engine import Engine
from input.data_structures.ISA_tool import ISA
from input.data_structures.aero import Aero

#------------ Instantiate classes and load values from JSON -----------------------
WingClass = Wing()
MatClass = Material()
EngClass = Engine()
AeroClass =  Aero()
ISAClass = ISA(const.h_cruise)

WingClass.load()
MatClass.load()
EngClass.load()
AeroClass.load()
#-----------------------------------------------------------------------------------------

#--------------------------- Assign correct values to global values in wingbox_georgina -----------


# wb.taper = WingClass.taper
# wb.rho = MatClass.rho
# wb.W_eng = EngClass.mass_pertotalengine
# wb.E = MatClass.E
# wb.poisson = MatClass.poisson
# wb.pb= MatClass.pb
# wb.beta= MatClass.beta
# wb.g= MatClass.g
# wb.sigma_yield = MatClass.sigma_yield
# wb.m_crip = MatClass.m_crip
# wb.sigma_uts = MatClass.sigma_uts
# wb.n_max= const.n_max_req
# wb.y_rotor_loc = EngClass.y_rotor_loc


x0=np.array([0.01, 0.003, 0.07, 0.003,0.009,0.006, 0.003])    # :param x0: Initial estimate Design vector X = [b, cr, tsp, trib, L, bst, hst, tst, wst, t]


    
Optclass = wb.Wingbox(WingClass, EngClass, MatClass, AeroClass)
print(np.sum(Optclass.post_buckling(1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3)))

optimizertest = wb.WingboxOptimizer(x0,WingClass, EngClass, MatClass, AeroClass)
