import sys
import os
import pathlib as pl
from scipy.optimize import minimize
import numpy as np

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

taper = WingClass.taper
rho = ISAClass.density() #kg/m^3 for Aluminium
W_eng = EngClass.mass_pertotalengine #kg
E = MatClass.E #MPa
poisson = MatClass.poisson
pb=2.5 #post-buckling ratio
beta= MatClass.beta #Constant for crippling
g=5 #Constant associated to z-stringers
sigma_yield = MatClass.sigma_yield #MPa
m_crip = MatClass.m_crip
sigma_uts = MatClass.sigma_uts  #MPa
n_max=const.n_max_req


x0=np.array([7, 1.5, 0.003, 0.003, 0.12, 0.07, 0.003,0.003,0.004,0.0022])


res = wingbox_optimization(x0, MatClass, WingClass, EngClass)


print(res)


