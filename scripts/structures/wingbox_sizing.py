import sys
import os
import pathlib as pl
from scipy.optimize import minimize

sys.path.append(str(list(pl.Path(__file__).parents)[2]))


from modules.structures.wingbox_georgina import *
import input.data_structures.GeneralConstants as const
from input.data_structures.material import Material
from input.data_structures.wing import Wing
from input.data_structures.engine import Engine
from input.data_structures.ISA_tool import ISA


taper = Wing.taper
rho = ISA.density() #kg/m^3 for Aluminium
W_eng = Engine.weight #kg
E = Material.E #MPa
poisson = Material.poisson
pb=2.5 #post-buckling ratio
beta= Material.beta #Constant for crippling
g=5 #Constant associated to z-stringers
sigma_yield = Material.sigma_yield #MPa
m_crip = Material.m_crip
sigma_uts = Material.sigma_uts  #MPa
n_max=const.n_max_req


x0=np.array([7, 1.5, 0.003, 0.003, 0.12, 0.07, 0.003,0.003,0.004,0.0022])


res = wingbox_optimization(x0)


print(res)


