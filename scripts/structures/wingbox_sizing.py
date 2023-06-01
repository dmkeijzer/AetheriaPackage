import sys
import os
import pathlib as pl
from scipy.optimize import minimize

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from modules.structures.wingbox_georgina import *

x0=np.array([7, 1.5, 0.003, 0.003, 0.12, 0.07, 0.003,0.003,0.004,0.0022])


res = wingbox_optimization(x0)


print(res)


