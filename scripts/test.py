import sys
import os
import pathlib as pl
from scipy.optimize import minimize
import numpy as np
import pickle

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

from  modules.structures.stress_forms import *
import input.GeneralConstants as const
from input.data_structures.material import Material
from input.data_structures.wing import Wing
from input.data_structures.engine import Engine
from input.data_structures.ISA_tool import ISA
from input.data_structures.fuselage import Fuselage
from input.data_structures.hor_tail import HorTail
from modules.stab_ctrl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size


WingClass = Wing()
FuseClass = Fuselage()
HorClass = HorTail()

WingClass.load()
FuseClass.load()
HorClass.load()

print(np.tan(0))

#print(WingClass.x_lewing)
#print(wing_location_horizontalstab_size(WingClass, FuseClass, HorClass, False))
#print()
#print('done')
