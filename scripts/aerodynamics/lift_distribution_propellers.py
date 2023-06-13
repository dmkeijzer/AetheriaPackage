
import os
import json
import sys
import numpy as np
import sys
import pathlib as pl
import matplotlib.pyplot as plt
import scipy.stats as stat

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.avlwrapper import Geometry, Surface, Section, NacaAirfoil, Control, Point, Spacing, Session, Case, Parameter

from input.data_structures.wing import Wing
from input.data_structures.aero import Aero
from input.data_structures.engine import Engine
import input.data_structures.GeneralConstants as const
from slipstream_effect_cruise import *
from modules.aero.avl_access import *

AeroClass = Aero()
WingClass = Wing()
EngineClass = Engine()

AeroClass.load()
WingClass.load()
EngineClass.load()

lift_array = get_strip_array(WingClass, AeroClass, plot= False)
cl = np.flip(lift_array[1])
b = WingClass.span

x = np.linspace(0, b/2, 24)

slipstream = CL_slipstream_final/4
prop_lift = prop_lift_var/6

# First Engine
x1 = EngineClass.x_rotor_loc[0]
pdf1 = stat.norm.pdf(x, loc= x1, scale= 0.4)
sumpdf1 = np.sum(pdf1)
pdf1 = pdf1/sumpdf1

# Second Engine
x2 = EngineClass.x_rotor_loc[2]
pdf2 = stat.norm.pdf(x, loc= x2, scale= 0.4)
sumpdf2 = np.sum(pdf2)
pdf2 = pdf2/sumpdf2


#term for extra visuals
term = 10
sum = cl + term * pdf1 * (slipstream+prop_lift) + term * pdf2 * (slipstream+prop_lift)

# print(pdf)
plt.plot(np.flip(x), sum, label="CL wing plus propellers")
plt.plot(np.flip(x), cl, label="CL wing")
plt.xlabel("X-position over half-span [m]")
plt.ylabel("CL [-]")
plt.ylim([0.,0.65])
plt.legend()
plt.grid()
plt.show()