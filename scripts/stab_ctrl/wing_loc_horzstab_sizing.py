import os
import pathlib as pl
import json
from potato_plot import J1loading
import numpy as np
import sys
import pathlib as pl
import os
import numpy as np
import json
import pandas as pd
import time

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))
import matplotlib.pyplot as plt
from input.data_structures import *

#with open() as jsonFile:
 #   data = json.load(jsonFile)

WingClass = Wing()
FuseClass = Fuselage()
HorTailClass = HorTail()


WingClass.load()
FuseClass.load()
HorTailClass.load()


Cma1 = WingClass.cm_alpha
Cma2 = HorTailClass.cm_alpha_h
CLa1 = WingClass.cL_alpha
CLa2 = HorTailClass.cL_alpha_h
Cm1 = WingClass.cm
Cm2 = HorTailClass.cm_h
CL1 = WingClass.cL_approach
CL2 = HorTailClass.cL_approach_h
x2 = FuseClass.length_fuselage
c1 = WingClass.chord_mac
c2 = HorTailClass.chord_mac_h
downwash = HorTailClass.downwash
V2_V1_ratio = 1



x1vec = np.arange(0, x2, 0.01)
log = np.zeros((1,6))

for x1 in x1vec:
    cglims = J1loading(x1,x2)[0]
    ShSstab = (-CLa1 * (cglims["rearcg"] - x1) - Cma1 * c1) / (Cma2 * (1 - downwash) * V2_V1_ratio ** 2 * c2 - CLa2 * (1-downwash) * V2_V1_ratio**2 * (x2 - cglims["rearcg"]))
    ShSctrl = (-Cm1 * c1 - CL1 * (cglims["frontcg"] - x1))/(Cm2* V2_V1_ratio**2 * c2 - CL2*V2_V1_ratio**2 * (x2 - cglims["frontcg"]))
    ShS = max(ShSctrl, ShSstab)
    x_np_nom = -Cma1 + (CLa1 * x1) / (c1) - Cma2 * (1 - downwash) * (ShS) * (c2 / c1) * (V2_V1_ratio) ** 2 + CLa2 * (1 - downwash) * (x2 / c1) * (ShS) * (V2_V1_ratio) ** 2
    x_np_den = CLa1 / c1 + CLa2 / c1 * (1 - downwash) * (ShS) * (V2_V1_ratio) ** 2
    x_np = x_np_nom / x_np_den
    x_cg = (-Cm1 - Cm2 * (c2 / c1) * (ShS) * V2_V1_ratio ** 2 + CL1 * x1 / c1 + CL2 * (x2 / c1) * (ShS) * V2_V1_ratio ** 2) / (CL1 / c1 + (CL2 / c1) * (ShS) * V2_V1_ratio ** 2)
    log = np.vstack((log, [x1, ShS, x_np, x_cg, cglims["frontcg"], cglims["rearcg"]]))
log = log[1:,:]

plt.subplot(1,2,1)
plt.plot(log[:,2], log[:,0])
plt.plot(log[:,3], log[:,0])
plt.plot(log[:,4], log[:,0])
plt.plot(log[:,5], log[:,0])
plt.ylabel("wing x location [m]")
plt.xlabel("horz flight cg range lims [m]")
plt.subplot(1,2,2)
plt.plot(log[:,0], log[:,1])
plt.ylabel("S_h/S [-]")
plt.xlabel("wing x location [m]")
plt.show()

