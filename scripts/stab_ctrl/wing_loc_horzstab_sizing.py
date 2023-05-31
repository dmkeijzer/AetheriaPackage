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
CLaw = WingClass.cL_alpha
CLa2 = HorTailClass.cL_alpha_h
Cm1 = WingClass.cm
Cm2 = HorTailClass.cm_h
CL1 = WingClass.cL_approach
CL2 = HorTailClass.cL_approach_h
x2 = FuseClass.length_fuselage
c1 = WingClass.chord_mac
c2 = HorTailClass.chord_mac_h
downwash = HorTailClass.downwash
w_fus = FuseClass.width_fuselage
h_fus = FuseClass.height_fuselage
x_lemac_x_rootchord = WingClass.X_lemac
b = WingClass.span
c_root = WingClass.chord_root
S = WingClass.surface
Cma1 = 0
Cma2 = 0
CLa1 = CLaw * (1+2.15 * w_fus / b) * (S-w_fus * c_root) / S + np.pi/2 * w_fus**2/S
V2_V1_ratio = 1



x1vec = np.arange(0, x2, 0.0002)
log = np.zeros((1,8))

for x_wing in x1vec:
    x1 = x_wing - 1.8/CLa1 * w_fus*h_fus*(x_wing - x_lemac_x_rootchord)/S
    cglims = J1loading(x1,x2)[0]
    ShSstab = (-CLa1 * (cglims["rearcg"] - x1) - Cma1 * c1) / (Cma2 * (1 - downwash) * V2_V1_ratio ** 2 * c2 - CLa2 * (1-downwash) * V2_V1_ratio**2 * (x2 - cglims["rearcg"]))
    ShSctrl = (-Cm1 * c1 - CL1 * (cglims["frontcg"] - x1))/(Cm2* V2_V1_ratio**2 * c2 - CL2*V2_V1_ratio**2 * (x2 - cglims["frontcg"]))
    ShS = max(ShSctrl, ShSstab)
    if ShS > 1: #or ShS < 0:
        continue
    x_np_nom = -Cma1 + (CLa1 * x1) / (c1) - Cma2 * (1 - downwash) * (ShS) * (c2 / c1) * (V2_V1_ratio) ** 2 + CLa2 * (1 - downwash) * (x2 / c1) * (ShS) * (V2_V1_ratio) ** 2
    x_np_den = CLa1 / c1 + CLa2 / c1 * (1 - downwash) * (ShS) * (V2_V1_ratio) ** 2
    x_aft = x_np_nom / x_np_den
    x_front = (-Cm1 - Cm2 * (c2 / c1) * (ShS) * V2_V1_ratio ** 2 + CL1 * x1 / c1 + CL2 * (x2 / c1) * (ShS) * V2_V1_ratio ** 2) / (CL1 / c1 + (CL2 / c1) * (ShS) * V2_V1_ratio ** 2)
    log = np.vstack((log, [x_wing, ShS, x_aft, x_front, cglims["frontcg"], cglims["rearcg"], ShSctrl, ShSstab]))
log = log[1:,:]
# rows_to_delete = np.where(log[:,3] > log[:,2])[0]
# log = np.delete(log, rows_to_delete, axis=0)

plt.subplot(1,2,1)
plt.plot(log[:,2], log[:,0], label="Stab line")
plt.plot(log[:,3], log[:,0], label="Ctrl line")
# plt.plot(log[:,4], log[:,0])
# plt.plot(log[:,5], log[:,0])
plt.ylabel("wing x location [m]")
plt.xlabel("horz flight cg range lims [m]")
plt.legend()
plt.subplot(1,2,2)
plt.plot(log[:,0], log[:,1])
plt.ylabel("S_h/S [-]")
plt.xlabel("wing x location [m]")
plt.show()
plt.plot(log[:,2], log[:,1], label="Stab line")
plt.plot(log[:,3], log[:,1], label="Ctrl line")
plt.legend()
plt.show()

minShS = min(log[:,1])
print(minShS)
x1_minShS = log[np.where(log[:,1] == minShS)[0],0]
print(x1_minShS)


print(J1loading(x1_minShS, x2)[0]["frontcg"], J1loading(x1_minShS, x2)[0]["rearcg"])
x1 = x1_minShS - 1.8/CLa1 * w_fus*h_fus*(x1_minShS - x_lemac_x_rootchord)/S
ShS = minShS
x_np_nom = -Cma1 + (CLa1 * x1) / (c1) - Cma2 * (1 - downwash) * (ShS) * (c2 / c1) * (V2_V1_ratio) ** 2 + CLa2 * (1 - downwash) * (x2 / c1) * (ShS) * (V2_V1_ratio) ** 2
x_np_den = CLa1 / c1 + CLa2 / c1 * (1 - downwash) * (ShS) * (V2_V1_ratio) ** 2
x_aft = x_np_nom / x_np_den
x_front = (-Cm1 - Cm2 * (c2 / c1) * (ShS) * V2_V1_ratio ** 2 + CL1 * x1 / c1 + CL2 * (x2 / c1) * (ShS) * V2_V1_ratio ** 2) / (CL1 / c1 + (CL2 / c1) * (ShS) * V2_V1_ratio ** 2)
print(x_front, x_aft)