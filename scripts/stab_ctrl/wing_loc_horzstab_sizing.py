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

AeroClass = Aero()
WingClass = Wing()
FuseClass = Fuselage()


AeroClass.load()
WingClass.load()
FuseClass.load()


Cma1 = AeroClass.cm_alpha
Cma2 = 0
CLa1 = AeroClass.cL_alpha
CLa2 = 1
Cm1 = AeroClass.cm
Cm2 = 1
CL1 = 1
CL2 = 1
x2 = FuseClass.lf
c1 = WingClass.chord_mac
c2 = 1
downwash = 1
V2_V1_ratio = 1

x1vec = np.arange(0, lfus, 0.01)
log = np.array([[]])

for x1 in x1vec:
    cglims = J1loading(x1,x2)[0]
    ShSstab = (-CLa1 * (cglims["rearcg"] - x1) - Cma1 * c1) / (Cma2 * (1 - downwash) * V2_V1_ratio ** 2 * c2 - CLa2 * (1-downwash) * V2_V1_ratio**2 * (x2 - cglims["rearwing"]))
    ShSctrl = (-Cm1 * c1 - CL1 * (cglims["frontcg"] - x1))/(Cm2* V2_V1_ratio**2 * c2 - CL2*V2_V1_ratio**2 * (x2 - cglims["frontcg"]))
    ShS = max(ShSctrl, ShSstab)
    x_np_nom = -Cma1 + (CLa1 * x1) / (c1) - Cma2 * (1 - downwash) * (ShS) * (c2 / c1) * (V2_V1_ratio) ** 2 + CLa2 * (1 - downwash) * (x2 / c1) * (ShS) * (V2_V1_ratio) ** 2
    x_np_den = CLa1 / c1 + CLa2 / c1 * (1 - downwash) * (ShS) * (V2_V1_ratio) ** 2
    x_np = x_np_nom / x_np_den
    x_cg = (-Cm1 - Cm2 * (c2 / c1) * (ShS) * V2_V1_ratio ** 2 + CL1 * x1 / c1 + CL2 * (x2 / c1) * (ShS) * V2_V1_ratio ** 2) / (CL1 / c1 + (CL2 / c1) * (ShS) * V2_V1_ratio ** 2)
    log = np.append(log, np.array([[x1], [ShS], [x_cg], [x_np]]))

print(log)

with open(os.path.join(download_dir, dict_name), "w") as jsonFile:
    json.dump(data, jsonFile, indent=6)