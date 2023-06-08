import json
import sys
import pathlib as pl
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.stab_crtl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size
from scripts.stab_ctrl.vee_tail_rudder_elevator_sizing import *
from input.data_structures import *

WingClass = Wing().load()
FuseClass = Fuselage().load()
HorTailClass = HorTail().load()


CLh = -0.05
log = np.zeros((0,3))

while True:
    wingloc_ShS, delta_cg_ac = wing_location_horizontalstab_size(WingClass, FuseClass, HorTailClass, CLh_approach=CLh)
    wingloc = wingloc_ShS[0,0]
    ShS = wingloc_ShS[0,1]
    l_v = FuseClass.length_fuselage * (1 - wingloc)
    Vh_V2 = 0.95
    CLh_cr = (WingClass.cm_ac + WingClass.cL_cruise * (delta_cg_ac)/WingClass.chord_mac) / (Vh_V2 * ShS * l_v / WingClass.chord_mac)
    v_tail = get_control_surface_to_tail_chord_ratio(WingClass, FuseClass, HorTailClass, CLh, l_v, Cn_beta_req=-0.0571, beta_h=1, eta_h=0.95, total_deflection=20 * np.pi / 180, design_cross_wind_speed=5.14, step=0.1 * np.pi / 180)
    if type(v_tail[-1]) == "str":
        break
    log = np.vstack((log, np.array([CLh, HorTailClass.surface, CLh_cr ** 2])))
    CLh = CLh - 0.005

print(log[-1,:])
plt.subplot(121)
plt.plot(log[:,0], log[:,1])
plt.xlabel("CLh_approach values")
plt.ylabel("Sh values")

plt.subplot(122)
plt.plot(log[:,0], log[:,2])
plt.xlabel("CLh_approach values")
plt.ylabel("CLh_cruise^2 values (~CD)")

plt.show()


