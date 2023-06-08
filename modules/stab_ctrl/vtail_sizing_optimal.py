import json
import sys
import pathlib as pl
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.stab_ctrl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size
from modules.stab_ctrl.vee_tail_rudder_elevator_sizing import *
from input.data_structures import *

# WingClass = Wing()
# FuseClass = Fuselage()
# HorTailClass = HorTail()
#
# WingClass.load()
# HorTailClass.load()
# FuseClass.load()

def size_vtail_opt(WingClass, HorTailClass, FuseClass, VTailClass, StabClass, CLh_initguess = -0.4, CLh_step = 0.01, plot = False):
    CLh = CLh_initguess

    log = np.zeros((0,4))

    while True:
        wingloc_ShS, delta_cg_ac = wing_location_horizontalstab_size(WingClass, FuseClass, HorTailClass, CLh_approach=CLh)
        WingClass.load()
        FuseClass.load()
        HorTailClass.load()
        wingloc = wingloc_ShS[0,0]
        ShS = wingloc_ShS[0,1]
        l_v = FuseClass.length_fuselage * (1 - wingloc)
        Vh_V2 = 0.95
        CLh_cr = (WingClass.cm_ac + WingClass.cL_cruise * (delta_cg_ac)/WingClass.chord_mac) / (Vh_V2 * ShS * l_v / WingClass.chord_mac)
        v_tail = get_control_surface_to_tail_chord_ratio(WingClass, FuseClass, HorTailClass, CLh, l_v, Cn_beta_req=-0.0571, beta_h=1, eta_h=0.95, total_deflection=20 * np.pi / 180, design_cross_wind_speed=5.14, step=0.1 * np.pi / 180)
        if type(v_tail[-1]) is str:
            break
        log = np.vstack((log, np.array([CLh, HorTailClass.surface, CLh_cr ** 2], v_tail[-2])))
        print(CLh)
        CLh = CLh - CLh_step
    CLh = CLh + CLh_step
    wing_location_horizontalstab_size(WingClass, FuseClass, HorTailClass, CLh_approach=CLh)
    WingClass.load()
    FuseClass.load()
    HorTailClass.load()
    wingloc = wingloc_ShS[0, 0]
    ShS = wingloc_ShS[0, 1]
    l_v = FuseClass.length_fuselage * (1 - wingloc)
    Vh_V2 = 0.95
    CLh_cr = (WingClass.cm_ac + WingClass.cL_cruise * (delta_cg_ac) / WingClass.chord_mac) / (
                Vh_V2 * ShS * l_v / WingClass.chord_mac)
    v_tail = get_control_surface_to_tail_chord_ratio(WingClass, FuseClass, HorTailClass, CLh, l_v, Cn_beta_req=-0.0571,
                                                     beta_h=1, eta_h=0.95, total_deflection=20 * np.pi / 180,
                                                     design_cross_wind_speed=5.14, step=0.1 * np.pi / 180)
    VTailClass.load()
    VTailClass.CL_cruise = CLh_cr
    VTailClass.length_wing2vtail = l_v
    VTailClass.rudder_max = np.radians(v_tail[0])
    VTailClass.elevator_min = np.radians(v_tail[1])
    VTailClass.dihedral = v_tail[5]
    VTailClass.surface = v_tail[6]
    VTailClass.c_control_surface_to_c_vee_ratio = v_tail[7]
    VTailClass.ruddervator_efficiency = v_tail[2]
    VTailClass.dump()

    StabClass.load()
    StabClass.Cm_de = v_tail[3]
    StabClass.Cn_dr = v_tail[4]
    StabClass.dump()


    if plot:
        print(log[-1,:])
        plt.subplot(131)
        plt.plot(log[:,0], log[:,3])
        plt.xlabel("CLh_approach values")
        plt.ylabel("Sh values")

        plt.subplot(132)
        plt.plot(log[:,0], log[:,2])
        plt.xlabel("CLh_approach values")
        plt.ylabel("CLh_cruise^2 values (~CD)")

        plt.subplot(133)
        plt.plot(log[:,0], log[:,2] * log[:,3])
        plt.xlabel("CLh_approach values")
        plt.ylabel("CLh_cruise^2 values * Sh (~D)")

        plt.show()


