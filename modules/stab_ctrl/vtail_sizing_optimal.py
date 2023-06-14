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

def size_vtail_opt(WingClass, HorTailClass, FuseClass, VTailClass, StabClass, Aeroclass, b_ref, stepsize=1e-2,  CLh_initguess = -0.1, CLh_step = 0.01, plot = False):
    CLh = CLh_initguess

    log = np.zeros((0,5))
    for A_h in np.arange(2, 8, 0.2):
        CLh = CLh_initguess
        while True:
            wingloc_ShS, delta_cg_ac = wing_location_horizontalstab_size(WingClass, FuseClass, HorTailClass,Aeroclass,StabClass,  A_h, CLh_approach=CLh, stepsize= stepsize)
            WingClass.load()
            FuseClass.load()
            HorTailClass.load()
            StabClass.load()
            Aeroclass.load()
            wingloc = wingloc_ShS[0,0]
            l_v = FuseClass.length_fuselage * (1 - wingloc)
            axial_induction_factor=0.2
            Vh_V2 = 0.95*(1+axial_induction_factor)**2
            v_tail = get_control_surface_to_tail_chord_ratio(WingClass, FuseClass, HorTailClass, Aeroclass, CLh, l_v, Cn_beta_req=0.0571, beta_h=1, eta_h=0.95, total_deflection=20 * np.pi / 180, design_cross_wind_speed=5.14, step=0.1 * np.pi / 180)
            CLvee_cr_N = (Aeroclass.cm_ac + Aeroclass.cL_cruise * (delta_cg_ac)/WingClass.chord_mac) / (Vh_V2 * v_tail[-2]/WingClass.surface *np.cos(v_tail[-3]) * l_v / WingClass.chord_mac)
    
            if type(v_tail[-1]) is str:
                break
            log = np.vstack((log, np.array([CLh, v_tail[6], np.sqrt(A_h * v_tail[6]), CLvee_cr_N ** 2 * v_tail[6]/A_h ,A_h])))
            CLh = CLh - CLh_step
        
    log = log[np.where(log[:,2] > b_ref)[0], :]
    log = log[np.where(log[:,2] < 5)[0], :]
    log = log[np.where(log[:,3] == np.min(log[:,3]))[0], :]
    CLh = log[0,0]
    b_vee = log[0,2]
    Ah = log[0,4]
    wing_location_horizontalstab_size(WingClass, FuseClass, HorTailClass,Aeroclass,StabClass, CLh_approach=CLh,A_h=Ah, stepsize = 1e-1)
    WingClass.load()
    FuseClass.load()
    HorTailClass.load()
    wingloc = wingloc_ShS[0, 0]
    l_v = FuseClass.length_fuselage * (1 - wingloc)
    axial_induction_factor=0.005
    Vh_V2 = 0.95*(1+axial_induction_factor)**2


    v_tail = get_control_surface_to_tail_chord_ratio(WingClass, FuseClass, HorTailClass, Aero = Aeroclass, CL_h= CLh,l_v= l_v, Cn_beta_req=0.0571,
                                                     beta_h=1, eta_h=0.95, total_deflection=20 * np.pi / 180,
                                                     design_cross_wind_speed=5.14, step=0.1 * np.pi / 180)
    CLvee_cr_N = (Aeroclass.cm_ac + Aeroclass.cL_cruise * (delta_cg_ac) / WingClass.chord_mac) / (
                Vh_V2 * v_tail[-2] / WingClass.surface * np.cos(v_tail[-3]) * l_v / WingClass.chord_mac)
    VTailClass.load()
    VTailClass.CL_cruise = CLvee_cr_N
    VTailClass.length_wing2vtail = l_v
    VTailClass.rudder_max = np.radians(v_tail[0])
    VTailClass.elevator_min = np.radians(v_tail[1])
    VTailClass.dihedral = v_tail[5]
    VTailClass.surface = v_tail[6]
    VTailClass.c_control_surface_to_c_vee_ratio = v_tail[7]
    VTailClass.ruddervator_efficiency = v_tail[2]
    VTailClass.span = b_vee
    VTailClass.dump()
    print(VTailClass.surface)

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
        plt.ylabel("CL of vee tail during cruise normal to its surface (~CD)")

        plt.subplot(133)
        plt.plot(log[:,0], log[:,2] * log[:,3])
        plt.xlabel("CLvee_cruise_N^2 values * Svee (~D)")

        plt.show()

    return WingClass, HorTailClass, FuseClass, VTailClass, StabClass


