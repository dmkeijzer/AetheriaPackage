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
import input.GeneralConstants as const

# WingClass = Wing()
# FuseClass = Fuselage()
# HorTailClass = HorTail()
#
# WingClass.load()
# HorTailClass.load()
# FuseClass.load()

def size_vtail_opt(WingClass, FuseClass, VTailClass, StabClass, Aeroclass, AircraftClass, PowerClass, EngineClass, b_ref, stepsize=1e-2,  CLh_initguess = -0.1, CLh_step = 0.01, plot = False):
    CLh = CLh_initguess

    dict_log = {
        "clh_lst": [],
        "S_vee_lst": [],
        "span_vee_lst": [],
        "trim_drag_lst": [],
        "aspect_ratio_lst": [],
        "wing_pos_lst": [],
        "ctrl_surf_lst": [],
        "cl_vee_cr_lst": []
    }


    for A_h in np.arange(2, 8, 0.2):
        VTailClass.aspect_ratio = A_h
        CLh = CLh_initguess
        while True:
            shs, wing_loc, cg_front_bar, cg_aft_bar, cg_dict_margin = wing_location_horizontalstab_size(WingClass, FuseClass,Aeroclass, VTailClass, AircraftClass, PowerClass, EngineClass, StabClass,  A_h, CLh_approach=CLh, stepsize= stepsize)
            l_v = FuseClass.length_fuselage * (1 - wing_loc)
            Vh_V2 = 0.95*(1 + const.axial_induction_factor1)**2
            control_surface_data = get_control_surface_to_tail_chord_ratio(WingClass, FuseClass,VTailClass, Aeroclass, CLh, l_v, Cn_beta_req=0.0571, beta_h=1, eta_h=0.95, total_deflection=20 * np.pi / 180, design_cross_wind_speed=5.14, step=0.1 * np.pi / 180)
            CLvee_cr_N = (Aeroclass.cm_ac + Aeroclass.cL_cruise * (abs(cg_aft_bar - cg_front_bar))) / (Vh_V2 * control_surface_data["S_vee"]/WingClass.surface *np.cos(control_surface_data["dihedral"]) * l_v / WingClass.chord_mac)
    
            if type(control_surface_data["control_surface_ratio"]) is str:
                break

            dict_log["clh_lst"].append(CLh)
            dict_log["S_vee_lst"].append(control_surface_data["S_vee"])
            dict_log["span_vee_lst"].append(np.sqrt(A_h * control_surface_data["S_vee"]))
            dict_log["trim_drag_lst"].append(CLvee_cr_N ** 2 * control_surface_data["S_vee"]/A_h)
            dict_log["aspect_ratio_lst"].append(A_h)
            dict_log["wing_pos_lst"].append((shs, wing_loc, cg_front_bar, cg_aft_bar, cg_dict_margin))
            dict_log["ctrl_surf_lst"].append(control_surface_data)
            dict_log["cl_vee_cr_lst"].append(CLvee_cr_N)

            #Move to next step
            CLh = CLh - CLh_step
        
    design_idx = np.argmin(np.array(dict_log["trim_drag_lst"])[dict_log["span_vee_lst"] > b_ref])

    CLh = dict_log["clh_lst"][design_idx]
    b_vee = dict_log["span_vee_lst"][design_idx]
    Ah = dict_log["aspect_ratio_lst"][design_idx]
    Shs, wing_loc, cg_front_bar, cg_aft_bar, cg_dict= dict_log["wing_pos_lst"][design_idx]
    ctrl_surf_data = dict_log["ctrl_surf_lst"][design_idx]
    cl_vee_cr = dict_log["cl_vee_cr_lst"][design_idx]

    l_v = FuseClass.length_fuselage * (1 - wing_loc)
    Vh_V2 = 0.95*(1+const.axial_induction_factor2)**2

    AircraftClass.oem_cg = cg_dict["oem_cg"]
    AircraftClass.wing_loc = wing_loc
    AircraftClass.cg_front = cg_dict["frontcg"]
    AircraftClass.cg_rear = cg_dict["rearcg"]
    AircraftClass.cg_front_bar = cg_front_bar
    AircraftClass.cg_rear_bar =  cg_aft_bar
    StabClass.cg_front_bar = cg_front_bar
    StabClass.cg_rear_bar =  cg_aft_bar
    
    VTailClass.cL_h_cruise = cl_vee_cr 
    VTailClass.length_wing2vtail = l_v
    VTailClass.rudder_max = np.radians(ctrl_surf_data["max_rudder_angle"])
    VTailClass.elevator_min = np.radians(ctrl_surf_data["min_elevator_angle"])
    VTailClass.dihedral = ctrl_surf_data["dihedral"]
    VTailClass.surface = ctrl_surf_data["S_vee"]
    VTailClass.c_control_surface_to_c_vee_ratio = ctrl_surf_data["control_surface_ratio"]
    VTailClass.ruddervator_efficiency = ctrl_surf_data["tau"]
    VTailClass.span = b_vee
    VTailClass.aspect_ratio = Ah

    StabClass.Cm_de = control_surface_data["cm_de"]
    StabClass.Cn_dr = control_surface_data["cn_dr"]

    return WingClass, FuseClass, VTailClass, StabClass


