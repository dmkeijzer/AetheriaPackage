import sys
import pathlib as pl

import json
import os
import numpy as np
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))
# Import from modules and input folder
import input.data_structures.GeneralConstants  as const
from  modules.aero.prop_wing_interaction  import *
from input.data_structures.ISA_tool import ISA


# import CL_cruise from json files




def slipstream_cruise(WingClass, AeroClass, mission):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
                data = json.load(jsonFile)
        atm = ISA(const.h_cruise)
        t_cr = atm.temperature()
        rho_cr = atm.density()
        mhu = atm.viscosity_dyn()
        # print(rho_cr)
        diskarea = mission.MTOM / 120 
        diameter_propellers = 2*np.sqrt(diskarea / np.pi )
        D = diameter_propellers

        # Angles
        i_cs_var = 0.0733 # calculated from lift at cruise
        angle_of_attack_fuse = 0
        angle_of_attack_prop = angle_of_attack_fuse + i_cs_var

        # ---------- CRUISE ---------
        drag_cruise = 0.5*rho_cr*WingClass.surface*const.v_cr*const.v_cr*AeroClass.cd_cruise
        # drag_cruise = 0
        # Thrust coefficient
        C_T_var = 0.15

        #change in V
        V_delta_var = V_delta(C_T=C_T_var, S_W=WingClass.surface, n_e=3, D=diameter_propellers, V_0=const.v_cr)

        # effective Diameter
        D_star_var = D_star(D=diameter_propellers, V_0=const.v_cr, V_delta=V_delta_var)

        A_eff_var = A_s_eff(b_W=WingClass.span, S_W=WingClass.surface, n_e=3, D=diameter_propellers, V_0=const.v_cr, V_delta=V_delta_var)[0]

        # DATCOM
        CL_eff_alpha_var = CL_effective_alpha(mach=AeroClass.mach_cruise, A_s_eff= A_eff_var, sweep_half=-WingClass.sweep_LE)

        # angles
        angles = alpha_s(CL_wing=AeroClass.cL_cruise, CL_alpha_s_eff=CL_eff_alpha_var, i_cs = i_cs_var, angle_of_attack= angle_of_attack_prop, alpha_0=data['alpha_zero_L'], V_0=const.v_cr, V_delta=V_delta_var, delta_alpha_zero_f=0)
        alpha_s_var = angles[0]

        sin_epsilon = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=AeroClass.cL_cruise, A_w=data["A"])[0]
        sin_epsilon_s = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=AeroClass.cL_cruise, A_w=data["A"])[1]

        CL_slipstream_final = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_cruise)[0]
        CL_old = AeroClass.cL_cruise

        CL_ws_var = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_cruise)[1]
        CL_wing_section = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_cruise)[2]
        CL_s_section = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_cruise)[3]

        prop_lift_var = prop_lift_thrust(T=drag_cruise, rho=rho_cr, V_0=const.v_cr, S_W=WingClass.surface, angle_of_attack=angle_of_attack_prop)

        CL_total_cruise = CL_ws_var + prop_lift_var

        downwash_angle_wing = np.sin(sin_epsilon)
        downwash_angle_prop = np.sin(sin_epsilon_s)
        average_downwash_angle = (downwash_angle_prop*diameter_propellers*3 + downwash_angle_wing*(WingClass.span-3*diameter_propellers))/WingClass.span

        AeroClass.downwash_angle = average_downwash_angle
        AeroClass.downwash_angle_wing = downwash_angle_wing
        AeroClass.downwash_angle_prop = downwash_angle_prop
        AeroClass.cL_plus_slipstream = CL_total_cruise
        #     print(CL_wing_section, CL_s_section, CL_old, prop_lift_var)
        return AeroClass

