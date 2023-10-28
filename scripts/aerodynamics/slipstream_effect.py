import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import json
import os
import numpy as np

# Import from modules and input folder
import input.GeneralConstants  as const
from  modules.aero.prop_wing_interaction  import *
from input.data_structures.ISA_tool import ISA

os.chdir(str(list(pl.Path(__file__).parents)[2]))
# import CL_cruise from json files

# Define the directory and filenames of the JSON files
TEST = int(input("\n\nType 1 if you want to write the JSON data to your downlo1ad folder instead of the repo, type 0 otherwise:\n")) # Set to true if you want to write to your downloads folders instead of rep0
dict_directory = "input/data_structures"
file_name = "aetheria_constants.json"

with open(os.path.join(dict_directory, file_name)) as f:
    data = json.load(f)
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

atm = ISA(const.h_cruise)
t_cr = atm.temperature()
rho_cr = atm.density()
mhu = atm.viscosity_dyn()

diameter_propellers = 2*np.sqrt(data['diskarea']/(np.pi*6))
D = diameter_propellers
# ---------- CRUISE ---------
drag_cruise = 0.5*rho_cr*data['S']*const.v_cr*const.v_cr*data['cd']

# Thrust coefficient
C_T_var = C_T(T=drag_cruise, rho=const.rho_cr, V_0=const.v_cr, S_W=data["S"])

#change in V
V_delta_var = V_delta(C_T=C_T_var, S_W=data['S'], n_e=4, D=D, V_0=const.v_cr)

# effective Diameter
D_star_var = D_star(D=1.9, V_0=const.v_cr, V_delta=V_delta_var)

A_eff_var = A_s_eff(b_W=data["b"], S_W=data['S'], n_e=4, D=D, V_0=const.v_cr, V_delta=V_delta_var)[0]

# DATCOM
CL_eff_alpha_var = CL_effective_alpha(mach=data["mach_cruise"], A_s_eff= A_eff_var, sweep_half=-data["sweep_le"])

# angles
angles = alpha_s(CL_wing=data["cL_cruise"], CL_alpha_s_eff=CL_eff_alpha_var, alpha_0=data['alpha_zero_L'], V_0=const.v_cr, V_delta=V_delta_var, delta_alpha_zero_f=0)
alpha_s_var = angles[0]
sin_epsilon = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=data["cL_cruise"], A_w=data["A"])[0]
sin_epsilon_s = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=data["cL_cruise"], A_w=data["A"])[1]

CL_slipstream_final = CL_ws(S_W=data["S"], b_W=data["b"], n_e=4, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_cr, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=data["cL_cruise"])[1]



prop_lift_var = prop_lift_thrust(T=drag_cruise, rho=rho_cr, V_0=const.v_cr, S_W=data['S'], angle_of_attack= angles[1])

CL_total_cruise = CL_slipstream_final + prop_lift_var
print(CL_total_cruise)


# ----------- CLmax ---------

# drag_clmax = 0.5*const.rho_sl*data['S']*const.v_stall*const.v_stall*data['cd_stall']

# slipstream_lift_stall = prop_lift_slipstream(mach=data['mach_stall'], sweep_half=-data['sweep_le'], T=drag_clmax, S_W=data['S'], n_e=4, D=diameter_propellers, b_W=data['b'], V_0=const.v_stall, CL_wing=data['cLmax_flaps60'], i_cs=0, rho=const.rho_sl, delta_alpha_zero_f=1.0, alpha_0=data['alpha_zero_L'])

# prop_lift_stall = prop_lift_thrust(T=drag_clmax, rho=const.rho_cr, V_0=const.v_stall, S_W=data['S'], angle_of_attack=slipstream_lift_var[2])
# CL_total_stall = slipstream_lift_stall[1] + prop_lift_stall
# print(CL_total_stall)

