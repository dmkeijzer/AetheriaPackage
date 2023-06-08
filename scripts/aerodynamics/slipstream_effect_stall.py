import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import json
import os
import numpy as np

# Import from modules and input folder
import input.data_structures.GeneralConstants  as const
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



# ----------- CLmax ---------------------------------------------------------------------------------
print("---------- CLmax -------------")
atm = ISA(const.h_transition)
t_cr = atm.temperature()
rho_stall = atm.density()
mhu = atm.viscosity_dyn()
# print(rho_stall)
diameter_propellers = 2*np.sqrt(data['diskarea']/(np.pi*6))
D = diameter_propellers

i_cs_var = 0.0549661449027131 # calculated from lift at cruise

angle_of_attack_stall = 18.2*np.pi/180
angle_of_attack_prop  = angle_of_attack_stall 

# ---------- CRUISE ---------
drag_stall = 0.5*rho_stall*data['S']*const.v_stall*const.v_stall*data['cd']
# drag_stall = 10000
# Thrust coefficient
C_T_var = C_T(T=drag_stall, rho=rho_stall, V_0=const.v_stall, S_W=data["S"])

#change in V
V_delta_var = V_delta(C_T=C_T_var, S_W=data['S'], n_e=3, D=diameter_propellers, V_0=const.v_stall)

# effective Diameter
D_star_var = D_star(D=diameter_propellers, V_0=const.v_stall, V_delta=V_delta_var)

A_eff_var = A_s_eff(b_W=data["b"], S_W=data['S'], n_e=3, D=diameter_propellers, V_0=const.v_stall, V_delta=V_delta_var)[0]

# DATCOM
CL_eff_alpha_var = CL_effective_alpha(mach=data["mach_stall"], A_s_eff= A_eff_var, sweep_half=-data["sweep_le"])

# angles
angles = alpha_s(CL_wing=data["cLmax"], CL_alpha_s_eff=CL_eff_alpha_var, i_cs = i_cs_var, angle_of_attack= angle_of_attack_stall, alpha_0=0, V_0=const.v_stall, V_delta=V_delta_var, delta_alpha_zero_f=(data["delta_alpha_zero_L_flaps60"]))
alpha_s_var = angles[0]
sin_epsilon = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=data["cLmax"], A_w=data["A"])[0]
sin_epsilon_s = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=data["cLmax"], A_w=data["A"])[1]

CL_slipstream_final = CL_ws(S_W=data["S"], b_W=data["b"], n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_stall, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=data["cLmax"])[0]
CL_old = data["cLmax"]
CL_ws_var = CL_ws(S_W=data["S"], b_W=data["b"], n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_stall, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=data["cLmax"])[1]
CL_wing_section = CL_ws(S_W=data["S"], b_W=data["b"], n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_stall, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=data["cLmax"])[2]
CL_s_section = CL_ws(S_W=data["S"], b_W=data["b"], n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_stall, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=data["cLmax"])[3]

prop_lift_var = prop_lift_thrust(T=drag_stall, rho=rho_stall, V_0=const.v_stall, S_W=data['S'], angle_of_attack=(angle_of_attack_stall))

CL_total_cruise = CL_ws_var + prop_lift_var

print("CL_tot:", CL_total_cruise)
print("CL_S:", CL_s_section)
print("CL_wing_section:", CL_wing_section)
print("CL_prop:", prop_lift_var)
print("CL_old:", CL_old)
print("CL percentage increase:", 100*(CL_total_cruise-CL_old)/CL_old)
print(np.sin(epsilon))




