import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import json
import os
import numpy as np

# Import from modules and input folder
import input.data_structures.GeneralConstants as const
from  modules.aero.prop_wing_interaction  import *
from input.data_structures.ISA_tool import ISA

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))
# import CL_cruise from json files

AeroClass = Aero()
FuselageClass = Fuselage()
VTailClass = VeeTail()
HorTailClass = HorTail()
AeroClass.load()
FuselageClass.load()
VTailClass.load()
HorTailClass.load()

# ----------- CLmax ---------------------------------------------------------------------------------

atm = ISA(const.h_transition)
t_cr = atm.temperature()
rho_stall = atm.density()
mhu = atm.viscosity_dyn()
# print(rho_stall)
diameter_propellers = 2*np.sqrt(const.diskarea/(np.pi*6))
D = diameter_propellers

i_cs_var = 0.0549661449027131 # calculated from lift at cruise

angle_of_attack_stall = 18.2*np.pi/180
angle_of_attack_prop  = angle_of_attack_stall 

# ---------- STALL ---------
drag_stall = 0.5*rho_stall*WingClass.surface*const.v_stall*const.v_stall*AeroClass.cd_stall
# drag_stall = 10000
# Thrust coefficient
C_T_var = C_T(T=drag_stall, rho=rho_stall, V_0=const.v_stall, S_W=WingClass.surface)

#change in V
V_delta_var = V_delta(C_T=C_T_var, S_W=WingClass.surface, n_e=3, D=diameter_propellers, V_0=const.v_stall)

# effective Diameter
D_star_var = D_star(D=diameter_propellers, V_0=const.v_stall, V_delta=V_delta_var)

A_eff_var = A_s_eff(b_W=WingClass.span, S_W=WingClass.surface, n_e=3, D=diameter_propellers, V_0=const.v_stall, V_delta=V_delta_var)[0]

# DATCOM
CL_eff_alpha_var = CL_effective_alpha(mach=AeroClass.mach_stall, A_s_eff= A_eff_var, sweep_half=-WingClass.sweep_LE)

# angles
angles = alpha_s(CL_wing=AeroClass.cL_max, CL_alpha_s_eff=CL_eff_alpha_var, i_cs = i_cs_var, angle_of_attack= angle_of_attack_stall, alpha_0=0, V_0=const.v_stall, V_delta=V_delta_var, delta_alpha_zero_f=(data["delta_alpha_zero_L_flaps60"]))
alpha_s_var = angles[0]
sin_epsilon = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=AeroClass.cL_max, A_w=WingClass.aspectratio)[0]
sin_epsilon_s = sin_epsilon_angles(CL_alpha_s_eff=CL_eff_alpha_var, alpha_s=alpha_s_var, A_s_eff=A_eff_var, CL_wing=AeroClass.cL_max, A_w=WingClass.aspectratio)[1]

CL_slipstream_final = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_stall, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_max)[0]
CL_old = AeroClass.cL_max
CL_ws_var = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_stall, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_max)[1]
CL_wing_section = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_stall, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_max)[2]
CL_s_section = CL_ws(S_W=WingClass.surface, b_W=WingClass.span, n_e=3, D_star=D_star_var, sin_epsilon=sin_epsilon, V_0=const.v_stall, V_delta=V_delta_var, sin_epsilon_s=sin_epsilon_s, CL_wing=AeroClass.cL_max)[3]

prop_lift_var = prop_lift_thrust(T=drag_stall, rho=rho_stall, V_0=const.v_stall, S_W=WingClass.surface, angle_of_attack=(angle_of_attack_stall))

CL_total_cruise = CL_ws_var + prop_lift_var

print("CL_tot:", CL_total_cruise)
print("CL_S:", CL_s_section)
print("CL_wing_section:", CL_wing_section)
print("CL_prop:", prop_lift_var)
print("CL_old:", CL_old)
print("CL percentage increase:", 100*(CL_total_cruise-CL_old)/CL_old)


downwash_angle_wing = np.sin(sin_epsilon)
downwash_angle_prop = np.sin(sin_epsilon_s)
average_downwash_angle = (downwash_angle_prop*diameter_propellers*3 + downwash_angle_wing*(data['b']-3*diameter_propellers))/data['b']

AeroClass.downwash_angle_wing_stall = downwash_angle_wing 
AeroClass.downwash_angle_prop_stall = downwash_angle_prop 
AeroClass.downwash_angle_stall = average_downwash_angle 



    



