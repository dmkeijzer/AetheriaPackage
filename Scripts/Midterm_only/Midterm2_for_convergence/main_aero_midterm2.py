import sys
sys.path.append("../")
from Preliminary_Lift.Airfoil import *
from Preliminary_Lift.Drag import *
from Aero_tools import ISA
from Preliminary_Lift.Wing_design import *
from Preliminary_Lift.Airfoil_analysis import airfoil_stats
import os
import json
import matplotlib.pyplot as plt
root_path = os.path.join(os.getcwd(), os.pardir)
import constants as const

datafile = open(os.path.join(root_path, "data/inputs_config_1.json"), "r")
data = json.load(datafile)
datafile.close()
FP = data["Flight performance"]
STR = data["Structures"]
AR = 4


W = 2602*const.g  # STR["MTOW"] #[N]
Vcruise = 62  # FP["V_cruise"] #[m/s]
Wing_loading = W/14.9  # FP["WS"]

#C ruise conditions
h = 1000  # cruise height[m]
atm_flight = ISA(h)
rho = atm_flight.density()  # cte.rho
mu = atm_flight.viscosity_dyn()
a = atm_flight.soundspeed()
M = Mach(Vcruise, a)

# Wing planform
S_ref = 14.9  # [m**2] PLACEHOLDER
# print("S", S_ref)
b = np.sqrt(AR*S_ref) # Due to reqs

# For double wing
s1 = 0.5
s2 = 1-s1
# Sweep
sweepc41 = 0
sweepc42 = 0

# Other paramters
b_d = b  # fixed due to span limitations
h_d = 1.4  # Vertical gap between wings. Based on fuselage size
l_h = 5  # Horizontal gap between wings. Based on fuselage size
e_ref = e_OS(AR)
e = e_factor('tandem', h_d, b_d, e_ref)
# Fuselage dimensions
l1 = const.l_nosecone
l2 = const.l_cylinder
l3 = const.l_tailcone
w_max = const.w_fuselage
h_max = const.h_fuselage
d_eq = np.sqrt(h_max*w_max)
# Winglets
h_wl1 = 0
h_wl2 = 0
Wing_params = wing_design(AR, s1, sweepc41, s2, sweepc42, M, S_ref, l_h, h_d, w_max, h_wl1, h_wl2)
b = Wing_params.wing_planform_double()[0][0]
C_r = Wing_params.wing_planform_double()[0][1]
C_t = Wing_params.wing_planform_double()[0][2]
# print(b, C_r, C_t)
MAC = Wing_params.wing_planform_double()[0][3]
SweepLE = Wing_params.sweep_atx(0)[0]
Slope1 = Wing_params.liftslope()[1]

CLmax = Wing_params.CLmax_s()

# For Drag estimation
k = 0.634 * 10**(-5)  # Smooth paint from adsee 2 L2
flamf = 0.1  # From ADSEE 2 L2 GA aircraft
IF_f = 1    # From ADSEE 2 L2
IF_w = 1.1   # From ADSEE 2 L2
IF_v = 1.04  # From ADSEE 2 L2
flamw = 0.35  # From ADSEE 2 L2 GA aircraft
u = 8.43 * np.pi/180  # fuselage upsweep
Abase = 0
# Airfoil
airfoil = airfoil_stats()
tc = 0.12  # NACA0012 for winglets and Vtail
xcm = 0.3  # NACA0012 for winglets and Vtail
CL_CDmin = airfoil[2]
CL_lst = np.arange(-0.5, 1.7, 0.100)
# Other parameters
S_v = 0.6
S_t = 0


Drag = componentdrag('tandem',S_ref,l1,l2,l3,d_eq,Vcruise,rho,MAC,AR,e,Mach(Vcruise,a),k,flamf,flamw,mu,tc,xcm,0,SweepLE,u,0,h_d,IF_f,IF_w, IF_v, CL_CDmin,Abase, S_v, s1, s2, h_wl1, h_wl2)


CL_design = Drag.CL_des()[0]
Cd_des = Drag.Cd_w(CL_design)
# print("CL_des", CL_design)
# print("Cd", Cd_des)
# Stall
stall = Wing_params.CLmax_s()
CLmax = stall[0]

CDs = Drag.CD(CLmax)
CDs_f = Drag.CD0_f
CDs_w = CDs - CDs_f
# Post stall
Afus = np.pi * d_eq**2/4
post_stall = Wing_params.post_stall_lift_drag(tc, CDs, CDs_f, Afus)

alpha_lst = np.arange(0, 89, 0.1)
Cl_alpha_curve = Wing_params.CLa(tc, CDs, CDs_f, Afus, alpha_lst)

CD_a_w = Wing_params.CDa_poststall(tc, CDs, CDs_f, Afus, alpha_lst, "wing", Drag.CD)
CD_a_f = Wing_params.CDa_poststall(tc, CDs, CDs_f, Afus, alpha_lst, "fus", Drag.CD)

# plt.plot(alpha_lst, CD_a_w)
# plt.show()










