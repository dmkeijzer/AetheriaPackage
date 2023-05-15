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

datafile = open(os.path.join(root_path, "data/inputs_config_1.json"), "r")
data = json.load(datafile)
datafile.close()
FP = data["Flight performance"]
STR = data["Structures"]
AR = 5


W = STR["MTOW"] #[N]
Vcruise = 60#FP["V_cruise"] #[m/s]
Wing_loading = FP["WS"]

#Cruise conditions
h = 400 # cruise height[m]
atm_flight  = ISA(h)
rho = atm_flight.density() # cte.rho
mu = atm_flight.viscosity_dyn()
a = atm_flight.soundspeed()
M = Mach(Vcruise,a)

#Wing planform
S_ref = W/Wing_loading #[m**2] PLACEHOLDER
print("S", S_ref)
b = np.sqrt(AR*S_ref) # Due to reqs

# For double wing
s1=0.5
s2=1-s1
#Sweep
sweepc41= 0
sweepc42=0

#Other paramters
b_d = b  # fixed due to span limitations
h_d = 1.4  #  Vertical gap between wings. Based on fuselage size
l_h = 5 # Horizontal gap between wings. Based on fuselgae size

#Fuselage dimensions
l1 = 2.5
l2 = 2
l3 = 2.7
w_max = 1.38
h_max = 1.705
d_eq = np.sqrt(h_max*w_max)
#Winglets
h_wl1 =0
h_wl2 = 0
Wing_params = wing_design(AR,s1,sweepc41,s2,sweepc42,M,S_ref, l_h,h_d,w_max,h_wl1,h_wl2)
b = Wing_params.wing_planform_double()[0][0]
C_r = Wing_params.wing_planform_double()[0][1]
C_t = Wing_params.wing_planform_double()[0][2]
print(b,C_r,C_t)
MAC = Wing_params.wing_planform_double()[0][3]
SweepLE = Wing_params.sweep_atx(0)[0]
Slope1 = Wing_params.liftslope()[1]

CLmax = Wing_params.CLmax_s()

#For Drag estimation
k = 0.634 * 10**(-5) # Smooth paint from adsee 2 L2
flamf =0.1  # From ADSEE 2 L2 GA aircraft
IF_f = 1    # From ADSEE 2 L2
IF_w = 1.1   # From ADSEE 2 L2
IF_v = 1.04 #From ADSEE 2 L2
flamw = 0.35 # From ADSEE 2 L2 GA aircraft
u = 8.43 *np.pi/180 # fuselage upsweep
Abase = 0
# Airfoil
airfoil = airfoil_stats()
tc = 0.12 #NACA0012 for winglets and Vtail
xcm = 0.3 #NACA0012 for winglets and Vtail
CL_CDmin = airfoil[2]
CL_lst = np.arange(0,1.3,0.001)
#Other parameters
S_v = 0.6

h_list = b* np.array([0, 0.025,0.05,0.1,0.15,0.2])

for h_wl1 in h_list:
    Drag = componentdrag('tandem',S_ref,l1,l2,l3,d_eq,Vcruise,rho,MAC,AR,Mach(Vcruise,a),k,flamf,flamw,mu,tc,xcm,0,SweepLE,u,C_t,h_d,IF_f,IF_w, IF_v, CL_CDmin,Abase, S_v, s1, s2, h_wl1, h_wl1)
    if h_wl1 == 0:
        CD_lst0 = Drag.CD(CL_lst)
        print("lmao1")
    if h_wl1/b == 0.025:
        CD_lst1 = Drag.CD(CL_lst)
        print("lmao2")
    if h_wl1/b == 0.05:
        CD_lst2 = Drag.CD(CL_lst)
        print("lmao3")
    if h_wl1/b == 0.1:
        CD_lst3 = Drag.CD(CL_lst)
        print("lmao4")
    if h_wl1/b == 0.15:
        CD_lst4 = Drag.CD(CL_lst)
        print("lmao4")
    if h_wl1/b == 0.2:
        CD_lst5 = Drag.CD(CL_lst)
        print("lmao5")

#plt.plot(CL_lst, CD_lst0)
#plt.plot(CL_lst, CD_lst1)
#plt.plot(CL_lst, CD_lst2)
#plt.plot(CL_lst, CD_lst3)
#plt.plot(CL_lst, CD_lst4)
#plt.plot(CL_lst, CD_lst5)
#plt.show()

h_lst2 = b*np.arange(0,0.2,0.01)
CL_lst = []
LD_lst = []
CL_desmin = Wing_loading/(0.5*rho*(58**2))
for h_wl1 in h_lst2:
    Drag = componentdrag('tandem',S_ref,l1,l2,l3,d_eq,Vcruise,rho,MAC,AR,Mach(Vcruise,a),k,flamf,flamw,mu,tc,xcm,0,SweepLE,u,C_t,h_d,IF_f,IF_w, IF_v, CL_CDmin,Abase, S_v, s1, s2, h_wl1, h_wl1)
    CL_des = Drag.CL_des()[0]
    LDmax = Drag.CL_des()[1]
    CL_lst.append(CL_des)
    LD_lst.append(LDmax)

plt.plot(h_lst2/b, LD_lst)
#plt.axhline(CL_desmin)
plt.show()
plt.plot(h_lst2/b, CL_lst)
plt.axhline(CL_desmin)
plt.show()