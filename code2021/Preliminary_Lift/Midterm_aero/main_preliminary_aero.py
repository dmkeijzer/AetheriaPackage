import sys
sys.path.append("../")
from Preliminary_Lift.Airfoil import *
from Preliminary_Lift.Drag import *
from constants import *
from Aero_tools import ISA
import os
import json
root_path = os.path.join(os.getcwd(), os.pardir)
conf = 1

if conf == 1:
    datafile = open(os.path.join(root_path, "data/inputs_config_1.json"), "r")
    data = json.load(datafile)
    datafile.close()
    FP = data["Flight performance"]
    STR = data["Structures"]
    AR = 7
if conf == 2:
    datafile = open(os.path.join(root_path, "data/inputs_config_2.json"), "r")
    data = json.load(datafile)
    datafile.close()
    FP = data["Flight performance"]
    STR = data["Structures"]
    AR = 7
if conf == 3:
    datafile = open(os.path.join(root_path, "data/inputs_config_3.json"), "r")
    data = json.load(datafile)
    datafile.close()
    FP = data["Flight performance"]
    STR = data["Structures"]
    AR = 10.5
# A/C
W = STR["MTOW"] #[N]
Vcruise = 60#FP["V_cruise"] #[m/s]

Wing_loading = FP["WS"]
# ISA
h = 400 # cruise height[m]
atm_flight  = ISA(h)
rho = atm_flight.density() # cte.rho
mu = atm_flight.viscosity_dyn()
a = atm_flight.soundspeed()

# Wing Planform Parameter

S_ref = W/Wing_loading #[m**2] PLACEHOLDER
print("S", S_ref)
b = np.sqrt(AR*S_ref) # Due to reqs
  #PLACEHOLDER
taper = 0.4
sweepc4 =0
# For double wing configurations
s1=0.5
s2=1-s1

S1 = S_ref*s1 #[m**2]  PLACEHOLDER
S2= S_ref*s2 #[m**2]  PLACEHOLDER

sweepc41= 0
sweepc42=0
taper1= 0.4
taper2= 0.4

#Lift-Drag estimation parameters
Cfe = 0.0045
Swet_ratio = 4.5
b_d = b  # fixed due to span limitations
h_d = 0.2*b  # preliminary: for a 0.2 h/b ratio
e_ref = e_OS(AR)
deda = 0.1 # 10%, from Daniel Schitanz, Scholtz

# Airfoil data
NASA_LANGLEY = [6.188, 1.979, -0.065, 0.00445, 0.293, 0.17,0.65] # Lift slope [1/rad], CL_max, C_m cruise, Cd_min, CL for Cdmin, t/c, xcm
EPPLER335 = [6.245, 1.61330, 0.0489, 0.00347, 0.241,0.126,0.199]  # Lift slope [1/rad], CL_max, C_m cruise, Cd_min, CL for Cdmin, t/c, xcm

#Preliminary Lift-Drag Results
CD_0 = C_D_0(Cfe, Swet_ratio)

e_conv = e_factor('normal', h_d, b_d, e_ref)
e_tan = e_factor('tandem', h_d,b_d,e_ref)
e_box = e_factor('box', h_d, b_d, e_ref)

LD_conv = LoD_ratio('cruise', CD_0, AR, e_conv), LoD_ratio('loiter', CD_0, AR, e_conv)
LD_tan = LoD_ratio('cruise', CD_0, AR, e_tan), LoD_ratio('loiter', CD_0, AR, e_tan)
LD_box = LoD_ratio('cruise', CD_0, AR, e_box), LoD_ratio('loiter', CD_0, AR, e_box)

Wing_planform_params_single = wing_planform(AR,S_ref,sweepc4,taper)
Wing_planform_params_double =  wing_planform_double(AR, S1, sweepc41, taper1, S2, sweepc42, taper2)

MAC = Wing_planform_params_single[4]
c_t_double = Wing_planform_params_double[0][2]
"""
Method 1 to find CLdes
CL_Design_conv = C_L('cruise', CD_0,AR,e_conv)
CL_Design_box =C_L('cruise', CD_0,AR,e_box)
CL_Design_tan = C_L('cruise', CD_0,AR,e_tan)

Cl_des_conv = CL_Design_conv/(np.cos(sweep_atx(0,Wing_planform_params_single[1],14,taper,sweepc4)))**2
Cl_des_box = CL_Design_box/(np.cos(sweep_atx(0,Wing_planform_params_double[0][1],14,taper,sweepc4)))**2
Cl_des_tan = CL_Design_tan/(np.cos(sweep_atx(0,Wing_planform_params_double[0][1],14,taper,sweepc4)))**2

V_cruise_conv = np.sqrt(Wing_loading/(0.5*rho*CL_Design_conv))
V_cruise_box = np.sqrt(Wing_loading/(0.5*rho*CL_Design_box))
V_cruise_tan = np.sqrt(Wing_loading/(0.5*rho*CL_Design_tan))
"""
#Method 2
C_L_des = CL_des(rho,Vcruise,W,S_ref)
print('CL=',C_L_des)
Cl_des_conv = C_L_des/(np.cos(sweep_atx(0,Wing_planform_params_single[1],b,taper,sweepc4)))**2
Cl_des_box = C_L_des/(np.cos(sweep_atx(0,Wing_planform_params_double[0][1],b,taper,sweepc4)))**2
Cl_des_tan = C_L_des/(np.cos(sweep_atx(0,Wing_planform_params_double[0][1],b,taper,sweepc4)))**2

Re_Number = Re( rho, Vcruise, Wing_planform_params_single[3], mu), Re( rho, Vcruise, Wing_planform_params_double[0][3], mu), Re( rho, Vcruise, Wing_planform_params_double[1][3], mu)
print('Reynolds number',Re_Number)
#Wing performance
sweepc2_single = sweep_atx(0.5,Wing_planform_params_single[1],Wing_planform_params_single[0],taper,sweepc4)
sweepc2_double = sweep_atx(0.5,Wing_planform_params_double[0][1],Wing_planform_params_double[0][0],taper,sweepc4)

Clda_conv = liftslope('normal', AR, sweepc2_single, Mach(Vcruise,a), EPPLER335[0], s1, s2, deda) # 2pi airfoil slope assumed as placeholder
Clda_double = liftslope('double', AR, sweepc2_double, Mach(Vcruise,a), NASA_LANGLEY[0], s1, s2, deda) # includes in order: total clda, clda wing1, clda wing 2

C_L_max_conv = 0.9* EPPLER335[1]  # From ADSEE-II L2
C_L_max_double = s1*0.9* NASA_LANGLEY[1]+ s2*0.9*1.930  #Due to downwash Clmax for second wing is lower

#Drag estimations
k = 0.634 * 10**(-5) # Smooth paint from adsee 2 L2
flamf =0.1  # From ADSEE 2 L2 GA aircraft
IF_f = 1    # From ADSEE 2 L2
IF_w = 1.1   # From ADSEE 2 L2
flamw = 0.35 # From ADSEE 2 L2 GA aircraft
u = 0.229 #np.pi/180# fuselage upsweep
Abase = 0.04
sweep_xcm_single = sweep_atx(EPPLER335[6],Wing_planform_params_single[1],Wing_planform_params_single[0],taper,sweepc4)
sweep_xcm_double = sweep_atx(NASA_LANGLEY[6],Wing_planform_params_double[0][1],Wing_planform_params_single[0],taper,sweepc4)

LEsweep_single = sweep_atx(0,Wing_planform_params_single[1],b,taper,sweepc4)
LEsweep_double = sweep_atx(0,Wing_planform_params_double[0][1],b,taper,sweepc4)
C_L_finite_single = EPPLER335[4]*(np.cos(LEsweep_single)**2)
C_L_finite_double = NASA_LANGLEY[4]*(np.cos(LEsweep_double)**2)
print("e_conv",e_conv)
class2drag_box = componentdrag('box',S_ref,2,0,2,np.sqrt(1.3*1.6),Vcruise,rho,MAC*0.5,AR,e_box,Mach(Vcruise,a),k,flamf,flamw,mu,NASA_LANGLEY[5],NASA_LANGLEY[6],0,u,c_t_double,h_d,IF_f,IF_w,C_L_des,C_L_finite_double, Abase)

class2drag_tan = componentdrag('tandem',S_ref,2,0,2,np.sqrt(1.3*1.6),Vcruise,rho,MAC*0.5,AR,e_tan,Mach(Vcruise,a),k,flamf,flamw,mu,NASA_LANGLEY[5],NASA_LANGLEY[6],0,u,c_t_double,h_d,IF_f,IF_w,C_L_des,C_L_finite_double,Abase)

class2drag_wing = componentdrag('wing',S_ref,2,0,2,np.sqrt(1.3*1.6),Vcruise,rho,MAC,AR,e_conv,Mach(Vcruise,a),k,flamf,flamw,mu,EPPLER335[5],EPPLER335[6],0,u,0,0,IF_f,IF_w,C_L_des,C_L_finite_single, Abase)

if conf == 1:
    C_D = class2drag_tan.CD()
    C_Dpolar = class2drag_tan.Drag_polar()
    Drag = class2drag_tan.Drag()
    AR_final = AR
    e_final = e_tan
    C_r, C_t, MAC = Wing_planform_params_double[0][1:4]
    LE_sweep = LEsweep_double
    LoD = C_L_des/ C_D
    C_D_u = class2drag_tan.CD_upsweep()
    C_D_b = class2drag_tan.CD_base()
    C_D0 = class2drag_tan.CD0()
if conf == 2:
    C_D = class2drag_box.CD()
    C_D_u = class2drag_box.CD_upsweep()
    C_D_b = class2drag_box.CD_base()
    C_D0 = class2drag_box.CD0()
    C_Dpolar = class2drag_box.Drag_polar()
    Drag = class2drag_box.Drag()
    AR_final = AR
    e_final = e_box
    C_r, C_t, MAC = Wing_planform_params_double[0][1:4]
    LE_sweep = LEsweep_double
    LoD = C_L_des / C_D
    swetf = class2drag_box.Swet_f()
if conf == 3:
    C_D = class2drag_wing.CD()
    C_Dpolar = class2drag_wing.Drag_polar()
    Drag = class2drag_wing.Drag()
    AR_final = AR
    e_final = e_conv
    C_r, C_t , MAC = Wing_planform_params_single[1:4]
    LE_sweep = LEsweep_single
    LoD = C_L_des / C_D
    C_D_u = class2drag_wing.CD_upsweep()
    C_D_b = class2drag_wing.CD_base()
    C_D0 = class2drag_wing.CD0()

print(e_conv)
print("AR= ", AR)
print("e_OS", e_final)
print("C_r,C_t, MAC=", C_r, C_t, MAC)
print("LE sweep=", LE_sweep)

print("Lift slope conv =", Clda_conv)
print("Lift slope double =", Clda_double)

print("C_L_max_single=", C_L_max_conv)
print("C_L_max_double=", 0.9* NASA_LANGLEY[1],  0.9*1.930)

print("C_D", C_D)
print("Drag polar", C_Dpolar)
print("C_D0", C_D0)
print("C_Du", C_D_u)
print("C_Db", C_D_b)
print("C_L for minimum drag", C_L_finite_single, C_L_finite_double)
print("Lift over drag", LoD)
#print(swetf)