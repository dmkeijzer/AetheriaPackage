import numpy as np

from Code2021.Final_optimization.Aero_tools import ISA
from Code2021.Preliminary_Lift.Airfoil import Mach
from drag import *

AR1 = 9.5
AR2 = 10

W = 2510.  # STR["MTOW"] #[N]
Vcruise = 83.33  # FP["V_cruise"] #[m/s]
Vstall = 40

# Cruise conditions
h = 400  # cruise height[m]
atm_flight = ISA(h)
rho = atm_flight.density()  # cte.rho
mu = atm_flight.viscosity_dyn()
a = atm_flight.soundspeed()
M = Mach(Vcruise, a)
print("Mach", M)

# Wing planform
S_ref = 14  # W/Wing_loading #[m**2] PLACEHOLDER


CLmax = 1.5856096132929682
Wing_loading = 0.5 * rho * Vstall * Vstall * CLmax

# For double wing
s1 = 0.5

s2 = 1-s1
b = np.sqrt(0.5*(AR1*s1+s2*AR2)*S_ref)  # Due to reqs
print(b)
# Sweep
sweepc41 = 0
sweepc42 = 0

# Other paramters
b_d = b  # fixed due to span limitations
h_d = 1.4  # Vertical gap between wings. Based on fuselage size
l_h = 5.6  # Horizontal gap between wings. Based on fuselgae size
i1 = -0.0

# Fuselage dimensions
l1 = 2.5
l2 = 2.14888
l3 = 2.7
w_max = 1.38
h_max = 1.70
d_eq = np.sqrt(h_max*w_max)
# Winglets
h_wl1 = 0.5  # 0.5
h_wl2 = 0.5  # 0.5
k_wl = 2.0
# 7 9 0.45454545454545453 0 0.5454545454545454 0 0.1961932635918894 18.379085418840855 7.0 1.4 1.38 0.5 0.5 2.0 0
Wing_params = wing_design(AR1, AR2, s1, sweepc41, s2, sweepc42,
                          M, S_ref, l_h, h_d, w_max, h_wl1, h_wl2, k_wl, i1)
b = Wing_params.wing_planform_double()[0][0]
C_r = Wing_params.wing_planform_double()[0][1]
C_t = Wing_params.wing_planform_double()[0][2]

MAC = Wing_params.wing_planform_double()[1][3]
print("YE", b, C_r, C_t, MAC)
SweepLE = Wing_params.sweep_atx(0)[0]
# downwash(Wing_params.wing_planform_double()[0][0] , Wing_params.AR1,
deda = 0.352
#  Wing_params.wing_planform_double()[0][1], Wing_params.wing_planform_double()[0][2], Wing_params.sweepc41, 5, Wing_params.h_ht,
#  Wing_params.lh, Wing_params.wing_planform_double()[1][0] ,
#  Wing_params.wing_planform_double()[0][1], Wing_params.wing_planform_double()[0][2], Wing_params.sweepc41,
#  70)  # deps_da(self.sweepc41, wg[0][0], self.lh, self.h_ht, self.AR_i, slope1)


Slope1 = Wing_params.liftslope(deda)[1]

CLmax = Wing_params.CLmax_s(deda)

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
tca = 0.17
xcm = 0.3  # NACA0012 for winglets and Vtail
CL_CDmin = airfoil[2]
CL_lst = np.arange(-0.5, 1.8, 0.025)
# Other parameters
S_v = 1.614
S_t = 0


Drag = componentdrag('tandem', S_ref, l1, l2, l3, d_eq, Vcruise, rho, MAC, AR1, AR2, Mach(Vcruise, a), k, flamf,
                     flamw, mu, tc, xcm, 0, SweepLE, u, 0, h_d, IF_f, IF_w, IF_v, CL_CDmin, Abase, S_v, s1, s2, h_wl1, h_wl2, k_wl)
