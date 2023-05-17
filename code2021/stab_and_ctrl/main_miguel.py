"""
In this file, every python script is runned.

Author: Michal Cuadrat-Grzybowski
"""

import numpy as np
# from stab_and_ctrl.Scissor_Plots import Wing_placement_sizing
from stab_and_ctrl.Vertical_tail_sizing import VT_sizing
from stab_and_ctrl.Aileron_Sizing import Control_surface
from stab_and_ctrl.Elevator_sizing import Elevator_sizing
from stab_and_ctrl.Stability_derivatives import Stab_Derivatives
from structures.Weight import *
from stab_and_ctrl.Model_cruise import Aircraft
from Aero_tools import ISA
import constants as const
from matplotlib import pyplot as plt

# example values based on inputs_config_1.json
W = 3024.8012022968796*9.80665
W = 2793.7948931207516*9.80665
h = 1000
ISA_atm = ISA(h)
T = ISA_atm.temperature()
lfus = 7.348878876267166
hfus = 1.705
wfus = 1.38
xcg = 2.604
V0 = 72.18676185339652
M0= V0/(np.sqrt(1.4*287*T))
Vstall = 40
Pbr = 110024/1.2 * 0.9 /12
# M0 = V0 / np.sqrt(const.gamma * const.R * 288.15)
CD0 = 0.03254
CLfwd = 1.9044877866958172
CLrear =  1.2569734761848055
CLdesfwd = 0.5158389632834982
CLdesrear = 0.5158389632834982
Clafwd = 6.028232202020173
Clarear=Clafwd
Cd0fwd = 0.00347 # Airfoil drag coefficient [-]
Cd0rear = Cd0fwd
CD0fwd = 0.008558896967176247 # Wing drag coefficient [-]
CD0rear = CD0fwd
Cmacfwd = -0.0645
Cmacrear = -0.0645
Sfwd = 9.910670535618632
Srear = 9.910670535618632
# Srear = 8.417113787320769
S = Srear+Sfwd
Gamma = 0
Lambda_c4_fwd = 0.0*np.pi/180
Lambda_c4_rear = 0.0*np.pi/180
cfwd = 1.265147796494402
crear = 1.265147796494402
bfwd = 8.209297146662843
brear = 8.209297146662843
Afwd =bfwd**2/Sfwd
Arear = brear**2/Srear
efwd = 0.65
erear = 0.65
taper = 0.45
b = np.sqrt(0.5*(Srear/S*Arear+Sfwd/S*Afwd)*S)
c = Srear/S*crear+Sfwd/S*cfwd
e = 1.1302
def Sweep(AR, taper, Sweepm, n, m):
    """
    Inputs
    :param AR: Aspect Ratio of VT
    :param Sweepm: Sweep at mth chord [rad]
    :param n: (example quarter chord: n =25)
    :param m: mth chord (example half chord: m=50)
    :return: Sweep at nth chord [rad]
    """
    tanSweep_m = np.tan(Sweepm)
    tanSweep_n = tanSweep_m - 4 / (AR) * (n - m) / 100 * (1 - taper) / (1 + taper)
    return np.arctan(tanSweep_n)

Sweep_c2_fwd = Sweep(Afwd,taper,0,50,25)
Sweep_c2_rear = Sweep(Arear,taper,0,50,25)
def C_L_a(Cla, M0, A, Lambda_half, eta=0.95):
    """
    Inputs:
    :param M: Mach number
    :param b: wing span
    :param S: wing area
    :param Lambda_half: Sweep angle at 0.5*chord
    :param eta: =0.95
    :return: Lift curve slope for tail AND wing using DATCOM method
    """
    M = M0
    beta = np.sqrt(1 - M ** 2)
    # print("Lambda_1/2c = ",Lambda_half)
    value = Cla * A / (2 + np.sqrt(4 + ((A * beta / eta) ** 2) * (1 + (np.tan(Lambda_half) / beta) ** 2)))
    return value
# xcg2 = xcg*0.5
# xcg3 = xcg*2
CLafwd =C_L_a(Clafwd, M0,Afwd,Sweep_c2_fwd)
CLarear = C_L_a(Clarear,M0,Arear,Sweep_c2_rear)
# n_rot_f = 6
# n_rot_r = 6
# rot_y_range_f = [0.5 * bfwd * 0.1, 0.5 * bfwd * 0.9]
# rot_y_range_r = [0.5 * brear * 0.1, 0.5 * brear * 0.9]
# K = 4959.86
# ku = 0.1
Zcg = 0.70

d = 0
dy = 0.3
# wps = Wing_placement_sizing(W,h, lfus, hfus, wfus, V0, CD0fwd, CLfwd,
#                  CLrear,CLdesfwd,CLdesrear, Clafwd,Clarear,Cmacfwd, Cmacrear, Sfwd, Srear,
#                  Afwd, Arear, Gamma, Lambda_c4_fwd, Lambda_c4_rear, cfwd,
#                  crear, bfwd, brear, efwd, erear, taper, n_rot_f, n_rot_r,
#                  rot_y_range_f, rot_y_range_r, K, ku,Zcg,d,dy,Pbr,1)


elevator_effect = 1.4
dx = 0.1

#### Plotting Vertical Tail ####
r1 = 0.50292
r2 = r1
P_br_cruise_per_engine = 13627.720621056835
T_cr = 153.63377687614096*12
nE = 12
nf = 3
Tt0 = T_cr # Thrust required for cruise
lv = lfus-xcg # Initial estimate of l_v
brbv = np.linspace(0.75,1,150)
crcv = np.linspace(0.1,0.4,150)

ARv = 1.4
sweepTE =np.deg2rad(25)
vt_sizing = VT_sizing(W,h,xcg,lfus,hfus,wfus,V0,Vstall,CD0,CLdesfwd,CLdesrear,CLafwd,CLarear,
                 Sfwd,Srear,Afwd,Arear,Lambda_c4_fwd,Lambda_c4_rear,cfwd,crear,bfwd,brear,taper,ARv,sweepTE)
sweepc4 = vt_sizing.Sweep(ARv, sweepTE,25,100)
print("sweep = ",np.rad2deg(sweepc4))

if isinstance(ARv,(float,int)) and isinstance(sweepTE,(float,int)):
    vt_sizing.plotting(r1,r2,nf,nE,Tt0,br_bv=brbv,cr_cv=crcv,ARv=ARv,sweepTE=sweepTE)
    vt_sizing.plotting(r1,r2,nf,nE, Tt0, br_bv=1.0, cr_cv=0.24,ARv=ARv,sweepTE=sweepTE)
    Sv = vt_sizing.final_VT_rudder(r1,r2,nf,nE,Tt0,br_bv=1.0,cr_cv=0.24,ARv=ARv,sweepTE=sweepTE)[0]
    Svstab = vt_sizing.VT_stability(ARv,sweepTE)
    Svctrl = vt_sizing.VT_controllability(r1,r2,nf,nE,Tt0,br_bv=1.0, cr_cv=0.24,ARv=ARv,sweepTE=sweepTE)
    bv =  vt_sizing.final_VT_rudder(r1,r2,nf,nE,Tt0,br_bv=1.0,cr_cv=0.24,ARv=ARv,sweepTE=sweepTE)[3]
    cvr = vt_sizing.final_VT_rudder(r1,r2,nf,nE,Tt0,br_bv=1.0,cr_cv=0.24,ARv=ARv,sweepTE=sweepTE)[1]
    cvt = vt_sizing.final_VT_rudder(r1,r2,nf,nE, Tt0, br_bv=1.0, cr_cv=0.24, ARv=ARv, sweepTE=sweepTE)[2]
    print("Stability outside: Sv = ",Svstab)
    print("Controllability outside: Sv = ", Svctrl)
    print()
    print("Final: Sv, bv, cv_root, cv_tip =", Sv,bv,cvr,cvt)
else:
    vt_sizing.plotting(r1,r2,nf,nE,Tt0,br_bv=0.8,cr_cv=0.35,ARv=ARv,sweepTE=sweepTE)

#### Plotting Elevator ####
elevator = Elevator_sizing(W,h,xcg,Zcg,dy, lfus,hfus,wfus,V0,Vstall,CD0,0,CLfwd,CLrear,CLafwd,CLarear,Clafwd,Clarear, Cd0fwd, Cd0rear,
                          Sfwd,Srear,Afwd,Arear,0,0,cfwd,crear,bfwd,brear,taper,dCLfwd=0.1*CLfwd,taper_e=0.45)
beb = np.linspace(10,100,200)
SeS = np.linspace(0.1,0.4,200)
cec = np.linspace(0.1,0.4,200)
de_max = 10
elevator.plotting(SeS,beb,cec, de_max)
be_b = 86.8
ce_c = 0.25
Se_S = ce_c*be_b/100*(bfwd/Sfwd)*cfwd
print(Se_S)
# Se_S = 0.26
elevator.plotting(Se_S,be_b,ce_c, de_max)

#### Plotting Aileron ####
aileron = Control_surface(V0,Vstall,CLfwd,CLrear,CLafwd,CLarear,Clafwd,Clarear,Cd0fwd,Cd0rear,
                         Sfwd,Srear,Afwd,Arear,cfwd,crear,bfwd,brear,taper,eta_rear=1)
b1 = np.linspace(1/bfwd*100,100,150)
b2 = 99
Sa_S = np.linspace(0.05,0.2,150)
ca_c = np.linspace(0.10,0.25,150)
# elevon.plotting(0.15,b1,b2)
aileron.plotting(Sa_S,b1,b2,Se_S, be_b, True)
aileron.plotting(Sa_S=0.115423,b1=47.0294,b2=99.0,Se_S=Se_S,be_b=be_b,rear=True)


#### Initial conditions #####
theta0 = 0
alpha0 = 0
q0 = 0
b0 = 0
phi0 = 0
p0 = 0
r0 = 0


# wps.plotting(0, lfus, dx, elevator_effect, d)

CL0 = 0.5158389632834982
A = Afwd/2
CD0_a = CD0+CL0**2/(np.pi*A*e)


n_ult = 3.2 * 1.5  # 3.2 is the max we found, 1.5 is the safety factor
Pmax = 17  # this is defined as maximum perimeter in Roskam, so i took top down view of the fuselage perimeter
lf = 7.348878876267166  # length of fuselage
m_pax = 88  # average mass of a passenger according to Google
n_prop = 12  # number of engines
n_pax = 5  # number of passengers (pilot included)
pos_fus = 2.9395515505068666  # fuselage centre of mass away from the nose
pos_lgear = 3.875  # Oklanding gear position away from the nose
pos_frontwing, pos_backwing = 0.5, 6.1  # positions of the wings away from the nose
m_prop = [502.6006543358783/12]*12  # list of mass of engines (so 30 kg per engine with nacelle and propeller)
pos_prop = [-0.01628695, -0.01628695, -0.01628695, -0.01628695, -0.01628695, -0.01628695,
  5.58371305,  5.58371305,  5.58371305,  5.58371305,  5.58371305,  5.58371305]  # 8 on front wing and 8 on back wing
wing = Wing(W/9.80665, Sfwd, Srear, n_ult, Afwd, Arear,[pos_frontwing, pos_backwing])
fuselage = Fuselage(W/9.80665, Pmax, lfus, n_pax, pos_fus)
lgear = LandingGear(W/9.80665, pos_lgear)
props = Propulsion(n_prop, m_prop, pos_prop)
weight = Weight(m_pax, wing, fuselage, lgear, props, cargo_m=35, cargo_pos=6.5, battery_m=886.1868116321529, battery_pos=0.5,
                p_pax=[1.75, 3.75, 3.75, 6, 6])

# Iyy = 15368.81327 , Ixx = 7771.42196, Izz = 19319.20998, Ixz = 1155.55423
# Kxx2 = 0.04236, Kyy2 = 3.52711, Kzz2 = 0.10530, Kxz = 0.00630
Ixx, Iyy, Izz, Ixz = weight.MMI(wmac = [cfwd, crear], toc = [0.17, 0.17], vpos_wing = [0.3, 1.7])
print("Iyy = %.5f , Ixx = %.5f, Izz = %.5f, Ixz = %.5f"%(Iyy, Ixx, Izz, Ixz))
print("Kxx2 = %.5f, Kyy2 = %.5f, Kzz2 = %.5f, Kxz = %.5f"%(Ixx/(W/9.80665*b**2),Iyy/(W/9.80665*c**2),Izz/(W/9.80665*b**2),Ixz/(W/9.80665*b**2)))

Ka = 1.429



if isinstance(ARv,float) and isinstance(sweepTE,float):
    #print("---------------------For xcg:-----------------------")
    print("-------------------------------------------------------------------")
    print("-----------------------Stability Derivatives-----------------------")
    print("-------------------------------------------------------------------")
    stability_derivatives = Stab_Derivatives(W,h,lfus,hfus,wfus, d,dy,xcg,Zcg,cfwd,crear,Afwd,Arear,Vstall,
                     V0,Tt0,CLdesfwd,CLdesrear,CD0_a,CL0,theta0,alpha0,
                     Clafwd,Clarear, Cd0fwd, Cd0rear, CLafwd,CLarear,Sfwd,Srear,0,0,
                     efwd,erear,Lambda_c4_fwd,Lambda_c4_rear,taper, 0.4,
                     bv,Sv,ARv,sweepTE,P_br_cruise_per_engine,CD0fwd,eta_rear=1,eta_v=0.95)
    G1 = np.deg2rad(-0.5)
    G2 = np.deg2rad(-4.0)
    print("q-derivatives:",stability_derivatives.q_derivatives())
    print("alpha-derivatives:",stability_derivatives.alpha_derivatives())
    print("u-derivatives:",stability_derivatives.u_derivatives())
    print("alpha_dot derivatives:",stability_derivatives.alpha_dot_derivatives())
    print("p-derivatives:",stability_derivatives.p_derivatives())
    print("r-derivatives:", stability_derivatives.r_derivatives())
    print("beta-derivatives:", stability_derivatives.beta_derivatives(G1,G2))
    print("------------------Control Derivatives----------------------")
    print("de-derivatives:", stability_derivatives.de_derivatives(Se_S=Se_S,be_b=be_b))
    print("da-derivatives:",stability_derivatives.da_derivatives(Sa_S=0.115423,b1=47.0294,b2=99.0))
    print("dr-derivatives:", stability_derivatives.dr_derivatives(cr_cv=0.24,br_bv=1.0))
    print("Kyy2 = %.5f"%(Iyy/(W/9.80665*c**2)))
    stability_derivatives.asym_stability_req(Ixx,Izz,Ixz,Sa_S=0.115423,b1=47.0294,b2=99.0,cr_cv=0.24,br_bv=1.0,g1=G1,g2 = G2)
    C_X, C_Z, C_m, C_Y, C_l,C_n = \
            stability_derivatives.return_stab_derivatives(Se_S = Se_S,be_b = be_b,
                                                          Sa_S=0.115423,b1=47.0294,b2=99.0,cr_cv=0.24,br_bv=1.0,
                                                          Ixx=Ixx, Iyy=Iyy,Izz=Izz, Ixz=Ixz,g1=G1,g2=G2)
    print("----------------------Responses--------------------------")
    aircraft = Aircraft(W, h, S, c, b, V0, theta0, alpha0, q0, b0, phi0, p0, r0, Iyy, Ixx, Izz, Ixz, C_X, C_Z,C_m, CL0,
                            C_Y, C_l, C_n,Ka)
    sys = aircraft.compute_sym_sys(V0,Iyy/(W/9.80665*b**2), C_X,C_Z,C_m)
    aircraft.plot_open_loop(sys,0,0)
    aircraft.plot_results(V0)
    # print("---------------------For 2*xcg:-------------------------")
    # stability_derivatives = Stab_Derivatives(W, h, lfus, hfus, wfus, d, dy, xcg3, Zcg, cfwd, crear, Afwd, Arear, Vstall,
    #                                          V0, Tt0, CLdesfwd, CLdesrear, CD0_a, CL0, theta0, alpha0,
    #                                          Clafwd, Clarear, Cd0fwd, Cd0rear, CLafwd, CLarear, Sfwd, Srear, 0,
    #                                          -2 * np.pi / 180,
    #                                          efwd, erear, Lambda_c4_fwd, Lambda_c4_rear, taper, 0.4,
    #                                          bv, Sv, ARv, sweepTE, Pbr, CD0, eta_rear=0.90, eta_v=0.9)
    #
    # print("q-derivatives:", stability_derivatives.q_derivatives())
    # print("alpha-derivatives:", stability_derivatives.alpha_derivatives())
    # print("u-derivatives:", stability_derivatives.u_derivatives())
    # print("alpha_dot derivatives:", stability_derivatives.alpha_dot_derivatives())
    # print("p-derivatives:", stability_derivatives.p_derivatives())
    # print("r-derivatives:", stability_derivatives.r_derivatives())
    # print("beta-derivatives:", stability_derivatives.beta_derivatives())
    # print("-----Control Derivatives----------")
    # print("de-derivatives:", stability_derivatives.de_derivatives(Se_S=0.15, be_b=0.99))
    # print("da-derivatives:", stability_derivatives.da_derivatives(Sa_S=0.145, b1=55, b2=99.0))
    # print("dr-derivatives:", stability_derivatives.dr_derivatives(cr_cv=0.4, br_bv=0.85))
    # print("Kyy2 = %.5f" % (Iyy / (W / 9.80665 * c ** 2)))
    # print("Iyy = %.5f " % (Iyy))
    # stability_derivatives.asym_stability_req(Ixx, Izz, Ixz, 0.145, 55, 97.5, 0.4, 0.85)
    # C_X, C_Z, C_m, C_Y, C_l, C_n = \
    #     stability_derivatives.return_stab_derivatives(Se_S=0.15, be_b=0.99,
    #                                                   Sa_S=0.145, b1=b1, b2=99.0, cr_cv=0.4, br_bv=0.85,
    #                                                   Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixz=Ixz)
    #
    # print("---------------------For 0.5*xcg:-----------------------")
    # stability_derivatives = Stab_Derivatives(W, h, lfus, hfus, wfus, d, dy, xcg2, Zcg, cfwd, crear, Afwd, Arear, Vstall,
    #                                          V0, Tt0, CLdesfwd, CLdesrear, CD0_a, CL0, theta0, alpha0,
    #                                          Clafwd, Clarear, Cd0fwd, Cd0rear, CLafwd, CLarear, Sfwd, Srear, 0,
    #                                          -2 * np.pi / 180,
    #                                          efwd, erear, Lambda_c4_fwd, Lambda_c4_rear, taper, 0.4,
    #                                          bv, Sv, ARv, sweepTE, Pbr, CD0, eta_rear=0.90, eta_v=0.9)
    #
    # print("q-derivatives:", stability_derivatives.q_derivatives())
    # print("alpha-derivatives:", stability_derivatives.alpha_derivatives())
    # print("u-derivatives:", stability_derivatives.u_derivatives())
    # print("alpha_dot derivatives:", stability_derivatives.alpha_dot_derivatives())
    # print("p-derivatives:", stability_derivatives.p_derivatives())
    # print("r-derivatives:", stability_derivatives.r_derivatives())
    # print("beta-derivatives:", stability_derivatives.beta_derivatives())
    # print("-----Control Derivatives----------")
    # print("de-derivatives:", stability_derivatives.de_derivatives(Se_S=0.15, be_b=0.99))
    # print("da-derivatives:", stability_derivatives.da_derivatives(Sa_S=0.145, b1=55, b2=99.0))
    # print("dr-derivatives:", stability_derivatives.dr_derivatives(cr_cv=0.4, br_bv=0.85))
    # print("Kyy2 = %.5f" % (Iyy / (W / 9.80665 * c ** 2)))
    # print("Iyy = %.5f " % (Iyy))
    # stability_derivatives.asym_stability_req(Ixx, Izz, Ixz, 0.145, 55, 97.5, 0.4, 0.85)
    # C_X, C_Z, C_m, C_Y, C_l, C_n = \
    #     stability_derivatives.return_stab_derivatives(Se_S=0.15, be_b=0.99,
    #                                                   Sa_S=0.145, b1=b1, b2=99.0, cr_cv=0.4, br_bv=0.85,
    #                                                   Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixz=Ixz)
    # closed = aircraft.sym_sys_tuned
    # aircraft.plot_results(V0,closed)
    # aircraft.plot_results(V0)





