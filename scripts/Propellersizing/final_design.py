import matplotlib.pyplot as plt
import numpy as np
import BEM2023 as BEM
import Blade_potting2023 as BP
import code2021.Final_optimization.Aero_tools as at
import code2021.PropandPower.engine_sizing_positioning as esp
from input.data_structures.performanceparameters import PerformanceParameters


# Constants from optimisation

# From constants file
g0 = 9.80665

# # Wing parameters
MTOM=2150

n_prop = 6

flighttime = 1.5504351809662442
takeofftime = 262.839999999906

V_cruise = 83.333
h_cruise = 2400

T_cr_per_engine = 240

TW_ratio = 1.225


# Atmospherical parameters
ISA = at.ISA(h_cruise)
rho = ISA.density()
dyn_vis = ISA.viscosity_dyn()
soundspeed = ISA.soundspeed()

# Fixed blade parameters
xi_0 = 0.1

prop_radius = 0.65
prop_diameter = prop_radius*2
totarea = np.pi*(prop_radius)**2*n_prop

print("Propeller radius:", prop_radius, "m")
print("Disk loading:", MTOM/totarea, "kg/m^2")
print("")

# Design parameters
B = 6
rpm_cruise = 1000
T_factor = 2.2

blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho, dyn_vis, V_cruise, N_stations=25, a=soundspeed,
                       RN_spacing=100000, T=T_cr_per_engine * T_factor)
# Design the blade


# Initial estimate: zeta = 0

zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)

C_T = T_cr_per_engine/(rho*(rpm_cruise/60)**2*prop_diameter**4)

print("Average solidity of the propeller", np.average(solidity))

print("############# Blade design #############")
print("T_cr", T_cr_per_engine, "N")
print("")
print("C_T:", C_T, "[-]")
print("")
print("Displacement velocity ratio (zeta):", zeta, "[-]")
print("Slipstream speed:", V_e, "m/s")
print("Freestream velocity:", V_cruise, "m/s")
print("")
print("Advance ratio:", blade_cruise.J(), "[-]")
print("")
# [cs, betas, alpha, E, eff, Tc, self.Pc]
print("Chord per station:", design[0])
print("")
print("Pitch per station in [deg]:", np.rad2deg(design[1]))
print("")
print("AoA per station in [deg]:", np.rad2deg(design[2]))
print("")
print("Radial coordinates [m]:", design[3])
print("")
print("D/L ratio per station:", design[4])
print("")
print("Propeller efficiency:", design[5])
print("")
print("Thrust coefficient:", design[6])
print("")
print("Power coefficient:", design[7])
print("")
blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho, dyn_vis, V_cruise, N_stations=25, a=soundspeed,
                       RN_spacing=100000, T=T_cr_per_engine)
# Design the blade


# Initial estimate: zeta = 0

zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)

print("Propeller required power in cruise:", design[7]/2 * rho * V_cruise**3 * np.pi * prop_radius**2, "W")
print("")
# print("Cls, Cds", coefs)
# print("")

blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho, dyn_vis, V_cruise, N_stations=25, a=soundspeed,
                       RN_spacing=100000, T=T_cr_per_engine*T_factor)
# Design the blade


# Initial estimate: zeta = 0

zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)

print("############# Off-design analysis #############")
print("")

max_M_tip = 0.75
omega_max = max_M_tip*soundspeed/prop_radius

rpm_max = omega_max/0.10472

print("Maximum allowable rpm:", rpm_max, "rpm")
print("")

# Hover constants
V_h = 2

# Hover design parameters
# delta_pitch_hover = 54

Omega_hover = rpm_max * 2 * np.pi / 60
n_hover = Omega_hover / (2 * np.pi)
# delta_pitch_hover = 45

# Atmospheric parameters still assumed at 1 km
# ISA = at.ISA(500)
# rho = ISA.density()
# dyn_vis = ISA.viscosity_dyn()
# soundspeed = ISA.soundspeed()

# Initial estimate for RN

max_T = 0
# best_combo = []
Ts = []
Vs = []
dif = 10000
# for rpm_cr in range(800, 1400, 10):
#     print(rpm_cr)
#
#     # print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
#     hover_blade = BEM.OffDesignAnalysisBEM(V_cruise, B, prop_radius, design[0], design[1], design[3],
#                                            coefs[0], coefs[1], rpm_cr, rho, dyn_vis, soundspeed, RN)
#
#
#     # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
#     hover_performance = hover_blade.analyse_propeller()
#     T = hover_performance[0][0]
#     Ts.append(T)
#     Vs.append(V_h)
#     print(T)
#
#     # See if D/L is minimum. If so, save the values
#     if np.abs(T-T_cr_per_engine) < dif:
#         best_combo = [rpm_cr, T]
#         dif = np.abs(T-T_cr_per_engine)
#
#
# plt.plot(Vs, Ts)
# plt.show()
# print(RN)
# hover_blade = BEM.OffDesignAnalysisBEM(V_cruise, B, prop_radius, design[0], design[1],
#                                        design[3], coefs[0], coefs[1], 1090, rho, dyn_vis, soundspeed, RN)
#
# # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
# hover_performance = hover_blade.analyse_propeller()
# # T = hover_performance[0][0]

rs=[]
ps=[]

# max_T = 0
# # best_combo = []
# Ts = []
# Vs = []
#
# for i in range(50,100,5):
#     for t in range(21,51):
# for r in range(800,1300,50):
#     prop_radius = 0.6
#     max_M_tip = 0.75
#     T_factor = 2.5
#     omega_max = max_M_tip * soundspeed / prop_radius
#
#     rpm_max = omega_max / 0.10472
#     rpm_cruise=r
#
#     blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho, dyn_vis, V_cruise, N_stations=25, a=soundspeed,
#                        RN_spacing=100000, T=T_cr_per_engine * T_factor)
#     # Design the blade
#
#     # Initial estimate: zeta = 0
#
#     zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)
#
#     RN = Omega_hover * design[0] * rho / dyn_vis
#
#     for delta_pitch_max_T in range(15, 65):
#
#         # print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
#         maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0],
#                                               design[1] - np.deg2rad(delta_pitch_max_T), design[3],
#                                               coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)
#
#         # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
#         maxT_performance = maxT_blade.analyse_propeller()
#         print(delta_pitch_max_T)
#         T = maxT_performance[0][0]
#         Ts.append(T)
#         Vs.append(V_h)
#
#
#         # See if D/L is minimum. If so, save the values
#         if T > max_T:
#             best_combo = [delta_pitch_max_T, T]
#
#             max_T = T
#
#     # print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
#     maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1] - np.deg2rad(best_combo[0]),
#                                           design[3], coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)
#
#     # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
#     maxT_performance = maxT_blade.analyse_propeller()
#
#     print(r,best_combo)
#     print((maxT_performance[1][1] * rho * n_hover**3 * (prop_diameter)**5)*6)
#     print(maxT_performance[0][2])
#



# print("Required cruise thrust:", T_cr_per_engine, "N")
# print("")
# print("Cruise thrust:", hover_performance[0][0], "N")
# print("")
# print("With:")
# print("     Cruise speed:", V_cruise, "m/s")
# print("     Cruise rpm:", rpm_cruise, "rpm")
# print("     Blade pitch change:", 0, "deg")
# print("")
# print("Cruise efficiency:", hover_performance[0][2], "[-]")
# print("")
# print("Necessary cruise power:", 0.001 * hover_performance[1][1] * rho * n_hover**3 * prop_diameter**5, "kW")
# print("")
# print("AoA per station:", np.rad2deg(hover_performance[2][0]))
# print("")
# print("Cl per station:", hover_performance[2][1])
# print("Cd per station:", hover_performance[2][2])
# print("")



# print("############# Off-design analysis #############")
# print("")
#
# max_M_tip = 0.75
# omega_max = max_M_tip*soundspeed/prop_radius
#
# rpm_max = omega_max/0.10472
#
# print("Maximum allowable rpm:", rpm_max, "rpm")
# print("")
#
# # Hover constants
# V_h = 10
#
# # Hover design parameters
# # delta_pitch_hover = 54
# rpm_hover = 2500
# Omega_hover = rpm_hover * 2 * np.pi / 60
# n_hover = Omega_hover / (2 * np.pi)
# # delta_pitch_hover = 45
#
# # Atmospheric parameters still assumed at 1 km
# ISA = at.ISA(100)
# rho = ISA.density()
# dyn_vis = ISA.viscosity_dyn()
# soundspeed = ISA.soundspeed()
#
# # Initial estimate for RN
RN = Omega_hover * design[0] * rho / dyn_vis
#
# max_T = 0
# # best_combo = []
# Ts = []
# Vs = []
# counter=1
# for delta_pitch_hover in range(1, 50):
#
#     # print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
#     hover_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1]-np.deg2rad(delta_pitch_hover), design[3],
#                                            coefs[0], coefs[1], rpm_hover, rho, dyn_vis, soundspeed, RN)
#
#
#     # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
#     hover_performance = hover_blade.analyse_propeller()
#     T = hover_performance[0][0]
#     Ts.append(T)
#     Vs.append(V_h)
#     print(T)
#
#     # See if D/L is minimum. If so, save the values
#     if T > max_T:
#         best_combo = [delta_pitch_hover, T]
#         max_T = T
#     print(counter)
#     counter+=1
#
# # plt.plot(Vs, Ts)
# # plt.show()
#
# hover_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1] - np.deg2rad(best_combo[0]),
#                                        design[3], coefs[0], coefs[1], rpm_hover, rho, dyn_vis, soundspeed, RN)
#
# # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
# hover_performance = hover_blade.analyse_propeller()
# # T = hover_performance[0][0]
#
# print("Minimum thrust to hover:", MTOM*g0/n_prop, "N")
# print("")
# print("Hover thrust:", hover_performance[0][0], "N")
# print("")
# print("With:")
# print("     Hover speed:", V_h, "m/s")
# print("     Hover rpm:", rpm_hover, "rpm")
# print("     Blade pitch change:", best_combo[0], "deg")
# print("")
# # print("Hover efficiency:", hover_performance[0][2], "[-]")
# # print("")
# # print("Necessary hover power:", 0.001 * hover_performance[1][1] * rho * n_hover**3 * prop_diameter**5, "kW")
# # print("")
# print("AoA per station:", np.rad2deg(hover_performance[2][0]))
# print("")
# print("Cl per station:", hover_performance[2][1])
# print("Cd per station:", hover_performance[2][2])

print("############# Off-design analysis: Max thrust #############")
print("")
# Max thrust

max_T = 0
# best_combo = []
Ts = []
Vs = []

for delta_pitch_max_T in range(15, 60):

    # print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
    maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1]-np.deg2rad(delta_pitch_max_T), design[3],
                                          coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)



    # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
    maxT_performance = maxT_blade.analyse_propeller()
    T = maxT_performance[0][0]
    Ts.append(T)
    Vs.append(V_h)
    print(delta_pitch_max_T)


    # See if D/L is minimum. If so, save the values
    if T > max_T:
        best_combo = [delta_pitch_max_T, T]

        max_T = T


# print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1] - np.deg2rad(best_combo[0]),
                                      design[3], coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)

# Outputs: [T, Q, eff], [C_T, C_P], [alphas]
maxT_performance = maxT_blade.analyse_propeller()




print("Required maximum thrust (T/W ratio of 1.225):", 1.225*MTOM*g0/n_prop, "N")
print("")
print("Max thrust:", maxT_performance[0][0], "N")
print("")
print("Carmelo Value", 0.8*maxT_performance[0][0]+1.2*maxT_performance[0][0]*np.sqrt(1+maxT_performance[0][0]/(2*1.225*np.pi*prop_radius**2)), "N")
print("With:")
print("     Hover speed:", V_h, "m/s")
print("     Max rpm:", rpm_max, "rpm")
print("     Blade pitch change:", best_combo[0], "deg")
print("")
print("Max T efficiency:", maxT_performance[0][2], "[-]")
# print("")
# print("Necessary max T power:", 0.001 * hover_performance[1][1] * rho * n_hover**3 * prop_diameter**5, "kW")
# print("")
print("AoA per station:", np.rad2deg(maxT_performance[2][0]))
print("")
print("Cl per station:", maxT_performance[2][1])
print("Cd per station:", maxT_performance[2][2])
print("")
print("Total propeller required power in hover (all 6):", 0.001* (maxT_performance[1][1] * rho * n_hover**3 * (prop_diameter)**5)*6, "kW")
# Load blade plotter
plotter = BP.PlotBlade(design[0], design[1], design[3], prop_radius, xi_0)

# Plot blade
plotter.plot_blade()
plotter.plot_3D_blade()

# Polinomial regression for smooth distribution
coef_chords = np.polynomial.polynomial.polyfit(design[3], design[0], 5)
coef_pitchs = np.polynomial.polynomial.polyfit(design[3], design[1], 5)

radial_stations_Koen = np.array([(1/10 + 1/11*9/10), (1/10 + 2/11*9/10), (1/10 + 3/11*9/10), (1/10 + 4/11*9/10),
                                 (1/10 + 5/11*9/10), (1/10 + 6/11*9/10), (1/10 + 7/11*9/10), (1/10 + 8/11*9/10),
                                 (1/10 + 9/11*9/10), (1/10 + 10/11*9/10), (1/10 + 11/11*9/10)-0.001])*prop_radius


chord_fun = np.polynomial.polynomial.Polynomial(coef_chords)
pitch_fun = np.polynomial.polynomial.Polynomial(coef_pitchs)


# ls=[]
# ys=[]
# for rpm in range(1000,1500,50):
#     rpm_cruise = rpm
#     blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho, dyn_vis, V_cruise, N_stations=25, a=soundspeed,
#                            RN_spacing=100000, T=T_cr_per_engine * T_factor)
#     zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)
#     maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1] - np.deg2rad(best_combo[0]),
#                                           design[3], coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)
#
#     # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
#     maxT_performance = maxT_blade.analyse_propeller()
#
#     val = (hover_performance[1][1] * rho * V_cruise**3 * np.pi * prop_radius**2)
#     print(val)
#     ls.append(rpm)
#     ys.append(val)
# plt.plot(ls,val)
# plt.show()



# koen_chords = chord_fun(radial_stations_Koen) * 1000
# koen_pitch = np.rad2deg(pitch_fun(radial_stations_Koen))
#
# print("Koen chords:", koen_chords)
# print("Koen pitchs:", koen_pitch)
# print("Hub radius:", xi_0)
#


# Constants from optimisation
# MAC1 = 1.265147796494402
# MAC2 = 1.265147796494402
# taper = 0.45
# rootchord1 = 1.6651718350228892
# rootchord2 = 1.6651718350228892
# thicknessChordRatio = 0.17
# xAC = 0.25
# MTOM_nc = 2793.7948931207516
# MTOM = 3024.8012022968796
# S1 = 9.910670535618632
# S2 = 9.910670535618632
# span1 = 8.209297146662843
# span2 = 8.209297146662843
# nmax = 3.2
# Pmax = 17
# lf = 7.348878876267166
# m_pax = 88
# n_prop = 12
# n_pax = 5
# pos_fus = 2.9395515505068666
# pos_lgear = 3.875
# pos_frontwing = 0.5
# pos_backwing = 6.1
# zpos_frontwing = 0.3
# zpos_backwing = 1.7
# m_prop = 502.6006543358783
# pos_prop = [-0.01628695 -0.01628695 -0.01628695 -0.01628695 -0.01628695 -0.01628695
#   5.58371305  5.58371305  5.58371305  5.58371305  5.58371305  5.58371305]
# Mac1 = -0.002866846692576361
# Mac2 = -0.002866846692576361
# flighttime = 1.5504351809662442
# takeofftime = 262.839999999906
# enginePlacement = [0.18371305 0.18371305 0.18371305 0.18371305 0.18371305 0.18371305
#  5.78371305 5.78371305 5.78371305 5.78371305 5.78371305 5.78371305]
# T_max = 34311.7687171136
# p_pax = [1.75, 3.75, 3.75, 6, 6]
# battery_pos = 0.5
# cargo_m = 35
# cargo_pos = 6.5
# battery_m = 886.1868116321529
# P_max = 1809362.3556091622
# vol_bat = 334.7816843943688
# price_bat = 30130.351595493197
# h_winglet_1 = 0.5
# h_winglet_2 = 0.5
# V_cruise = 72.18676185339652
# h_cruise = 1000
# CLmax = 1.5856096132929682
# CD0fwd = 0.008558896967176247
# CD0fwd = 0.008558896967176247
# CD0 = 0.005425868411297487
# Clafwd = 6.028232202020173
# Clarear = 6.028232202020173
# CL_cr_fwd = 0.5158389632834982
# CL_cr_rear = 0.5158389632834982
# CL_cr = 0.5158389632834982
# P_br_cruise_per_engine = 13627.720621056835
# T_cr_per_engine = 153.63377687614096
# x_cg_MTOM_nc = 8.131453112800873



def propcalc(radius, MTOM, clcd, v_cruise, h_cruise, dyn_vis, soundspeed, rho_cruise, TW = 1.225, rho_hover = 1.225):
    prop_radius = radius
    n_prop = 6
    T_cr_per_engine = (MTOM * g0 / clcd)/n_prop
    xi_0 = 0.1
    B = 6
    rpm_cruise = 1000
    T_factor = 2.2
    V_h = 2

    ISA = at.ISA(h_cruise)
    rho = ISA.density()
    dyn_vis = ISA.viscosity_dyn()
    soundspeed = ISA.soundspeed()

    blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho_cruise, dyn_vis, V_cruise, N_stations=25, a=soundspeed, RN_spacing=100000, T=T_cr_per_engine)

    zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)

    power_tot_cruise = (design[7] / 2 * rho_cruise * V_cruise ** 3 * np.pi * prop_radius ** 2) * 6

    blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho_cruise, dyn_vis, V_cruise, N_stations=25, a=soundspeed,
                           RN_spacing=100000, T=T_cr_per_engine * T_factor)

    zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)



    max_M_tip = 0.75
    omega_max = max_M_tip * soundspeed / prop_radius

    rpm_max = omega_max / 0.10472


    Omega_hover = rpm_max * 2 * np.pi / 60
    n_hover = Omega_hover / (2 * np.pi)
    RN = Omega_hover * design[0] * rho / dyn_vis

    max_T = 0


    for delta_pitch_max_T in range(75):

        # print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
        maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1] - np.deg2rad(delta_pitch_max_T),
                                              design[3],
                                              coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)

        # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
        maxT_performance = maxT_blade.analyse_propeller()
        T = maxT_performance[0][0]


        # See if D/L is minimum. If so, save the values
        if T > max_T:
            best_combo = [delta_pitch_max_T, T]

            max_T = T


    maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1] - np.deg2rad(best_combo[0]),
                                          design[3], coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)

    # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
    maxT_performance = maxT_blade.analyse_propeller()

    #OUTPUTS

    C_T_cruise = T_cr_per_engine/(rho*(rpm_cruise/60)**2*prop_diameter**4)
    chords_per_station = design[0]
    prop_eff = design[5]
    design_thrust = T_cr_per_engine * T_factor

    max_thrust_per_engine = maxT_performance[0][0]
    power_tot_hover = (maxT_performance[1][1] * rho * n_hover ** 3 * (prop_diameter) ** 5) * 6

    return C_T_cruise, chords_per_station, prop_eff, design_thrust, power_tot_cruise, max_thrust_per_engine, power_tot_hover

print(propcalc(prop_radius, MTOM, 15, V_cruise, h_cruise, dyn_vis, soundspeed, rho))