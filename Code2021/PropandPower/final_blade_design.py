

import matplotlib.pyplot as plt
import numpy as np
import BEM as BEM
import Blade_plotter as BP
import Aero_tools as at
import engine_sizing_positioning as esp
import Final_optimization.constants_final as const


# Constants from optimisation

# From constants file
g0 = const.g

# Wing parameters
MTOM = 3024.8012022968796
S1 = 9.910670535618632
S2 = 9.910670535618632
span1 = 8.209297146662843
span2 = 8.209297146662843

n_prop = 12
m_prop = 502.6006543358783

flighttime = 1.5504351809662442
takeofftime = 262.839999999906

V_cruise = 72.18676185339652
h_cruise = 1000

# Thrust and power values
T_max = 34311.7687171136
P_max = 1809362.3556091622
P_br_cruise_per_engine = 13627.720621056835
T_cr_per_engine = 153.63377687614096

TW_ratio = MTOM*g0/T_max


# Atmospherical parameters
ISA = at.ISA(h_cruise)
rho = ISA.density()
dyn_vis = ISA.viscosity_dyn()
soundspeed = ISA.soundspeed()

# Fixed blade parameters
xi_0 = 0.1

eng_sizing = esp.PropSizing(span1, const.w_fuselage, n_prop, const.c_fp, const.c_pp, MTOM, xi_0)
prop_radius = eng_sizing.radius()
prop_diameter = eng_sizing.diameter()

print("Propeller radius:", prop_radius, "m")
print("Disk loading:", eng_sizing.disk_loading(), "kg/m^2")
print("")

# Design parameters
B = 6
rpm_cruise = 1350
T_factor = 2.6

# Design the blade

blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho, dyn_vis, V_cruise, N_stations=25, a=soundspeed,
                       RN_spacing=100000, T=T_cr_per_engine*T_factor)

# Initial estimate: zeta = 0

zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)

print("Average solidity of the propeller", np.average(solidity))

print("############# Blade design #############")
print("T_cr", T_cr_per_engine, "N")
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
print("Propeller required power in cruise:", design[7]/2 * rho * V_cruise**3 * np.pi * prop_radius**2, "W")
print("")
# print("Cls, Cds", coefs)
# print("")

print("############# Off-design analysis #############")
print("")

max_M_tip = 0.75
omega_max = max_M_tip*soundspeed/prop_radius

rpm_max = omega_max/0.10472

print("Maximum allowable rpm:", rpm_max, "rpm")
print("")

# Hover constants
V_h = 10

# Hover design parameters
# delta_pitch_hover = 54
rpm_hover = 4000
Omega_hover = rpm_hover * 2 * np.pi / 60
n_hover = Omega_hover / (2 * np.pi)
# delta_pitch_hover = 45

# Atmospheric parameters still assumed at 1 km
# ISA = at.ISA(500)
# rho = ISA.density()
# dyn_vis = ISA.viscosity_dyn()
# soundspeed = ISA.soundspeed()

# Initial estimate for RN
RN = Omega_hover * design[0] * rho / dyn_vis

max_T = 0
# best_combo = []
Ts = []
Vs = []
dif = 10000
# for rpm_cr in range(1000, 1200, 10):
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
#     # print(T)
#
#     # See if D/L is minimum. If so, save the values
#     if np.abs(T-T_cr_per_engine) < dif:
#         best_combo = [rpm_cr, T]
#         dif = np.abs(T-T_cr_per_engine)
#
# plt.plot(Vs, Ts)
# plt.show()

hover_blade = BEM.OffDesignAnalysisBEM(V_cruise, B, prop_radius, design[0], design[1],
                                       design[3], coefs[0], coefs[1], 1090, rho, dyn_vis, soundspeed, RN)

# Outputs: [T, Q, eff], [C_T, C_P], [alphas]
hover_performance = hover_blade.analyse_propeller()
# T = hover_performance[0][0]

print("Required cruise thrust:", T_cr_per_engine, "N")
print("")
print("Cruise thrust:", hover_performance[0][0], "N")
print("")
print("With:")
print("     Cruise speed:", V_h, "m/s")
print("     Cruise rpm:", 1100, "rpm")
print("     Blade pitch change:", 0, "deg")
print("")
print("Cruise efficiency:", hover_performance[0][2], "[-]")
print("")
print("Necessary cruise power:", 0.001 * hover_performance[1][1] * rho * n_hover**3 * prop_diameter**5, "kW")
print("")
print("AoA per station:", np.rad2deg(hover_performance[2][0]))
print("")
print("Cl per station:", hover_performance[2][1])
print("Cd per station:", hover_performance[2][2])


print("############# Off-design analysis #############")
print("")

max_M_tip = 0.75
omega_max = max_M_tip*soundspeed/prop_radius

rpm_max = omega_max/0.10472

print("Maximum allowable rpm:", rpm_max, "rpm")
print("")

# Hover constants
V_h = 10

# Hover design parameters
# delta_pitch_hover = 54
rpm_hover = 4000
Omega_hover = rpm_hover * 2 * np.pi / 60
n_hover = Omega_hover / (2 * np.pi)
# delta_pitch_hover = 45

# Atmospheric parameters still assumed at 1 km
ISA = at.ISA(500)
rho = ISA.density()
dyn_vis = ISA.viscosity_dyn()
soundspeed = ISA.soundspeed()

# Initial estimate for RN
RN = Omega_hover * design[0] * rho / dyn_vis

max_T = 0
# best_combo = []
Ts = []
Vs = []
for delta_pitch_hover in range(1, 55):

    # print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
    hover_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1]-np.deg2rad(delta_pitch_hover), design[3],
                                           coefs[0], coefs[1], rpm_hover, rho, dyn_vis, soundspeed, RN)


    # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
    hover_performance = hover_blade.analyse_propeller()
    T = hover_performance[0][0]
    Ts.append(T)
    Vs.append(V_h)
    # print(T)

    # See if D/L is minimum. If so, save the values
    if T > max_T:
        best_combo = [delta_pitch_hover, T]
        max_T = T

# plt.plot(Vs, Ts)
# plt.show()

hover_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1] - np.deg2rad(best_combo[0]),
                                       design[3], coefs[0], coefs[1], rpm_hover, rho, dyn_vis, soundspeed, RN)

# Outputs: [T, Q, eff], [C_T, C_P], [alphas]
hover_performance = hover_blade.analyse_propeller()
# T = hover_performance[0][0]

print("Minimum thrust to hover:", MTOM*g0/n_prop, "N")
print("")
print("Hover thrust:", hover_performance[0][0], "N")
print("")
print("With:")
print("     Hover speed:", V_h, "m/s")
print("     Hover rpm:", rpm_hover, "rpm")
print("     Blade pitch change:", best_combo[0], "deg")
print("")
# print("Hover efficiency:", hover_performance[0][2], "[-]")
# print("")
# print("Necessary hover power:", 0.001 * hover_performance[1][1] * rho * n_hover**3 * prop_diameter**5, "kW")
# print("")
print("AoA per station:", np.rad2deg(hover_performance[2][0]))
print("")
print("Cl per station:", hover_performance[2][1])
print("Cd per station:", hover_performance[2][2])

print("############# Off-design analysis: Max thrust #############")
print("")
# Max thrust

max_T = 0
# best_combo = []
Ts = []
Vs = []
for delta_pitch_max_T in range(1, 75):

    # print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
    maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1]-np.deg2rad(delta_pitch_max_T), design[3],
                                          coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)


    # Outputs: [T, Q, eff], [C_T, C_P], [alphas]
    maxT_performance = maxT_blade.analyse_propeller()
    T = maxT_performance[0][0]
    Ts.append(T)
    Vs.append(V_h)
    # print(T)

    # See if D/L is minimum. If so, save the values
    if T > max_T:
        best_combo = [delta_pitch_max_T, T]
        max_T = T

# print(design[1], design[1]-np.deg2rad(delta_pitch_hover))
maxT_blade = BEM.OffDesignAnalysisBEM(V_h, B, prop_radius, design[0], design[1] - np.deg2rad(best_combo[0]),
                                      design[3], coefs[0], coefs[1], rpm_max, rho, dyn_vis, soundspeed, RN)

# Outputs: [T, Q, eff], [C_T, C_P], [alphas]
maxT_performance = maxT_blade.analyse_propeller()

print("Required maximum thrust (T/W ratio of 1.5):", 1.5*MTOM*g0/n_prop, "N")
print("")
print("Max thrust:", maxT_performance[0][0], "N")
print("")
print("With:")
print("     Hover speed:", V_h, "m/s")
print("     Max rpm:", rpm_max, "rpm")
print("     Blade pitch change:", best_combo[0], "deg")
print("")
# print("Max T efficiency:", hover_performance[0][2], "[-]")
# print("")
# print("Necessary max T power:", 0.001 * hover_performance[1][1] * rho * n_hover**3 * prop_diameter**5, "kW")
# print("")
print("AoA per station:", np.rad2deg(hover_performance[2][0]))
print("")
print("Cl per station:", hover_performance[2][1])
print("Cd per station:", hover_performance[2][2])

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

koen_chords = chord_fun(radial_stations_Koen) * 1000
koen_pitch = np.rad2deg(pitch_fun(radial_stations_Koen))

print("Koen chords:", koen_chords)
print("Koen pitchs:", koen_pitch)
print("Hub radius:", xi_0)



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
