import matplotlib.pyplot as plt
import numpy as np
import BEM as BEM
import Blade_plotter as BP
import PropandPower.prelim_ADT as ADT
import constants as const
import Aero_tools as at

g0 = const.g
rho0 = const.rho_0

ISA = at.ISA(1000)

rho = ISA.density()
dyn_visc = ISA.viscosity_dyn()
a = ISA.soundspeed()


"""
Check the exit speed of BEM with ADT, they should be similar
"""

MTOM = 3000

n_prop = 12
# B = 20
rpm = 2500
R = 0.5029
xi_0 = 0.1

A_prop = np.pi * R**2 - (np.pi * (R*xi_0)**2)
A_tot = A_prop * n_prop

DiskLoad = MTOM / A_tot

V_cr = 72
T_cr_per_eng = 400

ActDisk = ADT.ActDisk_verif(V_cr, T_cr_per_eng*n_prop, rho, A_tot)

print("Cruise exit speed (ADT):", ActDisk.v_e_cr())

# Bs = []
# Ves = []
# for B in range(3, 26):
#     blade = BEM.BEM(B, R, rpm, xi_0, rho, dyn_visc, V_cr, 100, a, 100000, T=T_cr_per_eng)
#
#     blade_design = blade.optimise_blade(0)
#
#     # Check exit speed
#     Ves.append(blade_design[0]*V_cr + V_cr)
#     Bs.append(B)
#
# # print("Cruise exit speed (BEM)", blade_design[0]*V_cr + V_cr)
# #
# # print("Ratio:", ActDisk.v_e_cr()/(blade_design[0]*V_cr + V_cr))
# # print("")
#
# # Plot the propeller exit speed against the number of blades
# plt.ylim(70, 95)
# plt.plot(Bs, Ves, label='Blade Element Momentum Theory', color='tab:orange')
# plt.hlines(ActDisk.v_e_cr(), Bs[0], Bs[-1], label='Actuator Disk Theory')
# plt.xlabel("B [-]")
# plt.ylabel("Slipstream speed [m/s]")
# plt.legend()
# plt.show()
#
# print("#######################################")
# print("")
#
#
"""
Plot efficiency against J

J = V/(nD)
"""

# Fix n and D, change only V
D = 2*R
n = rpm / 60
B = 5

Js = []
effs = []
for V in range(90, 200, 5):
    blade = BEM.BEM(B, R, rpm, xi_0, rho, dyn_visc, V, 100, a, 100000, T=T_cr_per_eng)
    blade_design = blade.optimise_blade(0)

    # Check the advance ratio
    J = V/(n*D)
    Js.append(J)

    # Compute the efficiency
    eff = blade_design[1][5]
    effs.append(eff)

# Plot efficiency against advance ratio
plt.plot(Js, effs)
plt.xlabel("Advance ratio, J = V/(nD) [-]")
plt.ylabel(r'$\eta$ [-]')

plt.show()

"""
Plot efficiency against thrust for constant speed and rpm

"""
# V = 75
# B = 5
# rpm = 2000
# R = 0.6
#
# Ts = []
# effs = []
#
# for T_cr_per_eng in range(100, 350, 10):
#     blade = BEM.BEM(B, R, rpm, xi_0, rho, dyn_visc, V, 100, a, 100000, T=T_cr_per_eng)
#     blade_design = blade.optimise_blade(0)
#
#     # Check the thrust level
#     Ts.append(T_cr_per_eng)
#
#     # Compute the efficiency
#     eff = blade_design[1][5]
#     effs.append(eff)
#
# # Plot efficiency against advance ratio
# plt.plot(Ts, effs)
# plt.xlabel("Thrust [N]")
# plt.ylabel(r'$\eta$ [-]')
#
# plt.show()

"""
Check off design analysis
Eff against J
J = V/nD

"""
MTOM = 3000

n_prop = 12
B = 5
rpm = 1500
R = 0.55
xi_0 = 0.1

# A_prop = np.pi * R**2 - (np.pi * (R*xi_0)**2)
# A_tot = A_prop * n_prop

V_cruise = 74
N_stations = 30
RN_spacing = 100000

T_cr_per_eng = 27.55 * 12

propeller = BEM.BEM(B, R, rpm, xi_0, rho, dyn_visc, V_cruise, N_stations, a, RN_spacing, T=T_cr_per_eng)


# Zeta init
zeta_init = 0
zeta, design, V_e, coefs, solidity = propeller.optimise_blade(zeta_init)

# print("Displacement velocity ratio (zeta):", zeta)
# print("")
# print("Advance ratio:", propeller.J())
# print("")
# # [cs, betas, alpha, E, eff, Tc, self.Pc]
# print("Chord per station:", design[0])
# print("")
# print("Pitch per station in [deg]:", np.rad2deg(design[1]))
# print("")
# print("AoA per station in [deg]:", np.rad2deg(design[2]))
# print("")
# print("Radial coordinates [m]:", design[3])
# print("")
# print("D/L ratio per station:", design[4])
# print("")
# print("Propeller efficiency:", design[5])
# print("")
# print("Thrust coefficient:", design[6])
# print("")
# print("Power coefficient:", design[7])
# print("")
# print("Exit speed per station:", V_e)
# print("")
# print("Average exit speed per station:", np.average(V_e))
# print("")
# print("Propulsive efficiency:", 2/(1 + np.average(V_e)/V_cruise))
# print("")
# print("Cls, Cds", coefs)
# print("")
print("T_cr", T_cr_per_eng)

# plt.subplot(211)
# plt.plot(design[3], coefs[0])
# plt.subplot(212)
# plt.plot(design[3], design[0])
# plt.show()

# # Load blade plotter
# plotter = BP.PlotBlade(design[0], design[1], design[3], R, xi_0)
#
# # Plot blade
# plotter.plot_blade()
# plotter.plot_3D_blade()

# ----------- Analyse in hover -------------
print("")
print("----------- Analyse in hover -------------")
# ISA = at.ISA(1000)
# a = ISA.soundspeed()
# rho = ISA.density()
# dyn_visc = ISA.viscosity_dyn()

# # Polinomial regression for smooth distribution
# coef_chords = np.polynomial.polynomial.polyfit(design[3], design[0], 5)
# coef_pitchs = np.polynomial.polynomial.polyfit(design[3], design[1], 5)
#
# chord_fun = np.polynomial.polynomial.Polynomial(coef_chords)
# pitch_fun = np.polynomial.polynomial.Polynomial(coef_pitchs)
#
# new_chords = chord_fun(design[3])
# new_pitch = pitch_fun(design[3])

# M_tip = 0.6
# omega = M_tip*a/R
#
# rpm = omega/0.10472
# print("Propeller rpm at hover:", rpm)

# rpm = 3000
Omega = rpm * 2 * np.pi / 60


RN = (Omega * design[3]) * design[0] * rho / dyn_visc

# # TODO: J has to be the same as in cruise for max efficiency (Larrabee)
# # zeta_new, [cs, betas, alpha, stations_r, E, eff, self.Tc, Pc], Ves, [Cl, Cd]  #-np.deg2rad(30)
# blade_hover = BEM.OffDesignAnalysisBEM(V_cruise, B, R, design[0], design[1]-np.deg2rad(0), design[3], coefs[0],
#                                        coefs[1], rpm, rho, dyn_visc, a, RN)
#
# blade_hover_analysis = blade_hover.analyse_propeller()
#
# print(blade_hover_analysis)

D = 2*R

Js = []
effs = []
for rpm in range(1000, 5500, 50):

    n = rpm / 60
    blade_hover = BEM.OffDesignAnalysisBEM(V_cruise, B, R, design[0], design[1] - np.deg2rad(0), design[3], coefs[0],
                                           coefs[1], rpm, rho, dyn_visc, a, RN)

    blade_hover_analysis = blade_hover.analyse_propeller()

    # Check the advance ratio
    J = V_cruise/(n*D)
    Js.append(J)

    # Compute the efficiency
    eff = blade_hover_analysis[0][2]
    effs.append(eff)

# Plot efficiency against advance ratio
plt.plot(Js, effs)
plt.xlabel("Advance ratio, J = V/(nD) [-]")
plt.ylabel(r'$\eta$ [-]')

plt.show()
