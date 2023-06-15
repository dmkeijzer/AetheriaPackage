import numpy as np
import scripts.Propellersizing.BEM2023 as BEM
import code2021.Final_optimization.Aero_tools as at
import Blade_plotter as BP
import matplotlib.pyplot as plt

# ISA = at.ISA(1000)
ISA = at.ISA(2400)
a = ISA.soundspeed()
rho = ISA.density()
dyn_visc = ISA.viscosity_dyn()

# Path to save graphs
path = '../PropandPower/Figures/'

# Midterm
# B, R, rpm, xi_0, rho, dyn_vis, V_fr, N_stations, a, RN_spacing, T=None, P=None
# B = 5
xi_0 = 0.1
R = 0.65
A_prop = np.pi*R**2
MTOM = 2150

# M_t_max = 0.6
# rpm = M_t_max*a*60 / (np.pi * 2*R)
# rpm = 2000
# rpm = 1500

V_cruise = 90
V_h = 1
N_stations = 30
RN_spacing = 100000

T_cr_per_eng = 240
T_h_per_eng = MTOM*9.80665 / 6
T_max_per_eng = 1.225 * MTOM*9.80665 / 5

# Range of variables for sensitivity analysis
Bs = np.arange(2, 10)
rpms = np.arange(1000, 4001, 100)
X, Y = np.meshgrid(rpms, Bs[::-1])  # Reorder Bs, idk why it is necessary

Z = np.ones(np.shape(X))

# for y in range(len(Bs)):
#     for x in range(len(rpms)):
#
#         # Check combinations of number of blades and rpm
#         B = Bs[::-1][y]  # Reorder Bs here too
#         rpm = rpms[x]
#
#         # Load the propeller
#         propeller = BEM.BEM(B, R, rpm, xi_0, rho, dyn_visc, V_cruise, N_stations, a, RN_spacing, T=T_cr_per_eng)
#
#         zeta, design, V_e, coefs = propeller.optimise_blade(0)
#
#         # The parameter of interest is the propeller efficiency
#         Z[y][x] = design[5]

# Plot sensitivity plot
# cont = plt.contourf(X, Y, Z, cmap='coolwarm', levels=20)
# cbar = plt.colorbar(cont, orientation="vertical")
#
# cbar.set_label(r'$\eta$ [-]', fontsize=14)
# plt.ylabel("B", fontsize=12)
# plt.xlabel("Rotational speed [rpm]", fontsize=12)
#
# # Save figures
# plt.tight_layout()
# plt.savefig(path + 'sensitivity_design_BEM_B_rpm' + '.pdf')
#
# plt.show()

# Range of variables for sensitivity analysis
B = 5
rpm = 2500

Vs = np.arange(75,95, 2)
rs = np.arange(0.55, 0.7, 0.02)
X, Y = np.meshgrid(rs, Vs[::-1])  # Reorder Vs, idk why it is necessary

Z = np.ones(np.shape(X))
counter=0
for y in range(len(Vs)):
    for x in range(len(rs)):

        # Check combinations of number of blades and rpm
        V_cruise = Vs[::-1][y]  # Reorder Bs here too
        R = rs[x]

        # Load the propeller
        propeller = BEM.BEM(B, R, rpm, xi_0, rho, dyn_visc, V_cruise, N_stations, a, RN_spacing, T=T_cr_per_eng)

        zeta, design, V_e, coefs, solidity = propeller.optimise_blade(0)

        # The parameter of interest is the propeller efficiency
        Z[y][x] = design[5]
        counter+=1
        print(counter)

# Plot sensitivity plot
cont = plt.contourf(X, Y, Z, cmap='coolwarm', levels=20)
cbar = plt.colorbar(cont, orientation="vertical")

cbar.set_label(r'$\eta$ [-]', fontsize=14)
plt.ylabel("V [m/s]", fontsize=12)
plt.xlabel("R [m]", fontsize=12)


plt.scatter(R, V_cruise, marker='x', color='k', label='Design point')
# point = plt.scatter(6, 1.4, color='black', label= 'Wigeon', marker = 'x')

# Save figures
plt.tight_layout()
#plt.savefig(path + 'sensitivity_design_BEM_V_R' + '.pdf')
plt.legend()

plt.show()

# Off design analysis
B = 5
D = 2*R
rpm = 1500

xi_0 = 0.1





# Base propeller
propeller = BEM.BEM(B, R, rpm, xi_0, rho, dyn_visc, V_cruise, N_stations, a, RN_spacing, T=T_cr_per_eng)


# Zeta init
zeta_init = 0
zeta, design, V_e, coefs, solidity = propeller.optimise_blade(zeta_init)


Omega = rpm * 2 * np.pi / 60

RN = (Omega * design[3]) * design[0] * rho / dyn_visc


rpms = np.arange(1500, 4501, 100)
deltas = np.arange(0, 15.1, 0.5)
X, Y = np.meshgrid(rpms, deltas[::-1])  # Reorder Vs, idk why it is necessary

Z = np.ones(np.shape(X))
Z2 = np.ones(np.shape(X))

effs = []
thrust = []

for y in range(len(deltas)):
    # for x in range(len(rpms)):

        # Check combinations of number of blades and rpm
        delta = deltas[::-1][y]  # Reorder deltas here too
        # rpm = rpms[x]
        rpm = 2000

        n = rpm / 60
        blade_hover = BEM.OffDesignAnalysisBEM(V_cruise, B, R, design[0], design[1] - np.deg2rad(delta), design[3],
                                               coefs[0], coefs[1], rpm, rho, dyn_visc, a, RN)

        blade_hover_analysis = blade_hover.analyse_propeller()

        # # The parameter of interest is the thrust
        # Z[y][x] = blade_hover_analysis[0][0]
        # Z2[y][x] = blade_hover_analysis[0][2]

        thrust.append(blade_hover_analysis[0][0])
        effs.append(blade_hover_analysis[0][2])

# Plot efficiency against advance ratio
plt.plot(deltas, effs)
plt.xlabel(r'$\Delta \beta$')
plt.ylabel(r'$\eta$ [-]')

plt.show()

# Plot efficiency against advance ratio
plt.plot(deltas, thrust)
plt.xlabel(r'$\Delta \beta$')
plt.ylabel('T [N]')

plt.show()

# # Plot sensitivity plot
# cont = plt.contourf(X, Y, Z, cmap='coolwarm', levels=20)
# cbar = plt.colorbar(cont, orientation="vertical")
#
# cbar.set_label('Thrust [N]', fontsize=12)
# plt.ylabel(r'$\Delta \beta$ [deg]', fontsize=12)
# plt.xlabel("Rotational speed [rpm]", fontsize=12)
#
# # Save figures
# plt.tight_layout()
# plt.savefig(path + 'sensitivity_offdesign_BEM_rpm_delta_T' + '.pdf')
#
# plt.show()
#
#
# # Plot sensitivity plot
# cont = plt.contourf(X, Y, Z2, cmap='coolwarm', levels=20)
# cbar = plt.colorbar(cont, orientation="vertical")
#
# cbar.set_label(r'$\eta$ [-]', fontsize=14)
# plt.ylabel(r'$\Delta \beta$ [deg]', fontsize=12)
# plt.xlabel("Rotational speed [rpm]", fontsize=12)
#
# # Save figures
# plt.tight_layout()
# plt.savefig(path + 'sensitivity_offdesign_BEM_rpm_delta_eff' + '.pdf')
#
# plt.show()
