import numpy as np
import scripts.Propellersizing.BEM2023 as BEM
import scripts.Propellersizing.Blade_potting2023 as BP
import code2021.Final_optimization.Aero_tools as at





# ISA = at.ISA(1000)
ISA = at.ISA(2400)
a = ISA.soundspeed()
rho = ISA.density()
dyn_visc = ISA.viscosity_dyn()


B = 6
xi_0 = 0.1
R = 0.7
A_prop = np.pi*R**2
MTOM = 2150

# M_t_max = 0.6
# rpm = M_t_max*a*60 / (np.pi * 2*R)
#rpm should be so that Ct = 0.09-0.13
rpm = 1000
# rpm = 1500

V_cruise = 90
V_h = 52.87
N_stations = 30
RN_spacing = 100000

CLCD = 14.7778

T_cr_per_eng = MTOM*9.80665/CLCD/6*3
P_cr_per_eng =135000/6
T_h_per_eng = MTOM*9.80665 / 6

propeller = BEM.BEM(B, R, rpm, xi_0, rho, dyn_visc, V_cruise, N_stations, a, RN_spacing, T=T_cr_per_eng)



# Zeta init
zeta_init = 0
zeta, design, V_e, coefs, solidity = propeller.optimise_blade(zeta_init)

C_T = T_cr_per_eng/(rho*(rpm/60)**2*(R*2)**4)


print("C_T:", C_T, "[-]")

print("Displacement velocity ratio (zeta):", zeta)
print("")
print("Advance ratio:", propeller.J())
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
print("Exit speed per station:", V_e)
print("")
print("Average exit speed per station:", np.average(V_e))
print("")
print("Propulsive efficiency:", 2/(1 + np.average(V_e)/V_cruise))
print("")
print("Cls, Cds", coefs)
print("")
print("Solidity:", solidity)
print("")
print("T_cr", T_cr_per_eng)

# Load blade plotter
plotter = BP.PlotBlade(design[0], design[1] ,design[3], R, xi_0, airfoil_name='wortman.dat')

# Plot blade
plotter.plot_blade()
plotter.plot_3D_blade()

# print("")
# print("----------- Analyse in hover -------------")
#
# rpm = 2500
# Omega = rpm * 2 * np.pi / 60
#
# V = 0
#
# RN = Omega * design[3] * design[0] * rho / dyn_visc
#
#
# blade_hover = BEM.OffDesignAnalysisBEM(V_cruise, B, R, design[0], design[1]-np.deg2rad(6.5), design[3], coefs[0],
#                                        coefs[1], rpm, rho, dyn_visc, a, RN)
#
# blade_hover_analysis = blade_hover.analyse_propeller()
#
# print(blade_hover_analysis)
#
# print("Needed T in hover per engine", T_h_per_eng)
