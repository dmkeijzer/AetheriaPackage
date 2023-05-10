import scipy.optimize as sc_opt
import numpy as np
import BEM as BEM
import Blade_plotter as BP
import Aero_tools as at

ISA = at.ISA(1000)

"""
------------ These should be a result of the integration code ------------
R -> Radius
rho_cr -> Density in cruise
dyn_vis_cr -> Dynamic viscosity in cruise
V_cr -> Cruise speed
a_cr -> Speed of sound during cruise
rho_h -> Density during hover
dyn_vis_h -> Dynamic viscosity during hover
V_h -> Speed during hover (it should be 0, but the program might not accept it, still working on it)
a_h -> Speed of sound during hover
t_cr -> Time in cruise
t_h -> Time in hover
t_tot -> Total time in cruise plus hover (transition idk, TODO: discuss with Egon)

------------ Prescribed ------------
RN_spacing=100000 -> Internal variable, do not modify
max_M_tip -> Maximum allowable Mach at the tip (0.7ish)
N_stations -> Number of stations in the blade (try 25-30, maybe 20 if it is too slow)
xi_0 -> Non-dimensional hub radius (r_hub/R)
"""

R = 0.55
xi_0 = 0.1
rho_cr = ISA.density()
dyn_vis_cr = ISA.viscosity_dyn()
V_cr = 74
N_stations = 25
RN_spacing = 100000
a_cr = ISA.soundspeed()

max_M_tip = 0.6
rho_h = 1.225
dyn_vis_h = at.ISA(0).viscosity_dyn()
V_h = 20
a_h = at.ISA(0).soundspeed()
T_cr = 100
T_h = 3000*9.80665

t_cr = 10
t_h = 1
t_tot = t_h+t_cr


def cost_function_blades(variables):
    #  Variables: [B, rpm hover, rpm cruise, delta_pitch hover]

    # Inputs:   B, R, rpm_cr, xi_0, rho_cr, dyn_vis_cr, V_cr, N_stations, a_cr, RN_spacing, max_M_tip, rho_h,
    #           dyn_vis_h, V_h, a_h, rpm_h, delta_pitch, T_cr, T_h
    blade = BEM.BEM(round(variables[0]*100000000), R, variables[1]*1000, xi_0, rho_cr, dyn_vis_cr, variables[2],
                    N_stations, a_cr, RN_spacing, T=T_cr)

    blade_optim = blade.optimise_blade(0)
    # [0] -> Blade in cruise (zeta_new, [cs, betas, alpha, stations_r, E, eff, self.Tc, Pc], Ves, [Cl, Cd])
    # [1] -> Blade in hover (T, Q, eff)
    # [2] -> Thrust_factor

    # Use % of the mission as weights for the optimisation
    return np.abs(1-blade_optim[1][5])


# Variables to optimise: B, rpm hover, rpm cruise, delta_pitch hover, maybe even thrust factor
#  B, rpm hover, rpm cruise, delta_pitch hover
# TODO: B has to be integer
initial_guess = np.array([5/100000000, 3000/1000, 52])

# Minimum and maximum bounds for the optimisation
# Maximum rpm for maximum tip Mach
omega_max = max_M_tip*a_h/R
rpm_max = omega_max/0.10472

# Min rpm for maximum tip Mach
omega_min = 0.3*a_cr/R
rpm_min = omega_min/0.10472

max_param = np.array([8/100000000, rpm_max/1000, 90])
min_param = np.array([3/100000000, rpm_min/1000, 45])

bounds = np.c_[min_param, max_param]

# Minimize cost function
minimum_cost = sc_opt.minimize(cost_function_blades, initial_guess, bounds=bounds)

# Optimisation results:
# B = int(minimum_cost[0])
print(1-minimum_cost['fun'], minimum_cost['x'])

blade = BEM.BEM(round(minimum_cost['x'][0] * 100000000), R, minimum_cost['x'][1] * 1000, xi_0, rho_cr, dyn_vis_cr,
                minimum_cost['x'][2], N_stations, a_cr, RN_spacing, T=T_cr)

blade_optim = blade.optimise_blade(0)
print(blade_optim)
# Load blade plotter
plotter = BP.PlotBlade(blade_optim[1][0], blade_optim[1][1], blade_optim[1][3], R, xi_0)

# Plot blade
plotter.plot_blade()
plotter.plot_3D_blade()


# def cost_function_blades(variables):
#     #  Variables: [B, rpm hover, rpm cruise, delta_pitch hover]
#     print(variables)
#     # Inputs:   B, R, rpm_cr, xi_0, rho_cr, dyn_vis_cr, V_cr, N_stations, a_cr, RN_spacing, max_M_tip, rho_h,
#     #           dyn_vis_h, V_h, a_h, rpm_h, delta_pitch, T_cr, T_h
#     blade = BEM.Optiblade(int(variables[0]), R, variables[2], xi_0, rho_cr, dyn_vis_cr, V_cr, N_stations, a_cr, RN_spacing,
#                           max_M_tip, rho_h, dyn_vis_h, V_h, a_h, variables[1], variables[3], T_cr, T_h/12)
#
#     blade_performance = blade.optimised_blade()
#     # [0] -> Blade in cruise (zeta_new, [cs, betas, alpha, stations_r, E, eff, self.Tc, Pc], Ves, [Cl, Cd])
#     # [1] -> Blade in hover (T, Q, eff)
#     # [2] -> Thrust_factor
#
#     # Use % of the mission as weights for the optimisation TODO: discuss with Egon whether this or just cruise is better
#     return np.abs((t_cr/t_tot) * (1 - blade_performance[0][1][5]) + (t_h/t_tot) * (1 - blade_performance[1][2]))
#
#
# # Variables to optimise: B, rpm hover, rpm cruise, delta_pitch hover, maybe even thrust factor
# #  B, rpm hover, rpm cruise, delta_pitch hover
# # TODO: B has to be integer
# initial_guess = np.array([5, 3500, 1900, np.deg2rad(25)])
#
# # Minimum and maximum bounds for the optimisation
# # Maximum rpm for maximum tip Mach
# omega_max = max_M_tip*a_h/R
# rpm_max = omega_max/0.10472
#
# # Min rpm for maximum tip Mach
# omega_min = 0.3*a_cr/R
# rpm_min = omega_min/0.10472
#
# max_param = np.array([8, rpm_max, rpm_max, np.deg2rad(30)])
# min_param = np.array([3, rpm_min, rpm_min, np.deg2rad(0)])
# print("Max", max_param)
# print("Min", min_param)
# bounds = np.c_[min_param, max_param]
#
# # Minimize cost function
# minimum_cost = sc_opt.minimize(cost_function_blades, initial_guess, bounds=bounds)
#
# # Optimisation results:
# # B = int(minimum_cost[0])
# print(minimum_cost)
