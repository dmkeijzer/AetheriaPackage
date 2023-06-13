import numpy as np
import os
import json
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import *
from input.data_structures import GeneralConstants as const
import scripts.Propellersizing.BEM2023 as BEM
from input.data_structures.ISA_tool import ISA
import time

Mission = PerformanceParameters()
Mission.load()

aero = Aero()
aero.load()

def propcalc(radius, clcd, mission: PerformanceParameters, h_cruise: float, T_factor: float = 2.2):
    prop_radius = radius
    n_prop = 6
    T_cr_per_engine = (mission.MTOM * const.g0 / clcd)/n_prop
    xi_0 = 0.1
    B = 6
    rpm_cruise = 900
    #T_factor = 2.2
    V_h = 2

    isa = ISA(h_cruise)
    rho = isa.density()
    dyn_vis = isa.viscosity_dyn()
    soundspeed = isa.soundspeed()
    rho_cruise = 1.225

    blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho_cruise, dyn_vis, mission.cruise_velocity, N_stations=25, a=soundspeed, RN_spacing=100000, T=T_cr_per_engine)

    zeta, design, V_e, coefs, solidity = blade_cruise.optimise_blade(0)

    power_tot_cruise = (design[7] / 2 * rho_cruise * mission.cruise_velocity ** 3 * np.pi * prop_radius ** 2) * 6

    blade_cruise = BEM.BEM(B, prop_radius, rpm_cruise, xi_0, rho_cruise, dyn_vis, mission.cruise_velocity, N_stations=25, a=soundspeed,
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
    prop_diameter = 2 * prop_radius
    C_T_cruise = T_cr_per_engine/(rho*(rpm_cruise/60)**2*prop_diameter**4)
    #chords_per_station = design[0]
    prop_eff = design[5]
    #design_thrust = T_cr_per_engine * T_factor

    max_thrust_per_engine = maxT_performance[0][0]
    power_tot_hover = (maxT_performance[1][1] * rho * n_hover ** 3 * (prop_diameter) ** 5) * 6

    mission.hoverPower = power_tot_hover
    mission.cruisePower = power_tot_cruise
    mission.max_thrust_per_engine = max_thrust_per_engine
    mission.prop_eff = prop_eff

    return C_T_cruise, mission

t0 = time.perf_counter()
ct,mission = propcalc(0.7, aero.ld_cruise, Mission, const.h_cruise)
t1 = time.perf_counter()

print(f'Run time: {t1-t0} s')