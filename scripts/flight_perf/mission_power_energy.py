""""
Author: Damien & Can & Lucas

"""

import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.flight_perf.EnergyPower import *
from modules.flight_perf.transition_model import *
import input.GeneralConstants as const
from  ISA_tool import ISA
from input.data_structures import *
WingClass = Wing()
AeroClass = Aero()
PerformanceClass = PerformanceParameters()
EngineClass = Engine()

WingClass.load()
AeroClass.load()
PerformanceClass.load()
EngineClass.load()
    
dict_directory = "input/data_structures"
dict_names = ["aetheria_constants converged.json"]
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)

    atm = ISA(const.h_cruise)
    rho_cr = atm.density()
    #==========================Energy calculation ================================= 

    #----------------------- Take-off-----------------------

    P_takeoff = powertakeoff(PerformanceClass.MTOM, const.g0, const.roc_hvr, EngineClass.total_disk_area, const.rho_sl)
    E_to = P_takeoff * const.t_takeoff
    

    #-----------------------Transition to climb-----------------------
    
    transition_simulation = numerical_simulation(l_x_1=2.4, l_x_2=-0.07, l_x_3=5.88, l_y_1=0.8, l_y_2=0.8, l_y_3=1.62+0.8, T_max=10500, y_start=30.5, mass=PerformanceClass.MTOM, g0=const.g0, S=WingClass.surface, CL_climb=AeroClass.cl_climb_clean,
                                alpha_climb=AeroClass.alpha_climb_clean, CD_climb=AeroClass.cd0_cruise,
                                Adisk=EngineClass.total_disk_area, lod_climb=AeroClass.ld_climb, eff_climb=PerformanceClass.prop_eff, v_stall=AeroClass.v_stall)
    E_trans_ver2hor = transition_simulation[0]
    transition_power_max = np.max(transition_simulation[5])
    final_trans_distance = transition_simulation[3][-1]
    final_trans_altitude = transition_simulation[1][-1]
    t_trans_climb = transition_simulation[2][-1]
    
    
    #----------------------- Horizontal Climb --------------------------------------------------------------------
    # print("-------- horizontal climb")
    average_h_climb = (const.h_cruise  - final_trans_altitude)/2
    rho_climb = ISA(average_h_climb).density()
    v_climb = const.roc_cr/np.sin(const.climb_gradient)
    v_aft= v_exhaust(PerformanceClass.MTOM, const.g0, rho_climb, EngineClass.total_disk_area, v_climb)
    prop_eff_var = propeff(v_aft, v_climb)
    climb_power_var = powerclimb(PerformanceClass.MTOM, const.g0, WingClass.surface, rho_climb, AeroClass.ld_climb, prop_eff_var, const.roc_cr)
    t_climb = (const.h_cruise  - final_trans_altitude) / const.roc_cr
    # print('climb', climb_power_var)
    E_climb = climb_power_var * t_climb
    
    
    #----------------------- Transition (from horizontal to vertical)-----------------------
    # print("--------------- transition to vertical")
    transition_simulation_landing = numerical_simulation_landing(vx_start=AeroClass.v_stall_flaps20, descend_slope=-0.04, mass=PerformanceClass.MTOM, g0=const.g0,
                                S=WingClass.surface, CL=AeroClass.cL_descent_trans_flaps20, alpha=AeroClass.alpha_descent_trans_flaps20,
                                CD=AeroClass.cd0_stall, Adisk=EngineClass.total_disk_area)
    E_trans_hor2ver = transition_simulation_landing[0]
    transition_power_max_landing = np.max(transition_simulation_landing[4])
    final_trans_distance_landing = transition_simulation_landing[3][-1]
    final_trans_altitude_landing = transition_simulation_landing[1][0]  
    t_trans_landing = transition_simulation_landing[2][-1]
    # print('trans landing', t_trans_landing)
    # print("trans",transition_power_max_landing)
        # ---------------------- Horizontal Descent-----------------------
    P_desc = powerdescend(PerformanceClass.MTOM, const.g0, WingClass.surface, rho_climb, AeroClass.ld_climb, prop_eff_var, const.rod_cr)
    t_desc = (const.h_cruise - final_trans_altitude_landing)/const.rod_cr # Equal descend as ascend
    E_desc = P_desc* t_desc
    d_desc = (const.h_cruise - final_trans_altitude_landing)/const.descent_slope
    v_descend = const.rod_cr/const.descent_slope
    # print("t_descs",P_desc)

    #-----------------------------Cruise-----------------------
    # print('-------- cruise')
    P_cr = powercruise(PerformanceClass.MTOM, const.g0, const.v_cr, AeroClass.ld_cruise, prop_eff_var)
    d_climb = final_trans_distance + (const.h_cruise  - final_trans_altitude)/np.tan(const.climb_gradient) #check if G is correct
    d_cruise = const.mission_dist - d_desc - d_climb - final_trans_distance - final_trans_distance_landing
    t_cr = (const.mission_dist - d_desc - d_climb - final_trans_distance - final_trans_distance_landing)/const.v_cr
    E_cr = P_cr * t_cr
    # print('cr', P_cr)

    # print('distance', (const.mission_dist - d_desc - d_climb - final_trans_distance_landing))
    # print(P_cr)
    #----------------------- Loiter cruise-----------------------
    # print('--------- loiter cruise')
    P_loit_cr = powerloiter(PerformanceClass.MTOM, const.g0, WingClass.surface, const.rho_cr, AeroClass.ld_climb, prop_eff_var)
    E_loit_hor = P_loit_cr * const.t_loiter
    # print('loit', const.t_loiter)

    #----------------------- Loiter vertically-----------------------
    # print('------ loiter vertically')
    P_loit_land = hoverstuffopen(PerformanceClass.MTOM*const.g0, const.rho_sl,EngineClass.total_disk_area, data["TW"])[1]
    E_loit_vert = P_loit_land * 30 # 30 sec for hovering vertically
    # print('t', 30)

    #----------------------- Landing----------------------- 
    # print('----------- landing')
    # landing_power_var = hoverstuffopen(PerformanceClass.MTOM*const.g0, const.rho_sl,PerformanceClass.MTOM/data["diskloading"],data["TW"])[1]
    energy_landing_var = 0

    #---------------------------- TOTAL ENERGY CONSUMPTION ----------------------------
    E_total = E_to + E_trans_ver2hor + E_climb + E_cr + E_desc + E_loit_hor + E_loit_vert + E_trans_hor2ver + energy_landing_var
    
    #---------------------------- Writing to JSON and printing result  ----------------------------
    data["mission_energy"] = E_total
    data["power_hover"] = transition_power_max
    # print('Pto',transition_power_max)
    data["power_climb"] = climb_power_var
    data["power_cruise"] = P_cr 
    

    data["takeoff_energy"] = E_to
    data["trans2hor_energy"] = E_trans_ver2hor
    data["climb_energy"] = E_climb
    data["cruise_energy"] = E_cr
    data["descend_energy"] = E_desc
    data["hor_loiter_energy"] = E_loit_hor
    data["trans2ver_energy"] = E_trans_hor2ver
    data["ver_loiter_energy"] = E_loit_vert
    
    
    # print(f"Energy consumption {data['name']} = {round(E_total/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption {data['name']} = {round(E_to/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption {data['name']} = {round(E_trans_hor2ver/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption {data['name']} = {round(E_climb/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption {data['name']} = {round(E_cr/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption {data['name']} = {round(E_desc/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption loit {data['name']} = {round((E_loit_hor)/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption {data['name']} = {round(E_trans_hor2ver/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption {data['name']} = {round(E_loit_vert/3.6e6, 1)} [Kwh]")
    # print(f"Energy consumption {data['name']} = {round(energy_landing_var/3.6e6, 1)} [Kwh]")


