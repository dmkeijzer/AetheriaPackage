
import numpy as np
import sys
import pathlib as pl
import os
import json
import matplotlib.pyplot as plt

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.flight_perf.EnergyPower import *
from modules.flight_perf.transition_model import *
import input.data_structures.GeneralConstants as const
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
    
TEST = int(input("\n\nType 1 if you want to write the JSON data to your download folder instead of the repo, type 0 otherwise:\n")) # Set to true if you want to write to your downloads folders instead of rep0
dict_directory = "input/data_structures"
dict_names = ["aetheria_constants converged.json"]
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)


    max_cruise_alt = 8000
    cruise_lst = []
    E_lst = []
    t_lst = []
    range_lst = []
    for i in range(300, max_cruise_alt, 100):
        cruise_alt = i
        cruise_lst.append(cruise_alt)

        atm = ISA(cruise_alt)
        rho_cr = atm.density()
        #==========================Energy calculation ================================= 

        #----------------------- Take-off-----------------------
        
        P_takeoff = powertakeoff(PerformanceClass.MTOM, const.g0, const.roc_hvr, EngineClass.prop_area*6, const.rho_sl)
        E_to = P_takeoff * const.t_takeoff
        

        #-----------------------Transition to climb-----------------------
        
        transition_simulation = numerical_simulation(l_x_1=3.7057, l_x_2=1.70572142*0.75, l_x_3=4.5, l_y_1=0.5, l_y_2=0.5, l_y_3=0.789+0.5, T_max=8700, y_start=30.5, mass=PerformanceClass.MTOM, g0=const.g0, S=WingClass.surface, CL_climb=AeroClass.cl_climb_clean,
                                    alpha_climb=AeroClass.alpha_climb_clean, CD_climb=AeroClass.cdi_climb_clean + AeroClass.cd0_stall,
                                    Adisk=EngineClass.prop_area*6, lod_climb=AeroClass.ld_climb, eff_climb=data['prop_eff'], v_stall=AeroClass.v_stall)
        E_trans_ver2hor = transition_simulation[0]
        transition_power_max = np.max(transition_simulation[0])
        final_trans_distance = transition_simulation[3][-1]
        final_trans_altitude = transition_simulation[1][-1]
        t_trans_climb = transition_simulation[2][-1]
        

        #----------------------- Horizontal Climb --------------------------------------------------------------------
        # print("-------- horizontal climb")
        average_h_climb = (cruise_alt  - final_trans_altitude)/2
        rho_climb = ISA(average_h_climb).density()
        v_climb = const.roc_cr/const.climb_gradient
        v_aft= v_exhaust(PerformanceClass.MTOM, const.g0, rho_climb, EngineClass.prop_area*6, v_climb)
        prop_eff_var = propeff(v_aft, v_climb)
        climb_power_var = powerclimb(PerformanceClass.MTOM, const.g0, WingClass.surface, rho_climb, AeroClass.ld_climb, prop_eff_var, const.roc_cr)
        t_climb = (cruise_alt  - final_trans_altitude) / const.roc_cr
        print('v', v_climb)
        E_climb = climb_power_var * t_climb
        print('E', E_climb)
        
        #----------------------- Transition (from horizontal to vertical)-----------------------
        # print("--------------- transition to vertical")
        transition_simulation_landing = numerical_simulation_landing(vx_start=data['v_stall_flaps20'], descend_slope=-0.04, mass=PerformanceClass.MTOM, g0=const.g0,
                                    S=WingClass.surface, CL=data['cl_descent_trans_flaps20'], alpha=data['alpha_descent_trans_flaps20'],
                                    CD=data["cdi_descent_trans_flaps20"]+data['cd0'], Adisk=EngineClass.prop_area*6)
        E_trans_hor2ver = transition_simulation_landing[0]
        transition_power_max_landing = np.max(transition_simulation_landing[4])
        final_trans_distance_landing = transition_simulation_landing[3][-1]
        final_trans_altitude_landing = transition_simulation_landing[1][0]  
        t_trans_landing = transition_simulation_landing[2][-1]
        # print('t', t_trans_landing)


            # ----------------------- Horizontal Descend-----------------------
        P_desc = powerdescend(PerformanceClass.MTOM, const.g0, WingClass.surface, rho_climb, AeroClass.ld_climb, prop_eff_var, const.rod_cr)
        t_desc = (cruise_alt - final_trans_altitude_landing)/const.rod_cr # Equal descend as ascend
        E_desc = P_desc* t_desc
        d_desc = (cruise_alt - final_trans_altitude_landing)/const.descent_slope
        v_descend = const.rod_cr/const.descent_slope
        print("v_desc",v_descend)

        #-----------------------------Cruise-----------------------
        # print('-------- cruise')
        P_cr = powercruise(PerformanceClass.MTOM, const.g0, const.v_cr, AeroClass.ld_cruise, prop_eff_var)
        d_climb = final_trans_distance + (cruise_alt  - final_trans_altitude)/np.tan(const.climb_gradient) #check if G is correct
        d_cruise = const.mission_dist - d_desc - d_climb - final_trans_distance - final_trans_distance_landing
        t_cr = (const.mission_dist - d_desc - d_climb - final_trans_distance - final_trans_distance_landing)/const.v_cr
        E_cr = P_cr * t_cr
        # print('t', t_cr)
        # print('distance', (const.mission_dist - d_desc - d_climb - final_trans_distance_landing))

        #----------------------- Loiter cruise-----------------------
        # print('--------- loiter cruise')
        P_loit_cr = powerloiter(PerformanceClass.MTOM, const.g0, WingClass.surface, const.rho_cr, AeroClass.ld_climb, prop_eff_var)
        E_loit_hor = P_loit_cr * const.t_loiter
        # print('t', const.t_loiter)

        #----------------------- Loiter vertically-----------------------
        # print('------ loiter vertically')
        P_loit_land = hoverstuffopen(PerformanceClass.MTOM*const.g0, const.rho_sl, PerformanceClass.MTOM/EngineClass.prop_area*6,data["TW"])[1]
        E_loit_vert = P_loit_land * 30 # 30 sec for hovering vertically
        # print('t', 30)

        #----------------------- Landing----------------------- 
        # print('----------- landing')
        # landing_power_var = hoverstuffopen(PerformanceClass.MTOM*const.g0, const.rho_sl, PerformanceClass.MTOM/EngineClass.prop_area*6,data["TW"])[1]
        energy_landing_var = 0

        #---------------------------- TOTAL ENERGY CONSUMPTION ----------------------------
        E_total = E_to + E_trans_ver2hor + E_climb + E_cr + E_desc + E_loit_hor + E_loit_vert + E_trans_hor2ver + energy_landing_var
        E_lst.append((E_total/3.6e6)-5)
        t_lst.append(t_climb+ t_cr + t_desc + t_trans_climb + t_trans_landing+ const.t_takeoff)
        print(cruise_alt)

        Range = 400*210/(E_total/3.6e6)
        range_lst.append(Range)
    plt.plot(cruise_lst, E_lst)
    plt.xlabel('Cruise altitude [m]', fontsize = 12)
    plt.ylabel("Energy consumption [kWh]", fontsize = 12)
    plt.grid()
    plt.show()

    opt = np.where(np.min(E_lst))
    print(opt)
    # print(cruise_lst[opt])