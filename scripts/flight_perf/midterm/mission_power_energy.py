""""
Author: Damien & Can & Lucas

"""

import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[3]))
os.chdir(str(list(pl.Path(__file__).parents)[3]))

from modules.flight_perf.EnergyPower import *
import input.data_structures.GeneralConstants as const
    
TEST = int(input("\n\nType 1 if you want to write the JSON data to your download folder instead of the repo, type 0 otherwise:\n")) # Set to true if you want to write to your downloads folders instead of rep0
dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)

    #==========================Energy calculation ================================= 

    #----------------------- Take-off-----------------------
    # if data["name"] == "L1":
    #     P_loit_land = hoverstuffduct(data["mtom"]*1.1*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]
    # else:
    #     P_loit_land = hoverstuffopen(data["mtom"]*1.1*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]

   # P_takeoff = data["mtom"]*const.g0*const.roc_hvr + 1.2*data["mtom"]*const.g0*((-const.roc_hvr/2) + np.sqrt((const.roc_hvr**2/4)+(data["mtom"]*const.g0/(2*const.rho_cr*data["diskarea"]))))
    P_takeoff = powertakeoff(data["mtom"], const.g0, const.roc_hvr, data["diskarea"], const.rho_sl)
    E_to = P_takeoff * const.t_takeoff
    
    #----------------------- Horizontal Climb --------------------------------------------------------------------
    v_aft= v_exhaust(data["mtom"], const.g0, const.rho_cr, data["mtom"]/data["diskloading"], const.v_cr)
    prop_eff_var = propeff(v_aft, const.v_cr)
    climb_power_var = powerclimb(data["mtom"], const.g0, data["S"], const.rho_sl, data["ld_climb"], prop_eff_var, const.roc_cr)
    t_climb = (const.h_cruise  - const.h_transition) / const.roc_cr
    E_climb = climb_power_var * t_climb
    
    #-----------------------Transition (after climb because it needs the power)-----------------------
    E_trans_ver2hor = (P_takeoff + climb_power_var)*const.t_trans / 2

    #-----------------------------Cruise-----------------------
    P_cr = powercruise(data["mtom"], const.g0, const.v_cr, data["ld_cr"], prop_eff_var)
    d_climb = (const.h_cruise - const.h_transition)/np.tan(data["G"])
    d_desc = (const.h_cruise - const.h_transition)/np.tan(data['G'])
    t_cr = (const.mission_dist-d_desc-d_climb)/const.v_cr
    E_cr = P_cr * t_cr

    # -----------------------Descend-----------------------
    P_desc = powerdescend(data["mtom"], const.g0, data["S"], const.rho_cr, data["ld_climb"], prop_eff_var, const.rod_cr)
    t_desc = (const.h_cruise - const.h_transition)/const.rod_cr # Equal descend as ascend
    E_desc = P_desc* t_desc

    #----------------------- Loiter cruise-----------------------
    P_loit_cr = powerloiter(data["mtom"], const.g0, data["S"], const.rho_cr, data["ld_climb"], prop_eff_var)
    E_loit_hor = P_loit_cr * const.t_loiter

    #----------------------- Loiter vertically-----------------------
    if data["name"] == "L1":
        P_loit_land = hoverstuffduct(data["mtom"]*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"])[1]
    else:
        P_loit_land = hoverstuffopen(data["mtom"]*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"])[1]
    E_loit_vert = P_loit_land * 30 # 30 sec for hovering vertically

    #----------------------- Landing----------------------- 
    if data["name"] == "L1":
        landing_power_var = hoverstuffduct(data["mtom"]*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"])[1]
    else:
        landing_power_var = hoverstuffopen(data["mtom"]*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"])[1]
    energy_landing_var = P_loit_land * const.t_takeoff

    #----------------------- Transition (from horizontal to vertical)-----------------------
    E_trans_hor2ver = (landing_power_var + P_desc)*const.t_trans / 2

    #---------------------------- TOTAL ENERGY CONSUMPTION ----------------------------
    E_total = E_to + E_trans_ver2hor + E_climb + E_cr + E_desc + E_loit_hor + E_loit_vert + E_trans_hor2ver + energy_landing_var

    #---------------------------- Writing to JSON and printing result  ----------------------------
    data["mission_energy"] = E_total
    data["power_hover"] = P_takeoff
    data["power_climb"] = climb_power_var
    data["power_cruise"] = P_cr 
    data["diskarea"] = data["mtom"]/data["diskloading"]

    data["takeoff_energy"] = E_to
    data["trans2hor_energy"] = E_trans_ver2hor
    data["climb_energy"] = E_climb
    data["cruise_energy"] = E_cr
    data["descend_energy"] = E_desc
    data["hor_loiter_energy"] = E_loit_hor
    data["trans2ver_energy"] = E_trans_hor2ver
    data["ver_loiter_energy"] = E_loit_vert
    data["land_energy"] = energy_landing_var
    
    

    if TEST:
        with open(os.path.join(download_dir, dict_name), "w") as jsonFile:
            json.dump(data, jsonFile, indent= 6)
    else:
        with open(os.path.join(dict_directory, dict_name), "w") as jsonFile:
            json.dump(data, jsonFile, indent= 6)
    
    print(f"Energy consumption {data['name']} = {round(E_total/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption {data['name']} = {round(E_to/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption {data['name']} = {round(E_trans_hor2ver/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption {data['name']} = {round(E_climb/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption {data['name']} = {round(E_cr/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption {data['name']} = {round(E_desc/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption loit {data['name']} = {round((E_loit_hor+E_loit_vert)/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption {data['name']} = {round(E_trans_hor2ver/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption {data['name']} = {round(E_loit_vert/3.6e6, 1)} [Kwh]")
    print(f"Energy consumption {data['name']} = {round(energy_landing_var/3.6e6, 1)} [Kwh]")


