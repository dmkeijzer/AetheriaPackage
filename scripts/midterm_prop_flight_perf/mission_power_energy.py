""""
Author: Damien & Can

"""

import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from modules.midterm_prop_flight_perf.EnergyPower import *
import input.GeneralConstants as const
    
dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)

    #==========================Energy calculation ================================= 

    #----------------------- Take-off-----------------------
    if data["name"] == "L1":
        takeoff_power_var = hoverstuffduct(data["mtom"]*1.1*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]
    else:
        takeoff_power_var = hoverstuffopen(data["mtom"]*1.1*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]
    E_to = takeoff_power_var * const.t_takeoff



    #----------------------- Horizontal Climb --------------------------------------------------------------------
    v_aft= v_exhaust(data["mtom"], const.g0, const.rho_cr, data["mtom"]/data["diskloading"], const.v_cr)
    prop_eff_var = propeff(v_aft, const.v_cr)
    climb_power_var = powerclimb(data["mtom"], const.g0, const.roc_cr/data["G"], data["ld_climb"], prop_eff_var, const.roc_cr)
    t_climb = (const.h_cruise/const.h_transition)/const.roc_cr
    E_climb = climb_power_var * t_climb
    
    #-----------------------Transition (after climb because it needs the power)-----------------------
    E_trans_ver2hor = (takeoff_power_var + climb_power_var)*const.t_trans / 2

    #-----------------------------Cruise-----------------------
    P_cr = powercruise(data["mtom"], const.g0, const.v_cr, data["ld_cr"], prop_eff_var)
    t_cr = (const.mission_dist)/const.v_cr
    E_cr = P_cr * t_cr

    # -----------------------Descend-----------------------
    P_desc = P_cr*0.2
    t_desc = (const.h_cruise/const.h_transition)/const.rod_cr # Equal descend as ascend
    E_desc = P_desc* t_desc

    #----------------------- Loiter-----------------------
    P_loit_cr = powerloiter(data["mtom"], const.g0, const.v_cr, data["ld_cr"], prop_eff_var)
    E_loit_cr = P_loit_cr * const.t_loiter

    #----------------------- Landing----------------------- 
    if data["name"] == "L1":
        landing_power_var = hoverstuffduct(data["mtom"]*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]
    else:
        landing_power_var = hoverstuffopen(data["mtom"]*const.g0, const.rho_sl, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]
    energy_landing_var = takeoff_power_var * const.t_takeoff

    #----------------------- Transition (from horizontal to vertical)-----------------------
    E_trans_hor2ver = (landing_power_var + P_desc)*const.t_trans / 2

    #---------------------------- TOTAL ENERGY CONSUMPTION ----------------------------
    E_total = E_to + E_trans_ver2hor + E_climb + E_cr + E_desc + E_trans_hor2ver + energy_landing_var

    #---------------------------- Writing to JSON ----------------------------

    print(f"Energy consumption {data['name']} = {round(E_total/3.6e6, 1)} [Kwh]")
