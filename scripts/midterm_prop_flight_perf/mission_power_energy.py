""""
Author: Damien & Can

"""

import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from Modules.midterm_prop_flight_perf.EnergyPower import *
import input.GeneralConstants as const
    
dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)
    if data["name"] == "L1":
        hoverpower, maxpower, hoverenergy, maxenergy = hoverstuffduct(data["mtom"], atm.rho(0), data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)
    else:
        hoverpower, maxpower, hoverenergy, maxenergy = hoverstuffopen(data["mtom"], atm.rho(0), data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)

    v_aft= v_exhaust(data["mtom"], const.rho_cr, data["mtom"]/data["diskloading"], const.v_cr)

    prop_eff = propeff(v_aft, const.v_cr)


    # ------------ Energy calculation ----------

    # Take-off
    if data["name"] == "L1":
        takeoff_power_var = hoverstuffduct(data["mtom"]*1.1*const.g0, const.rho0, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]
    else:
        takeoff_power_var = hoverstuffopen(data["mtom"]*1.1*const.g0, const.rho0, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)
    energy_takeoff_var = takeoff_power_var * t_takeoff

    # Horizontal Climb
    climb_power_var = powerclimb(data["mtom"], const.g0, v_climb, lod_climb, prop_eff)
    t_climb = (const.h_cruise/const.h_transition)/const.ROC
    energy_climb_var = climb_power_var * const.t_climb
    
    # Transition (after climb because it needs the power)
    energy_transition_vert2hor_var = (takeoff_power_var + climb_power_var)*const.t_trans / 2

    # Cruise
    powercruise_var = powercruise(data["mtom"], const.g0, const.v_cr, data["lift_over_drag"], prop_eff, const.range)
    cruise_energy_var = powercruise_var * const.t_cr

    # Descend
    power_descend_var = powercruise_var *0.2
    energy_descend_var = power_descend_var* const.t_descend

    # Loiter
    power_loiter_var = powerloiter(data["mtom"], const.g0, const.v_climb, const.lod_climb, propeff)
    energy_loiter_var = power_loiter_var * const.t_loiter

    # Landing 
    if data["name"] == "L1":
        landing_power_var = hoverstuffduct(data["mtom"]*0.9, atm.rho(0), data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]
    else:
        landing_power_var = hoverstuffopen(data["mtom"]*0.9, atm.rho(0), data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)
    energy_landing_var = takeoff_power_var * t_takeoff

    # Transition (from horizontal to vertical)
    energy_transition_hor2vert_var = (landing_power_var + descend_power_var)*const.t_trans / 2

    # ----- TOTAL ENERGY CONSUMPTION -----
    energy_consumption = energy_takeoff_var + energy_transition_vert2hor_var + energy_climb_var + cruise_energy_var + energy_descend_var + energy_transition_hor2vert_var + energy_landing_var
