""""
Author: Damien & Can

"""

import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from  modules.midterm_prop_flight_perf.EnergyPower import *
import input.GeneralConstants as const
    
dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)
    if data["name"] == "L1":
        hoverpower, maxpower, hoverenergy, maxenergy = hoverstuffduct(data["mtow"], const.rho0, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)
    else:
        hoverpower, maxpower, hoverenergy, maxenergy = hoverstuffopen(data["mtow"], const.rho0, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)

    v_aft= v_exhaust(data["mtom"], const.rho_cr, data["mtom"]/data["diskloading"], const.v_cr)

    prop_eff = propeff(v_aft, const.v_cr)


    # ------------ Energy calculation ----------

    # Take-off
    if data["name"] == "L1":
        takeoff_power_var = hoverstuffduct(data["mtow"]*1.1, const.rho0, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)[0]
    else:
        takeoff_power_var = hoverstuffopen(data["mtow"]*1.1, const.rho0, data["mtom"]/data["diskloading"],data["TW"]*data["mtom"]*const.g0)
    energy_takeoff = takeoff_power_var * t_takeoff

    # Horizontal Climb


    # Transition (after climb because it needs the power)
    energy_transition = (takeoff_power_var + power_climb_var)*t_trans / 2

    



    # Cruise
    powercruise_var = powercruise(data["MTOW"], const.v_cr, data["lift_over_drag"], prop_eff, const.range)
    cruise_energy = powercruise_var * const.t_cr

    # Descend

    # Loiter


