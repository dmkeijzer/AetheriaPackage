import matplotlib.pyplot as plt
from warnings import warn
import numpy as np
import sys
import os
import pathlib as pl
import pdb

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

import input.GeneralConstants as const
from input.data_structures.fuellCell import FuelCell

def loading_diagram(wing_loc, lf, fuselage, wing, vtail, aircraft, power, engine):

    dict_mass_loc = { 
        "wing": (wing.wing_weight, wing_loc),
        "vtail": (vtail.vtail_weight, wing_loc + vtail.length_wing2vtail),
        "engine": (engine.totalmass, np.mean(engine.x_rotor_loc)),
        "fuel_cell": (FuelCell.mass, fuselage.length_cockpit + fuselage.length_cabin + FuelCell.depth/2),
        "battery": (power.battery_mass, wing_loc), # Battery was placed in the wing
        "cooling": (power.cooling_mass, fuselage.length_cockpit +  fuselage.length_cabin + power.h2_tank_length ), # Battery was placed in the wing
        "tank": (power.h2_tank_mass, fuselage.length_cockpit +  fuselage.length_cabin + power.h2_tank_length/2 ), # Battery was placed in the wing
        "landing_gear": (aircraft.lg_mass,  lf*const.cg_fuselage ), # For now, assume it coincides with the cg of the fuselage
        "fuselage": (fuselage.fuselage_weight, lf*const.cg_fuselage),
        "misc": (aircraft.misc_mass, lf*const.cg_fuselage),
    }


    oem_mass = np.sum(x[0] for x in dict_mass_loc.values())
    oem_cg = np.sum([x[0]*x[1] for x in dict_mass_loc.values()])/oem_mass

    # Initalize lists anc create set up
    loading_array = np.array([[5.056, 125],  # payload
                    [1.723, 77], # pilot
                    [3.453, 77], # passengers row 1
                    [3.453, 77], # passengers row 1
                    [4.476, 77], # passengers row 2
                    [4.476, 77]]) # passengers row 2

    #------------------ front to back -----------------------------------------
    mass_array = [oem_mass]
    mass_pos_array = [oem_cg]

    for i in loading_array:
        mass_array.append(mass_array[-1] + i[1])
        mass_pos_array.append((mass_array[-1]*mass_pos_array[-1] + i[0]*i[1])/(mass_array[-1] + i[0]))

    #----------------------- back to front -----------------------------------
    mass_array2 = [oem_mass]
    mass_pos_array2 = [oem_cg]

    for j in reversed(loading_array[[1,2,3,4,5,0]]):
        mass_array2.append(mass_array2[-1] + j[1])
        mass_pos_array2.append((mass_array2[-1]*mass_pos_array2[-1] + j[0]*j[1])/(mass_array2[-1] + j[0]))

    #------------------------------------ log results --------------------------------------------
    res = {
        "frontcg": min(mass_pos_array), 
        "rearcg": max(mass_pos_array2),
        "oem_cg": oem_cg
        }
        
    res_margin = {
        "frontcg": min(mass_pos_array)-0.1*(max(mass_pos_array2)-min(mass_pos_array)),
        "rearcg": max(mass_pos_array2)+0.1*(max(mass_pos_array2)-min(mass_pos_array)),
        "oem_cg": oem_cg
                 }

    return res, res_margin
