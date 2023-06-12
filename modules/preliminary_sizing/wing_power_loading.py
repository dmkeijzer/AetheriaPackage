
import numpy as np
import json
import pathlib as pl
import sys
import os
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.preliminary_sizing.powerloading  import *
import input.data_structures.GeneralConstants as const

def get_wing_power_loading(perf_par, wing cont_factor=1.1):
    """ Returns the wing loading and thrust of the weight based on performance parameters

    :param perf_par: performance parameter class from data structues
    :type perf_par: PerformanceParameters class
    :param wing:  wing class from data structues
    :type wing:  Wing class
    """    

    #Check if it"s lilium or not to define the variable that will say to vertical_flight what formula to use.
    WS_range = np.arange(1,4000,1)
    #data["WS"],data["TW"],data["WP_cruise"],data["WP_hover"] = plot_wing_power_loading_graphs(data["eff"], data["StotS"], data["diskloading"], data["name"],WS_range,i)


    #CALCULATE ALL THE VALUES FOR THE GRAPHS
    TW_range = powerloading_thrustloading(WS_range,const.rho_sl,perf_par.rate_of_climb_hover, perf_par.Stots)  
    #if data["name"] == "J1":   
    #    TW_range = TW_range*1.3     #Added 30% extra thrust to maintain stability
    CLIMBRATE = cont_factor*powerloading_climbrate(perf_par.prop_eff,perf_par.rate_of_climb_cruise, WS_range,const.rho_cr,wing.cd0, wing.e, wing.aspectratio)
    TURN_VCRUISE = cont_factor*powerloading_turningloadfactor(const.rho_cr, perf_par.cruise_velocity ,WS_range, perf_par.prop_eff ,wing.aspectratio,wing.e, data['loadfactor'],data['cd0'])
    TURN_VMAX = cont_factor*powerloading_turningloadfactor(const.rho_cr,data['v_max'],WS_range,data['eff'],data['A'],data['e'],data['loadfactor'],data['cd0'])
    VERTICALFLIGHT = cont_factor*powerloading_verticalflight(data['mtom'],TW_range,data['A_tot'],const.rho_sl,data['eff'],data['ducted_bool'],9.81)
    STALLSPEED = wingloading_stall(data['cLmax'],data['v_stall'], const.rho_sl)
    CLIMBGRADIENT = cont_factor*powerloading_climbgradient(data['e'],data['A'],data['cd0'],WS_range,const.rho_sl,data['eff'],data['G'])

    return perf_par

