
import numpy as np
import os
import json
import sys
import pathlib as pl
import pandas as pd

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import *
from modules.powersizing import PropulsionSystem
from input import GeneralConstants
from modules.stab_ctrl.vtail_sizing_optimal import size_vtail_opt
from modules.stab_ctrl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size # Well this should have probably been used
from modules.planform.planformsizing import wing_planform
from modules.preliminary_sizing.wing_power_loading_functions import get_wing_power_loading
from modules.structures.Flight_Envelope import get_gust_manoeuvr_loadings
from modules.aero.clean_class2drag import integrated_drag_estimation
from modules.aero.slipstream_cruise_function import slipstream_cruise
from modules.flight_perf.performance  import get_energy_power_perf, get_performance_updated
from modules.structures.fuselage_length import get_fuselage_sizing
from modules.structures.ClassIIWeightEstimation import get_weight_vtol
from modules.structures.wingbox_optimizer import GetWingWeight
# from modules.propellor.propellor_sizing import propcalc
from scripts.structures.vtail_span import span_vtail
import input.GeneralConstants as const
from scripts.power.finalPowersizing import power_system_convergences

def run_integration(file_path, counter_tuple=(0,0), optimizer_pointer= None):
    """ Runs an entire integraton loop

    :param label: Label required for writing to files
    :type label: str
    :param file_path: Path to the initial estmate
    :type file_path: str
    :param counter_tuple: tuple with two integers, defaults to (0,0)
    :type counter_tuple: tuple, optional
    :param counter_tuple: tuple with two integers, defaults to (0,0)
    :type counter_tuple: tuple, optional
    :param optimizer_pointer: Refers to the self method of the parent optimization class component
    :type optimizer_pointer:  VTOLOptimization (see aetheria_optimization)
    """    
    #----------------------------- Initialize classes --------------------------------
    if counter_tuple == (1,1):
        IonBlock = Battery(Efficiency= 0.9)
        Pstack = FuelCell()
        Tank = HydrogenTank(energyDensity=1.8, volumeDensity=0.6, cost= 16)
        mission = AircraftParameters.load(file_path)
        wing  =  Wing.load(file_path)
        engine = Engine.load(file_path)
        aero = Aero.load(file_path)
        fuselage = Fuselage.load(file_path)
        vtail = VeeTail.load(file_path)
        stability = Stab.load(file_path)
        power = Power.load(file_path)
    else:
        IonBlock = Battery(Efficiency= 0.9)
        Pstack = FuelCell()
        Tank = HydrogenTank(energyDensity=1.8, volumeDensity=0.6, cost= 16)
        mission = AircraftParameters.load(optimizer_pointer.json_path)
        wing  =  Wing.load(optimizer_pointer.json_path)
        engine = Engine.load(optimizer_pointer.json_path)
        aero = Aero.load(optimizer_pointer.json_path)
        fuselage = Fuselage.load(optimizer_pointer.json_path)
        vtail = VeeTail.load(optimizer_pointer.json_path)
        stability = Stab.load(optimizer_pointer.json_path)
        power = Power.load(optimizer_pointer.json_path)
    #----------------------------------------------------------------------------------

    # Preliminary Sizing
    get_wing_power_loading(mission, wing, engine, aero)
    get_gust_manoeuvr_loadings(mission, aero)
    
    #planform sizing
    wing_planform(wing, mission.MTOM, mission.wing_loading_cruise)

    #-------------------- propulsion ----------------------------
    # mission, engine = propcalc( clcd= aero.ld_cruise, mission=mission, engine= engine, h_cruise= GeneralConstants.h_cruise)

    #-------------------- Aerodynamic sizing--------------------
    integrated_drag_estimation(wing, fuselage, vtail, aero) #
    # aero = slipstream_cruise(wing, engine, aero, mission) # FIXME the effect of of cl on the angle of attack

    #-------------------- Flight Performance --------------------
    get_energy_power_perf(wing, engine, aero, mission)
    # get_performance_updated(aero, mission, wing, enigine, power)

    #-------------------- power system sizing--------------------
    power_system_convergences(power, mission) #


    #-------------------- stability and control--------------------
    vtail.span = span_vtail(1,fuselage.diameter_fuselage,30*np.pi/180)
    size_vtail_opt(WingClass=  wing,
                                                                    AircraftClass= mission,
                                                                    PowerClass= power,
                                                                    EngineClass= engine,
                                                                    Aeroclass= aero,
                                                                    FuseClass= fuselage,
                                                                    VTailClass= vtail, 
                                                                    StabClass=stability,
                                                                    b_ref= vtail.span, #!!!!!!!!! please update value when we get it
                                                                    stepsize = 5e-2,
                                                                    ) 
    #------------- Structures------------------
    # Fuselage sizing
    get_fuselage_sizing(Tank,Pstack, mission, fuselage)

    #------------- weight_estimation------------------
    get_weight_vtol(mission, fuselage, wing, engine, vtail)

    #---------------------- dumping update parameters to json file ------------------
    mission.dump(optimizer_pointer.json_path)
    wing.dump(optimizer_pointer.json_path)
    engine.dump(optimizer_pointer.json_path)
    aero.dump(optimizer_pointer.json_path)
    fuselage.dump(optimizer_pointer.json_path)
    vtail.dump(optimizer_pointer.json_path)
    fuselage.dump(optimizer_pointer.json_path)
    stability.dump(optimizer_pointer.json_path)
    power.dump(optimizer_pointer.json_path)


    #--------------------------------- Log all variables from current iterations ----------------------------------

    for data_struct in [mission, wing, engine, aero, fuselage, vtail, stability, power]:
        save_path = os.path.join(optimizer_pointer.dir_path, "aetheria" + "_" + data_struct.label + "_hist.csv")
        data = data_struct.model_dump()
        
        if os.path.exists(save_path):
            pd.DataFrame(np.array(list(data.values()), dtype=object).reshape(1, -1)).to_csv(save_path, mode="a", header=False, index= False)
        else: 
            pd.DataFrame([data]).to_csv(save_path, columns= list(data.keys()), index=False)
                # Read the output from the subprocess



if __name__ == "__main__":
    
    run_integration()

