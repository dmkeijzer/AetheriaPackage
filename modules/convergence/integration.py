
import numpy as np
import os
import json
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import *
from modules.powersizing import PropulsionSystem
from input.data_structures import GeneralConstants
from modules.stab_ctrl.vtail_sizing_optimal import size_vtail_opt
from modules.stab_ctrl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size
from modules.planform.planformsizing import wing_planform
from modules.preliminary_sizing.wing_power_loading_functions import get_wing_power_loading
from modules.structures.Flight_Envelope import get_gust_manoeuvr_loadings
from modules.aero.drag_estimation_function import final_drag_estimation
from modules.aero.slipstream_cruise_function import slipstream_cruise
from modules.aero.slipstream_stall_function import slipstream_stall
from modules.flight_perf.performance  import get_energy_power_perf
from modules.structures.fuselage_length import get_fuselage_sizing
from modules.structures.ClassIIWeightEstimation import get_weight_vtol
from modules.propellor.propellor_sizing import propcalc


def run_integration():
    #----------------------------- Initialize classes --------------------------------
    IonBlock = Battery(Efficiency= 0.9)
    Pstack = FuelCell()
    Tank = HydrogenTank(energyDensity=1.8, volumeDensity=0.6, cost= 16)
    mission = PerformanceParameters()
    wing  =  Wing()
    engine = Engine()
    aero = Aero()
    horizontal_tail = HorTail()
    fuselage = Fuselage()
    vtail = VeeTail()
    stability = Stab()
    #----------------------------------------------------------------------------------

    #------------------------ Load cases for first time----------------------------------------
    mission.load()
    wing.load()  
    engine.load() 
    aero.load() 
    horizontal_tail.load() 
    fuselage.load() 
    vtail.load() 
    stability.load() 
    #----------------------------------------------------------------------

    # Preliminary Sizing
    mission, wing,  engine, aero = get_wing_power_loading(mission, wing, engine, aero)
    mission =  get_gust_manoeuvr_loadings(mission, aero)

    #planform sizing
    wing.load()
    wing = wing_planform(wing)
    wing.dump()


    #-------------------- Aerodynamic sizing--------------------
    wing, fuselage, vtail, aero, horizontal_tail =  final_drag_estimation(wing, fuselage, vtail, aero, horizontal_tail)
    aero, wing = slipstream_cruise(wing, aero) # TODO the effect of of cl on the angle of attack

    #-------------------- Flight Performance --------------------
    wing, engine, aero, mission = get_energy_power_perf(wing, engine, aero, mission)



    #-------------------- propulsion ----------------------------
    mission, engine = propcalc( clcd= aero.ld_cruise, mission=mission, engine= engine, h_cruise= GeneralConstants.h_cruise)


    #-------------------- power system sizing--------------------
    nu = np.arange(0,1.001,0.005)
    Totalmass, Tankmass, FCmass, Batterymass= PropulsionSystem.mass(echo= np.copy(nu),
                                Mission= mission,
                                Battery=IonBlock,
                                FuellCell= Pstack,
                                FuellTank= Tank )
    index_min_mass = np.where(Totalmass == min(Totalmass))
    NU = nu[index_min_mass][0]
    powersystemmass = Totalmass[index_min_mass][0]
    Batterymass = Batterymass[index_min_mass][0]
    coolingsmass = 15 + 35 #kg mass radiator (15 kg) and mass air (35) subsystem. calculated outside this outside loop and will not change signficant 
    PerformanceParameters.powersystem_mass = powersystemmass + coolingsmass

    #-------------------- stability and control--------------------
    #The function here loads and dumps a couple of times so to make sure nothing goes wrong therefore
    #before this class everything has to be dumped and loaded again

    #loading
    mission.dump()
    wing.dump()  
    engine.dump() 
    aero.dump() 
    horizontal_tail.dump() 
    fuselage.dump() 
    vtail.dump() 
    stability.dump() 


    #dumping
    mission.load()
    wing.load()  
    engine.load() 
    aero.load() 
    horizontal_tail.load() 
    fuselage.load() 
    vtail.load() 
    stability.load() 
    
    wing,horizontal_tail,fuselage,vtail, stability = size_vtail_opt(WingClass=  wing,
                                                                    HorTailClass= horizontal_tail,
                                                                    FuseClass= fuselage,
                                                                    VTailClass= vtail, 
                                                                    StabClass=stability,
                                                                    b_ref= 2, #!!!!!!!!! please update value when we get it
                                                                    stepsize = 5e-2 ) 

    #loading
    mission.dump()
    wing.dump()  
    engine.dump() 
    aero.dump() 
    horizontal_tail.dump() 
    fuselage.dump() 
    vtail.dump() 
    stability.dump()

    #------------- Structures------------------

    # Fuselage sizing
    fuselage = get_fuselage_sizing(Tank,Pstack, mission, fuselage)

    #------------- weight_estimation------------------
    mission, fuselage, wing, engine, vtail =  get_weight_vtol(mission, fuselage, wing, engine, vtail)


    #Final dump
    mission.dump()
    wing.dump()  
    engine.dump() 
    aero.dump() 
    horizontal_tail.dump() 
    fuselage.dump() 
    vtail.dump() 
    stability.dump()

run_integration()