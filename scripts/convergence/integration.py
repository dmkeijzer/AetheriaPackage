
import numpy as np
import os
import json
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import *
from modules.powersizing import PropulsionSystem
from modules.stab_ctrl.vtail_sizing_optimal import size_vtail_opt
from modules.stab_ctrl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size
from modules.planform.planformsizing import wing_planform
from modules.preliminary_sizing.wing_power_loading_functions import get_wing_power_loading

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
    #----------------------------------------------------------------------

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

    #planform sizing
    wing = wing_planform(wing)
    wing.dump()

    # Aerodynamic sizing

    
    #power system sizing
    mission.load()
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
    coolingsmass = 15 + 35 #kg mass radiator and mass air subsystem. calculated outside this outside loop and will not change signficant 


    PerformanceParameters.powersystem_mass = powersystemmass + coolingsmass
    mission.dump()

    #stability and control
    wing.load()
    horizontal_tail.load()
    fuselage.load()
    vtail.load()
    stability.load()

    wing,horizontal_tail,fuselage,vtail, stability = size_vtail_opt(WingClass=  wing,
                                                                    HorTailClass= horizontal_tail,
                                                                    FuseClass= fuselage,
                                                                    VTailClass= vtail, 
                                                                    Aeroclass= aero,
                                                                    StabClass=stability,                       
                                                                    b_ref= 2, #!!!!!!!!! please update value when we get it
                                                                    stepsize=  5e-2)
    # WingboxClass =  Wing()
run_integration()