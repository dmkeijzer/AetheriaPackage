#import statements
import numpy as np
import sys
import pathlib as pl
import os
import pdb
from matplotlib import pyplot as plt

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import *
from input.data_structures.aircraft_parameters import AircraftParameters
from modules.powersizing.powersystem import PropulsionSystem




def power_system_convergences(powersystem: Power, Mission: AircraftParameters):
    """
        does the computations for the final optimization/ convergences loop
    """
    IonBlock = Battery(Efficiency= 0.9)
    Pstack = FuelCell()
    Tank = HydrogenTank()
    #estimate power system mass
    nu = np.linspace(0,1,1000)
    Totalmass, Tankmass, FCmass, Batterymass= PropulsionSystem.mass(echo= np.copy(nu),
                                Mission= Mission,
                                Battery=IonBlock,
                                FuellCell= Pstack,
                                FuellTank= Tank )
    
    # font_size = 14
    # plt.title("Aetheria Power System distribution", fontsize = font_size)
    # plt.plot(nu, Totalmass + 80, linewidth = 3)
    # plt.yticks(np.linspace(0,1200,7), fontsize = font_size -2)
    # plt.xticks(np.linspace(0,1,6), fontsize = font_size -2, )
    # plt.xlabel(r"Fuel cell cruise fraction $ \nu $ [-]", fontsize = font_size)
    # plt.ylabel("Power system mass [kg]", fontsize = font_size)
    # plt.grid()
    # # plt.show()

    index_min_mass = np.argmin(Totalmass)
    NU = nu[index_min_mass]
    powersystemmass = Totalmass[index_min_mass] + 15 + 35 #kg mass radiator (15 kg) and mass air (35) subsystem. calculated outside this outside loop and will not change signficant 
    Batterymass = Batterymass[index_min_mass]
    powersystem.fuelcell_mass = FCmass


    powersystem.battery_energy = Batterymass * IonBlock.EnergyDensity * 3.6e6
    powersystem.battery_volume = Batterymass * IonBlock.EnergyDensity / IonBlock.VolumeDensity
    powersystem.battery_power = Batterymass * IonBlock.PowerDensity * 1e3
    powersystem.battery_mass = Batterymass
    # powersystem.fuelcell_mass = Pstack.mass
    powersystem.fuelcell_volume = Pstack.volume
    powersystem.h2_tank_mass = Tankmass[index_min_mass]
    powersystem.nu_FC_cruise_fraction = NU
    Mission.powersystem_mass = powersystemmass
    powersystem.powersystem_mass = powersystemmass


    return powersystem, Mission
