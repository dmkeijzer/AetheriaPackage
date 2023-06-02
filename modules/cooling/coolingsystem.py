import os
import sys
import pathlib as pl
from dataclasses import dataclass
import numpy as np


sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import Battery, FuelCell

@dataclass
class Fluid:
    """Stores coolant parameters"""
    viscosity: float = None
    thermal_conductivity: float = None
    heat_capacity: float = None
    density: float = None


def mass_flow(heat, delta_temperature: float, heat_capacity: float ) -> float:
    return heat / (delta_temperature * heat_capacity)

def exchange_effectiveness(Cr: float, NTU: float) -> float: 
    """
    :param: Cr[-]: c_min / c_max 
    :param: NTU [-]: Number of transfer units
    :return: epsilon: exchange effectiveness
    """
    return 1 - np.exp( (1/Cr )*  NTU ** (0.22) * ( np.exp(-1 * Cr* NTU ** (0.78) -1) ) )

#inputs
diameter_pipe = 10e-2
area_inlet = 5e-3 #m^2
T_air = 40 #celsius
T_max_coolant = 80 #celsius
air_speed = 300/3.6

# initialising 
bat = Battery(Efficiency= 0.9)
fc = FuelCell()
fc.load()

water = Fluid(viscosity = 8.9e-4, thermal_conductivity = 0.600, heat_capacity = 4184, density=1.225)
air = Fluid(heat_capacity=1005, density= 1.225)

#calculating heat
heatbattery = bat.heat(power= 520)
heatfuelcell = fc.heat(power= 101)
totalHeat = heatfuelcell + heatbattery


#knowns 
mass_flow_hot = 0.6 #kg/s
c_hot = water.heat_capacity * mass_flow_hot
c_cold = air.heat_capacity * air.density * area_inlet * 300/3.6


c_min = min(c_hot,c_cold)
c_max = max(c_cold, c_hot)
cr = c_min/c_max

Q_max = c_max * (T_max_coolant-T_air) 
mass_flow_air  = heatfuelcell / 40 * 1000 / air.heat_capacity
print(mass_flow_air)
P_fan = mass_flow_air **3 /4 / (air.density * area_inlet * 0.7)
print(P_fan)
