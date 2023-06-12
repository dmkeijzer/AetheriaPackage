import os
import sys
import pathlib as pl
import numpy as np


sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import Battery, FuelCell
from input.data_structures.fluid import Fluid
from modules.cooling.coolingsystem import CoolingsystemPerformance, RadiatorPerformance
from input.data_structures.radiator import Radiator


#constants
T_air = 40 #celsius
T_h_in = 80
dT = 10 #k

#inputs
Z_HX = 0.1
H_HX = 0.4
W_HX = 0.2

#designs parameters
area_inlet = 0.05  #m^2
air_speed =  50 #300/3.6
overall_heat_transfer_capacity = np.arange(0.5,6,0.1) * 1500 #m^2



# initialising 
bat = Battery(Efficiency= 0.9)
fc = FuelCell()
fc.load()


coolant = Fluid(viscosity = 0.355e-3, thermal_conductivity = 0.65, heat_capacity = 4184, density=997)
air = Fluid(heat_capacity=1005, density= 1.225,viscosity=18e-6, thermal_conductivity=25.87e-3)

HX = Radiator(W_HX= W_HX, H_HX= H_HX, Z_HX= Z_HX)
HX.load()
HX = RadiatorPerformance.hx_geometry(HX)

#calculating heat
heatbattery = bat.heat(power= 520e3)
heatfuelcell = fc.heat(power= 101e3)
totalHeat = heatfuelcell + heatbattery


#mass flow and heat capacity rate
mass_flow_hot = heatfuelcell / coolant.heat_capacity /dT
pfan = CoolingsystemPerformance.power_fan_airspeed(air_speed,fan_area= area_inlet  ,density = air.density)
c_hot = coolant.heat_capacity * mass_flow_hot
mass_flow_cold = air.density * area_inlet * air_speed
c_cold = air.heat_capacity * mass_flow_cold

#calculate calculate heat radiated
R_tot_HX = RadiatorPerformance.cooling_radiator(HX= HX, mass_flow_cold= mass_flow_cold, mass_flow_hot = mass_flow_hot,air = air, coolant = coolant)
Q_expelled = CoolingsystemPerformance.calculate_heat_expelled(c_hot = c_hot, c_cold = c_cold, 
                                                   T_hot_in = T_h_in, T_cold_in= T_air, 
                                                   overall_heat_transfer_capacity= 1/R_tot_HX )
mass_HX = RadiatorPerformance.mass_radiator(HX,2700)
pfan = CoolingsystemPerformance.power_fan_airspeed(air_speed, area_inlet * 0.7, air.density)
print(f"fan power: {pfan}")
print(f"Mass radiator: {mass_HX}")
print(f"heat expelled: {Q_expelled}")
print(f"heat fuelcell: {heatfuelcell}")
print(f"total heat: {totalHeat}")



