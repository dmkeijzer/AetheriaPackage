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
from input.data_structures import Power


#constants
T_air = 45 #celsius
T_h_in = 65
dT = 10 #k

#inputs
Z_HX = 0.05
H_HX = 1.5
W_HX = 0.5

#designs parameters
area_inlet = 0.15  #m^2
air_speed =  35#300/3.6




# initialising 
bat = Battery(Efficiency= 0.9)
fc = FuelCell()
power = Power()
power.load()


coolant = Fluid(viscosity = 0.355e-3, thermal_conductivity = 0.65, heat_capacity = 4184, density=997)
air = Fluid(heat_capacity=1005, density= 1.225,viscosity=18e-6, thermal_conductivity=25.87e-3)

HX = Radiator(W_HX= W_HX, H_HX= H_HX, Z_HX= Z_HX)
HX.load()
HX = RadiatorPerformance.hx_geometry(HX)

#calculating heat
heatbattery = bat.heat(power= power.battery_power)
heatfuelcell = fc.heat(power= fc.maxpower * 1e3)
totalHeat = heatfuelcell + heatbattery


#mass flow and heat capacity rate
mass_flow_hot = heatfuelcell / coolant.heat_capacity /dT
c_hot = coolant.heat_capacity * mass_flow_hot
mass_flow_cold = air.density * area_inlet * air_speed
c_cold = air.heat_capacity * mass_flow_cold

#calculate calculate heat radiated
R_tot_HX, pressure_drop = RadiatorPerformance.cooling_radiator(HX= HX, mass_flow_cold= mass_flow_cold, mass_flow_hot = mass_flow_hot,air = air, coolant = coolant)
Q_expelled, epsilon = CoolingsystemPerformance.calculate_heat_expelled(c_hot = c_hot, c_cold = c_cold, 
                                                   T_hot_in = T_h_in, T_cold_in= T_air, 
                                                   overall_heat_transfer_capacity= 1/R_tot_HX )
mass_HX = RadiatorPerformance.mass_radiator(HX,2700)
pfan = CoolingsystemPerformance.power_fan_massflow(mass_flow_cold,air.density, np.pi * (W_HX/2  * 0.8)**2)
print(f"--------------------")
print(f"max power fuel cell {fc.maxpower} kW")
print(f"Battery power: {power.battery_power/1e3} kW")
print(f"\n -----------------")
print(f"fan power: {round(pfan/1e3,2)} KW")
print(f"heat expelled: {round(Q_expelled/1e3,2)} kW")
print(f"Mass radiator: {round(mass_HX,2)} kg")
print(f"pressure drop: {round(pressure_drop,2)} Pa")
print(f'inlet diameter: {round(np.sqrt(area_inlet/np.pi)*100,2)} cm')
print(f"Pump power: {pressure_drop/(coolant.density * 0.75) * mass_flow_hot  } W")
print(f'\n -----------------')

print(f"heat fuelcell: {round(heatfuelcell/1e3,2)} kW")
print(f"heat battery: {round(1.1*heatbattery/1e3,2)} kW")
print(f"total heat: {round(totalHeat/1e3,2)} kW")
print(f"mass flow coolant: {mass_flow_hot} kg /s")
print(f"mass flow air: {mass_flow_cold} kg/s")
print(f"epsilon : {epsilon}")

