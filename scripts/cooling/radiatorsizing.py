import os
import sys
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt


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


#HX characteristics 
h_tube = 1.5e-3
t_tube = 0.1e-3
t_channel = 0.2e-3
s_fin = 1e-3
l_fin = 2e-3
h_fin = 2e-3
t_fin = 0.1e-3



#inputs
Z_HX = 0.1
H_HX = 1.5
W_HX = 0.9





#designs parameters
area_inlet = 0.5  #m^2
air_speed =  300/3.6
overall_heat_transfer_capacity = np.arange(0.5,6,0.1) * 1500 #m^2


# initialising 
bat = Battery(Efficiency= 0.9)
fc = FuelCell()
fc.load()

water = Fluid(viscosity = 0.355e-3, thermal_conductivity = 0.65, heat_capacity = 4184, density=1.225)
air = Fluid(heat_capacity=1005, density= 1.225,viscosity=18e-6, thermal_conductivity=25.87e-3)

HX = Radiator(W_HX= W_HX, H_HX= H_HX, Z_HX= Z_HX,  h_tube = h_tube, t_tube = t_tube, t_channel = t_channel, t_fin = t_fin,
              s_fin= s_fin, h_fin= h_fin, l_fin= l_fin)



#calculating heat
heatbattery = bat.heat(power= 520e3)
heatfuelcell = fc.heat(power= 101e3)
totalHeat = heatfuelcell + heatbattery


#knowns 
mass_flow_hot = heatfuelcell / water.heat_capacity /dT
pfan = CoolingsystemPerformance.power_fan_airspeed(air_speed,fan_area= area_inlet  ,density = air.density)
c_hot = water.heat_capacity * mass_flow_hot
mass_flow_cold = air.density * area_inlet * air_speed
c_cold = air.heat_capacity * mass_flow_cold

#heat exchanger sizing
HX = RadiatorPerformance.hx_geometry(HX,Z_HX,H_HX,W_HX)
print(HX)



#calculate thermal resistance
R_tot_HX = RadiatorPerformance.cooling_radiator(HX,mass_flow_cold,mass_flow_hot,air,water)

Q_expelled = CoolingsystemPerformance.calculate_heat_expelled(c_hot = c_hot, c_cold = c_cold, 
                                                   T_hot_in = T_h_in, T_cold_in= T_air, 
                                                   overall_heat_transfer_capacity= 1/R_tot_HX )

mass_HX = RadiatorPerformance.mass_radiator(HX,2700)
print(f"Mass radiator: {mass_HX}")
print(f"heat expelled: {Q_expelled}")
print(f"heat fuelcell: {heatfuelcell}")
Q_expelled = CoolingsystemPerformance.calculate_heat_expelled(c_hot = c_hot, c_cold = c_cold, 
                                                   T_hot_in = T_h_in, T_cold_in= T_air, 
                                                   overall_heat_transfer_capacity= overall_heat_transfer_capacity )



massflow_air_hover = air_speed * area_inlet * air.density
heatdifference = Q_expelled - totalHeat
heatcheck = np.argwhere(heatdifference > 0)
#area_min = overall_heat_transfer_capacity[heatcheck][0]

##print(area_min)

##print(h_c_cold)


