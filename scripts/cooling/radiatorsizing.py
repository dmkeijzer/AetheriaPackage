import os
import sys
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import Battery, FuelCell
from input.data_structures.fluid import Fluid
from modules.cooling.coolingsystem import *
from input.data_structures.radiator import Radiator


#constants
T_air = 40 #celsius
T_h_in = 90
dT = 4 #k
thermal_conductivity_aluminium = 230 # W/(m K)


#HX characteristics 
h_tube = 2.5e-3
t_tube = 0.2e-3
t_fin = 0.1e-3
t_channel = 0.2e-3

HX_alpha = 0.5
HX_gamma = 0.086
HX_delta = 0.032


#inputs
Z_HX = 0.2
H_HX = 3.5
W_HX = 2





#designs parameters
area_inlet = 0.5  #m^2
air_speed = 300/3.6
overall_heat_transfer_capacity = np.arange(0.5,6,0.1) * 1500 #m^2


# initialising 
bat = Battery(Efficiency= 0.9)
fc = FuelCell()
fc.load()

water = Fluid(viscosity = 8.9e-4, thermal_conductivity = 0.600, heat_capacity = 4184, density=1.225)
air = Fluid(heat_capacity=1005, density= 1.225,viscosity=18e-6, thermal_conductivity=25.87)

HX = Radiator( h_tube = h_tube, t_tube = t_tube, t_channel = t_channel, t_fin = t_fin,
              HX_alpha=  HX_alpha, HX_gamma= HX_gamma, HX_delta = HX_delta)



#calculating heat
heatbattery = bat.heat(power= 520e3)
heatfuelcell = fc.heat(power= 101e3)
totalHeat = heatfuelcell + heatbattery


#knowns 
mass_flow_hot = heatfuelcell / water.heat_capacity /dT
pfan = Coolingsystem.power_fan_airspeed(air_speed,fan_area= area_inlet  ,density = air.density)
c_hot = water.heat_capacity * mass_flow_hot
mass_flow_cold = air.density * area_inlet * air_speed
c_cold = air.heat_capacity * mass_flow_cold


#heat exchanger sizing
HX = hx_geometry(HX,Z_HX,H_HX,W_HX)

#cold side
massflux_cold = mass_flux(mass_flow_cold,HX.A_fs_cross)
Dh_cold = hydralic_diameter_HX(HX.A_cold, HX.A_fs_cross,W_HX)
Re_cold = reynolds_HX(mass_flux= massflux_cold, hydraulic_diameter=Dh_cold,viscosity= air.viscosity)
colburn = colburn_factor(radiator= HX, reynolds= Re_cold)
Pr_cold = prandtl_heat(air.heat_capacity, viscosity= air.viscosity, thermal_conductivity= air.thermal_conductivity)
h_c_cold = heat_capacity_cold(colburn, massflux_cold, air.heat_capacity, Pr_cold)
print(f"cold contact area: {HX.A_cold}")
print(f"hc_cold: {h_c_cold}")
#hot side
mass_flux_hot = mass_flux(mass_flow_hot,HX.A_cross_hot)
dh_hot = hydralic_diameter_HX(HX.A_hot, HX.A_cross_hot, Z_HX)
Re_hot = reynolds_HX(mass_flux_hot, dh_hot,water.viscosity)
Pr_hot = prandtl_heat(water.heat_capacity, water.viscosity, water.thermal_conductivity)
friction_factor_hot = calculate_flam(Re_hot,1)
h_c_hot = heat_capacity_hot(Re_hot, Pr_hot, friction_factor_hot, dh_hot, water.thermal_conductivity)
print(f"\nhot contact area: {HX.A_hot}")
print(f"hc_hot: {h_c_hot}\n")




#calculate thermal resistance
eta_fin = fin_efficiency(h_c_cold, 230 , HX )
eta_surface = surface_efficiency(HX,eta_fin)
R_tot = hx_thermal_resistance(HX,h_c_hot, h_c_hot,eta_surface)
print(1/R_tot)

Q_expelled = Coolingsystem.calculate_heat_expelled(c_hot = c_hot, c_cold = c_cold, 
                                                   T_hot_in = T_h_in, T_cold_in= T_air, 
                                                   overall_heat_transfer_capacity= 1/R_tot )

print(f"heat expelled: {Q_expelled}")

Q_expelled = Coolingsystem.calculate_heat_expelled(c_hot = c_hot, c_cold = c_cold, 
                                                   T_hot_in = T_h_in, T_cold_in= T_air, 
                                                   overall_heat_transfer_capacity= overall_heat_transfer_capacity )



massflow_air_hover = air_speed * area_inlet * air.density
pfan = Coolingsystem.power_fan_massflow(massflow_air_hover/3,air.density, area_inlet/4) *3
heatdifference = Q_expelled - totalHeat
heatcheck = np.argwhere(heatdifference > 0)
area_min = overall_heat_transfer_capacity[heatcheck][0]


print(h_c_cold)


