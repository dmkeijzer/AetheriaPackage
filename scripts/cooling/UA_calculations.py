


import os
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.cooling.coolingsystem import fin_efficiency, surface_efficiency, hx_geometry, hx_thermal_resistance
from input.data_structures.radiator import Radiator
from input.data_structures.fluid import Fluid


#inputs
Z_HX = 1
H_HX = 3
W_HX = 1
air = Fluid(heat_capacity=1005, density= 1.225,viscosity=18e-6, thermal_conductivity=25.87)

#HX characteristics 
h_tube = 2.5e-3
t_tube = 0.2e-3
t_fin = 0.1e-3
t_channel = 0.2e-3

HX_alpha = 0.5
HX_gamma = 0.086
HX_delta = 0.032
thermal_conductivity_aluminium = 230 # W/(m K)

HX = Radiator( h_tube = h_tube, t_tube = t_tube, t_channel = t_channel, t_fin = t_fin,
              HX_alpha=  HX_alpha, HX_gamma= HX_gamma, HX_delta = HX_delta)

h_c_hot = 1000
h_c_cold = 800

#model
HX = hx_geometry(HX,Z_HX,H_HX,W_HX)



print(prandtl_heat(air.heat_capacity , air.viscosity, air.thermal_conductivity))



eta_fin = fin_efficiency(h_c_cold, 230 , HX )
eta_surface = surface_efficiency(HX,eta_fin)
R_heat = hx_thermal_resistance(HX, h_c_cold, h_c_hot, eta_surface)

print(eta_surface)


print(1/R_heat)