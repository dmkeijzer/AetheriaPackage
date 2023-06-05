import os
import sys
import pathlib as pl

#sys.path.append(str(list(pl.Path(__file__).parents)[2]))
#os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import Battery
bat = Battery(EnergyDensity=0.34, PowerDensity=3.8, VolumeDensity=0.85, Depth_of_discharge=1, CostDensity=100, End_of_life_cycle=0.8, Energy=34, Power=500)

# def battery_mass_energy():
#     return bat

#def f(battery : Battery):
    #specificE = battery.EnergyDensity

print(Battery.energymass())