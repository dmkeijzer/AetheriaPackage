import os 
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
from input import data_structures as  ds

#battery
bat = ds.Battery()
bat.load()
print(bat)
print()

#fuelcell
fc = ds.FuelCell()
fc.load()
print(fc)
print()

#wing
wing = ds.Wing()
wing.load()
print(wing)
print()

#fuselage 
fuselage = ds.Fuselage()
fuselage.load()
print(fuselage)
print()

#horizon tail
hortail = ds.HorTail()
hortail.load()
print(hortail)
print()

#hydrogen tank 
tank = ds.HydrogenTank()
tank.load()
print(tank)
print()

#performance parameters
performance = ds.PerformanceParameters()
performance.load()
print(performance)
print()
