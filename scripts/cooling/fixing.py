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

Re = 1e3
alpha = 0.5
delta = 0.01
gamma = 0.1

j = RadiatorPerformance.colburn_factor(reynolds=Re, alpha= alpha, delta= delta, gamma= gamma)
print(f"delta: {delta} -> j: {j}")

delta = 0.02
j = RadiatorPerformance.colburn_factor(reynolds=Re, alpha= alpha, delta= delta, gamma= gamma)
print(f"delta: {delta} -> j: {j}")