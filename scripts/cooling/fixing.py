import os
import sys
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from AetheriaPackage.data_structs import *
from AetheriaPackage.power import CoolingsystemPerformance, RadiatorPerformance

Re = 1e3
alpha = 0.5
delta = 0.01
gamma = 0.1

j = RadiatorPerformance.colburn_factor(reynolds=Re, alpha= alpha, delta= delta, gamma= gamma)
print(f"delta: {delta} -> j: {j}")

delta = 0.02
j = RadiatorPerformance.colburn_factor(reynolds=Re, alpha= alpha, delta= delta, gamma= gamma)
print(f"delta: {delta} -> j: {j}")
