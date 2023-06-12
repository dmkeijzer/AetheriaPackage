import sys
import os
import pathlib as pl
from scipy.optimize import minimize
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import modules.structures.fuselage_length as fl
import input.data_structures.GeneralConstants as const

# inputs

m = 2500
V = 0.5

#def fuselage_calculations(aircraftmass: float, h2tankVolume: float):

h0 = 1.8
b0 = 1.6
Beta = 0.5
ARe = 2
n = 2
l_tank = np.linspace(1, 5, 40)
l_fuelcell = 0.3
l_cockpit = 2
l_cabin = 2.5
l_fcs = 2 * l_fuelcell

s_p, s_y, e_0, e_d, v0, s0 = 0.5*10**6, 1.2*10**6, 0.038, 0.9, 9.1, 0.5
crash_box_height, crash_box_area = fl.crash_box_height_convergerence(s_p, s_y, e_0, e_d, v0, s0, m)
h0 = 1.6 + crash_box_height

V_tank = 4/3*np.pi*r_tank**3 + np.pi*r_tank**2*l_cyl

''' 
print('Tank length: ', l_tank)
print("Tail: ", l_tail)
print("Upsweep: ", upsweep)
print("Tank radius: ", r_tank)
print("hc: ", hc)
print("bc: ", bc)
print("hf: ", hf)
print("bf: ", bf)

print("V_tank:", V_tank)
'''
l_fuselage = l_cockpit + l_cabin + l_fcs + l_tail


#things to return
# l_fuselage
# upsweep
# l_tank
# r_tank
# bc, hc


