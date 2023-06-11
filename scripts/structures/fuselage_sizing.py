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
s, A = fl.simple_crash_box(m, 20, 3.15*10**6, 9.1)
#h0 = 1.6 + s

h0 = 1.8
b0 = 1.6
Beta = 0.5
V = 0.5
ARe = 2.82
n = 2
l_tank = np.linspace(1, 5, 40)
l_fuelcell = 0.3
l_cockpit = 2
l_cabin = 2.5
l_fcs = 2 * l_fuelcell

l_tail, upsweep, bc, hc, hf, bf, AR, l_tank = fl.minimum_tail_length(h0, b0, Beta, V, l_tank, ARe, n)
r_tank = bc/(2*n)

l_cyl = l_tank - 2*r_tank

V_tank = 4/3*np.pi*r_tank**3 + np.pi*r_tank**2*l_cyl

print('Tank length: ', l_tank)
print("Tail: ", l_tail)
print("Upsweep: ", upsweep)
print("Tank radius: ", r_tank)
print("hc: ", hc)
print("bc: ", bc)
print("hf: ", hf)
print("bf: ", bf)

print("V_tank:", V_tank)

l_fuselage = l_cockpit + l_cabin + l_fcs + l_tail

print(l_fuselage)

