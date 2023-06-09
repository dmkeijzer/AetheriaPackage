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

# define inputs
h0 = 1.8
b0 = 1.6
AR0 = b0/h0
ARe = 2

V = 0.5

l_tank = np.linspace(0.3, 6, 41)
Beta = np.linspace(0.3, 0.6, 41)
print(l_tank)
print(Beta)

l_tank = 0.87
Beta = 0.3075

# initialise
AR = AR0
l_tail = []

l_t, upsweep, bc, hc, hf, bf, AR = fl.converge_tail_length(h0, b0, Beta, V, l_tank, AR, ARe, AR0)

print("Tail: ", l_t)
print("Upsweep: ", upsweep)
print("Tank radius: ", bc/4)
print("Height crashed: ", hc)
print("hf: ", hf)
print("bf: ", bf)
print("AR:", AR)