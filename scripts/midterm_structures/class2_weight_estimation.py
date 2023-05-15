import numpy as np
import sys
import os
import pathlib as pl
import json
# Path handling
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

# Importing modules and input data
from modules.midterm_structures.ClassIIWeightEstimation import *
from input.GeneralConstants import *

#Getting JSON files
with open('input/J1_constants.json', 'r') as f:
    J1 = json.load(f)

with open('input/L1_constants.json', 'r') as f:
    L1 = json.load(f)

with open('input/W1_constants.json', 'r') as f:
    W1 = json.load(f)

# Computing masses

#J1
weight_J1 = VtolWeightEstimation()
weight_J1.add_component(Wing(J1["mtom"], J1["S"], n_ult, J1["A"]))
weight_J1.add_component(Fuselage(J1["name"], J1["mtom"], np.pi*J1["w_fuse"], J1["l_fuse"], npax ))
weight_J1.add_component(LandingGear(J1["mtom"]))
weight_J1.add_component(Engines(J1["p_max"], p_density))
weight_J1.add_component(HorizontalTail(J1["mtom"], J1["S_h"], J1["A_h"], J1["t_r_h"]))
weight_J1.add_component(Nacelle(J1["mtom"]))

mass_J1 = weight_J1.compute_mass()

#L1
weight_L1 = VtolWeightEstimation()
weight_L1.add_component(Wing(L1["mtom"], L1["S1"], n_ult, L1["A1"]))
weight_L1.add_component(Wing(L1["mtom"], L1["S2"], n_ult, L1["A2"]))
weight_L1.add_component(Fuselage(L1["name"], L1["mtom"], np.pi*L1["w_fuse"], L1["l_fuse"], npax ))
weight_L1.add_component(LandingGear(L1["mtom"]))
weight_L1.add_component(Engines(L1["p_max"], p_density))
weight_L1.add_component(Nacelle(L1["mtom"]))

mass_L1 = weight_L1.compute_mass()

# W1
weight_W1 = VtolWeightEstimation()
weight_W1.add_component(Wing(W1["mtom"], W1["S1"], n_ult, W1["A1"]))
weight_W1.add_component(Wing(W1["mtom"], W1["S2"], n_ult, W1["A2"]))
weight_W1.add_component(Fuselage(W1["name"], W1["mtom"], np.pi*W1["w_fuse"], W1["l_fuse"], npax ))
weight_W1.add_component(LandingGear(W1["mtom"]))
weight_W1.add_component(Engines(W1["p_max"], p_density))
weight_W1.add_component(Nacelle(W1["mtom"]))

mass_W1 = weight_W1.compute_mass()

print(mass_J1, [i.mass for i in weight_J1.components])
print(mass_L1, [i.mass for i in weight_L1.components])
print(mass_W1, [i.mass for i in weight_W1.components])
