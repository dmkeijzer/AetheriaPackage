import numpy as np
import sys
import os
import pathlib as pl
import json
# Path handling
sys.path.append(str(list(pl.Path(__file__).parents)[2]))

# Importing modules and input data
from modules.midterm_structures.ClassIIWeightEstimation import *
from input.GeneralConstants import *
os.chdir(str(list(pl.Path(__file__).parents)[2]))

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
 weight_J1.add_component(Fuselage(J1["mtom"], ))

#L1
weight_L1 = VtolWeightEstimation()
weight_L1.add_component(Wing(L1["mtom"], L1["S1"], n_ult, L1["A1"]))
weight_L1.add_component(Wing(L1["mtom"], L1["S2"], n_ult, L1["A2"]))
# weight_L1.add_component()

# W1
weight_W1 = VtolWeightEstimation()
weight_W1.add_component(Wing(W1["mtom"], W1["S1"], n_ult, W1["A1"]))
weight_W1.add_component(Wing(W1["mtom"], W1["S2"], n_ult, W1["A2"]))
# weight_W1.add_component()