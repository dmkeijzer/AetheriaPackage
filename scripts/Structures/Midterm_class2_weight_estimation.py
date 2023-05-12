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

#J1
weight_J1 = VtolWeightEstimation()
weight_J1.add_component()

#L1
weight_L1 = VtolWeightEstimation()
weight_L1.add_component(L1["mtom"], L1["S1"], L1["S1"], n_ult)
weight_L1.add_component()

# W1
weight_W1 = VtolWeightEstimation()
weight_W1.add_component()
weight_W1.add_component()