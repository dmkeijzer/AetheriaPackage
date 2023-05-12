import numpy as np
import sys
import os
import pathlib as pl
import json

# Path handling
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
from modules.midterm_structures.ClassIIWeightEstimation import *
os.chdir(str(list(pl.Path(__file__).parents)[2]))

#Getting JSON files
with open('input/J1_constants.json', 'r') as f:
    J1 = json.load(f)

with open('input/L1_constants.json', 'r') as f:
    L1 = json.load(f)

with open('input/W1_constants.json', 'r') as f:
    W1 = json.load(f)

weight_J1 = VtolWeightEstimation()
# weight_J1.add_component()