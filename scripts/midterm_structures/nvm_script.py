# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import matplotlib.pyplot as plt
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
from modules.midterm_structures.nvm import wing_root_cruise, wing_root_hover
from input.GeneralConstants import *

dict_directory = str(list(pl.Path(__file__).parents)[2])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration

for i in range(len(dict_name)):                                             #iterate over each value
    Vx_cr, Vz_cr, Mx_cr, Mz_cr, T_cr = wing_root_cruise(dict_directory,dict_name[i], PRINT=True, ULTIMATE=False)
    Vz_vf, Mx_vf, T_vf = wing_root_hover(dict_directory, dict_name[i], PRINT=True)

