# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import matplotlib.pyplot as plt
import json


# import modules
sys.path.append(str(list(pl.Path(__file__).parents)[3]))
from modules.structures.nvm import wing_root_cruise, wing_root_hover
from input.GeneralConstants import *

# define directory of json files
dict_directory = str(list(pl.Path(__file__).parents)[3])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration

# give user option to overwrite values in json files
write_bool = int(input("Do you want to overwrite the current loading values? type 1 if you want to do this.")) == 1


for i in range(len(dict_name)):                                             #iterate over each value
    Vx_cr, Vz_cr, Mx_cr, Mz_cr, T_cr = wing_root_cruise(dict_directory,dict_name[i], PRINT=True, ULTIMATE=True)
    Vz_vf, Mx_vf, T_vf = wing_root_hover(dict_directory, dict_name[i], PRINT=True)
    with open(dict_directory + "\\" + dict_name[i], "r") as jsonFile:
        data = json.loads(jsonFile.read())

    data["Vz_vf"], data["Mx_vf"], data["T_vf"], = Vz_vf, Mx_vf, T_vf
    data["Vx_cr"], data["Vz_cr"], data["Mx_cr"], data["Mz_cr"], data["T_cr"], = Vx_cr, Vz_cr, Mx_cr, Mz_cr, T_cr

    if write_bool==True:
        with open(dict_directory+"\\"+dict_name[i], "w") as jsonFile:
            json.dump(data, jsonFile,indent=2)
        print("Old files were overwritten.")
