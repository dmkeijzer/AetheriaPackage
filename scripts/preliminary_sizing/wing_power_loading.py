# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import json
import matplotlib

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

import matplotlib.pyplot as plt

from modules.preliminary_sizing import *
import input.GeneralConstants as const


write_bool = int(input("\n\nType 1 if you want to write the JSON data to your download folder instead of the repo, type 0 otherwise:\n"))
WS_range = np.arange(1,4000,1)
ylim = [0,0.15]



#FIRST EASY PRELIMINARY DESIGN
dict_directory = str(list(pl.Path(__file__).parents)[2])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration
for i in range(len(dict_name)):                                             #iterate over each value
    plot_wing_power_loading_graphs(dict_directory,dict_name[i],i)