# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import matplotlib.pyplot as plt

from modules.midterm_structures import *

b = 40
h = 30
t = 3
def root_wingbox_geometry_cruise(dict_directory,dict_name,i):
    with open(dict_directory+"\\"+dict_name, "r") as jsonFile:
        data = json.loads(jsonFile.read())
    if data["tandem_bool"] ==1:
        c_root = data["c_root2"]
        wing_weight = data["wing2_weight"]
        b = data["b2"]
    else:
        c_root = data["c_root"]
        wing_weight = data["wing_weight"]
        b = data["b"]
        
    h_root = c_root*t_c_ratio
    
    t = 5
    h_wingbox = 0.8*h_root
    w_wingbox = 0.6*c_root
    
    i_xx = i_xx_thinwalled(w_wingbox,h_wingbox,3E-3)
    i_yy = i_yy_thinwalled(w_wingbox,h_wingbox,3E-3)
    j_z = j_z_thinwalled(w_wingbox,h_wingbox,3E-3)
    
    cross_section = h_wingbox*w_wingbox - ((h_wingbox-t*2)*(w_wingbox-t*2))
    
    torsional_shear = 
    shear_y = 
    
    max_normal = bending_stress(moment_x,moment_y,i_xx,i_yy,i_xy,c_root/2,h_root/2)
    
    return 
    

dict_directory = str(list(pl.Path(__file__).parents)[2])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration
for i in range(len(dict_name)):                                             #iterate over each value
    root_wingbox_geometry(dict_directory,dict_name[i],i)
