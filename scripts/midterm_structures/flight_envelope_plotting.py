# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import matplotlib.pyplot as plt
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
from modules.midterm_structures import Flight_Envelope
from input.GeneralConstants import *


def flightEnvelope(dict_directory, dict_name, PRINT=False):
    with open(dict_directory + "\\" + dict_name, "r") as jsonFile:
        data = json.loads(jsonFile.read())

    nm = Flight_Envelope.plotmaneuvrenv(data['WS'], data['v_cruise'], data['cLmax'], n_min, n_max)
    ng = Flight_Envelope.plotgustenv(data['v_stall'], data['v_cruise'], data['clalpha'], data['WS'], TEXT=True)

    data['n_max'], data['n_ult'] = max(nm, ng), max(nm, ng)*1.5

    write_bool = int(input("Do you want to overwrite the current loading values? type 1 if you want to do this.")) == 1
    if write_bool==True:
        with open(dict_directory+"\\"+dict_name, "w") as jsonFile:
            json.dump(data, jsonFile,indent=2)
        print("Old files were overwritten.")

    output_directory = str(list(pl.Path(__file__).parents)[2])+"\\output\\midterm_structures\\"
    plt.savefig(output_directory+str('Vn-')+str(data['name'])+".png")
    # plt.plot([0, 40, 80, 100, 100, 80, 40, 0],[1, 3, 4, 2, -1, -1.5, -1, 1])
    plt.show()
    if PRINT:
        print("Max load factor for", str(data['name']),":", round(max(nm, ng),2))
    #print("nm: ", round(nm, 2))
    #print("ng: ", round(ng, 2))
    #print("Vb: ", round(data['v_stall']*np.sqrt(ng)))

dict_directory = str(list(pl.Path(__file__).parents)[2])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration

for i in range(len(dict_name)):                                             #iterate over each value
    flightEnvelope(dict_directory,dict_name[i], PRINT=True)
