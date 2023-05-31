# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import matplotlib.pyplot as plt
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
from modules.structures import Flight_Envelope
from input.GeneralConstants import *

write_bool = int(input("\n\nType 1 if you want to write the JSON data to your download folder instead of the repo, type 0 otherwise:\n"))
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

def flightEnvelope(dict_directory, dict_name, PRINT=False):
    with open(dict_directory + "\\" + dict_name, "r") as jsonFile:
        data = json.loads(jsonFile.read())

    if data["name"] == "J1":
        cl_alpha = data["cl_alpha"]
    else:
        cl_alpha = data["cl_alpha2"]

    nm = Flight_Envelope.plotmaneuvrenv(data['WS'], data['v_cruise'], data['cLmax'], n_min_req, n_max_req)
    ng = Flight_Envelope.plotgustenv(data['v_stall'], data['v_cruise'], cl_alpha, data['WS'], TEXT=True)

    data['n_max'], data['n_ult'] = max(nm, ng), max(nm, ng)*1.5


    if write_bool:
        with open(download_dir+"\\"+dict_name, "w") as jsonFile:
            json.dump(data, jsonFile,indent=2)
        print("Data written to downloads folder.")
    else:
        with open(dict_directory+"\\"+dict_name, "w") as jsonFile:
            json.dump(data, jsonFile,indent=2)
        print("Old files were overwritten.")

    output_directory = str(list(pl.Path(__file__).parents)[2])+"\\output\\midterm_structures\\"
    plt.savefig(output_directory+str('Vn-')+str(data['name'])+".png")
    # plt.plot([0, 40, 80, 100, 100, 80, 40, 0],[1, 3, 4, 2, -1, -1.5, -1, 1])
    if PRINT:
        print("Max load factor for", str(data['name']),":", round(max(nm, ng),2))
        print("Ultimate load factor for", str(data['name']), ":", round(max(nm, ng)*1.5, 2))
        plt.show()
    #print("nm: ", round(nm, 2))
    #print("ng: ", round(ng, 2))
    #print("Vb: ", round(data['v_stall']*np.sqrt(ng)))

dict_directory = str(list(pl.Path(__file__).parents)[2])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration

for i in range(len(dict_name)):                                             #iterate over each value
    flightEnvelope(dict_directory,dict_name[i], PRINT=False)
