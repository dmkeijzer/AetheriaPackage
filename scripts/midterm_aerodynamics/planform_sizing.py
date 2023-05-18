import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.midterm_planform.planformsizing import *
import input.GeneralConstants as const

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

TEST = input("\n\nType 1 if you want to write the JSON data to your download folder instead of the repo, type 0 otherwise:\n") # Set to true if you want to write to your downloads folders instead of rep0
dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)
    
    new_S = (data["mtom"]*const.g0)/data["WS"]

    if data["name"] == "J1":

        WingClass = Wing(
                        surface= new_S,
                        taper=data["taper"],
                        aspectratio= data["A"],
                        quarterchord_sweep= data["sweep_1/4"])
        wing_planform(WingClass)
        # winglet_correction(WingClass, winglet_factor(data["h_wl"], data["b"], const.k_wl)) igonirign for now can't find correction factor

        data["S"] = new_S
        data["b"] = WingClass.span
        data["c_root"] = WingClass.chord_root
        data["c_tip"] = WingClass.chord_tip
        data["mac"] = WingClass.chord_mac
        data["y_mac"] = WingClass.y_mac
        data["sweep_le"] = WingClass.tan_sweep_LE
        data["sweep_1/4"] = WingClass.quarterchord_sweep
        data["x_lemac"] = WingClass.X_lemac
    
    else:
        
        WingClass1 = Wing(
                        surface= new_S*data["S1/S"],
                        taper=data["taper1"],
                        aspectratio= data["A1"],
                        quarterchord_sweep= data["sweep1_1/4"])

        WingClass2 = Wing(
                        surface= new_S*data["S2/S"],
                        taper=data["taper2"],
                        aspectratio= data["A2"],
                        quarterchord_sweep= data["sweep2_1/4"])

        wing_planform(WingClass1)
        wing_planform(WingClass2)
        # winglet_correction(WingClass, winglet_factor(data["h_wl"], data["b"], const.k_wl)) igonirign for now can't find correction factor
        
        data["S"] = new_S
        data["S1"] = new_S*data["S1/S"]
        data["S2"] = new_S*data["S2/S"]

        data["b1"] = WingClass1.span
        data["b2"] = WingClass2.span

        data["c_root1"] = WingClass1.chord_root
        data["c_root2"] = WingClass2.chord_root

        data["c_tip1"] = WingClass1.chord_tip
        data["c_tip2"] = WingClass2.chord_tip

        data["mac1"] = WingClass1.chord_mac
        data["mac2"] = WingClass2.chord_mac

        data["y_mac1"] = WingClass1.y_mac
        data["y_mac2"] = WingClass2.y_mac

        data["sweep_le1"] = WingClass1.tan_sweep_LE
        data["sweep_le2"] = WingClass2.tan_sweep_LE

        data["sweep1_1/4"] = WingClass1.quarterchord_sweep
        data["sweep2_1/4"] = WingClass2.quarterchord_sweep

        data["x_lemac1"] = WingClass1.X_lemac
        data["x_lemac2"] = WingClass2.X_lemac

    if TEST:
        with open(os.path.join(download_dir, dict_name), "w") as jsonFile:
            json.dump(data, jsonFile, indent= 6)
    else:
        with open(os.path.join(dict_directory, dict_name), "w") as jsonFile:
            json.dump(data, jsonFile, indent= 6)

