from dataclasses import dataclass
import json
import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class Fuselage():
    length_fuselage: float = None # Length of the fuseglage
    length_cabin: float = None # Length of the cabin
    diameter_fuselage: float = None # Diameter of the fuselage
    upsweep: float = None #  Upsweep of the fuselage
    h_wing: float = None # Height of the wing 

    def load(self):
        """ Initializes the class automatically from the JSON file
        """        
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.length_fuselage = data["l_fuse"]
        self.diameter_fuselage = data["d_fuselage"]
        self.upsweep = data["upsweep"]
        self.h_wing = data["h_wings"]

    def dump(self):
        """Dumps values into the json file"""
        data = {
            "l_fuse": self.length_fuselage,
            "d_fuselage": self.diameter_fuselage,
            "upsweep": self.upsweep,
            "h_wings": self.h_wing
        }

        with open(r"output/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)