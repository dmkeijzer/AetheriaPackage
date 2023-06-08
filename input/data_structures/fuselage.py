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
    width_fuselage: float = None
    height_fuselage: float = None
    mass_fuselage: float = None

    def load(self):
        """ Initializes the class automatically from the JSON file
        """        
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.length_fuselage = data["l_fuse"]
        self.diameter_fuselage = data["d_fuselage"]
        self.upsweep = data["upsweep"]
        self.h_wing = data["h_wings"]
        self.width_fuselage = data["w_fuselage"]
        self.height_fuselage = data["h_fuselage"]
        self.mass_fuselage = data['fuselage_weight']



    def dump(self):
        """Dumps values into the json file"""
        data = {
            "l_fuse": self.length_fuselage,
            "d_fuselage": self.diameter_fuselage,
            "upsweep": self.upsweep,
            "h_wings": self.h_wing,
            'fuselage_weight': self.mass_fuselage
        }

        with open(r"output/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)