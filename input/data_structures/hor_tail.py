
from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class HorTail():
    surface: float = None #  Surface area
    aspect_ratio: float = None #  aspect ratio of horizontal tail
    t_r_h: float = None #  maximum thickness at the root of the horizontal tail
    b_h: float = None # span of horizontal tail 

    def load(self):
        """ Initializes the class automatically from the JSON file
        """        
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.surface = data["S_h"]
        self.aspect_ratio= data["A_h"]
        self.t_r_h = data["t_r_h"]
        #self.b_h = data["b_h"]

    def dump(self):
        """Dumps values into json file"""
        data = {
        "S_h": self.surface,
        "A_h": self.aspect_ratio,
        "t_r_h": self.t_r_h,
        # "b_h": self.b_h
        }

        with open(r"output/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)