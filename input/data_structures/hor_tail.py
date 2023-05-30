
from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class HorTail():
    S_h: float = None #  Surface area
    A_H: float = None #  aspect ratio of horizontal tail
    t_r_h: float = None #  maximum thickness at the root of the horizontal tail
    b_h: float = None # span of horizontal tail 

    def load(self):
        """ Initializes the class automatically from the JSON file
        """        
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.lf = data["l_fuse"]
        self.d_fuselage = data["d_fuselage"]
        self.upsweep = data["upsweep"]
        self.h_wing = data["h_wings"]