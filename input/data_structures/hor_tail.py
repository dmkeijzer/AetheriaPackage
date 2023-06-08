
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
    #cm_alpha_h: float = None
    #cL_alpha_h: float = None
    #cm_h: float = None
    #cL_approach_h: float = None
    #chord_mac_h: float = None
    downwash: float = None
    hortailsurf_wingsurf: float = None
    sweep_halfchord_h: float = None
    downwash_angle: float = None
    taper_h: float = None
    def load(self):
        """ Initializes the class automatically from the JSON file
        """        
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.surface = data["S_h"]
        self.aspect_ratio= data["A_h"]
        self.t_r_h = data["t_r_h"]
        #self.b_h = data["b_h"]
        #self.cm_alpha_h = data["cm_alpha_h"]
        #self.cL_alpha_h = data["cl_alpha_h"]
        #self.cm_h = data["cmac_h"]
        #self.cL_approach_h = data["cL_approach_h"]
        #self.chord_mac_h = data["mac_h"]
        self.downwash = data["depsda"]
        self.hortailsurf_wingsurf = data["hortailsurf_wingsurf"]
        self.sweep_halfchord_h = data["sweep_halfchord_h"]
        self.downwash_angle = data["downwash_angle"]
        self.taper_h = 1

    def dump(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        """Dumps values into json file"""

        data["S_h"] = self.surface
        data["A_h"] = self.aspect_ratio
        data["t_r_h"] = self.t_r_h
        data["hortailsurf_wingsurf"] = self.hortailsurf_wingsurf

        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)