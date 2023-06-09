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
    limit_fuselage: float = None # Length of the fuseglage
    length_cabin: float = 2.7 # Length of the cabin
    height_cabin: float = 1.6 # Length of the cabin
    diameter_fuselage: float = None # Diameter of the fuselage
    upsweep: float = None #  Upsweep of the fuselage
    h_wing: float = None # Height of the wing
    width_fuselage_inner: float = None
    width_fuselage_outer: float = None
    height_fuselage_inner: float = None
    height_fuselage_outer: float = None
    volume_fuselage: float = None
    crash_box_area: float = None
    length_cockpit: float = None
    length_tail: float = None
    fuselage_weight: float = None

    # Crash diameter stuff

    bc: float = None # width crash area
    hc: float = None # height crash area
    bf: float = None # width crash area
    hf: float = None # height crash area

    @property
    def max_perimeter(self):
        #TODO Please disucss a better explanation with Jorrick
        return 2*self.height_fuselage_outer + 2*self.width_fuselage_outer


    def load(self):
        """ Initializes the class automatically from the JSON file
        """ 
        os.chdir(str(list(pl.Path(__file__).parents)[2]))
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.length_fuselage = data["l_fuse"]
        self.limit_fuselage = data["l_fuse"]
        self.diameter_fuselage = data["d_fuselage"]
        self.upsweep = data["upsweep"]
        self.h_wing = data["h_wings"]
        self.width_fuselage_inner = data["w_fuselage_inner"]
        self.width_fuselage_outer = data["w_fuselage_outer"]
        self.height_fuselage_inner = data["h_fuselage_inner"]
        self.height_fuselage_outer = data["h_fuselage_outer"]
        self.volume_fuselage = data["volume_fuselage"]
        self.length_cockpit = data['l_cockpit']
        self.length_tail = data['l_tail']
        self.fuselage_weight = data["fuselage_weight"] 
        self.crash_box_area= data["crash_box_area"] 
        



    def dump(self):
        """Dumps values into the json file"""
        os.chdir(str(list(pl.Path(__file__).parents)[2]))
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        data["l_fuse"] = self.length_fuselage
        data["limit_fuse"] = self.limit_fuselage
        data["d_fuselage"] = self.diameter_fuselage
        data["upsweep"] =  self.upsweep
        data["h_wings"] = self.h_wing
        data["w_fuselage_inner"] = self.width_fuselage_inner
        data["w_fuselage_outer"] = self.width_fuselage_outer
        data["h_fuselage_inner"] = self.height_fuselage_inner
        data["h_fuselage_outer"] = self.height_fuselage_outer
        data["volume_fuselage"] = self.volume_fuselage
        data['l_cockpit'] = self.length_cockpit
        data['l_tail'] = self.length_tail
        data["bc"] = self.bc
        data["bf"] = self.bf
        data["hc"] = self.hc
        data["hf"] = self.hf
        data["fuselage_weight"] = self.fuselage_weight
        data["crash_box_area"] = self.crash_box_area


        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)

