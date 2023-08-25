from dataclasses import dataclass
import json
import sys
import pathlib as pl
import os
from pydantic import BaseModel, FilePath

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

import input.GeneralConstants as const

class Fuselage(BaseModel):
    length_fuselage: float 
    length_cabin: float = 2.7 # Length of the cabin
    height_cabin: float = 1.6 # Length of the cabin
    height_fuselage_outer: float  = 1.6 + const.fuselage_margin
    length_cockpit: float 
    length_tail: float 
    diameter_fuselage: float 
    upsweep: float 
    limit_fuselage: float | None = None # Length of the fuseglage
    h_wing: float | None = None # Height of the wing
    width_fuselage_inner: float | None = None
    width_fuselage_outer: float | None = None
    height_fuselage_inner: float | None = None
    volume_fuselage: float | None = None
    crash_box_area: float | None = None
    fuselage_weight: float | None = None

    # Crash diameter stuff

    bc: float = None # width crash area
    hc: float = None # height crash area
    bf: float = None # width crash area
    hf: float = None # height crash area

    @property
    def max_perimeter(self):
        #TODO Please disucss a better explanation with Jorrick
        return 2*self.height_fuselage_outer + 2*self.width_fuselage_outer


    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        return cls(**data["Fuselage"])

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Fuselage"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

