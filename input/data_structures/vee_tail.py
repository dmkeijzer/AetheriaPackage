from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl
import numpy as np
from pydantic import BaseModel, FilePath

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

class VeeTail(BaseModel):
    surface: float 
    quarterchord_sweep: float 
    Vh_V2: float | None  = None
    length_wing2vtail: float | None  = None
    rudder_max: float | None  = None
    elevator_min: float | None  = None
    dihedral: float | None  = None
    taper: float | None  = 1
    c_control_surface_to_c_vee_ratio: float | None  = None
    ruddervator_efficiency: float | None  = None
    span: float | None  = None
    vtail_weight: float | None  = None
    thickness_to_chord: float | None  = None


    @property
    def aspectratio(self):
        return self.span**2/self.surface
    
    @property
    def chord_root(self):
        return  2 *self.surface / ((1 + self.taper) * self.span)

    @property
    def chord_tip(self):
        return self.chord_root * self.taper

    @property
    def chord_mac(self):
        return  (2 / 3) * self.chord_root  * ((1 + self.taper + self.taper ** 2) / (1 + self.taper))


    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        return cls(**data["Veetail"])
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Veetail"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


