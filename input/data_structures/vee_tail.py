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
    label : str = "Vtail"
    surface: float 
    quarterchord_sweep: float 
    length_wing2vtail: float | None  = None
    virtual_hor_surface: float | None  = None
    virtual_ver_surface: float | None  = None
    rudder_max: float | None  = None
    elevator_min: float | None  = None
    dihedral: float | None  = None
    taper: float | None  = 1
    c_control_surface_to_c_vee_ratio: float | None  = None
    cL_cruise: float | None  = None
    max_clh: float | None  = None
    ruddervator_efficiency: float | None  = None
    span: float | None  = None
    vtail_weight: float | None  = None
    thickness_to_chord: float | None  = 0.12
    aspect_ratio: float | None = None
    chord_root: float | None = None
    chord_tip: float | None = None
    chord_mac: float | None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Veetail"])
        except:
            raise Exception(f"There was an error when loading in {cls}")
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Veetail"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


