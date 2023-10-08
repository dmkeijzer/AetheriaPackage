from dataclasses import dataclass
import json
import sys
from pydantic import BaseModel, FilePath
import pathlib as pl
import numpy as np
import os
sys.path.append(str(list(pl.Path(__file__).parents)[0]))



class Engine(BaseModel):
    label : str = "Engine"
    x_rotor_loc: list 
    y_rotor_loc: list 
    pylon_length: float 
    total_disk_area: float   

    totalmass: float | None   = None
    mass_perpowertrain: float | None  = None
    mass_pernacelle: float | None  = None
    mass_pertotalengine: float | None  = None
    #nacelle_width: float = None
    thrust_coefficient: float | None  = None
    thrust_per_engine: float | None  = None
    hub_radius: float | None  = None
    prop_radius: float | None  = None
    prop_area: float | None  = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Engine"])
        except:
            raise Exception(f"There was an error when loading in {cls}")

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Engine"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

