from dataclasses import dataclass
import json
import sys
from pydantic import BaseModel, FilePath
import pathlib as pl
import numpy as np
import os
sys.path.append(str(list(pl.Path(__file__).parents)[0]))



class Engine(BaseModel):
    """
    This class is to estimate the parameters of a Fuel Cell.

    :param no_engines: Number of engines [-]
    :param totalmass: Total mass of all the engines combined, so each nacelle and powertrain [kg]
    :param mass_perpowertrain: Mass of each powertrain [kg]
    :param mass_pernacelle: Mass of each nacelle [kg]
    :param mass_pertotalengine: Mass of each engines, this consists of a nacelle and a powertrain [kg]
    """
    x_rotor_loc: list 
    y_rotor_loc: list 
    pylon_length: float 

    totalmass: float | None   = None
    mass_perpowertrain: float | None  = None
    mass_pernacelle: float | None  = None
    mass_pertotalengine: float | None  = None
    #nacelle_width: float = None
    total_disk_area: float | None  = None
    thrust_coefficient: float | None  = None
    thrust_per_engine: float | None  = None
    hub_radius: float | None  = None
    prop_radius: float | None  = None
    prop_area: float | None  = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        return cls(**data["Engine"])

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Engine"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

