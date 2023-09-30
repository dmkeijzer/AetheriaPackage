from pydantic import BaseModel, FilePath
from warnings import warn
import json
import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

class Power(BaseModel):
    label : str = "Power"
    battery_mass: float | None = None
    fuelcell_mass: float | None = None
    cooling_mass: float | None = None
    h2_tank_mass: float | None = None
    nu_FC_cruise_fraction: float | None = None
    battery_power : float | None = None
    battery_energy : float | None = None
    battery_volume: float | None = None
    fuelcell_volume : float | None = None
    h2_tank_volume : float | None = None
    h2_tank_length : float | None = None
    powersystem_mass: float | None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["Power"])
        except:
            raise Exception(f"There was an error when loading in {cls}")
        
    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Power"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)
