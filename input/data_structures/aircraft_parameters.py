from dataclasses import dataclass
import json
import sys 
import pathlib as pl
from pydantic import BaseModel, FilePath
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))


@dataclass
class AircraftParameters(BaseModel):
    MTOM: float 
    Stots: float # Total area of wing reference area
    prop_eff: float  # Propulsive efficiency
    glide_slope: float 


    #energy 
    mission_energy: float | None = None
    takeoff_energy: float | None = None
    climb_energy: float | None = None
    cruise_energy: float | None = None
    descend_energy: float | None = None
    hor_loiter_energy: float | None = None
    trans2hor_energy: float | None = None
    trans2ver_energy: float | None = None
    ver_loiter_energy: float | None = None

    #power
    cruisePower : float = None
    hoverPower : float = None
    climbPower : float = None
    
    #performance
    v_stall: float = None
    v_approach: float = None
    OEM: float = None
    wing_loading_cruise: float = None
    turn_loadfactor: float = None # Turning load factor
    v_max: float = None
    max_thrust_per_engine: float = None

    # Load factors
    n_max: float = None
    n_ult : float = None

    #CG
    cg : float | None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        return cls(**data["AircraftParameters"])


    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["AircraftParameters"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

