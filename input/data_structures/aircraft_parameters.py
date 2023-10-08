from dataclasses import dataclass
import json
import sys 
import pathlib as pl
from pydantic import BaseModel, FilePath
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import input.GeneralConstants as const


class AircraftParameters(BaseModel):
    label: str = "Aircraft"
    MTOM: float 
    Stots: float # Total area of wing reference area
    prop_eff: float  # Propulsive efficiency
    glide_slope: float 


    #energy 
    mission_energy: float | None = None
    mission_time: float | None = None
    takeoff_energy: float | None = None
    climb_energy: float | None = None
    cruise_energy: float | None = None
    descend_energy: float | None = None
    hor_loiter_energy: float | None = None

    #power
    cruisePower : float | None = None
    hoverPower : float | None = None
    max_thrust: float | None = None
    TW_max: float | None = None
    
    #performance
    v_stall: float = const.v_stall
    v_approach: float | None = None
    OEM: float | None = None
    wing_loading_cruise: float | None = None
    turn_loadfactor: float | None = None # Turning load factor
    v_max: float | None = None
    max_thrust_per_engine: float | None = None

    # Load factors
    n_max: float | None = None
    n_ult : float | None = None

    #CG and weight
    oem_cg : float | None = None
    cg_front : float | None = None
    cg_rear : float | None = None
    cg_front_bar : float | None = None
    cg_rear_bar : float | None = None
    wing_loc: float | None = None
    oem_mass : float | None = None
    powersystem_mass: float | None = None
    misc_mass: float | None = None
    lg_mass: float | None = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data["AircraftParameters"])
        except:
            raise Exception(f"There was an error when loading in {cls}")


    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["AircraftParameters"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)

