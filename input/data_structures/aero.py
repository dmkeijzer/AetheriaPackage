
from dataclasses import dataclass
import json
import sys
import os
from pydantic import BaseModel, FilePath
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.aero.midterm_datcom_methods import datcom_cl_alpha
import input.GeneralConstants as const

class Aero(BaseModel):
    label : str = "Aero"
    cd0_cruise: float 
    cL_max: float 
    cL_max_flaps60: float 
    cL_alpha: float 
    e: float 
    v_stall_flaps20: float = const.v_stall_flaps20
    cL_descent_trans_flaps20: float = const.cl_descent_trans_flaps20
    alpha_descent_trans_flaps20: float  = const.alpha_descent_trans_flaps20
    cdi_climb_clean: float  = const.cdi_climb_clean
    alpha_climb_clean: float   = const.alpha_climb_clean
    cl_climb_clean: float   = const.cl_climb_clean
    ld_climb: float  = const.ld_climb
    cL_alpha0_approach: float   = const.cL0_approach
    alpha_approach: float   = const.alpha_approach
    cd_cruise: float | None = None
    cd_upsweep: float | None  = None
    cd_base: float | None  = None
    cL_cruise: float | None  = None
    cm_ac: float | None  = None
    cm_alpha: float | None  = None
    alpha_zero_L: float | None  = None
    ld_cruise: float | None  = None
    downwash_angle: float   = const.downwash_angle
    downwash_angle_wing: float = const.downwash_angle_wing
    downwash_angle_prop: float  = const.downwash_angle_prop
    downwash_angle_stall: float   = const.downwash_angle_stall
    downwash_angle_wing_stall: float   = const.downwash_angle_wing_stall
    downwash_angle_prop_stall: float   = const.downwash_angle_prop_stall
    ld_stall: float | None  = None
    cd_stall: float | None  = None
    cd0_stall: float | None  = None
    mach_stall: float | None  = None
    deps_da: float | None  = None
    mach_cruise: float | None  = None
    cL_plus_slipstream: float | None  = None
    cL_plus_slipstream_stall: float | None  = None
    delta_alpha_zero_L_flaps60: float | None  = None
    cd_tot_cruise: float | None  = None

    @classmethod
    def load(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        return cls(**data["Aero"])

    def dump(self, file_path: FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)

        data["Aero"] = self.model_dump()

        with open(file_path, "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)


    

if __name__ == "__main__":
    AeroClass = Aero()
    AeroClass.load()

    print(AeroClass)

