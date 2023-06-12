
from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.aero.midterm_datcom_methods import datcom_cl_alpha
from input.data_structures.GeneralConstants import *

@dataclass
class Aero():
    cd_cruise: float = None
    cd0_cruise: float = None
    cd_upsweep: float = None
    cd_base: float = None
    cL_alpha: float = None
    cL_cruise: float = None
    cL_max: float = None
    cm_ac: float = None
    cm_alpha: float = None
    e: float = None
    cm_alpha: float = None
    cL_alpha: float = None
    cdi_climb_clean: float = None
    cl_climb_clean: float = None
    alpha_climb_clean: float = None
    ld_climb: float = None
    ld_cruise: float = None
    downwash_angle: float = None
    downwash_angle_wing: float = None
    downwash_angle_prop: float = None
    downwash_angle_stall: float = None
    downwash_angle_wing_stall: float = None
    downwash_angle_prop_stall: float = None
    ld_stall: float = None
    cd_stall: float = None
    cd0_stall: float = None
    mach_stall: float = None

    def load(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        self.cd_cruise =  data["cd"]
        self.cd0_cruise =  data["cd0"]
        self.cd_upsweep = data['cd_upsweep']
        self.cd_base = data['cd_base']
        self.cL_alpha =  datcom_cl_alpha(A=data["A"], mach=v_cr/a_cr, sweep_half=-data["sweep_le"])
        self.cL_cruise =  data["cL_cruise"]
        self.cL_max = data['cLmax']
        self.cm_ac  =  data["cm_ac"]
        self.e  =  data["e"]
        self.cm_alpha = data["cm_alpha"]
        self.cl_alpha = data["cl_alpha"]
        self.cdi_climb_clean = data["cdi_climb_clean"]
        # self.cl_climb_clean = data["cl_climb_clean"]
        self.alpha_climb_clean = data["alpha_climb_clean"] 
        self.ld_climb  = data["ld_climb"] 
        self.v_stall = data['v_stall']
        self.ld_cruise = data['ld_cr']
        self.downwash_angle = data['downwash_angle']
        self.downwash_angle_wing = data['downwash_angle_wing']
        self.downwash_angle_prop = data['downwash_angle_prop']
        self.downwash_angle_stall = data['downwash_angle_stall']
        self.downwash_angle_wing_stall = data['downwash_angle_wing_stall']
        self.downwash_angle_prop_stall = data['downwash_angle_prop_stall']
        self.ld_stall = data['ld_stall']
        self.cd_stall = data['cd_stall']
        self.cd0_stall = data['cd0_stall']
        self.mach_stall = data['mach_stall']
        
        

    def dump(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        data["cd"] = self.cd_cruise
        data["cd0"] = self.cd0_cruise
        data['cd_upsweep'] = self.cd_upsweep
        data['cd_base'] = self.cd_base
        data["cL_cruise"] = self.cruise
        data['cL_max'] = self.cL_max
        data["cm_ac"] = self.cm_ac
        data["e"] = self.e
        data["cm_alpha"] = self.cm_alpha
        data["cl_alpha"] = self.cl_alpha
        data["cdi_climb_clean"] = self.cdi_climb_clean
        data["cl_climb_clean"] = self.cl_climb_clean
        data["alpha_climb_clean"] = self.alpha_climb_clean
        data["ld_climb"] = self.ld_climb
        data['v_stall'] = self.v_stall
        data['ld_cr'] = self.ld_cruise
        data['downwash_angle'] = self.downwash_angle 
        data['downwash_angle_wing'] = self.downwash_angle_wing 
        data['downwash_angle_prop'] = self.downwash_angle_prop 
        data['downwash_angle_stall'] = self.downwash_angle_stall
        data['downwash_angle_wing_stall'] = self.downwash_angle_wing_stall
        data['downwash_angle_prop_stall'] = self.downwash_angle_prop_stall 
        data['ld_stall'] = self.ld_stall
        data['cd_stall'] = self.cd_stall
        data['cd0_stall'] = self.cd0_stall
        data['mach_stall'] = self.mach_stall

        with open(r"output/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)
    

if __name__ == "__main__":
    AeroClass = Aero()
    AeroClass.load()

    print(AeroClass)

