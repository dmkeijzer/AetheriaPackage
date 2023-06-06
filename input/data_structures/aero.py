
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
    cd: float = None
    cd0: float = None
    cL_alpha: float = None
    cL_cruise: float = None
    cm_ac: float = None
    cm_alpha: float = None
    e: float = None

    def load(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        self.cd =  data["cd"]
        self.cd0 =  data["cd0"]
        self.cL_alpha =  datcom_cl_alpha(A=data["A"], mach=v_cr/a_cr, sweep_half=-data["sweep_le"])
        self.cL_cruise =  data["cL_cruise"]
        self.cm_ac  =  data["cm_ac"]
        self.e  =  data["e"]
        self.cm_alpha = data["cm_alpha"]
        self.cl_alpha = data["cl_alpha"]

    def dump(self):
        data = {
            "cd": self.cd,
            "cd0": self.cd0,
            "cL_cruise":self.cruise,
            "cm_ac": self.cm_ac,
            "e": self.e,
            "cm_alpha": self.cm_alpha,
            "cl_alpha": self.cl_alpha
        }
    
        with open(r"output/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)
    

if __name__ == "__main__":
    AeroClass = Aero()
    AeroClass.load()

    print(AeroClass)


