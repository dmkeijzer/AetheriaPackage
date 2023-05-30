
from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class Aero():
    cd: float = None
    cd0: float = None
    cL_alpha: float = None
    cL_cruise: float = None
    cm: float = None
    cm_alpha: float = None
    e: float = None

    def load(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        self.cd =  data["cd"]
        self.cd0 =  data["cd0"]
        self.cL_alpha =  data["clalpha"]
        self.cL_cruise =  data["cL_cruise"]
        self.cm  =  data["cm"]
        self.e  =  data["e"]
        self.cm_alpha = data["cm_alpha"]
    

if __name__ == "__main__":
    AeroClass = Aero()
    AeroClass.load()

    print(AeroClass)

