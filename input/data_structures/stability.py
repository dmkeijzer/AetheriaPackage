from dataclasses import dataclass
import json
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class Stab():

    Cm_de: float = None
    Cn_dr: float = None

    def load(self):
        """ Initializes the class automatically from the JSON file
        """
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.Cm_de = data["Cm_de"]
        self.Cn_dr = data["Cn_dr"]


    def dump(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        """Dumps values into json file"""

        data["Cn_dr"] = self.Cn_dr
        data["Cm_de"] = self.Cm_de


        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)