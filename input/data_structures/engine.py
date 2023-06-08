from dataclasses import dataclass
import json
import sys
import pathlib as pl
import os
sys.path.append(str(list(pl.Path(__file__).parents)[0]))



@dataclass
class Engine:
    """
    This class is to estimate the parameters of a Fuel Cell.

    :param no_engines: Number of engines [-]
    :param totalmass: Total mass of all the engines combined, so each nacelle and powertrain [kg]
    :param mass_perpowertrain: Mass of each powertrain [kg]
    :param mass_pernacelle: Mass of each nacelle [kg]
    :param mass_pertotalengine: Mass of each engines, this consists of a nacelle and a powertrain [kg]
    """
    no_engines: int = None
    totalmass: float  = None
    mass_perpowertrain: float = None
    mass_pernacelle: float = None
    mass_pertotalengine: float = None
    x_rotor_loc: float = None
    y_rotor_loc: float = None
    nacelle_width: float = None


    def load(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)
        self.no_engines = int(len(data["x_rotor_loc"]))
        self.totalmass = data["powertrain_weight"] + data["nacelle_weight"]
        self.mass_perpowertrain = data["powertrain_weight"]/self.no_engines
        self.mass_pernacelle = data["nacelle_weight"]/self.no_engines
        self.mass_pertotalengine = self.totalmass/self.no_engines
        self.x_rotor_loc = data["x_rotor_loc"]
        self.y_rotor_loc = data["y_rotor_loc"]
        self.nacelle_width = data["nacelle_width"]

    def dump(self):

        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)


        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)
