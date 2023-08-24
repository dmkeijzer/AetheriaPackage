from dataclasses import dataclass
import json
import sys
import pathlib as pl
import numpy as np
import os
sys.path.append(str(list(pl.Path(__file__).parents)[0]))



@dataclass
class Engine():
    """
    This class is to estimate the parameters of a Fuel Cell.

    :param no_engines: Number of engines [-]
    :param totalmass: Total mass of all the engines combined, so each nacelle and powertrain [kg]
    :param mass_perpowertrain: Mass of each powertrain [kg]
    :param mass_pernacelle: Mass of each nacelle [kg]
    :param mass_pertotalengine: Mass of each engines, this consists of a nacelle and a powertrain [kg]
    """
    totalmass: float  = None
    mass_perpowertrain: float = None
    mass_pernacelle: float = None
    mass_pertotalengine: float = None
    x_rotor_loc: float = None
    y_rotor_loc: float = None
    #nacelle_width: float = None
    total_disk_area: float = None
    thrust_coefficient: float = None
    thrust_per_engine: float = None
    hub_radius: float = None
    prop_radius: float = None
    prop_area: float = None
    pylon_length: float = None


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
        #self.nacelle_width = data["nacelle_width"]
        self.total_disk_area = data["diskarea"]
        self.thrust_coefficient = data['C_T']
        self.thrust_per_engine = data["max_thrust_per_engine"]
        self.hub_radius = data['hub_radius']
        self.prop_area = data['mtom'] / (120 * 6)
        self.prop_radius = np.sqrt(self.prop_area / np.pi)
        self.pylon_length = data["pylon_length"]

    def dump(self):

        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        data["A_tot"] = self.total_disk_area
        data["x_rotor_loc"][2] = data["x_lewing"] + data["c_root"] * 0.25 + data["c_tip"]*0.20
        data["x_rotor_loc"][3] = self.x_rotor_loc[2]
        data["y_rotor_loc"][2] = data["b"]/2
        data["y_rotor_loc"][3] = -self.y_rotor_loc[2]
        data['C_T'] = self.thrust_coefficient
        data['hub_radius'] = self.hub_radius
        data['prop_radius'] = self.prop_radius
        data['prop_area'] = self.prop_area

        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)
