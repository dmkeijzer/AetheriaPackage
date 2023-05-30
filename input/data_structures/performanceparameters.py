from dataclasses import dataclass
import json
import GeneralConstants as constants
import sys 
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))


@dataclass
class PerformanceParameters:
    """Datastructure for performance parameters
        param: energyRequired [kWh]
        param: cruisePower [kW]
        param: hoverPower [kW]
        param: climPower [kW]
        param: rate_of_climb_cruise [m/s]
        param: rate_of_descend_cruise [m/s]
    """
    #energy & power
    energyRequired: float = None
    cruisePower : float = None
    hoverPower : float = None
    climbPower : float = None
    
    #performance
    rate_of_climb_cruise: float = None
    rate_of_descent_cruise : float = None
    rate_of_climb_hover: float = None
    cruise_velocity : float = None

    def load(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        self.hoverPower = data["power_hover"]
        self.cruisePower = data["power_cruise"]
        self.energyRequired = data["mission_energy"]
        self.climbPower = data["power_cruise"]

        self.rate_of_climb_cruise = constants.roc_cr
        self.rate_of_descent_cruise = constants.rod_cr
        self.rate_of_climb_hover = constants.roc_hvr
        self.cruise_velocity = constants.v_cr

    
    def dump(self):
        data = {
            "power_hover": self.hoverPower,
            "power_cruise": self.cruisePower,
            "mission_energy": self.energyRequired,
            "power_climb": self.climbPower
        }
    
        with open(r"output/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)