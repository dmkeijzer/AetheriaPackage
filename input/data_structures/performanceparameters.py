from dataclasses import dataclass
import json
import GeneralConstants as constants
import sys 
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))


@dataclass
class PerformanceParameters:
    """Datastructure for performance parameters
        param: energyRequired [J]
        param: cruisePower [W]
        param: hoverPower [W]
        param: climPower [W]
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
    v_stall: float = None
    v_approach: float = None
    MTOM: float = None
    wing_loading_cruise: float = None
    Stots: float = None # Total area of wing reference area
    prop_eff: float = None # Propulsive efficiency
    turn_loadfactor: float = None # Turning load factor
    v_max: float = None
    glide_slope: float = None
    max_thrust_per_engine: float = None

    powersystem_mass: float = None


    def load(self):
        os.chdir(str(list(pl.Path(__file__).parents)[2]))
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
        self.MTOM = data["mtom"]
        self.wing_loading_cruise = data["WS"]
        self.Stots = data["StotS"]
        self.prop_eff = data["prop_eff"]
        self.turn_loadfactor = data["turn_loadfactor"]
        self.v_max = data["v_max"]
        self.v_stall = data["v_stall"]
        self.G = data["G"]
        self.max_thrust_per_engine = data['max_thrust_per_engine']

        self.powersystem_mass = data['h2_weight']

    def dump(self):
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        data["power_hover"] = self.hoverPower
        data["power_cruise"] = self.cruisePower
        data["mission_energy"] = self.energyRequired
        data["power_climb"] = self.climbPower
        data["mtom"] = self.MTOM
        data["WS"] = self.wing_loading_cruise
        data["prop_eff"] = self.prop_eff
        data["turn_loadfactor"] = self.turn_loadfactor
        data["v_max"] =  self.v_max
        data["v_stall"] = self.v_stall
        data["G"] = self.G
        data['max_thrust_per_engine'] = self.max_thrust_per_engine

        data['h2_weight'] = self.powersystem_mass


    
        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=6)