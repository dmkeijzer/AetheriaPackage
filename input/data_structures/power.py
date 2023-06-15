from dataclasses import dataclass
import json
import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

@dataclass
class Power:
    battery_mass: float = None
    fuelcell_mass: float = None
    cooling_mass: float = None
    h2_tank_mass: float = None
    nu_FC_cruise_fraction: float = None
    battery_power : float = None
    battery_energy : float = None
    battery_volume: float = None
    fuelcell_volume : float = None
    h2_tank_volume : float = None

    def load(self):
            """ Initializes the class automatically from the JSON file
            """
            os.chdir(str(list(pl.Path(__file__).parents)[2]))
            with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
                data = json.load(jsonFile)
            self.battery_mass = data["battery_mass"]
            self.fuelcell_mass = data["fuelcell_mass"]
            self.cooling_mass = data["cooling_mass"]
            self.h2_tank_mass = data["h2_tank_mass"]
            self.nu_FC_cruise_fraction = data["nu_FC_cruise_fraction"]
            self.battery_power = data["battery_power"]
            self.battery_energy = data["battery_energy"]
            self.battery_volume = data["battery_volume"]
            self.fuelcell_volume = data["fuelcell_volume"]
            self.h2_tank_volume = data["h2_tank_volume"]

    def dump(self):
        """Dumps values into the json file"""
        os.chdir(str(list(pl.Path(__file__).parents)[2]))
        with open(r"input/data_structures/aetheria_constants.json") as jsonFile:
            data = json.load(jsonFile)

        data["battery_mass"] = self.battery_mass
        data["fuelcell_mass"] = self.fuelcell_mass
        data["cooling_mass"] = self.cooling_mass
        data["h2_tank_mass"] = self.h2_tank_mass
        data["nu_FC_cruise_fraction"] = self.nu_FC_cruise_fraction
        data["battery_power"] = self.battery_power
        data["battery_energy"] = self.battery_energy
        data["battery_volume"] = self.battery_volume
        data["fuelcell_volume"] = self.fuelcell_volume
        data["h2_tank_volume"] = self.h2_tank_volume
        
        with open(r"input/data_structures/aetheria_constants.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)
