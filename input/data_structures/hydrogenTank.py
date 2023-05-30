# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""
from dataclasses import dataclass
import json
@dataclass
class HydrogenTank:
    """
    This class is to estimate the parameters of a Hydrogen tank.

    :param EnergyDensity  [kWh/kg]
    :param VolumeDensity [kWh/l]
    :param cost [US$/kWh]
    :param energy [kWh]
    """
    EnergyDensity: float = None
    VolumeDensity: float = None
    cost: float = None
    energy: float = None
    
    def load(self):
        jsonfilename = "aetheria_constants.json"
        with open(jsonfilename) as jsonfile:
            data = json.open(jsonfile)
            raise NotImplementedError


    def mass(self) -> float:
        """
        :return: Mass of the battery
        """
        return self.energy / self.EnergyDensity

    def volume(self) -> float:
        """
        :return: Volume of the battery [m^3]
        """
        return self.energy / self.VolumeDensity * 0.001

    def price(self) -> float:
        """
        :return: Approx cost of the battery [US$]
        """
        return self.energy * self.cost