# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""
from dataclasses import dataclass
import GeneralConstants as constants
@dataclass
class HydrogenTank:
    """
    This class is to estimate the parameters of a Hydrogen tank.

    :param EnergyDensity  [kWh/kg]
    :param VolumeDensity [kWh/l]
    :param cost [US$/kWh] (not loaded)
    :param energy [kWh] (not loaded)
    """
    energyDensity: float = None
    volumeDensity: float = None
    cost: float = None
    energy: float = None
    
    def load(self):
        self.energyDensity = constants.EnergyDensityTank
        self.volumeDensity = constants.VolumeDensityTank


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