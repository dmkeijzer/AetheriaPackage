# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""
from dataclasses import dataclass
import GeneralConstants as constants
import sys 
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[0]))


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
    
    def load(self):
        self.energyDensity = constants.EnergyDensityTank
        self.volumeDensity = constants.VolumeDensityTank


    def mass(self,energy) -> float:
        """
        :return: Mass of the battery
        """
        return energy / self.energyDensity

    def volume(self,energy) -> float:
        """
        :return: Volume of the battery [m^3]
        """
        return energy / self.volumeDensity * 0.001

    def price(self,energy) -> float:
        """
        :return: Approx cost of the battery [US$]
        """
        return energy * self.cost