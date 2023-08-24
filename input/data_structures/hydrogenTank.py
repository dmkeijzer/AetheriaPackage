# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""
from dataclasses import dataclass
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
    """
    energyDensity: float = None
    volumeDensity: float = None
    cost: float = None
    
    def load(self):
        self.energyDensity = 1.8  # 1.8 kWh/kg
        self.volumeDensity = 0.6  #0.6 kWg/l
        self.cost = 16  # 16 USD / wH


    def mass(self,energy) -> float:
        """
        :return: Mass of the battery
        """
        return energy / self.energyDensity

    def volume(self,energy) -> float:
        """
        :return: Volume of the tank [m^3]
        """
        return energy / self.volumeDensity * 0.001

    def price(self,energy) -> float:
        """
        :return: Approx cost of the battery [US$]
        """
        return energy * self.cost


if __name__ == "__main__":
    test = HydrogenTank()
    test.load()
    print(test.volume(786963069.9958103/3.6e6))
