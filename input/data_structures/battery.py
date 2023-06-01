# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""
from dataclasses import dataclass
import numpy as np
import os
import sys 
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))


@dataclass
class Battery:
    """
    This class is to estimate the parameters of a Hydrogen tank.

    :param EnergyDensity: Specific energy density [kWh/kg]
    :param Energy: Required total energy for the tank [kWh]
    :param Power: Power output of the tank [kW]
    :param PowerDensity: Power density [kW/kg]
    :param VolumeDensity: Specific volumetric density [kWh/l]
    :param CostDensity: Cost density [US$/kWh]
    :param Efficiency: Efficiency of the tank
    :param Depth_of_discharge: Depth of discharge (DOD)
    :param ChargingEfficiency: Charging efficiency
    :param End_of_life
    """
    #densities
    EnergyDensity : float = 0.34
    PowerDensity : float  = 3.8
    VolumeDensity : float = .85
    CostDensity : float = None

    #extra parameters
    Efficiency : float = None
    Depth_of_discharge : float = 1
    End_of_life : float = 0.8
    ChargingEfficiency : float = None



    def energymass(self, Energy):
        """
        :return: Mass of the battery [kg]
        """
        return Energy/ self.EnergyDensity /self.Efficiency
    
    def powermass(self, Power):
        """
        :return: Mass of the battery [kg]
        """
        return Power/ self.PowerDensity / self.Depth_of_discharge /self.End_of_life


    def volume(self, Energy):
        """
        :param energy: Required total energy for the battery [kWh]
        :param vol_en_den: Volumetric energy density of the battery [kWh/l]
        :return: Volume of the battery [m^3]
        """
        return Energy /self.VolumeDensity * 0.001

    def price(self, Energy):
        """
        :param energy: Required total energy for the battery [kWh]
        :param cost: Cost per Wh of the battery [US$/kWh]
        :return: Approx cost of the battery [US$]
        """
        return Energy *self.CostDensity

if __name__ == "__main__":
    bat = Battery()
    print