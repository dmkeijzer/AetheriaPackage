# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""
from dataclasses import dataclass

@dataclass
class Battery:
    """This class is to estimate the parameters of a battery"""

    """ 
    :param: Energy Density of the battery [kWh/kg]
    :param:: Volume Density [kWh/l]
    :param: Power Density of the battery [kW/kg]
    :param: CostDensity: Cost per Wh of the battery [US$/kWh]"""
    EnergyDensity : float = None
    Energy : float = None
    VolumeDensity : float = None
    PowerDensity : float  = None
    CostDensity : float = None
    Efficiency : float = None
    DOD : float = None
    ChargingEfficiency : float = None


    def energymass(self):
        """
        :return: Mass of the battery [kg]
        """
        return self.Energy/ self.EnergyDensity


    def volume(self):
        """
        :param energy: Required total energy for the battery [kWh]
        :param vol_en_den: Volumetric energy density of the battery [kWh/l]
        :return: Volume of the battery [m^3]
        """
        return self.Energy /self.VolumeDensity * 0.001

    def price(self):
        """
        :param energy: Required total energy for the battery [kWh]
        :param cost: Cost per Wh of the battery [US$/kWh]
        :return: Approx cost of the battery [US$]
        """
        return self.Energy *self.Cost

