# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""


class HydrogenTankSizing:
    """This class is to estimate the parameters of a Hydrogen tank"""
    def __init__(self, sp_en_den, vol_en_den, cost):
        self.EnergyDensity = sp_en_den #[kWh/kg]
        self.VolumeDensity= vol_en_den #[kg/l]
        self.cost = cost

    def mass(self,energy):
        """
        :param energy: Required total energy for the tank [Wh]
        :param sp_en_den: Specific energy density of the tank + hydrogen in it[Wh/kg]
        :return: Mass of the battery
        """
        return energy / self.EnergyDensity

    def volume(self,energy):
        """
        :param energy: Required total energy from the hydrogen tank [Wh]
        :param vol_en_den: Volumetric energy density of the hydrogen tank [Wh/l]
        :return: Volume of the battery [m^3]
        """
        return energy/self.VolumeDensity * 0.001

    def price(self,energy) :
        """
        :param energy: Required total energy for the battery [Wh]
        :param cost: Cost per Wh of the battery [US$/Wh]
        :return: Approx cost of the battery [US$]
        """
        return energy * self.cost

