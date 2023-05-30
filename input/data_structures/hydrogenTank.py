# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""


class HydrogenTankSizing:
    """This class is to estimate the parameters of a Hydrogen tank"""
    def __init__(self, sp_en_den, vol_en_den, cost):
        """"
        :param: sp_en_den[kWh/kg]: Specific energy density 
        :param: vol_en_den[kWh/l]: specific volumetric density
        :param: cost[US$/kWh]: cost per kWh   """
        self.EnergyDensity = sp_en_den 
        self.VolumeDensity= vol_en_den #[kWh/l]
        self.cost = cost

    def mass(self,energy):
        """
        :param energy: Required total energy for the tank [kWh]
        :param sp_en_den: Specific energy density of the tank + hydrogen in it[kWh/kg]
        :return: Mass of the battery
        """
        return energy / self.EnergyDensity

    def volume(self,energy):
        """
        :param energy: Required total energy from the hydrogen tank [kWh]
        :param vol_en_den: Volumetric energy density of the hydrogen tank [kWh/l]
        :return: Volume of the battery [m^3]
        """
        return energy/self.VolumeDensity * 0.001

    def price(self,energy) :
        """
        :param energy: Required total energy for the battery [kWh]
        :param cost: Cost per kWh of the battery [US$/kWh]
        :return: Approx cost of the battery [US$]
        """
        return energy * self.cost

