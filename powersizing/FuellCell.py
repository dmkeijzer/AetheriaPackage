# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""


class HydrogenTank:
    """This class is to estimate the parameters of a Hydrogen tank"""
    def __init__(self, sp_en_den, vol_en_den, tot_energy, cost):
        self.sp_en_den = sp_en_den
        self.energy = tot_energy
        self.vol_en_den = vol_en_den
        self.cost = cost

    def mass(self):
        """
        :param energy: Required total energy for the tank [Wh]
        :param sp_en_den: Specific energy density of the tank[Wh/kg]
        :return: Mass of the battery
        """
        return self.energy / self.sp_en_den

    def volume(self):
        """
        :param energy: Required total energy for the battery [Wh]
        :param vol_en_den: Volumetric energy density of the battery [Wh/l]
        :return: Volume of the battery [m^3]
        """
        return self.energy/self.vol_en_den * 0.001

    def price(self):
        """
        :param energy: Required total energy for the battery [Wh]
        :param cost: Cost per Wh of the battery [US$/Wh]
        :return: Approx cost of the battery [US$]
        """
        return self.energy*self.cost

