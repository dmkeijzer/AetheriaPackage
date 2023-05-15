# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""



class BatterySizing:
    """This class is to estimate the parameters of a battery"""
    def __init__(self, sp_en_den, vol_en_den, sp_pow_den, cost,discharge_effiency,charging_efficiency,depth_of_discharge):
        """ 
        :param: sp_en_den: Energy Density of the battery [kWh/kg]
        :param: vol_en_den: Volumetric Density [kWh/l]
        :param: sp_power_den: Power Density of the battery [kW/kg]
        :param: CostDensity: Cost per Wh of the battery [US$/kWh]"""
        self.EnergyDensity = sp_en_den
        #self.Energy = tot_energy
        self.VolumeDensity = vol_en_den
        self.PowerDensity = sp_pow_den
        self.CostDensity = cost
        self.Efficiency = discharge_effiency
        self.DOD = depth_of_discharge
        self.ChargingEfficiency = charging_efficiency


    def energymass(self,Energy):
        """
        :param Energy: Required total energy for the battery [kWh]
        :param sp_en_den: Specific energy density of the battery [kWh/kg]
        :return: Mass of the battery [kg]
        """
        return Energy/ self.EnergyDensity


    def volume(self,Energy):
        """
        :param energy: Required total energy for the battery [kWh]
        :param vol_en_den: Volumetric energy density of the battery [kWh/l]
        :return: Volume of the battery [m^3]
        """
        return Energy /self.VolumeDensity * 0.001

    def price(self,Energy):
        """
        :param energy: Required total energy for the battery [kWh]
        :param cost: Cost per Wh of the battery [US$/kWh]
        :return: Approx cost of the battery [US$]
        """
        return Energy *self.Cost

