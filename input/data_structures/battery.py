# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""
from dataclasses import dataclass
import json 
import GeneralConstants as constants
import sys 
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[0]))



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
    """
    #power & energy
    Energy : float = None
    Power : float = None

    #densities
    EnergyDensity : float = None
    PowerDensity : float  = None
    VolumeDensity : float = None
    CostDensity : float = None

    #extra parameters
    Efficiency : float = None
    Depth_of_discharge : float = None
    ChargingEfficiency : float = None

    def load(self):
        #jsonfilename = "aetheria_constants.json"
        self.EnergyDensity = constants.EnergyDensityBattery
        self.PowerDensity = constants.PowerDensityBattery
        self.VolumeDensity = constants.VolumeDensityBattery

        self.Efficiency = constants.dischargeEfficiency
        self.Depth_of_discharge = constants.DOD
        self.ChargingEfficiency = constants.ChargingEfficiency
            

    def energymass(self):
        """
        :return: Mass of the battery [kg]
        """
        return self.Energy/ self.EnergyDensity
    
    def powermass(self):
        """
        :return: Mass of the battery [kg]
        """
        return self.Power/ self.PowerDensity


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

if __name__ == "__main__":
    bat = Battery()
    bat.load()
    print(bat.ChargingEfficiency)