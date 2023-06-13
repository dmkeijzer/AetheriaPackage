# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""

import numpy as np
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))


from input.data_structures import Battery, FuelCell , HydrogenTank
from input.data_structures import PerformanceParameters


def energy_cruise_mass(EnergyRequired: float , echo: float , Tank: HydrogenTank, Battery: Battery, FuellCell: FuelCell) -> list[float]:
    """Calculate the mass of the hydrogen tank + the hydrogen itself
        input:
            -EnergyRequired [Wh] : The total Energy required for the mission
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            -EnergyDensityHydrogen [Wh/kg]: The energy density of the tank + hydrogen in it
            -EnergyDensityBattery [Wh/kg]: The enegery density of the battery
            
        output:
            -Tankmass [kg]: The total mass of fuel tanks + the hydrogen in it
            -Batterymass [kg]: the battery mass 
    """
    
    
    #calculating energy required for the fuell cell and the battery
    Tankmass = Tank.mass(EnergyRequired * echo) / FuellCell.efficiency
    Batterymass = Battery.energymass((1-echo)*EnergyRequired) 
    
    return  Tankmass , Batterymass


def power_cruise_mass(PowerRequired: float, echo: float,  FuellCell:FuelCell, Battery:Battery ) -> list[float] :
    """Fuell Cell sizing and battery sizing for cruise conditions
        input:
            -PowerRequired [kW] : The power required during cruise
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            -PowerDensityFC [kW/kg]: The power density of the fuell cell
            -PowerDensityBattery [kW/kg]: The power battery of the battery
        return:
            -FCmass [kg]: Fuell cell mass
            -Batterymass[kg]: Battery mass
    """
    
    FCmass = FuellCell.mass
    Batterymass = PowerRequired * (1-echo) / Battery.PowerDensity / Battery.End_of_life
    
    for i in range(len(Batterymass)):
        Batterymass[i] = max(0,Batterymass[i])


    return FCmass, Batterymass

def hover_mass(PowerRequired: float ,MaxPowerFC: float, Battery: Battery) -> float :

    """Battery sizing for hover conditions
        input:
            -PowerRequired [kW] : The power required during hover
            -MaxPowerFC [kW]: The maximum power the Fuell Cell can deliver
            -PowerDensityBattery [kW/kg]: The po
        output:
            -Batterymass
    """
    BatteryMass = (PowerRequired - 0.9*MaxPowerFC) / Battery.PowerDensity /Battery.End_of_life 
    return  BatteryMass

def hover_energy_mass(PowerRequired: float ,MaxPowerFC: float, Battery: Battery, HoverTime:float) -> float:
    BatteryMass = (PowerRequired - MaxPowerFC) * HoverTime /3600 / Battery.EnergyDensity / Battery.Efficiency 
    return BatteryMass

class PropulsionSystem:

    def mass(echo: float , Mission: PerformanceParameters , Battery: Battery, FuellCell: FuelCell, FuellTank: HydrogenTank, hovertime: float = 60) -> list[float]: #,MaxPowerFC:float,PowerDensityFC: float , PowerDensityBattery: float, EnergyDensityTank: float  ) -> list[float]:
        """Calculate total mass of the propulsion system
        input: 
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            
        returns:
            -Totalmass [kg]
            -FCmass[kg]: Fuell Cell mass
            -Batterymass[kg]"""
            
        
        
        #Initial sizing for cruise phase
        Tankmass,  EnergyBatterymass = energy_cruise_mass(Mission.energyRequired / 3.6e6, echo, FuellTank, Battery, FuellCell) #convert to get to Wh
        FCmass, CruiseBatterymass = power_cruise_mass(Mission.cruisePower / 1e3, echo,FuellCell, Battery)
        #initial sizing for hovering phase
        HoverBatterymass = hover_mass(PowerRequired=Mission.hoverPower / 1e3 ,MaxPowerFC= FuellCell.maxpower,Battery= Battery)
        HoverEnergyBatterymass = hover_energy_mass(PowerRequired= Mission.hoverPower /1e3, MaxPowerFC= FuellCell.maxpower ,Battery= Battery,HoverTime= hovertime)

        #heaviest battery is needed for the total mass
        Batterymass = np.zeros(len(echo))
        #need to check which batterymass is limiting at each echo and hardcoded it because i did not trust np.maximum as it gave some weird results
        for i in range(len(echo)):
            Batterymass[i] = max([HoverBatterymass, 2* HoverEnergyBatterymass, CruiseBatterymass[i], EnergyBatterymass[i]])

        #returning total mass and all component masss 
        Totalmass = Tankmass + FCmass + Batterymass #estimation from pim de boer that power density of fuel cell halves for system power density
        return  Totalmass, Tankmass, FCmass, Batterymass

    def volume(echo:float, Battery: Battery, FuellCell: FuelCell, FuellTank: HydrogenTank,Tankmass: float, Batterymass:float) -> tuple[float]:

        #calculating component mass
        TankVolume = Tankmass  * FuellTank.energyDensity / FuellTank.volumeDensity * 0.001
        FuellCellVolume = FuellCell.volume
        BatteryVolume = Battery.EnergyDensity / Battery.VolumeDensity * Batterymass *0.001

        TotalVolume = TankVolume + FuellCellVolume + BatteryVolume

        return TotalVolume , TankVolume, FuellCellVolume, BatteryVolume


