# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""

import numpy as np
from battery import BatterySizing
from hydrogenTank import HydrogenTankSizing
from fuellCell import FuellCellSizing
from energypowerrequirement import MissionRequirements


def energy_cruise_mass(EnergyRequired: float , echo: float , Tank: HydrogenTankSizing, Battery: BatterySizing, FuellCell: FuellCellSizing) -> list[float]:
    """ Hydrogen tank sizing and battery sizing for the energy requirement of cruise condition
        input:
            -EnergyRequired[kWh]: The energy required for the entire mission
            -echo [-]: The percentage of power deliverd by the fuel cell during cruise, if over 1 than the fuell cell charges the  battery
            -EnergyDensityTank [kWh/kg]: The energy density of the hydrogen tank with the hydrogen inside it 
            -EnergyDensityBattery [kWh/kg]: The energy density of the battery
            """
    """Calculate the mass of the hydrogen tank + the hydrogen itself
        input:
            -EnergyRequired [kWh] : The total Energy required for the mission
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            -EnergyDensityHydrogen [kWh/kg]: The energy density of the tank + hydrogen in it
            -EnergyDensityBattery [kWh/kg]: The enegery density of the battery
            
        output:
            -Tankmass [kg]: The total mass of fuel tanks + the hydrogen in it
            -Batterymass [kg]: the battery mass 
    """
    
    
    #calculating energy required for the fuell cell and the battery
    Tankmass = EnergyRequired / FuellCell.Efficiency * echo / Tank.EnergyDensity
    Batterymass = EnergyRequired /Battery.Efficiency* (1 - echo) / Battery.EnergyDensity / Battery.DOD

    AddedMassRecharging = EnergyRequired * (echo - 1 ) / Tank.EnergyDensity / Battery.ChargingEfficiency
    
    return  Tankmass + np.maximum(np.zeros(len(echo)), AddedMassRecharging) , Batterymass


def power_cruise_mass(PowerRequired: float, echo: float,  FuellCell:FuellCellSizing, Battery:BatterySizing ) -> list[float] :
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
    
    FCmass = PowerRequired * echo / FuellCell.PowerDensity
    
    
    Batterymass = PowerRequired * (1-echo) / Battery.EnergyDensity
    return FCmass, np.maximum(Batterymass , np.zeros( len(Batterymass) ) )

def hover_mass(PowerRequired: float ,MaxPowerFC: float, Battery: BatterySizing) -> float :

    """Battery sizing for hover conditions
        input:
            -PowerRequired [kW] : The power required during hover
            -MaxPowerFC [kW]: The maximum power the Fuell Cell can deliver
            -PowerDensityBattery [kW/kg]: The po
        output:
            -Batterymass
    """
    return  (PowerRequired - MaxPowerFC) / Battery.PowerDensity

def hover_energy_mass(PowerRequired: float ,MaxPowerFC: float, Battery: BatterySizing, HoverTime:float) -> float:
    BatteryMass = (PowerRequired - MaxPowerFC) * HoverTime /3600 / Battery.EnergyDensity / Battery.DOD 
    return BatteryMass

class PropulsionSystem:

    def mass(echo: float , Mission: MissionRequirements, Battery: BatterySizing, FuellCell: FuellCellSizing, FuellTank: HydrogenTankSizing) -> list[float]: #,MaxPowerFC:float,PowerDensityFC: float , PowerDensityBattery: float, EnergyDensityTank: float  ) -> list[float]:
        """Calculate total mass of the propulsion system
        input: 
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            
        output:
            -Totalmass [kg]
            -FCmass[kg]: Fuell Cell mass
            -Batterymass[kg]"""
            
        
        
        #Initial sizing for cruise phase
        Tankmass,  EnergyBatterymass = energy_cruise_mass(Mission.EnergyRequired, echo, FuellTank, Battery, FuellCell)
        FCmass, CruiseBatterymass = power_cruise_mass(Mission.CruisePower, echo,FuellCell, Battery)
        
        #initial sizing for hovering phase
        MaxPowerFuellCell = Battery.PowerDensity * FCmass
        HoverBatterymass = hover_mass(Mission.HoverPower, MaxPowerFuellCell, Battery)
        HoverEnergyBatterymass = hover_energy_mass(Mission.HoverPower, MaxPowerFuellCell, Battery, 90)
        

        #heaviest battery is needed for the total mass
        Batterymass = np.zeros(len(echo))
        
        #need to check which batterymass is limiting at each echo and hardcoded it because i did not trust np.maximum as it gave some weird results
        for i in range(len(echo)):
            Batterymass[i] = max(HoverBatterymass[i], HoverEnergyBatterymass[i], CruiseBatterymass[i], EnergyBatterymass[i])
        #returning total mass and all component masss
        Totalmass = Tankmass + FCmass + Batterymass
        return  Totalmass, Tankmass, FCmass, Batterymass

    def volume(echo:float, Battery: BatterySizing, FuellCell: FuellCellSizing, FuellTank: HydrogenTankSizing,  Tankmass: float , FuellCellmass:float, Batterymass:float) -> list[float]:

        #calculating component mass
        TankVolume = Tankmass / FuellTank.VolumeDensity * 0.001
        FuellCellVolume = FuellCell.PowerDensity / FuellCell.VolumeDensity * FuellCellmass * 0.001
        BatteryVolume = Battery.EnergyDensity / Battery.VolumeDensity * Batterymass *0.001

        TotalVolume = TankVolume + FuellCellVolume + BatteryVolume

        return TotalVolume , TankVolume, FuellCellVolume, BatteryVolume
