# -*- coding: utf-8 -*-
"""

@author: Wessel Albers
"""

import numpy as np
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from modules.powersizing.battery import BatterySizing
from modules.powersizing.hydrogenTank import HydrogenTankSizing
from modules.powersizing.fuellCell import FuellCellSizing
from modules.powersizing.energypowerrequirement import MissionRequirements

def heatloss(power_electric: float, efficiency: float ) -> float:
    """
    :param: power_electric[kW]:  power required of the 
    :param: efficiency
    :return: heat[kW]"""
    return power_electric * (1 - efficiency) / efficiency

def coolingmass(heat: float, heatedensity:float) -> float:
    """
    :param: heat[kW]:  heat that has to be cooled
    :param: heatdensity[kW/kg]: specific weight of the cooling system
    :return: mass[kg]: mass of the cooling system """
    return heat / heatedensity


def energy_cruise_mass(EnergyRequired: float , echo: float , Tank: HydrogenTankSizing, Battery: BatterySizing, FuellCell: FuellCellSizing) -> list[float]:
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
    Tankmass = Tank.mass(EnergyRequired * echo) / FuellCell.Efficiency
    Batterymass = Battery.energymass((1-echo)*EnergyRequired) / Battery.DOD /Battery.Efficiency

    #extra weights needed because battery recharching process has losses
    #AddedMassRecharging = EnergyRequired * (echo - 1 ) / Tank.EnergyDensity / Battery.ChargingEfficiency / FuellCell.Efficiency
    
    return  Tankmass , Batterymass


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
    
    FCmass = FuellCell.mass(echo * PowerRequired ) 
    Batterymass = PowerRequired * (1-echo) / Battery.PowerDensity
    for i in range(len(Batterymass)):
        Batterymass[i] = max(0,Batterymass[i])


    return FCmass, Batterymass

def hover_mass(PowerRequired: float ,MaxPowerFC: float, Battery: BatterySizing) -> float :

    """Battery sizing for hover conditions
        input:
            -PowerRequired [kW] : The power required during hover
            -MaxPowerFC [kW]: The maximum power the Fuell Cell can deliver
            -PowerDensityBattery [kW/kg]: The po
        output:
            -Batterymass
    """
    BatteryMass =(PowerRequired - MaxPowerFC) / Battery.PowerDensity
    return  BatteryMass

def hover_energy_mass(PowerRequired: float ,MaxPowerFC: float, Battery: BatterySizing, HoverTime:float) -> float:
    BatteryMass = (PowerRequired - MaxPowerFC) * HoverTime /3600 / Battery.EnergyDensity / Battery.Efficiency 
    return BatteryMass

class PropulsionSystem:

    def mass(echo: float , Mission: MissionRequirements, Battery: BatterySizing, FuellCell: FuellCellSizing, FuellTank: HydrogenTankSizing, coolingdensity:float = 0, hovertime: float = 90) -> list[float]: #,MaxPowerFC:float,PowerDensityFC: float , PowerDensityBattery: float, EnergyDensityTank: float  ) -> list[float]:
        """Calculate total mass of the propulsion system
        input: 
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            
        returns:
            -Totalmass [kg]
            -FCmass[kg]: Fuell Cell mass
            -Batterymass[kg]"""
            
        
        
        #Initial sizing for cruise phase
        Tankmass,  EnergyBatterymass = energy_cruise_mass(Mission.EnergyRequired, echo, FuellTank, Battery, FuellCell)
        FCmass, CruiseBatterymass = power_cruise_mass(Mission.CruisePower, echo,FuellCell, Battery)


        #initial sizing for hovering phase
        MaxPowerFuellCell = FuellCell.PowerDensity * FCmass
        HoverBatterymass = hover_mass(PowerRequired=Mission.HoverPower,MaxPowerFC= MaxPowerFuellCell,Battery= Battery)
        HoverEnergyBatterymass = hover_energy_mass(PowerRequired= Mission.HoverPower, MaxPowerFC= MaxPowerFuellCell,Battery= Battery,HoverTime= hovertime)

        #heaviest battery is needed for the total mass
        Batterymass = np.zeros(len(echo))


        #need to check which batterymass is limiting at each echo and hardcoded it because i did not trust np.maximum as it gave some weird results
        for i in range(len(echo)):
            Batterymass[i] = max([HoverBatterymass[i], 2* HoverEnergyBatterymass[i], CruiseBatterymass[i], EnergyBatterymass[i]])

        #calculating heat produced by the 
        """"  
        FCheat = heatloss(FCmass * FuellCell.PowerDensity, FuellCell.Efficiency)
        BatteryHeat = heatloss(Batterymass * Battery.PowerDensity, Battery.Efficiency)
        totalheat = FCheat + BatteryHeat"""
        masscooling = FCmass  #coolingmass(totalheat,coolingdensity)
        
        #returning total mass and all component masss
        Totalmass = Tankmass + FCmass + Batterymass + masscooling
        return  Totalmass, Tankmass, FCmass, Batterymass, masscooling

    def volume(echo:float, Battery: BatterySizing, FuellCell: FuellCellSizing, FuellTank: HydrogenTankSizing,Tankmass: float, FuellCellmass:float, Batterymass:float) -> tuple[float]:

        #calculating component mass
        TankVolume = Tankmass  * FuellTank.EnergyDensity / FuellTank.VolumeDensity * 0.001
        FuellCellVolume = FuellCell.PowerDensity / FuellCell.VolumeDensity * FuellCellmass * 0.001
        BatteryVolume = Battery.EnergyDensity / Battery.VolumeDensity * Batterymass *0.001

        TotalVolume = TankVolume + FuellCellVolume + BatteryVolume

        return TotalVolume , TankVolume, FuellCellVolume, BatteryVolume


def onlyFuelCellSizing(mission: MissionRequirements, tank: HydrogenTankSizing, fuellcell: FuellCellSizing) -> tuple[float]:
    tankmass = tank.mass(mission.EnergyRequired) / fuellcell.Efficiency
    fuellcellmass = fuellcell.mass(mission.HoverPower) 
    tankvolume = tank.volume(mission.EnergyRequired) / fuellcell.Efficiency
    FuellCellVolume = fuellcell.PowerDensity / fuellcell.VolumeDensity * fuellcellmass * 0.001

    return tankmass, fuellcellmass, tankvolume, FuellCellVolume
