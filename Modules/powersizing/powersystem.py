# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:50:43 2023

@author: Wesse
"""

import numpy as np
from matplotlib import pyplot as plt

class HydrogenTank():
    
    def __init__(self, FuelWeight, TankWeight,H2_energyDensity):
        """Represent a tank of hydrogen
        inputs: 
            -FuelWeight [kg]: How much hydrogen can be stored in the tank 
            -TankWeight [kg]: The weight of the hydrogen tank itself"""
        self.hydrogen_weight = FuelWeight
        self.tank_weight = TankWeight
        self.EnergyDensity = FuelWeight*H2_energyDensity / (TankWeight + FuelWeight)
       
class FuelCell():
    
    def __init__(self, FuellCellWeight: float, PeakPower: float):
        """Represent a hydrogen fuel cell
        input:
            -Peakpower[kW]: Peak power of the fuel cell
            -"""
            
        self.maxPower = PeakPower
        self.weight = FuellCellWeight
        self.powerdensity = PeakPower/ FuellCellWeight

    
def hydrogen_tank_energydensity(tanks: list[HydrogenTank], energy_density_hydrogen: float)-> float:
    """Calculates an average energy density for the tanks
        input:
            -tanks: list of hydrogen tanks
            -energy_density_hydrogen[kWH/kg]: the energy of the hydrogen used for a certain operating pressure
        output:
            -energy_density[kWH/kg]: The enery density of the fuel tanks itself
            """
    energy_density = 0
    for tank in tanks:
        energy_density += (tank.hydrogen_weight * energy_density_hydrogen ) /  tank.tank_weight
    
    return energy_density / len(tanks)


def hydrogentank_weight(EnergyRequired: float , echo: float , EnergyDensityTank: float, EnergyDensityBattery: float) -> list[float]:
    """Calculate the weight of the hydrogen tank + the hydrogen itself
        input:
            -EnergyRequired [kWh] : The total Energy required for the mission
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            -EnergyDensityHydrogen [kWh/kg]: The energy density of the tank + hydrogen in it
            -EnergyDensityBattery [kWh/kg]: The enegery density of the battery
            
        output:
            -TankWeight [kg]: The total weight of fuel tanks + the hydrogen in it
            -BatteryWeight [kg]: the battery weight 
    """
    
    #filtering echo because battery weight cannot be negative
    if isinstance(echo,np.ndarray):
        echo[echo >= 1] = 1 
    elif echo > 1: echo = 1
    
    #calculating energy required for the fuell cell and the battery
    EnergyTank = EnergyRequired * echo
    EnergyBattery = EnergyRequired * (1-echo)
    
    #calculating weight
    TankWeight = EnergyTank * EnergyDensityTank
    BatteryWeight = EnergyBattery * EnergyDensityBattery
    
    
    return  TankWeight, BatteryWeight


def cruise_weight(PowerRequired: float, echo: float,  PowerDensityFC: float, PowerDensityBattery: float ) -> list[float] :
    """Fuell Cell sizing and battery sizing for cruise conditions
        input
            -PowerRequired [kW] : The power required during cruise
            -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
            -PowerDensityFC [kW/kg]: The power density of the fuell cell
            -PowerDensityBattery [kW/kg]: The power battery of the fuell cell
        output:
            -FCWeight [kg]: Fuell cell weight
            -BatteryWeight[kg]: Battery Weight
    """
    
    FCweight = PowerRequired * echo / PowerDensityFC
    
    #filtering echo because battery weight cannot be negative
    if isinstance(echo,np.ndarray):
        echo[echo >= 1] = 1 
    elif echo > 1: echo = 1
    BatteryWeight = PowerRequired * (1-echo) / PowerDensityBattery
    return FCweight, BatteryWeight

def hover_weight(PowerRequired: float ,MaxPowerFC: float, PowerDensityBattery: float ) -> float :

    """Battery sizing for hover conditions
        input:
            -PowerRequired [kW] : The power required during hover
            -MaxPowerFC [kW]: The maximum power the Fuell Cell can deliver
            -PowerDensityBattery [kW/kg]: The power battery of the fuell cell
        output:
            -BatteryWeight
    """
    return  (PowerRequired - MaxPowerFC) / PowerDensityBattery


def propulsionsystem_weight(echo: float , EnergyRequired:float, CruisePower: float, HoverPower: float) -> list[float]: #,MaxPowerFC:float,PowerDensityFC: float , PowerDensityBattery: float, EnergyDensityTank: float  ) -> list[float]:
    """Calculate total mass of the propulsion system
    input: 
        -echo [-]: The percentage of power deliverd by the fuel cell, if over 1 than the fuell cell charges the  battery
        -EnergyRequired [kWh] : The total Energy required for the mission
        -CruisePower [kW] : The power required during cruise
        -HoverPower [kW] : The power required during Hover
        
    output:
        -TotalWeight [kg]
        -FCWeight[kg]: Fuell Cell weight
        -BatteryWeight[kg]"""
        
    #input parameters 
    #MaxPowerFuellCell = 220 #kW
    PowerDensityBattery = 3 #kW/kg
    PowerDensityFuellCell = 125 / 42 #kW/kg
    EnergyDensityTank = 4 # kWh/kg
    EnergyDensityBattery = 0.3 #kWh/kg

    
    #Initial sizing of all componets
    TankWeight,  EnergyBatteryWeight = hydrogentank_weight(EnergyRequired, echo , EnergyDensityTank, EnergyDensityBattery)
    print(TankWeight + EnergyBatteryWeight)
    FCWeight, CruiseBatteryWeight = cruise_weight(CruisePower, echo, PowerDensityFuellCell, PowerDensityBattery)
    
    MaxPowerFuellCell = PowerDensityBattery * FCWeight
    HoverBatteryWeight = hover_weight(HoverPower, MaxPowerFuellCell, PowerDensityBattery)
    
    #heaviest battery is needed for the total weight
    BatteryWeight = np.maximum(HoverBatteryWeight,CruiseBatteryWeight, EnergyBatteryWeight)
    
    return TankWeight + FCWeight + BatteryWeight, TankWeight, FCWeight, BatteryWeight

if __name__ == "__main__":
    
    #inputs
    EnergyRequired = 540 #kWh
    CruisePower = 220 #kW
    HoverPower = 1230 #kW
    
    echo = np.arange(0,1.,0.05)
    
    TotalWeight, TankWeight, FuellCellWeight, BatteryWeight = propulsionsystem_weight(echo, EnergyRequired, CruisePower, HoverPower)
    
    System_Energy_Density = EnergyRequired/ TotalWeight
    
    
    plt.figure(0)
    plt.plot(echo*100,TotalWeight)
    plt.xlabel("% of fuell cell power during Cruise")
    plt.ylabel("Total Weight")
    
    ''' 
    plt.figure(1)
    plt.plot(echo*100,FuellCellWeight)
    plt.xlabel("% of fuell cell power during Cruise")
    plt.ylabel("FC Weight")
    
    plt.figure(2)
    plt.plot(echo*100,BatteryWeight)
    plt.xlabel("% of fuell cell power during Cruise")
    plt.ylabel("Battery Weight")
    
    plt.figure(3)
    plt.plot(echo*100,TankWeight*np.ones(len(echo)))
    plt.xlabel("% of fuell cell power during Cruise")
    plt.ylabel("Ta Weight")
    '''
    
    