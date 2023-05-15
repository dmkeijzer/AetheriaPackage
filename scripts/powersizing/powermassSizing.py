#import statements
import numpy as np
import matplotlib.pyplot as plt
import sys 
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
sys.path.append(os.path.join(list(pl.Path(__file__).parents)[2], "modules","powersizing"))

from modules.powersizing.battery import BatterySizing
from modules.powersizing.fuellCell import FuellCellSizing
from modules.powersizing.hydrogenTank import HydrogenTankSizing
from modules.powersizing.energypowerrequirement import MissionRequirements
from modules.powersizing.powersystem import PropulsionSystem, onlyFuelCellSizng

#plotfunction
def plotAll(echo, variable,variableUnit):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(echo, np.array(variable[0]).reshape(len(echo)))
    axs[0, 0].set_title('Total ' + variableUnit )
    axs[0, 0].grid()
    axs[0, 1].plot(echo, np.array(variable[1]).reshape(len(echo)))
    axs[0, 1].set_title('Tank + h2 ' + variableUnit)
    axs[0, 1].grid()
    axs[1, 0].plot(echo, np.array(variable[2]).reshape(len(echo)))
    axs[1, 0].set_title('Fuell Cell ' + variableUnit )
    axs[1, 0].grid()
    axs[1, 1].plot(echo, np.array(variable[3]).reshape(len(echo)))
    axs[1, 1].set_title('Battery ' + variableUnit )
    axs[1, 1].grid()
    
#-----------------------inputs-----------------
plotting = True
echo = np.arange(0,1.5,0.05)
DOD = 0.8
ChargingEfficiency = 0.7

#batteries
Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.4, vol_en_den=1, sp_pow_den=10,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
#HydrogenBat = BatterySizing(sp_en_den=1.85,vol_en_den=3.25,sp_pow_den=2.9,cost=0,discharge_effiency=0.6,charging_efficiency=1,depth_of_discharge=1)

#fuelcell input
VolumeDensityFuellCell = 3.25 #kW /l
PowerDensityFuellCell = 3.9 #kW/kg
effiencyFuellCell = 0.6

#Tank input
VolumeDensityTank = 0.5 #kg/l
EnergyDensityTank = 1.85 # kWh/kg

#input Flight performance params
totalEnergy = 340  #kWh
cruisePower = 80*1.5 #kW
hoverPower = 1900 #kW

#-----------------------Model-----------------
BatteryUsed = Liionbat
FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
FuelTank = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)
InitialMission = MissionRequirements(EnergyRequired= totalEnergy, CruisePower= cruisePower, HoverPower= hoverPower )


#calculating mass
Mass = PropulsionSystem.mass(np.copy(echo),
                                                            Mission= InitialMission, 
                                                            Battery = BatteryUsed, 
                                                            FuellCell = FirstFC, 
                                                            FuellTank= FuelTank)
TotalMass, TankMass, FuelCellMass, BatteryMass = Mass


#calculating Volume
Volumes = PropulsionSystem.volume(echo,
                                Battery =  BatteryUsed,
                                FuellCell = FirstFC, 
                                FuellTank = FuelTank,
                                Tankmass= TankMass,FuellCellmass= FuelCellMass, Batterymass = BatteryMass)

plotAll(echo,Volumes ,"Volume [m^3]")
plotAll(echo,Mass, "Mass [kg]")

#calculations for only the option with only the fuel cell
OnlyH2Tank, OnlyH2FC = onlyFuelCellSizng(InitialMission, FuelTank, FirstFC)

#print(OnlyH2FC, OnlyH2Tank)
print(OnlyH2Tank + OnlyH2FC)

print(np.min(TotalMass))
if plotting:

    plt.show()