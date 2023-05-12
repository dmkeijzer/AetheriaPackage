#import statements
import numpy as np
import matplotlib.pyplot as plt
from battery import BatterySizing
from fuellCell import FuellCellSizing
from hydrogenTank import HydrogenTankSizing
from energypowerrequirement import MissionRequirements
from powersystem import PropulsionSystem

#plotfunction
def plotAll(echo, variable,variableUnit):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(echo, np.array(variable[0]).reshape(len(echo)))
    axs[0, 0].set_title('Total ' + variableUnit )
    axs[0, 1].plot(echo, np.array(variable[1]).reshape(len(echo)))
    axs[0, 1].set_title('Tank + h2 ' + variableUnit)
    axs[1, 0].plot(echo, np.array(variable[2]).reshape(len(echo)))
    axs[1, 0].set_title('Fuell Cell ' + variableUnit )
    axs[1, 1].plot(echo, np.array(variable[3]).reshape(len(echo)))
    axs[1, 1].set_title('Battery ' + variableUnit )
    
#-----------------------inputs-----------------
plotting = True
echo = np.arange(0,1.5,0.1)
DOD = 0.8
ChargingEfficiency = 0.7

#batteries
Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2.4,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.4, vol_en_den=1, sp_pow_den=10,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)

#fuelcell input
VolumeDensityFuellCell = 3.25 #kW /l
PowerDensityFuellCell = 125 / 42 #kW/kg
effiencyFuellCell = 0.6

#Tank input
VolumeDensityTank = 0.5 #kg/l
EnergyDensityTank = 1.85 # kWh/kg

#input Flight performance params
totalEnergy = 500 #kWh
cruisePower = 80*1.5 #kW
hoverPower = 1400 #kW

#-----------------------Model-----------------
BatteryUsed = Lisulbat
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

if plotting:
    plt.show()