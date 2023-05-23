#import statements
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib as pl
import json
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.powersizing import BatterySizing , FuellCellSizing,  HydrogenTankSizing, MissionRequirements
from modules.powersizing import PropulsionSystem, onlyFuelCellSizing
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
designs = ["J1"]#,"L1","W1"]
plotting = False
designing = False
echo = np.arange(0,1.2,0.01)
DOD = 0.8
ChargingEfficiency = 0.7

#batteries
Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.5, vol_en_den=1, sp_pow_den=10,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)


#fuelcell input
VolumeDensityFuellCell = 3.25 #kW /l
PowerDensityFuellCell = 3 #kW/kg
effiencyFuellCell = 0.55

#Tank input
VolumeDensityTank = 0.8 #kWh/l
EnergyDensityTank = 1.85 # kWh/kg



BatteryUsed =Solidstatebat
FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
FuelTank = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)


#-----------------------Model-----------------

if plotting: 
    plt.figure
#input Flight performance params
for design in designs:
    path_to_json = "input/" + str(design) + "_constants.json"
    with open(path_to_json, "r") as file:
        ac = json.load(file)
        Mission = MissionRequirements(EnergyRequired= ac["mission_energy"] /3.6 / 1e6, #transfer joules to kWh
                                    CruisePower=ac["power_cruise"] / 1000, # divide 1000 to get to kW
                                    HoverPower= ac["power_hover"] / 1000) # divide 1000 to get to kW

    Mass = PropulsionSystem.mass(np.copy(echo),
                                                                Mission= Mission, 
                                                                Battery = BatteryUsed, 
                                                                FuellCell = FirstFC, 
                                                                FuellTank= FuelTank)
    TotalMass, tankMass, FuelCellMass, BatteryMass, coolingmass = Mass
    index = np.where(TotalMass == np.min(TotalMass))
    print(design + ":",BatteryMass[index] * BatteryUsed.EnergyDensity)
    if plotting:
        plt.plot(echo, TotalMass, label= design + " design")
        plt.xlabel(r'$\nu$ [-]')
        plt.ylabel("Mass power system [kg]")

if plotting:
    plt.legend()
    plt.show()

#calculating Volume
Volumes = PropulsionSystem.volume(echo, 
                                Battery =  BatteryUsed,
                                FuellCell = FirstFC, 
                                FuellTank = FuelTank,
                                Tankmass = tankMass ,FuellCellmass= FuelCellMass, Batterymass = BatteryMass)




#calculations for only the option with only the fuel cell
OnlyH2Tank, OnlyH2FC, Onlyh2tankVolume, Onlyh2FCVolume = onlyFuelCellSizing(Mission, FuelTank, FirstFC)
OnlyH2mass = OnlyH2Tank + OnlyH2FC
OnlyH2Volume = Onlyh2FCVolume + Onlyh2tankVolume

index = np.where(TotalMass == np.min(TotalMass))
print(BatteryMass[index])

print(np.min(TotalMass))
if designing:

    plotAll(echo,Volumes ,"Volume [m^3]")
    plotAll(echo,Mass, "Mass [kg]")

    ''' 
    font = "large"
    OnlyH2mass = np.ones(len(echo)) * OnlyH2mass
    plt.figure()
    plt.plot(echo,TotalMass, label ="Li-ion battery")
    plt.plot(echo,SolidMass[0], label = "Solid State Battery")
    plt.plot(echo, OnlyH2mass, label = "FC sized for hover" )
    plt.ylabel("Power system mass [kg]", fontsize = font)
    plt.xlabel("Fraction cruise power provide by Fuel Cell", fontsize = font)
    plt.legend(fontsize = font)


    OnlyH2Volume = np.ones(len(echo)) * OnlyH2Volume
    plt.figure()
    plt.plot(echo,Volumes[0], label ="Li-ion battery")
    plt.plot(echo,solidVolumes[0], label = "Solid State Battery")
    plt.plot(echo, OnlyH2Volume, label = "FC sized for hover" )
    plt.ylabel("Power system Volume [m^3]", fontsize = font)
    plt.xlabel("Fraction cruise power provide by Fuel Cell", fontsize = font)
    plt.legend(fontsize = font)
    '''
    plt.show()