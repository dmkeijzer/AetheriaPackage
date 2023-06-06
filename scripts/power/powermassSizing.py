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


def create_pie_chart(values, labels):
    plt.pie(values, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.show()  
#-----------------------inputs-----------------
designs = ["J1"]#,"L1","W1"]
plotting = False
designing = False
echo = np.arange(0,1.0001,0.01)
DOD = 0.8
ChargingEfficiency = 0.7

#batteries
Liionbat = BatterySizing(sp_en_den= 0.34, vol_en_den=0.85, sp_pow_den=3.8,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.90)
Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.5, vol_en_den=1, sp_pow_den=10,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)


#fuelcell input
VolumeDensityFuellCell = 3.25 #kW /l
PowerDensityFuellCell = 3 #kW/kg
effiencyFuellCell = 0.55

#Tank input
VolumeDensityTank = 0.6  #kWh/l
EnergyDensityTank = 1.8 # kWh/kg



BatteryUsed = Liionbat
FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
FuelTank = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)


#-----------------------Model-----------------
contingency = 1.1
if plotting: 
    plt.figure
#input Flight performance params
for design in designs:
    path_to_json = "input/" + str(design) + "_constants.json"
    print(type(path_to_json))
    with open(path_to_json) as file:
        ac = json.load(file)
    Mission = MissionRequirements(EnergyRequired= ac["mission_energy"] /3.6 / 1e6 * contingency, #transfer joules to kWh
                                    CruisePower=ac["power_cruise"] / 1000 * contingency, # divide 1000 to get to kW
                                    HoverPower= ac["power_hover"] / 1000 * contingency) # divide 1000 to get to kW

    Mass = PropulsionSystem.mass(np.copy(echo),
                                                                Mission= Mission, 
                                                                Battery = BatteryUsed, 
                                                                FuellCell = FirstFC, 
                                                                FuellTank= FuelTank)
    TotalMass, tankMass, FuelCellMass, BatteryMass, coolingmass = Mass

    Volumes = PropulsionSystem.volume(np.copy(echo),
                                      Battery= BatteryUsed,
                                      FuellCell= FirstFC,
                                      FuellTank= FuelTank,
                                      Tankmass=tankMass,FuellCellmass= FuelCellMass, Batterymass= BatteryMass)
    index = np.where(TotalMass == np.min(TotalMass))
    print(design + ":",FuelCellMass[index])
    print(Mission.EnergyRequired/TotalMass[index])
    
    if plotting:
        crosslines = echo[index] * np.ones(2)
        crossy = [np.max(TotalMass),np.min(BatteryMass)-50]
        plt.plot(echo, TotalMass, label= design + " design", linewidth = 4)


if plotting:

    # Example usage
    values = [FuelCellMass[index][0]*2,BatteryMass[index][0],tankMass[index][0]]
    labels = ['Fuel Cell System', 'Battery', 'Hydrogen Tank']

    font = 14
    plt.yticks(np.arange(0,1501,250),fontsize= 12)
    plt.xticks(fontsize = 12)
    plt.xlabel(r'Fuel cell cruise power fraction $\nu$ [-]',fontsize = font)
    plt.ylabel("Total mass power system " + r'$[kg] $',fontsize = font)
    plt.legend(fontsize = font)
    plt.grid()
    plt.show()
    create_pie_chart(values, labels)
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
totalVolume = Volumes[0]
print("Volume")
print(np.min(totalVolume[index]*1000))
print("Mass")
print(np.min(TotalMass))
print("echo")
print(echo[index])
print("batmas")
print(BatteryMass[index])
print("FC power")
print(FuelCellMass[index] * FirstFC.PowerDensity)
print("battery power ")
print(Mission.HoverPower-FuelCellMass[index] * FirstFC.PowerDensity)
print("Energy battery")
print(Mission.EnergyRequired * (1-echo[index]) /Liionbat.Efficiency)

print("tank mass")
print(tankMass[index])
print(Mission)




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