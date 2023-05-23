#import statements
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib as pl
import pandas as pd

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from modules.powersizing import BatterySizing
from modules.powersizing import FuellCellSizing
from modules.powersizing import HydrogenTankSizing
from modules.powersizing import MissionRequirements
from modules.powersizing import PropulsionSystem
def color_cell(value, min_value, max_value):
    # Define the color scale
    colors = [
        '\\cellcolor{red!30}',    # Color for lowest values
        '\\cellcolor{yellow!30}', # Color for intermediate values
        '\\cellcolor{green!30}'   # Color for highest values
    ]
    
    # Calculate the relative value
    relative_value = (value - min_value) / (max_value - min_value)
    
    # Determine the color index based on the relative value
    color_index = min(int(relative_value * len(colors)), len(colors) - 1)
    
    # Return the color specification for the cell
    return colors[color_index]

def dataframe_to_latex(dataframe):
    # Calculate the minimum and maximum values in the DataFrame
    min_value = dataframe.min().min()
    max_value = dataframe.max().max()
    
    # Create a copy of the DataFrame with formatted color cells
    colored_dataframe = dataframe.applymap(lambda x: f'{color_cell(x, min_value, max_value)} {x}')
    
    # Convert the colored DataFrame to LaTeX table format
    latex_table = colored_dataframe.to_latex(index=False, escape=False)
    
    # Add the necessary LaTeX code for coloring the table cells
    latex_table = latex_table.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline')
    latex_table = latex_table.replace('\\begin{tabular}', '\\begin{tabular}{|c|c|c|}')  # Adjust the number of columns accordingly
    
    return latex_table


def convert_array_to_strings(arr):
    # Convert the array to a list of strings
    str_list = [str(np.round(x,3)) for x in arr]
    return str_list


echo = np.arange(0,1.1,0.01)
DOD = 0.9
ChargingEfficiency = 1
N_loops = 30



#fuelcell input
VolumeDensityFuellCell = 3.25 #kW /l
PowerDensityFuellCell = 3 #kW/kg
effiencyFuellCell = 0.55

#Tank input
VolumeDensityTank = 0.8 #kWh/l
EnergyDensityTank = 0.06 * 33.3 # kWh/kg

#input Flight performance params
J1Mission = MissionRequirements(EnergyRequired= 270,CruisePower=170, HoverPower= 360)
#-----------------------Model-----------------

#batteries
Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.4, vol_en_den=1, sp_pow_den=7,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)




BatteryUsed = Solidstatebat
FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
FuelTank700Bbar = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)
FueltankLiquid = HydrogenTankSizing(sp_en_den=33.3* 0.25,vol_en_den=8/3.6,cost= 0)


#------first sensitivity analysis power density fc vs power density solidstate battery
PowersdensitiesFC = np.arange(2.5,3.6,0.1)
PowerdensitiesSolidState = np.arange(5,10.5,0.5)
#PowerdensitiesLiion = np.arange(0.3,0.5,0.01)

powerdensityFCm, Powerdensitiesbatm = np.meshgrid(PowersdensitiesFC,PowerdensitiesSolidState)

sensitivityPowerFCBat = np.zeros(powerdensityFCm.shape)

for i in range(powerdensityFCm.shape[0]):
    for j, powerdensities in enumerate(zip(powerdensityFCm[i],Powerdensitiesbatm[i])):
        powerdensityFC, powerdensitybat = powerdensities
        FirstFC.PowerDensity = powerdensityFC
        BatteryUsed.PowerDensity = powerdensitybat

        Mass = PropulsionSystem.mass(np.copy(echo),
                                                                    Mission= J1Mission, 
                                                                    Battery = BatteryUsed, 
                                                                    FuellCell = FirstFC, 
                                                                    FuellTank= FuelTank700Bbar)
        TotalMass, tankMass, FuelCellMass, BatteryMass, coolingmass = Mass
        sensitivityPowerFCBat[i,j] = np.round(np.min(TotalMass),2)


#converting to data frame
sppowerfc = convert_array_to_strings(PowersdensitiesFC)
sppowerbat = convert_array_to_strings(PowerdensitiesSolidState)
sensitivityPowerFCBat = pd.DataFrame(sensitivityPowerFCBat,index= sppowerbat,columns=sppowerfc)
#print(sensitivityPowerFCBat)

#------------------reset batteries and fuel cells
#batteries
Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.4, vol_en_den=1, sp_pow_den=10,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)


FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
FuelTank700Bbar = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)
FueltankLiquid = HydrogenTankSizing(sp_en_den=33.3* 0.25,vol_en_den=8/3.6,cost= 0)

#--------------Second Sensitivity analysis powerdensity FC and EnergyDensity tank 
PowersdensitiesFC = np.arange(2.5,3.6,0.1)
EnergyDensitiesTank = np.arange(0.06,0.1,0.005) * 33.3

powerdensityFCm, EnergyDensitiesTankm = np.meshgrid(PowersdensitiesFC,EnergyDensitiesTank)

sensitivityPowerFCEnergyTank = np.zeros(powerdensityFCm.shape)
for i in range(powerdensityFCm.shape[0]):
    for j, powerdensities in enumerate(zip(powerdensityFCm[i],EnergyDensitiesTankm[i])):
        powerdensityFC, energydensityH2 = powerdensities
        FirstFC.PowerDensity = powerdensityFC
        FuelTank700Bbar.EnergyDensity = energydensityH2

        Mass = PropulsionSystem.mass(np.copy(echo),
                                                                    Mission= J1Mission, 
                                                                    Battery = Liionbat, 
                                                                    FuellCell = FirstFC, 
                                                                    FuellTank= FuelTank700Bbar)
        TotalMass, tankMass, FuelCellMass, BatteryMass, coolingmass = Mass
        sensitivityPowerFCEnergyTank[i,j] = np.round(np.min(TotalMass),2)

sppowerfc = convert_array_to_strings(PowersdensitiesFC)
spenergyH2 = convert_array_to_strings(EnergyDensitiesTank)
sensitivityPowerFCEnergyTank = pd.DataFrame(sensitivityPowerFCEnergyTank,index= spenergyH2,columns=sppowerfc)
#print(sensitivityPowerFCEnergyTank)

#------------------reset batteries and fuel cells
#batteries
Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.4, vol_en_den=1, sp_pow_den=7,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
BatteryUsed = Solidstatebat

FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
FuelTank700Bbar = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)
FueltankLiquid = HydrogenTankSizing(sp_en_den=33.3* 0.25,vol_en_den=8/3.6,cost= 0)

#--------------third Sensitivity analysis powerdensity bat and EnergyDensity tank 
PowerdensitiesSolidState = np.arange(5,10.5,0.5)
EnergyDensitiesTank = np.arange(0.06,0.1,0.005) * 33.3
powerdensityBatm, EnergyDensitiesTankm = np.meshgrid(PowerdensitiesSolidState,EnergyDensitiesTank)

sensitivityPowerBatEnergyTank = np.zeros(powerdensityBatm.shape)
for i in range(sensitivityPowerBatEnergyTank.shape[0]):
    for j, powerdensities in enumerate(zip(powerdensityBatm[i],EnergyDensitiesTankm[i])):
        powerdensitybat, energydensityH2 = powerdensities
        BatteryUsed.PowerDensity = powerdensitybat
        FuelTank700Bbar.EnergyDensity = energydensityH2

        Mass = PropulsionSystem.mass(np.copy(echo),
                                                                    Mission= J1Mission, 
                                                                    Battery = BatteryUsed, 
                                                                    FuellCell = FirstFC, 
                                                                    FuellTank= FuelTank700Bbar)
        TotalMass, tankMass, FuelCellMass, BatteryMass, coolingmass = Mass
        sensitivityPowerBatEnergyTank[i,j] = np.round(np.min(TotalMass),2)

sppowerbat = convert_array_to_strings(PowerdensitiesSolidState)
spenergyH2 = convert_array_to_strings(EnergyDensitiesTank/33.3)
sensitivityPowerBatEnergyTank = pd.DataFrame(sensitivityPowerBatEnergyTank,index= spenergyH2,columns=sppowerbat)
print(sensitivityPowerBatEnergyTank)

