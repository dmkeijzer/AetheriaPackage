#import statements
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib as pl
import pandas as pd

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from Modules.powersizing import BatterySizing
from Modules.powersizing import FuellCellSizing
from Modules.powersizing import HydrogenTankSizing
from Modules.powersizing import MissionRequirements
from Modules.powersizing import PropulsionSystem



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
print(sensitivityPowerFCBat)

#------------------reset batteries and fuel cells
#batteries
Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.4, vol_en_den=1, sp_pow_den=7,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)


FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
FuelTank700Bbar = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)
FueltankLiquid = HydrogenTankSizing(sp_en_den=33.3* 0.25,vol_en_den=8/3.6,cost= 0)

#--------------Second Sensitivity analysis
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
spenergyH2 = convert_array_to_strings(EnergyDensitiesTank/33.3)
sensitivityPowerFCEnergyTank = pd.DataFrame(sensitivityPowerFCEnergyTank,index= spenergyH2,columns=sppowerfc)
print(sensitivityPowerFCEnergyTank)

