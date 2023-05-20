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


echo = np.arange(0,1.1,0.01)
DOD = 0.8
ChargingEfficiency = 1
N_loops = 30



#fuelcell input
VolumeDensityFuellCell = 3.25 #kW /l
PowerDensityFuellCell = 3 #kW/kg
effiencyFuellCell = 0.55

#Tank input
VolumeDensityTank = 0.8 #kWh/l
EnergyDensityTank = 1.85 # kWh/kg

#input Flight performance params
J1Mission = MissionRequirements(EnergyRequired= 270,CruisePower=170, HoverPower= 360)
#-----------------------Model-----------------

#batteries
Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)
Solidstatebat = BatterySizing(sp_en_den= 0.4, vol_en_den=1, sp_pow_den=7,cost =82.2, charging_efficiency= ChargingEfficiency, depth_of_discharge= DOD, discharge_effiency=0.95)




BatteryUsed = Solidstatebat
FirstFC = FuellCellSizing(PowerDensityFuellCell,VolumeDensityFuellCell,effiencyFuellCell, 0)
FuelTank = HydrogenTankSizing(EnergyDensityTank,VolumeDensityTank,0)

PowersdensitiesFC = np.arange(2.5,3.6,0.1)
PowerdensitiesSolidState = np.arange(5,10.5,0.5)

powerdensityFCm, Powerdensitiesbatm = np.meshgrid(PowersdensitiesFC,PowerdensitiesSolidState)

sensitivity = np.zeros(powerdensityFCm.shape)

for i in range(powerdensityFCm.shape[0]):
    for j, powerdensities in enumerate(zip(powerdensityFCm[i],Powerdensitiesbatm[i])):
        powerdensityFC, powerdensitybat = powerdensities
        FirstFC.PowerDensity = powerdensityFC
        BatteryUsed.PowerDensity = powerdensitybat
        label = "SPFC" + str(round(powerdensityFC,1)) + "SPBat" + str(round(powerdensitybat,1))

        Mass = PropulsionSystem.mass(np.copy(echo),
                                                                    Mission= J1Mission, 
                                                                    Battery = BatteryUsed, 
                                                                    FuellCell = FirstFC, 
                                                                    FuellTank= FuelTank)
        TotalMass, tankMass, FuelCellMass, BatteryMass, coolingmass = Mass
        sensitivity[i,j] = np.round(np.min(TotalMass),2)
        print(label, powerdensityFC, powerdensitybat, np.min(TotalMass))


def convert_array_to_strings(arr):
    str_list = []
    with np.nditer(arr, flags=['buffered'], op_flags=['readonly']) as it:
        for x in it:
            str_list.append(str(np.round(x,1)))
    return str_list

#converting to 
sppowerfc = convert_array_to_strings(PowersdensitiesFC)
sppowerbat = convert_array_to_strings(PowerdensitiesSolidState)
sensitivity = pd.DataFrame(sensitivity,index= sppowerbat,columns=sppowerfc)

print(sensitivity)

        

