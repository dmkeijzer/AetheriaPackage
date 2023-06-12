#import statements
import numpy as np
import sys
import pathlib as pl
import os
from matplotlib import pyplot as plt

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import Battery, FuelCell, HydrogenTank
from input.data_structures.performanceparameters import PerformanceParameters
from modules.powersizing.powersystem import PropulsionSystem
from input.data_structures.fluid import Fluid
from input.data_structures.radiator import Radiator

#loading data
IonBlock = Battery(Efficiency= 0.9)
Pstack = FuelCell()
Tank = HydrogenTank()
Mission = PerformanceParameters()

Tank.load()
Mission.load()

#estimate power system mass
nu = np.arange(0,1,0.01)
Totalmass, Tankmass, FCmass, Batterymass= PropulsionSystem.mass(echo= np.copy(nu),
                             Mission= Mission,
                             Battery=IonBlock,
                             FuellCell= Pstack,
                             FuellTank= Tank )

index_min_mass = np.where(Totalmass == min(Totalmass))
NU = nu[index_min_mass][0]
powersystemmass = Totalmass[index_min_mass][0]
Batterymass = Batterymass[index_min_mass][0]
fuelcellmass = Pstack.mass

plt.plot(nu, Totalmass)
plt.show()



print(f"nu: {NU}")
print(f"mass: {powersystemmass} kg")
print(f"battery mass: {Batterymass} kg")
