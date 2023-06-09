#import statements
import numpy as np
import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures import Battery, FuelCell, HydrogenTank
from input.data_structures.performanceparameters import PerformanceParameters
from input.data_structures.radiator import Radiator
from modules.powersizing.powersystem import PropulsionSystem

#loading data
IonBlock = Battery(Efficiency= 0.9)
Pstack = FuelCell()
Tank = HydrogenTank(energyDensity=1.8, volumeDensity=0.6, cost= 16)
Mission = PerformanceParameters()
Mission.load()

#estimate power system mass
nu = np.arange(0,1,0.005)
mass = PropulsionSystem.mass(echo= np.copy(nu),
                             Mission= Mission,
                             Battery=IonBlock,
                             FuellCell= Pstack,
                             FuellTank= Tank )[0]

print(mass)


