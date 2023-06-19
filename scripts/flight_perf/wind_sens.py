import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.flight_perf.EnergyPower import *
from modules.flight_perf.transition_model import *
import input.data_structures.GeneralConstants as const
from  ISA_tool import ISA
from input.data_structures import *
WingClass = Wing()
AeroClass = Aero()
PerformanceClass = PerformanceParameters()
EngineClass = Engine()

WingClass.load()
AeroClass.load()
PerformanceClass.load()
EngineClass.load()
    


import matplotlib.pyplot as plt


wind = np.arange(-10,11,1)

e_lst = []
r_lst = []
power1 = powercruise(PerformanceClass.MTOM, const.g0, 83.333, AeroClass.ld_cruise, 0.9)
for i in wind:
    v = 83.3333
    power2 = powercruise(PerformanceClass.MTOM, const.g0, v-i, AeroClass.ld_cruise, 0.9)
    frac = 300/power1
    time = 300/(v-i)
    energy = power2*time
    rang = frac*power2 + 100
    r_lst.append(rang)
    e_lst.append(energy)

plt.plot(wind, r_lst)
plt.ylabel('Range [km]')
plt.xlabel("Wind speed [m/s]")

# plt.ylim(0,500)
plt.grid()
plt.show()