# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from preliminary_sizing import *


WS_range = np.arange(500,4000,1)
ylim = [0,0.25]

TW_range = powerloading_thrustloading(no_engines,WS_range,rho0,ROC,StotS)

plt.plot(WS_range,powerloading_climbrate(eff_prop, ROC, WS_range, rho300, CL, CD),label="Climbrate")
plt.plot(WS_range,powerloading_turningloadfactor(rho300,V_cruise,WS_range,eff_prop,A,e,loadfactor,CD),label='Turnload@cruise speed')
plt.plot(WS_range,powerloading_turningloadfactor(rho300,V_max,WS_range,eff_prop,A,e,loadfactor,CD),label='Turnload@max speed')
plt.plot(WS_range,powerloading_verticalflight_ducted(TW_range, diskloading, rho0, eff_ductedfans),label='Vertical flight/TO (ducted fans)')
plt.plot(WS_range,powerloading_verticalflight_open(TW_range, diskloading, rho0, eff_openprop),label='Vertical flight/TO (open propellers)')
plt.vlines(wingloading_stall(CLmax, V_stall, rho0),ymin=ylim[0],ymax=ylim[1],label='Stall speed')
plt.legend()
plt.grid()
plt.xlabel('Wingloading W/S')
plt.ylabel('Powerloading W/P')
#plt.ylim(ylim)
print(wingloading_stall(CLmax, V_stall, rho0))