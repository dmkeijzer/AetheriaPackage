# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from preliminary_sizing import *

print(V_max)

WS_range = np.arange(0,1400,1)

#plt.plot(WS_range,powerloading_climbrate(eff_prop, ROC, WS_range, rho300, CL, CD),label="Climbrate")
plt.plot(WS_range,powerloading_turningloadfactor(rho300,V_cruise,WS_range,eff_prop,A,e,loadfactor,CD),label='turnload_cruise')
plt.plot(WS_range,powerloading_turningloadfactor(rho300,V_max,WS_range,eff_prop,A,e,loadfactor,CD),label='turnload_max')