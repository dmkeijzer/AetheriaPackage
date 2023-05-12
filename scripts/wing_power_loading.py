# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from preliminary_sizing import *


WS_range = np.arange(0,4000,1)
ylim = [0,0.25]


def plot_wing_power_loading_graphs(eff, StotS, diskloading, no_engines,name,WS_range,i):
    #Check if it's lilium or not to define the variable that will say to vertical_flight what formula to use.
    if name =='Lilium-like':
        ducted_bool=True
    else:
        ducted_bool=False
    #Set up plot
    plt.figure(i)
    
    #DETERMINE VALUES
    TW_range = powerloading_thrustloading(no_engines,WS_range,rho0,ROC,StotS)
    CLIMBRATE = powerloading_climbrate(eff, ROC, WS_range,rho_cruise,CD0,e,A)
    TURN_VCRUISE = powerloading_turningloadfactor(rho_cruise,V_cruise,WS_range,eff,A,e,loadfactor,CD0)
    TURN_VMAX = powerloading_turningloadfactor(rho_cruise,V_max,WS_range,eff,A,e,loadfactor,CD0)
    VERTICALFLIGHT = powerloading_verticalflight(TW_range,diskloading,rho0,eff,ducted_bool)
    STALLSPEED = wingloading_stall(CLmax, V_stall, rho0)
    
    plt.plot(WS_range,CLIMBRATE,label="Climbrate")
    plt.plot(WS_range,TURN_VCRUISE,label='Turnload@cruise speed')
    plt.plot(WS_range,TURN_VMAX,label='Turnload@max speed')
    plt.plot(WS_range,VERTICALFLIGHT,label='Vertical flight/TO')
    plt.vlines(STALLSPEED,ymin=ylim[0],ymax=ylim[1],label='Stall speed:CLmax=1.5')

    
    plt.legend()
    #plt.grid()
    plt.xlabel('Wingloading W/S')
    plt.ylabel('Powerloading W/P')
    plt.xlim([WS_range[100],WS_range[-1]])
    plt.ylim(ylim)
    plt.savefig('input_output/wing_power_loading_diagrams/'+str(name))

#FIRST EASY PRELIMINARY DESIGN
name = ['Joby-like',  'Lilium-like','Wigeon-like']
eff =  [eff_prop,eff_ductedfans,eff_prop]
StotS =[ 1.6,         1.6,             1.2]
diskloading = [50,    1200,           300 ]
no_engines = [6,       36,              12]
for i in range(3):
    plot_wing_power_loading_graphs(eff[i], StotS[i], diskloading[i], no_engines[i],name[i],WS_range,i)