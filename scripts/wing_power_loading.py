# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

from modules.preliminary_sizing import *


WS_range = np.arange(500,4000,1)
ylim = [0,0.25]


def plot_wing_power_loading_graphs(eff, StotS, diskloading, no_engines,name,WS_range,i):
    plt.figure(i)
    TW_range = powerloading_thrustloading(no_engines,WS_range,rho0,ROC,StotS)
    
    plt.plot(WS_range,powerloading_climbrate(eff, ROC, WS_range, rho_cruise, CL, CD),label="Climbrate")
    plt.plot(WS_range,powerloading_turningloadfactor(rho_cruise,V_cruise,WS_range,eff,A,e,loadfactor,CLmin,CDmin),label='Turnload@cruise speed')
    plt.plot(WS_range,powerloading_turningloadfactor(rho_cruise,V_max,WS_range,eff,A,e,loadfactor,CLmin,CDmin),label='Turnload@max speed')
    plt.plot(WS_range,powerloading_verticalflight_ducted(TW_range, diskloading, rho0, eff),label='Vertical flight/TO')
    #plt.plot(WS_range,powerloading_verticalflight_open(TW_range, diskloading, rho0, eff_openprop),label='Vertical flight/TO (open propellers)')
    plt.vlines(wingloading_stall(CLmax, V_stall, rho0),ymin=ylim[0],ymax=ylim[1],label='Stall speed')
    plt.legend()
    #plt.grid()
    plt.xlabel('Wingloading W/S')
    plt.ylabel('Powerloading W/P')
#    plt.ylim(ylim)
    plt.savefig(name)

#FIRST EASY PRELIMINARY DESIGN
name = ['Joby-like',  'Lilium-like','Wigeon-like']
eff =  [eff_openprop,eff_ductedfans,eff_openprop]
StotS =[ 1.6,         1.6,             1.2]
diskloading = [50,    1200,           300 ]
no_engines = [6,       36,              12]
for i in range(3):
    plot_wing_power_loading_graphs(eff[i], StotS[i], diskloading[i], no_engines[i],name[i],WS_range,i)