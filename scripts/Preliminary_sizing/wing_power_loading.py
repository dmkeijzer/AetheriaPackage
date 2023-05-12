# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import matplotlib.pyplot as plt

from modules.preliminary_sizing import *


WS_range = np.arange(1,4000,1)
ylim = [0,0.15]


def plot_wing_power_loading_graphs(eff, StotS, diskloading, name,WS_range,i):
    #Check if it"s lilium or not to define the variable that will say to vertical_flight what formula to use.
    if name =="L1":
        ducted_bool=True
    else:
        ducted_bool=False
    #Set up plot
    plt.figure(i)#make sure each plot has its own value
    
    #CALCULATE ALL THE VALUES FOR THE GRAPHS
    TW_range = powerloading_thrustloading(WS_range,rho0,data['roc'],data['stots'])
    CLIMBRATE = powerloading_climbrate(data['eff'], data['roc'], WS_range,rho_cruise,data['cd0'],data['e'],data['A'])
    TURN_VCRUISE = powerloading_turningloadfactor(rho_cruise,data['v_cruise'],WS_range,data['eff'],datap['A'],data['e'],data['loadfactor'],data['cd0'])
    TURN_VMAX = powerloading_turningloadfactor(rho_cruise,data['v_max'],WS_range,data['eff'],data['A'],data['e'],data['loadfactor'],data['cd0'])
    VERTICALFLIGHT = powerloading_verticalflight(TW_range,data['diskloading'],rho0,data['eff'],data['ducted_bool'])
    STALLSPEED = wingloading_stall(Aero.CLmax,Performance.V_stall, rho0)
    CLIMBGRADIENT = powerloading_climbgradient(Aero.e,Wing.A,Aero.CD0,WS_range,rho0,eff,Performance.G)
    
    #PLOT ALL THE LINES
    plt.plot(WS_range,CLIMBRATE,label="Climbrate")
    plt.plot(WS_range,TURN_VCRUISE,label="Turnload@cruise speed")
    plt.plot(WS_range,TURN_VMAX,label="Turnload@max speed")
    plt.plot(WS_range,VERTICALFLIGHT,label="Vertical flight/TO")
    plt.plot(WS_range,CLIMBGRADIENT,label="Climb gradient")
    plt.vlines(STALLSPEED,ymin=ylim[0],ymax=ylim[1],label="Stall speed:CLmax=1.5")

    #DETERMINE LOWEST
    lowest_area_y_novf = []
    lowest_area_y = []
    lowest_area_x = np.arange(0,int(STALLSPEED),1)
    for i in lowest_area_x:
        lowest_area_y.append(min(CLIMBRATE[i],TURN_VCRUISE[i],TURN_VMAX[i],CLIMBGRADIENT[i],VERTICALFLIGHT[i]))
        lowest_area_y_novf.append(min(CLIMBRATE[i],TURN_VCRUISE[i],TURN_VMAX[i],CLIMBGRADIENT[i]))
        
    #DETERMINE LIMITING FACTORS
    WS_max = STALLSPEED
    TW_max = powerloading_thrustloading(WS_max,rho0,Performance.ROC,StotS)
    WP_cruise = lowest_area_y_novf[-1]
    WP_hover = lowest_area_y[-1]
    
    #FILL AREAS IN GRAPH
    plt.fill_between(lowest_area_x,lowest_area_y, color = "Green", alpha = 0.3)
    plt.fill_between(lowest_area_x,lowest_area_y_novf, color = "Green", alpha = 0.2)
    
    
    #PLOT LIMITING DESIGN POINTS AND WRITE THE VALUES
    if lowest_area_y_novf[-1] == lowest_area_y[-1]:
        plt.plot(STALLSPEED,lowest_area_y_novf[-1],marker = "o",color = "green")
        plt.annotate((str(int(STALLSPEED))+", "+str(round(WP_cruise,8))),(STALLSPEED,WP_cruise+0.005))
    else: 
        plt.plot(STALLSPEED,lowest_area_y_novf[-1],marker = "o",color = "green")
        plt.annotate((str(int(STALLSPEED))+", "+str(round(WP_cruise,8))),(STALLSPEED,WP_cruise+0.005))
        plt.plot(STALLSPEED,lowest_area_y[-1],marker = "o",color = "red")
        plt.annotate((str(int(STALLSPEED))+", "+str(round(WP_hover,8))),(STALLSPEED,WP_hover-0.01))
    
    #GRAPH MAKE-UP
    plt.legend()
    plt.xlabel("Wingloading W/S")
    plt.ylabel("Powerloading W/P")
    plt.xlim([WS_range[100],WS_range[-1]])
    plt.ylim(ylim)
    output_directory = str(list(pl.Path(__file__).parents)[2])+"\\output\\wing_power_loading_diagrams\\"
    plt.savefig(output_directory+str(name)+".png")
    
    #PRINT VALUES
    print(name)
    print("WS = ",str(STALLSPEED))
    print("WP = ",str(round(lowest_area_y[-1],8)))
    print("WP_noverticalflight = ",str(round(lowest_area_y_novf[-1],8)))
    print("TW = ", str(round(TW_max,8))),'\n'
    return WS_max,TW_max,WP_cruise,WP_hover
    

#FIRST EASY PRELIMINARY DESIGN
dict_directory = str(list(pl.Path(__file__).parents)[2])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration
for i in range(len(dict_name)):                                             #iterate over each value
    with open(dict_directory+"\\"+dict_name[i], "r") as jsonFile:
        data = json.loads(jsonFile.read())
    data["WS"],data["TW"],data["WP_cruise"],data["WP_hover"] = plot_wing_power_loading_graphs(data["eff"], data["StotS"], data["diskloading"], data["name"],WS_range,i)
    
    with open(dict_directory+"\\"+dict_name[i], "w") as jsonFile:
        json.dump(data, jsonFile,indent=2)