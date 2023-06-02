import sys
import os
import numpy as np
import json
import pathlib as pl
import pandas as pd

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))
import matplotlib.pyplot as plt
from input.data_structures import *

HorTailClass = HorTail()
AeroClass = Aero()
PerformanceClass = PerformanceParameters()


HorTailClass.load()
AeroClass.load()
PerformanceClass.load()


CL_h=HorTailClass.cL_approach_h       ####Comes from the most negative tail obtainable from ADSEE3
CL_a_h=HorTailClass.cL_alpha_h
b_h=HorTailClass.b_h              ####THIS VALUE IS NONE RIGHT NOW but the parameter exists. IT NEEDS TO BE ADDED


#####MY INPUTS####
step=0.01
elevator_min=-20*np.pi/180
b_h=8                          ####REMOVE THIS AFTER b_h is no longer 'none' value.

####Create inputs
aoa=15*np.pi/180 ###angle of attack at landing
y_min=0    ###the y position that horizontal tail starts from
c_elevator_c_tail_ratio=0.5  ###CHANGE LATER AFTER TALKING TO STRUCTURES 
taper_h=0.4
c_r_h=1
S_h=4


#########Elevator sizing############

###The inner size of the elevator starts at the inner most possible location
###This is sized to provide controllability during landing condition.
## CL_h is the most negative tail lift coefficient by ADSEE 3.
##aoa is the angle of attack at controllability critial low speed condition
##CL_a_h is the lift curve gradient of the rear horizontal tail
##elevator_min is the most negative elevator deflection angle
##c_elevator_c_tail_ratio is the ratio of the elevator chord to the horizontal tail chord.
##y_min is the y location that the horizontal tail starts from
##b_h is the horizontal tail span.     



def size_elevator(CL_h,aoa,CL_a_h,elevator_min,c_elevator_c_tail_ratio,y_min,b_h,taper_h,c_r_h):
    CL_h_de_required=(CL_h-CL_a_h*aoa)/elevator_min
    print(CL_h_de_required)

    b_e=0        ###JUST FOR INITIALIZATION
    CL_h_de=0    ###JUST FOR INITIALIZATION
    y_max=y_min  ###JUST FOR INITIALIZATION
    CL_h_de=0    ###JUST FOR INITIALIZATION
    while CL_h_de_required>CL_h_de:
        b_e=(y_max-y_min)*2    ##To account for the two sides of the wing (multiplied by 2)
        chord_middle_of_elevator=c_r_h/(b_h/2)*(taper_h-1)*(y_max+y_min)/2+c_r_h
        Se_S=c_elevator_c_tail_ratio*(y_max-y_min)*chord_middle_of_elevator/S_h
        print(y_max,Se_S)
        tau_e=-6.624*Se_S**4+12.07*Se_S**3-8.292*Se_S**2+3.295*Se_S+0.004942
        CL_h_de=tau_e*b_e/b_h*CL_a_h
        y_max=y_max+step

    if y_max>b_h/2:
        print('The required horizontal elevator span is larger than the horizontal span')        
    
    return y_max, CL_h_de 
        
        
    
        
    
    




    
    
