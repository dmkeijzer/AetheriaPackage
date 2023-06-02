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

WingClass = Wing()
AeroClass = Aero()
PerformanceClass = PerformanceParameters()


WingClass.load()
AeroClass.load()
PerformanceClass.load()

b = WingClass.span
taper = WingClass.taper
S= WingClass.surface
Cd0= WingClass.cd0
c_r=WingClass.chord_root
CLa=WingClass.cL_alpha
V=PerformanceClass.cruise_velocity

###MY INPUTS:
step=0.01
roll_rate=60*np.pi/(180*3)
aileron_max=20*np.pi/180

###Create Inputs:
Cla=2*np.pi ###of the wing airfoil
ca_c_ratio=0.2

##########Aileron Sizing###########
####The outer side of the aileron starts at the tip to impart maximum roll with minimum surface.

def size_aileron(b,V,aileron_max,roll_rate,S,Cla,Cd0,c_r,taper,CLa,ca_c_ratio,step,S):  ##Aileron_max is the maximum allowable deflection
    y_outer=b/2
    y_inner=b/2-step
    Cl_da=1   #JUST FOR INITIALIZATION to make while condition true
    Cl_p=1    #JUST FOR INITIALIZATION to make while condition true
    Cl_da_Cl_p_ratio=0     #JUST FOR INITIALIZATION to make while condition true
    while Cl_da/Cl_p>Cl_da_Cl_p_ratio:
        Cl_da_Cl_p_ratio=-roll_rate/(2*V*aileron_max/b)
        Cl_p=-(Cla+Cd0)*c_r*b*(1+3*taper)/(24*S)
        Sa_S=2*ca_c_ratio*(y_outer-y_inner)*(c_r/b*(taper-1)*(y_outer+y_inner)+c_r)/S
        tau_a=-6.624*Sa_S**4+12.07*Sa_S**3-8.292*Sa_S**2+3.295*Sa_S+0.004942
        Cl_da=-CLa*tau_a*c_r/(S*b)*((y_inner**2/2+2/3*y_inner**3*(taper-1)/b)-((y_outer**2/2+2/3*y_outer**3*(taper-1)/b)))
        ###http://docsdrive.com/pdfs/medwelljournals/jeasci/2018/3458-3462.pdf
        y_inner=y_inner-step
        print(y_inner)
    return y_inner


