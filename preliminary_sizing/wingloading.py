# -*- coding: utf-8 -*-
import numpy as np
#Constants
g = 9.08665 #m/s^2

h_transition = 20 #m
t_TO = 20
acc_v = 2*h_transition/(t_TO*t_TO)

def wingloading_stall(CLmax,V_stall,rho):
    WS = CLmax*0.5*rho*V_stall*V_stall