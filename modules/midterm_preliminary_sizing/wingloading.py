# -*- coding: utf-8 -*-
import numpy as np

def wingloading_stall(CLmax,V_stall,rho):
    return CLmax*0.5*rho*V_stall*V_stall