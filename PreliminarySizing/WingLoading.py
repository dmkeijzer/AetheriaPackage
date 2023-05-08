# -*- coding: utf-8 -*-
''' 
Wing sizing is mainly done the same as for normal fixed-wing(FW) aircraft
since they are not used during the take-off and landing phase but are used
during cruise. The stall speed is chosen as 30 m/s as ...
''' 
import numpy as np

def tw_to_pw(TW,V,eff_prop):
    return TW*V/eff_prop

WS_stall = 0.5*V_stall*rho0*CLMax

