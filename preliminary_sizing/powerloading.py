# -*- coding: utf-8 -*-
import numpy as np

def powerloading_climbrate(eff_prop, ROC, WS,rho,CL,CD):
    return eff_prop*(ROC + np.sqrt(WS * 2 /rho)/(CL**(3/2)/CD))**(-1)

def powerloading_turningloadfactor(rho,V,WS,eff_prop,A,e,loadfactor,CD):
    return (CD*rho*V**3)/(2*WS) * eff_prop

def powerloading_thrustloading(no_engines,WS,rho,ROC,StotS):
    return 1.2*(1+1/(no_engines-1))*(1+(1/WS)*rho*ROC**2*StotS)

def powerloading_verticalflight_ducted(TW,diskloading,rho,eff_ductedfans):
    return(0.5*TW*np.sqrt(diskloading/rho))**(-1)*eff_ductedfans

def powerloading_verticalflight_open(TW,diskloading,rho,eff_openprop):
    return (TW*np.sqrt(diskloading/(2*rho)))**(-1)*eff_openprop

    