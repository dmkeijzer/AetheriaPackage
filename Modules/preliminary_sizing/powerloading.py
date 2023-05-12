# -*- coding: utf-8 -*-
import numpy as np

def powerloading_climbrate(eff_prop, ROC, WS,rho,CL,CD):
    return eff_prop*(ROC + (WS * 2 /rho)**(1/2)/(CL**(3/2)/CD))**(-1)

def powerloading_turningloadfactor(rho,V,WS,eff_prop,A,e,loadfactor,CLmin,CDmin):
    k=1/(e*A*np.pi)
    CD = CDmin + k * (loadfactor*WS/(0.5*rho*V**2)-CLmin)**2
    return (CD*rho*V**3)/(2*WS) * eff_prop

def powerloading_thrustloading(no_engines,WS,rho,ROC,StotS):
    return 1.2*(1+(1/WS)*rho*ROC**2*StotS)

def powerloading_verticalflight_ducted(TW,diskloading,rho,eff_ductedfans):
    return(0.5*TW*np.sqrt(diskloading/rho))**(-1)*eff_ductedfans

def powerloading_verticalflight_open(TW,diskloading,rho,eff_openprop):
    return (TW*np.sqrt(diskloading/(2*rho)))**(-1)*eff_openprop

    