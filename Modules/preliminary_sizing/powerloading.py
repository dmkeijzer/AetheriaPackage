# -*- coding: utf-8 -*-
import numpy as np

def powerloading_climbrate(eff, ROC, WS,rho,CD0,e,A):
    k = 1/(e*A*np.pi)
    CLCDratio = 3/(4*k)* np.sqrt(3*CD0/k)
    return (ROC+np.sqrt(2*WS/rho)*(1/CLCDratio))**(-1) * eff

def powerloading_turningloadfactor(rho,V,WS,eff,A,e,loadfactor,CD0):
    k = 1/(e*A*np.pi)
    n = loadfactor
    
    WP = (CD0*0.5*rho*V*V*V/WS + WS*n*n*k/(0.5*rho*V))**-1 *eff

    return WP

def powerloading_thrustloading(WS,rho,ROC,StotS):
    return 1.2*(1+(1/WS)*rho*ROC**2*StotS)

def powerloading_verticalflight(TW,diskloading,rho,eff,ducted_bool):
    if ducted_bool==True:
        return (0.5*TW*np.sqrt(diskloading/rho))**(-1)*eff
    else:
        return (TW*np.sqrt(diskloading/(2*rho)))**(-1)*eff
def powerloading_climbgradient(e,A,CD0,WS,rho,eff,G):
    CL = np.sqrt(np.pi*e*A*CD0)
    CD = 2*CD0
    WP = (np.sqrt(WS*2/(rho*CL))*(G+CD/CL))**(-1) * eff
    return WP