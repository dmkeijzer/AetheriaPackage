
import numpy as np
import json
import pathlib as pl
import sys
import os
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

import input.GeneralConstants as const

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
    #return 1.2*(1+np.ones(np.shape(WS)))*1.3

def powerloading_verticalflight(MTOM,TW,A_tot,rho,eff,ducted_bool):
    W = MTOM *const.g0
    T = TW * W
    
    if ducted_bool==True:
        return (0.5*np.sqrt((T*T*T)/(W*W*rho*A_tot)))**(-1)*eff
    else:
        return (np.sqrt((T*T*T)/(2*W*W*rho*A_tot)))**(-1)*eff    
       
def powerloading_climbgradient(e,A,CD0,WS,rho,eff,G):
    CL = np.sqrt(np.pi*e*A*CD0)
    CD = 2*CD0
    WP = (np.sqrt(WS*2/(rho*CL))*(G+CD/CL))**(-1) * eff
    return WP

def wingloading_stall(CLmax,V_stall,rho):
    return CLmax*0.5*rho*V_stall*V_stall

def get_wing_power_loading(perf_par, wing, engine, aero, cont_factor=1.1):
    """ Returns the wing loading and thrust of the weight based on performance parameters

    :param perf_par: performance parameter class from data structues
    :type perf_par: PerformanceParameters class
    :param wing:  wing class from data structues
    :type wing:  Wing class
    """    

    #Check if it"s lilium or not to define the variable that will say to vertical_flight what formula to use.
    WS_range = np.arange(1,4000,1)
    #data["WS"],data["TW"],data["WP_cruise"],data["WP_hover"] = plot_wing_power_loading_graphs(data["eff"], data["StotS"], data["diskloading"], data["name"],WS_range,i)


    #CALCULATE ALL THE VALUES FOR THE GRAPHS
    TW_range = powerloading_thrustloading(WS_range,const.rho_sl,const.roc_hvr, perf_par.Stots)  
    #if data["name"] == "J1":   
    #    TW_range = TW_range*1.3     #Added 30% extra thrust to maintain stability
    CLIMBRATE = cont_factor*powerloading_climbrate(perf_par.prop_eff,const.roc_cr, WS_range,const.rho_cr,aero.cd0_cruise ,aero.e,wing.aspect_ratio)
    TURN_VCRUISE = cont_factor*powerloading_turningloadfactor(const.rho_cr,const.v_cr,WS_range, perf_par.prop_eff ,wing.aspect_ratio,aero.e, perf_par.turn_loadfactor,aero.cd0_cruise)
    TURN_VMAX = cont_factor*powerloading_turningloadfactor(const.rho_cr,perf_par.v_max, WS_range, perf_par.prop_eff ,wing.aspect_ratio ,aero.e ,perf_par.turn_loadfactor,aero.cd0_cruise)
    VERTICALFLIGHT = cont_factor*powerloading_verticalflight(perf_par.MTOM ,TW_range, engine.total_disk_area ,const.rho_sl,perf_par.prop_eff ,False)
    STALLSPEED = wingloading_stall(aero.cL_max ,perf_par.v_stall, const.rho_sl)
    CLIMBGRADIENT = cont_factor*powerloading_climbgradient(aero.e ,wing.aspect_ratio ,aero.cd0_cruise,WS_range,const.rho_sl,perf_par.prop_eff ,const.climb_gradient)

    #DETERMINE LOWEST
    lowest_area_y_novf = []
    lowest_area_y = []
    lowest_area_x = np.arange(0,int(STALLSPEED),1)
    for i in lowest_area_x:
        lowest_area_y.append(min(CLIMBRATE[i],TURN_VCRUISE[i],TURN_VMAX[i],CLIMBGRADIENT[i],VERTICALFLIGHT[i]))
        lowest_area_y_novf.append(min(CLIMBRATE[i],TURN_VCRUISE[i],TURN_VMAX[i],CLIMBGRADIENT[i]))
        
    #DETERMINE LIMITING FACTORS
    margin = 0.95
    perf_par.wing_loading_cruise = STALLSPEED*margin
    perf_par.TW_max = powerloading_thrustloading(perf_par.wing_loading_cruise,const.rho_sl,const.roc_hvr, perf_par.Stots)
    WP_cruise = lowest_area_y_novf[-1]*margin
    WP_hover = lowest_area_y[-1]*margin
    aero.cL_cruise = 2/(const.rho_cr*const.v_cr**2)*perf_par.wing_loading_cruise
    return perf_par, wing, engine, aero


