import numpy as np

def v_exhaust(MTOM, g0, rho, atot, vinf):
    Vcr = np.sqrt(2*(MTOM*g0)/(rho*atot)+vinf**2)
    return Vcr

def vhover(MTOM, g0, rho,atot):
    Vhov = np.sqrt(2 * (MTOM*g0) / (rho * atot) )
    return Vhov

def propeff(vcr, vinf):
    prop = 2/(1+vcr/vinf)
    return prop

def hoverstuffopen(T, rho, atot,toverw ):
    Phopen = T**1.5/np.sqrt(2*rho*atot)
    PhopenMAX = (T * toverw) ** 1.5 / (2 * np.sqrt(rho * atot))
    energyhoveropen = 90/3600 * Phopen * 1.3
    energyhoveropenMAX = 90/3600 * PhopenMAX * 1.3
    return Phopen, PhopenMAX, energyhoveropen, energyhoveropenMAX

def hoverstuffduct(T, rho, atot,toverw ):
    Phduct = T**1.5/(2*np.sqrt(rho*atot))
    PhductMAX = (T*toverw) ** 1.5 / (2 * np.sqrt(rho * atot))
    energyhoverduct = 90 / 3600 * Phduct * 1.3
    energyhoverductMAX = 90/3600 * PhductMAX * 1.3
    return Phduct, PhductMAX, energyhoverduct, energyhoverductMAX

def powercruise(MTOM, g0 ,v_cr,lift_over_drag,propeff):
    powercruise = MTOM*g0*v_cr/(lift_over_drag*propeff)
    return powercruise

def powerclimb(MTOM, g0, S, rho, lod_climb, prop_eff, ROC):
    climb_power = MTOM*g0*(np.sqrt(2*MTOM*g0*(1/lod_climb)/(S*rho)) + ROC )/prop_eff
    return climb_power


def powerloiter(MTOM, g0, S, rho, lod_climb, prop_eff):
    loiter_power = MTOM*g0*(np.sqrt(2*MTOM*g0*(1/lod_climb)/(S*rho)))/prop_eff
    return loiter_power 


def powerdescend(MTOM, g0, S, rho, lod_climb, prop_eff, ROD):
    climb_power = MTOM*g0*(np.sqrt(2*MTOM*g0*(1/lod_climb)/(S*rho)) - ROD )/prop_eff
    return climb_power








