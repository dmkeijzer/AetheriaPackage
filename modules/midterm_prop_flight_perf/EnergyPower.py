import numpy as np

def v_exhaust(MTOM, g0, rho,atot, vinf):
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

def powerclimb(MTOM, g0, v_climb, lod_climb, prop_eff, ROC):
    climb_power = (MTOM*g0 * v_climb * (1/lod_climb) + ROC )/prop_eff
    return climb_power

def powerloiter(MTOM, g0, v_climb, lod_climb, propeff):
    loiter_power = (MTOM*g0 * v_climb * (1/lod_climb))/propeff
    return loiter_power 



# if __name__ == '__main__':




#     J1hoverpower, J1maxpower, J1hoverenergy, J1maxenergy = hoverstuffopen(MTOW, rho, J1area,J1toverw)




#     W1hoverpower, W1maxpower, W1hoverenergy, W1maxenergy = hoverstuffopen(MTOW, rho, W1area, W1toverw)




#     L1hoverpower, L1maxpower, L1hoverenergy, L1maxenergy = hoverstuffduct(MTOW, rho, L1area, L1toverw)





#     J1vcr = vcruise(MTOW, rho, J1area, vinfms)




#     L1vcr = vcruise(MTOW, rho, L1area, vinfms)




#     W1vcr = vcruise(MTOW, rho, W1area, vinfms)





#     J1prop = propeff(J1vcr, vinfms)




#     L1prop = propeff(L1vcr, vinfms)




#     W1prop = propeff(W1vcr, vinfms)





#     J1cruisepower, J1cruiseenergy = cruisestuff(MTOW,vinfms,J1clcd, J1prop, range)




#     L1cruisepower, L1cruiseenergy = cruisestuff(MTOW, vinfms,L1clcd, L1prop, range)




#     W1cruisepower, W1cruiseenergy = cruisestuff(MTOW, vinfms, W1clcd, W1prop, range)






