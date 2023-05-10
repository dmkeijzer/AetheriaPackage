import numpy as np
# import constants as cte
# WING PLANFORM
""""
b =span
"""
def wing_planform(AR,S, sweepc4, taper):

    b = np.sqrt(AR*S)
    c_r = 2*S/((1+taper)*b)
    c_t = taper * c_r
    c_MAC = (2/3)*c_r*((1+taper+taper**2)/(1+ taper))
    y_MAC = (b/6)*((1 + 2*taper)/(1+taper))
    tan_sweep_LE = 0.25 * (2 * c_r / b) * (1-taper) + np.tan(sweepc4)

    X_LEMAC = y_MAC*tan_sweep_LE

    return b,c_r,c_t,c_MAC, y_MAC, X_LEMAC

def wing_planform_double(AR, S1, sweepc41, taper1, S2, sweepc42, taper2):
    #Wing 1
    b1 = np.sqrt(2* AR * S1)
    c_r1 = 2 * S1 / ((1 + taper1) * b1)
    c_t1 = taper1 * c_r1
    c_MAC1 = (2 / 3) * c_r1 * ((1 + taper1 + taper1 ** 2) / (1 + taper1))
    y_MAC1 = (b1 / 6) * ((1 + 2 * taper1) / (1 + taper1))
    tan_sweep_LE1 = 0.25 * (2 * c_r1 / b1) * (1 - taper1) + np.tan(sweepc41)

    X_LEMAC1 = y_MAC1 * tan_sweep_LE1
    wing1 = [b1, c_r1, c_t1, c_MAC1, y_MAC1, X_LEMAC1]

    #Wing 2

    b2 = np.sqrt(2* AR * S2)
    c_r2 = 2 * S2 / ((1 + taper2) * b2)
    c_t2 = taper2 * c_r2
    c_MAC2 = (2 / 3) * c_r2 * ((1 + taper2 + taper2 ** 2) / (1 + taper2))
    y_MAC2 = (b2 / 6) * ((1 + 2 * taper2) / (1 + taper2))
    tan_sweep_LE2 = 0.25 * (2 * c_r2 / b2) * (1 - taper2) + np.tan(sweepc42)

    X_LEMAC2 = y_MAC2 * tan_sweep_LE2
    wing2 = [b2, c_r2, c_t2, c_MAC2, y_MAC2, X_LEMAC2]

    return wing1, wing2

def sweep_atx(x, c_r,b, taper,sweepc4):
    tan_sweep_LE = 0.25 * (2 * c_r / b) * (1 - taper) + np.tan(sweepc4)
    sweep = np.arctan(tan_sweep_LE - x * (2 * c_r / b) * (1 - taper) )
    return sweep
# Airfoil Selection


def taper_opt(sweepc4):
    return 0.45 * np.exp( -0.036 * sweepc4) # Eq. 7.4 Conceptual Design of a Medium Range Box Wing Aircraft


def CL_des(rho, V, W, S ):
    return W / S  / (0.5 * rho * V ** 2)  # no -ve lift contribution from tail trimming -> no 10% factor

def Re( rho, V, MAC, mu):
    return (rho * V * MAC)/mu

#Wing performance

def Mach(V,a):
    return V/a

def liftslope(type, AR, sweepc2, M, Clda_airfoil, s1, s2, deda):
    b = np.sqrt(1-M**2)
    SW = np.tan(sweepc2)
    ref_slope = Clda_airfoil * (AR/(2+ np.sqrt(4+((AR*b/0.95)**2 )*((1+SW**2)/(b**2)))))

    if type == 'normal':
        return ref_slope

    else:
        ratio = 2*(2+ np.sqrt((AR**2)*(1+SW**2)+4))/(2+ np.sqrt((4*AR**2)*(1+0.25*SW**2)+4))

        slope_iso = ratio*ref_slope
        slope_1 = slope_iso
        slope_2 = slope_iso *(1-deda)
        slope_tot = slope_1 *s1 +slope_2 *s2
        return slope_tot, slope_1, slope_2

