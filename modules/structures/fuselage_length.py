# -*- coding: utf-8 -*-
import numpy as np

def tank_radius(l_tank, V_tank):
    """WIDTH OF FUSELAGE AT THE END OF HYDROGEN TANK, BASED ON TANK RADIUS"

    :param l_tank: length of the tank in m
    :type l_tank: float
    :param V_tank: total volume of the tank in m^3
    :type V_tank:  float
    :return: tank radius in m
    :rtype: float
    """    
    roots = np.roots(np.pi/3, np.pi*l_tank, 0, V_tank)
    return [root.real for root in roots if root.real > 0 and root.imag == 0]

def width_fuselage_crushed(r): 
    """ WIDTH OF FUSELAGE AT THE END OF HYDROGEN TANK, BASED ON TANK RADIUS"

    :param r: tank radius in m
    :type r: float
    :return: width fuselage crushed in m
    :rtype: float
    """    
    return 4*r

def height_fuselage_crushed(r, Beta, h0, b0):
    """ HEIGHT OF FUSELAGE AT THE END OF HYDROGEN TANK, CONSTRAINED BY BETA

    :param r: radius of tank in meters
    :type r: float
    :param Beta: Crash diameter coefficient, defines the safe area after a crash
    :type Beta: float
    :param h0: Height of the cabin, thus the height at the start of the tailcone
    :type h0: float
    :param b0:Height of the cabin, thus the height at the start of the tailcone
    :type b0: _type_
    :return: _description_
    :rtype: _type_
    """    
    b_c = 4*r
    A = b0/h0
    A_f = b_c**2/(A*Beta**2)
    return np.sqrt(A_f/A)

def find_length_fuselage(h0, h_f, l_tank):
    """HEIGHT OF FUSELAGE AT THE END OF HYDROGEN TANK, CONSTRAINED BY BETA

    :param h0: _description_
    :type h0: _type_
    :param h_f: _description_
    :type h_f: _type_
    :param l_tank: _description_
    :type l_tank: _type_
    :return: _description_
    :rtype: _type_
    """    
    return (h0-h_f)/h_f*l_tank

def upsweep_angle(h0, h_f, l_tank):
    """ FIND UPSWEEP ANGLE

    :param h0: _description_
    :type h0: _type_
    :param h_f: _description_
    :type h_f: _type_
    :param l_tank: _description_
    :type l_tank: _type_
    :return: _description_
    :rtype: _type_
    """    
    return np.arctan2((h0-h_f)/l_tank)