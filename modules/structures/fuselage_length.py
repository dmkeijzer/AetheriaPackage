# -*- coding: utf-8 -*-
import numpy as np

""""TANK RADIUS FROM TANK LENGTH"""
def tank_radius(l_tank, V_tank):
    roots = np.roots(np.pi/3, np.pi*l_tank, 0, V_tank)
    return [root.real for root in roots if root.real > 0 and root.imag == 0]

"""WIDTH OF FUSELAGE AT THE END OF HYDROGEN TANK, BASED ON TANK RADIUS"""
def width_fuselage_crushed(r): return 4*r

"""HEIGHT OF FUSELAGE AT THE END OF HYDROGEN TANK, CONSTRAINED BY BETA"""
def height_fuselage_crushed(r, Beta, h0, b0):
    b_c = 4*r
    A = b0/h0
    A_f = b_c**2/(A*Beta**2)
    return np.sqrt(A_f/A)

"""FIND LENGTH OF FUSELAGE """
def find_length_fuselage(h0, h_f, l_tank): return (h0-h_f)/h_f*l_tank

"""FIND UPSWEEP ANGLE"""
def upsweep_angle(h0, h_f, l_tank): return np.arctan2((h0-h_f)/l_tank)