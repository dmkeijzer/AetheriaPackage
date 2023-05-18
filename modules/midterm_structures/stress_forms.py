# -*- coding: utf-8 -*-
import numpy as np


"""NORMAL STRESS"""
def bending_stress(moment_x,moment_y,i_xx,i_yy,i_xy,x,y):
    return((moment_x*i_yy-moment_y*i_xy)*y + (moment_y*i_xx - moment_x*i_xy)*x)/(i_xx*i_yy-i_xy*i_xy)
def normal_stress(force,area):
    return force/area
    

"""SHEAR STRESS"""
def torsion_circular_section(torque,dens,j_z):
    return torque*dens/j_z

def torsion_thinwalled_closed(torque,thickness,enclosed_area):
    return torque/(2*thickness*enclosed_area)
