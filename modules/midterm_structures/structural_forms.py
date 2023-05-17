# -*- coding: utf-8 -*-
import numpy as np

"""GEOMETRIES"""
#Moments of Inertia
def i_xx_solid(width,height):
    return width*height*height*height/12
def i_yy_solid(width,height):
    return width*width*width*height/12
def j_z_solid(width,height):
    return width*height*(width*width + height*height)/12

def i_xx_thinwalled(width,height,thickness):
    return 1/3 * width*height*height*thickness
def i_yy_thinwalled(width,height,thickness):
    return 1/3 * width*width*height*thickness
def j_z_thinwalled(width,height,thickness):
    return (height+width)*height*width*thickness/3



"""NORMAL STRESS"""
def bending_stress(moment_x,moment_y,i_xx,i_yy,i_xy,x,y):
    return((moment_x*i_yy-moment_y*i_xy)*y + (moment_y*i_xx - moment_x*i_xy)*x)/(i_xx*i_yy-i_xy*i_xy)
def normal_stress(force,area):
    return force/area
    

"""SHEAR STRESS"""
def torsion_circular(torque,dens,j_z):
    return torque*dens/j_z

def torsion_thinwalled_closed(torque,thickness,area):
<<<<<<< Updated upstream
    return torque/(2*thickness*area)

def shear_open_sect(v_x,v_y,i_xx,i_yy,i_xy,thickness,x,y):
    return 
=======
    return torque/(2*thickness*area)
>>>>>>> Stashed changes
