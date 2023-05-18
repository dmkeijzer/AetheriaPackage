# -*- coding: utf-8 -*-
import numpy as np

"""GEOMETRIES"""


# Moments of Inertia
def i_xx_solid(width, height):
    return width * height * height * height / 12

def i_yy_solid(width, height):
    return width * width * width * height / 12

def j_z_solid(width, height):
    return width * height * (width * width + height * height) / 12

def i_xx_thinwalled(width, height, thickness):
    return 1 / 3 * width * height * height * thickness

def i_yy_thinwalled(width, height, thickness):
    return 1 / 3 * width * width * height * thickness

def j_z_thinwalled(width, height, thickness):
    return (height + width) * height * width * thickness / 3



def enclosed_area_thinwalled(width,height,thickness):
    return (height-thickness)*(width-thickness)
