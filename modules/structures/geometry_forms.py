# -*- coding: utf-8 -*-
import numpy as np

"""GEOMETRIES"""


# Moments of Inertia
def i_xx_solid(width, height):
    return width * height * height * height / 12

def i_zz_solid(width, height):
    return width * width * width * height / 12

def j_y_solid(diameter):
    return diameter * diameter * diameter * diameter * np.pi/ 32

def i_xx_thinwalled(width, height, thickness):
    return 1 / 3 * width * height * height * thickness

def i_zz_thinwalled(width, height, thickness):
    return 1 / 3 * width * width * height * thickness




def enclosed_area_thinwalled(width,height,thickness):
    return (height-thickness)*(width-thickness)

def area_thinwalled(width,height,thickness):
    return (width*height - (width-thickness*2)*(height - thickness*2))