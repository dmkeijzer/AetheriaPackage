# -*- coding: utf-8 -*-
def i_xx_solid(b,h):
    return b*h*h*h/12
def i_yy_solid(b,h):
    return b*b*b*h/12
def j_z_solid(b,h):
    return b*h*(b*b + h*h)/12

def i_xx_thinwalled(b,h,t):
    return 1/3 * b*h*h*t
def i_yy_thinwalled(b,h,t):
    return 1/3 * b*b*h*t
def j_z_thinwalled(b,h,t):
    return (h+b)*h*b*t/3
