# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import sys
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
from input.data_structures.GeneralConstants import *

"""NORMAL STRESS"""
def bending_stress(moment_x,moment_z,i_xx,i_zz,i_xz,width,height):
    sigma_a = ((moment_x * i_zz - moment_z * i_xz) * height/2 + (moment_z * i_xx - moment_x * i_xz) * -width/2) / (i_xx * i_zz - i_xz * i_xz)
    sigma_b = ((moment_x * i_zz - moment_z * i_xz) * height/2 + (moment_z * i_xx - moment_x * i_xz) * width/2) / (i_xx * i_zz - i_xz * i_xz)
    sigma_c = ((moment_x * i_zz - moment_z * i_xz) * -height/2 + (moment_z * i_xx - moment_x * i_xz) * width/2) / (i_xx * i_zz - i_xz * i_xz)
    sigma_d = ((moment_x * i_zz - moment_z * i_xz) * -height/2 + (moment_z * i_xx - moment_x * i_xz) * -width/2) / (i_xx * i_zz - i_xz * i_xz)
    '''
    print(f"sigma_a = {sigma_a}")
    print(f"sigma_b = {sigma_b}")
    print(f"sigma_c = {sigma_c}")
    print(f"sigma_d = {sigma_d}")
    '''
    return max(sigma_a,sigma_b,sigma_c,sigma_d),min(sigma_a,sigma_b,sigma_c,sigma_d)

def normal_stress(force,area):
    return force/area
    

"""SHEAR STRESS"""
def torsion_circular_section(torque,dist,j_z):
    return torque*dist/j_z

def torsion_thinwalled_closed(torque,thickness,enclosed_area):
    return torque/(2*thickness*enclosed_area)
''' I fucked the directions here so to make it more clear i redid everything in the same equation below
def shear_thin_walled_rectangular_section(width,height,thickness,i_xx,i_zz,Vx,Vz):
    #Assumptions: Vx and Vz act at the centerpoint of the rectangle
    #The section is a rectangle with A in the bottom left corner and continueing counterclockwise with B, C and D respectively.
    #An imaginary cut is made at A to perform this analysis where first the basic shear flows are calculated below.
    def q_ab_b(s):
        return -Vz/i_xx*thickness*(height/2 * s - s*s/2) - Vx/i_zz*thickness*(-width/2*s)
    def q_bc_b(s):
        return Vz/i_xx*thickness*height/2*s - Vx/i_zz*thickness*(-width/2*s+s*s/2) + q_ab_b(height)
    def q_cd_b(s):
        return -Vz/i_xx*thickness*(-height/2*s+s*s/2)-Vx/i_zz*thickness*width/2*s + q_bc_b(width)
    def q_da_b(s):
        return-Vz/i_xx*thickness*height/2*s-Vx/i_zz*thickness*(width/2*s-s*s/2) + q_cd_b(height)

    #Now the line integrals are calculated for each shear flow.
    def int_q_ab_b(s):
        return -Vz/i_xx*thickness*(height/2 * s*s/2 - s*s*s/6) - Vx/i_zz*thickness*(-width/2*s*s/2)
    def int_q_bc_b(s):
        return Vz/i_xx*thickness*height/2*s*s/2 - Vx/i_zz*thickness*(-width/2*s*s/2+s*s*s/6) + q_ab_b(height)*s
    def int_q_cd_b(s):
        return -Vz / i_xx * thickness * (-height / 2 * s *s /2+ s * s *s/ 6) - Vx / i_zz * thickness * width / 2 * s * s / 2+ q_bc_b(width)*s
    def int_q_da_b(s):
        return -Vz / i_xx * thickness * height / 2 * s*s/2 - Vx / i_zz * thickness * (width / 2 * s*s/2 - s * s *s/ 6) + q_cd_b(height)*s

    #qs0 is determined by taking the sum of the moments around the center point of the rectangle.
    def qs0():
        return (width/2 * ((int_q_ab_b(height)-int_q_ab_b(0)) + (int_q_cd_b(height) - int_q_cd_b(0))) + height/2*((int_q_bc_b(width)-int_q_bc_b(0))+(int_q_da_b(width)-int_q_da_b(0))))/((height-thickness)*(width-thickness)*2)
        #return 0
    #The real shear flows for each section is determined
    def q_ab(s):
        print("qs0 = " ,qs0())
        return q_ab_b(s) - qs0()
    def q_bc(s):
        return q_bc_b(s) - qs0()
    def q_cd(s):
        return q_cd_b(s) - qs0()
    def q_da(s):
        return q_da_b(s) - qs0()

    #accumulate data
    x = np.arange(0,width,1E-5)
    y = np.arange(0,height,1E-5)
    tau_ab = q_ab(y)/(thickness)
    tau_bc = q_bc(x)/(thickness)
    tau_cd = q_cd(y)/(thickness)
    tau_da = q_da(x)/(thickness)
    plt.plot(y,tau_ab,label="AB")
    plt.plot(x,tau_bc,label="BC")
    plt.plot(y,tau_cd,label="CD")
    plt.plot(x,tau_da,label="DA")
    plt.legend()
    plt.xlim(0,1.5)
    print(f"AB goes from {tau_ab[0]} to {tau_ab[-1]}")
    print(f"BC goes from {tau_bc[0]} to {tau_bc[-1]}")
    print(f"CD goes from {tau_cd[0]} to {tau_cd[-1]}")
    print(f"DA goes from {tau_da[0]} to {tau_da[-1]}")
    print(f"Max shear stress = {max(max(abs(tau_ab)),max(abs(tau_bc)),max(abs(tau_cd)),max(abs(tau_da)))/1000000}[MPa]\n\n")
    plt.show()

    #I added the minus below since I assumed cw+ but it should be ccw+ so the magnitudes are correct but they should point the other way around
    return -tau_ab,-tau_bc,-tau_cd,-tau_da
'''

def shear_thin_walled_rectangular_section(width,height,thickness,i_xx,i_zz,Vx,Vz):
    #Assumptions: Vx and Vz act at the centerpoint of the rectangle
    #The section is a rectangle with A in the bottom left corner and continueing counterclockwise with B, C and D respectively.
    #An imaginary cut is made at A to perform this analysis where first the basic shear flows are calculated below.

    Vzt_ixx = Vz*thickness/i_xx
    Vxt_izz = Vx*thickness/i_zz

    def q_ab_b(s):
        return -Vzt_ixx*(height/2 * s) - Vxt_izz*(-width/2*s+s*s/2)
    def q_bc_b(s):
        return -Vzt_ixx*(height/2*s-s*s/2) - Vxt_izz*(width/2*s) + q_ab_b(width)
    def q_cd_b(s):
        return -Vzt_ixx*(-height/2*s) - Vxt_izz*(width/2*s-s*s/2) + q_bc_b(height)
    def q_da_b(s):
        return -Vzt_ixx*(-height/2*s + s*s/2) - Vxt_izz*(-width/2*s) + q_cd_b(width)

    #Now the line integrals are calculated for each shear flow.
    def int_q_ab_b(s):
        return -Vzt_ixx*(height/4*s*s) - Vxt_izz*(-width/4*s*s+s*s*s/6)
    def int_q_bc_b(s):
        return -Vzt_ixx*(height/4*s*s-s*s*s/6) - Vxt_izz*(width/4*s*s) + q_ab_b(width)*s
    def int_q_cd_b(s):
        return -Vzt_ixx*(-height/4*s*s) - Vxt_izz*(width/4*s*s-s*s*s/6) + q_bc_b(height)*s
    def int_q_da_b(s):
        return -Vzt_ixx*(-height/4*s*s + s*s*s/6) - Vxt_izz*(-width/4*s*s) + q_cd_b(width)*s

    #qs0 is determined by taking the sum of the moments around the center point of the rectangle.
    def qs0():
        return -(height/2 * ((int_q_ab_b(width)-int_q_ab_b(0)) + (int_q_cd_b(width) - int_q_cd_b(0))) + width/2*((int_q_bc_b(height)-int_q_bc_b(0))+(int_q_da_b(height)-int_q_da_b(0))))/((height-thickness)*(width-thickness)*2)
        #return 0
        #return -(-height/2 * ((int_q_cd_b(width)-int_q_cd_b(0)) + (int_q_ab_b(width)-int_q_ab_b(0))) - width/2 * ((int_q_bc_b(height)-int_q_bc_b(0))+(int_q_da_b(height)-int_q_da_b(0))))/(2*(height-thickness)*(width-thickness))
    #The real shear flows for each section is determined
    def q_ab(s):
        return q_ab_b(s) + qs0()
    def q_bc(s):
        return q_bc_b(s) + qs0()
    def q_cd(s):
        return q_cd_b(s) + qs0()
    def q_da(s):
        return q_da_b(s) + qs0()

    #accumulate data
    x = np.arange(0,width,1E-5)
    y = np.arange(0,height,1E-5)
    tau_ab = q_ab(x)/(thickness)
    tau_bc = q_bc(y)/(thickness)
    tau_cd = q_cd(x)/(thickness)
    tau_da = q_da(y)/(thickness)
    print(tau_ab)#,tau_bc,tau_cd,tau_da)
    return tau_ab,tau_bc,tau_cd,tau_da



def critical_buckling_stress(C,t,b): return C*(np.pi**2*E_alu)/(12*(1-nu_alu**2))*(t/b)**2

def wohlers_curve(C,m,S): return C/(S**m)

def paris_law(C,beta,load,m,a_f,a_0):
    def int_paris_law(a):
        return -2/((m-2)*a**((m-2)/2))
    return (1/(C * load **m)) * (1/((beta * np.sqrt(np.pi))**m)) * (int_paris_law(a_f)-int_paris_law(a_0))

