from math import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumulative_trapezoid
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import pathlib as pl
import os
import openmdao.api as om

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

import input.data_structures.GeneralConstants as const
from input.data_structures.engine import Engine
from input.data_structures.material import Material
from input.data_structures.wing import Wing
from input.data_structures.GeneralConstants import *
from modules.aero.avl_access import get_lift_distr

#------------ASSUMPTION----------
#Ribs carry no structural load. They only limit the buckling constraint on the stringers. So their thickness will be assumed to be 3 mm and their pitch will be optimized.
#Each stringer is assumed to be effective up until it's centerpoint is out of the center point of the spar.


class Wingbox():
    def __init__(self,wing,engine,material, aero):
        #Material
        self.poisson = material.poisson
        self.density = material.rho
        self.E = material.E
        self.pb = material.pb
        self.beta = material.beta
        self.g = material.g
        self.sigma_yield = material.sigma_yield
        self.m_crip = material.m_crip
        self.sigma_uts = material.sigma_uts
        self.shear_modulus = material.shear_modulus
        self.lift_func = get_lift_distr(wing, aero)
        self.engine = engine
        self.wing = wing
        self.t_rib = 4e-3 #[mm]
        self.x_lewing = wing.x_lewing
        self.thickness_to_chord = wing.thickness_to_chord


        #Wing
        self.taper = wing.taper
        self.n_max = n_max_req
        self.span = wing.span
        self.chord_root = wing.chord_root
       

        #Engine
        self.engine_weight = engine.mass_pertotalengine
        self.engine.dump()
        self.y_rotor_loc = engine.y_rotor_loc
        self.x_rotor_loc = engine.x_rotor_loc
        self.thrust_per_engine = engine.thrust_per_engine


        self.nacelle_w = engine.nacelle_width #TODO Check if it gets updated

        #STRINGERS
        self.n_str = 3
        self.str_array_root = np.linspace(0.15*self.chord_root,0.75*self.chord_root,self.n_str+2)

        self.y_mesh = 1000

        #GEOMETRY
        self.width = 0.6*self.chord_root
        self.str_pitch = 0.6*self.chord_root/(self.n_str+1) #THE PROGRAM ASSUMES THERE ARE TWO STRINGERS AT EACH END AS WELL
        
        #Set number of ribs in inboard and outboard section
        n_sections = 30


    #---------------General functions------------------
    # def find_nearest(array, value):
    #     difference_array = np.absolute(array-value)
    #     index = difference_array.argmin()
    #     return array[index],index


    #---------------Geometry functions-----------------
    def perimiter_ellipse(self,a,b):
        return np.pi *  ( 3*(a+b) - np.sqrt( (3*a + b) * (a + 3*b) ) ) #Ramanujans first approximation formula
    def l_sk(self,y):
        return np.sqrt(self.height(y) ** 2 + (0.25 * self.chord(y)) ** 2)

    def chord(self,y):
        return self.chord_root - self.chord_root * (1 - self.taper) * y * 2 / self.span

    def height(self,y):
        return self.thickness_to_chord * self.chord(y)
    
    def get_w_str(self,h_str):
        return 0.8*h_str

    def get_area_str(self, h_st,t_st):
        return t_st * (2 * self.get_w_str(h_st) + h_st)




    def I_st_x(self, h_st,t_st):
        Ast = self.get_area_str(h_st, t_st)
        i = t_st * h_st ** 3 / 12 + self.get_w_str(h_st) * t_st ** 3 / 12 + 2 * Ast * (0.5 * h_st) ** 2
        return i
    
    def I_st_z(self, h_st,t_st):
        Ast = self.get_area_str(h_st, t_st)
        i = (h_st*t_st ** 3)/12 + (t_st* self.get_w_str(h_st)**3)/12
        return i





    def w_sp(self,y):
        return 0.3 * self.height(y)




    def I_sp_x(self,t_sp,y):
        h = self.height(y)
        wsp = self.w_sp(y)
        return t_sp * (h - 2 * t_sp) ** 3 / 12 + 2 * wsp * t_sp ** 3 / 12 + 2 * t_sp * wsp * (
                0.5 * h) ** 2
    
    def I_sp_z(self,t_sp,y):
        h = self.height(y)
        wsp = self.w_sp(y)
        return ((h - 2*t_sp)*t_sp**3)/12 + (2*t_sp*wsp**3)/12
    
    def get_x_le(self,y):
        return self.x_lewing + 0.25*self.chord_root - 0.25*self.chord(y)
    
    def get_x_te(self,y):
        return self.x_lewing + 0.25*self.chord_root +0.75*self.chord(y)
    
    def get_y_le(self,x):
        return (x - self.x_lewing)/(0.25*self.chord_root*(1-self.taper)*2/self.span)
    
    def get_y_te(self,x):
        return (x - self.x_lewing - self.chord_root)/(-0.75*self.chord_root*(1-self.taper)*2/self.span)
    #!!!!
    def get_x_start_wb(self,y):
        return self.get_x_le(y) + 0.15*self.chord(y)
    
    def get_x_end_wb(self,y):
        return self.get_x_te(y) -0.25*self.chord(y)
    
    def get_y_start_wb(self,x):
        return (x-self.x_lewing-0.15*self.chord_root)/(0.1*self.chord_root*(1-self.taper)*2/self.span)
    
    def get_y_start_wb(self,x):
        return (x-self.x_lewing-0.75*self.chord_root)/(-0.5*self.chord_root*(1-self.taper)*2/self.span)
    

    def get_y_mesh(self):
        return np.linspace(0,self.span/2, self.y_mesh)   
    
    # def get_str_x(self,y):
    #     x_start_box = self.get_x_start_wb(y)
    #     x_end_box = self.get_x_end_wb(y)
    #     str_array = [x_start_box]  
    #     for x in self.str_array_root:
    #         if x > x_start_box and x < x_end_box:
    #             str_array = np.array(str_array,x)
    #     str_array = np.array(str_array,x_end_box)
    #     return str_array
    
    # def get_str_endings(self,y):
    #     str_at_y = self.get_str_x(y)
    #     str_at_tip = self.get_str_x(self.span/2)
    #     str_to_be_removed = np.unique(np.concatenate(str_at_y,str_at_tip))
    #     return 


    
    # def get_n_str(self,y):
    #     return len(self.get_str_x(y))

    def I_xx(self,t_sp,h_st,t_st,t_sk,y):#TODO implement dissappearing stringers
        h = self.height(y)
        # nst = n_st(c_r, b_st)
        Ist = self.I_st_x(h_st,t_st)
        Isp = self.I_sp_x(t_sp,y)
        A = self.get_area_str(h_st,t_st)
        return 2 * (Ist + A * (0.5 * h) ** 2) * self.n_str + 2 * Isp + 2 * (0.6 * self.chord(y) * t_sk ** 3 / 12 + t_sk * 0.6 * self.chord_root * (0.5 * h) ** 2)

    def I_zz(self,t_sp,h_st,t_st,t_sk,y):#TODO implement dissappearing stringers
        h = self.height(y)
        # nst = n_st(c_r, b_st)
        Ist = self.I_st_z(h_st,t_st)
        Isp = self.I_sp_z(t_sp,y)
        Ast = self.get_area_str(h_st,t_st)
        Asp = t_sp*self.w_sp(y)*2 + (h-2*t_sp)*t_sp
        centre_line = self.chord_root*0.25 + self.chord(y)*0.25
        position_stringers = np.ones((len(y),len(self.str_array_root)))*self.str_array_root
        distances_from_centre_line = position_stringers - np.transpose(np.ones((len(self.str_array_root),len(y))) * centre_line)
        moments_of_inertia = np.sum(distances_from_centre_line * Ast * 2,axis=1)
        moments_of_inertia += 2*(Isp + Asp * (self.chord(y)/2)*(self.chord(y)/2)) + 2* t_sk*self.chord(y)**3/12
        return moments_of_inertia


#----------------Load functions--------------------
    def aero(self, y):
        return  self.lift_func(y)
    
    def weight_from_tip(self,t_sp, h_st,t_st,t_sk,y):#TODO implement dissappearing stringers #TODO Include rib weights
        total_weight = np.zeros(len(y))

        total_weight += self.engine_weight

        weight_str = self.density * self.get_area_str(h_st,t_st) * (self.span/2- y) * self.n_str
        weight_skin = t_sk * ((0.6 * self.chord(self.span/2) + 0.6 * self.chord(y))* (self.span/2- y) / 2) * 2 + self.perimiter_ellipse(0.15*self.chord(y),self.height(y))*t_sk * 0.15 + np.sqrt((0.25*self.chord(y))**2 + (self.height(y))**2)*2*t_sk
        weight_spar_flanges = (self.w_sp(self.span/2) + self.w_sp(y))*(self.span/2- y)/2 * t_sp * self.density * 4
        weight_spar_web = (self.height(self.span/2) - 2*t_sp + self.height(y) - 2*t_sp) * (self.span/2- y) /2 * t_sp *self.density * 2
        
        total_weight += (weight_str + weight_skin + weight_spar_flanges + weight_spar_web)
        if y[-1]>self.y_rotor_loc[0]:
            difference_array = np.absolute(y-self.y_rotor_loc[0])
            index = difference_array.argmin()
            eng_array = np.zeros(index+1) + self.engine_weight + 1
            eng_array = np.append(eng_array,np.zeros(len(y)-len(eng_array)))
            total_weight += eng_array

        # self.find_nearest([1,2,3],3)
        return total_weight
    
    def shear_z_from_tip(self,t_sp, h_st,t_st,t_sk,y):
        return self.weight_from_tip(t_sp, h_st,t_st,t_sk,y) - self.aero(y)
    
    def torque_from_tip(self,y):
        if y[-1]>self.y_rotor_loc[0]:
            difference_array = np.absolute(y-self.y_rotor_loc[0])
            index = difference_array.argmin()
            torque_array = np.zeros(index+1) + (self.thrust_per_engine- self.engine_weight*9.81)*(self.x_rotor_loc[0]-self.get_x_start_wb(y[index]))
            torque_array = np.append(torque_array,np.zeros(len(y)-len(torque_array)))
        return torque_array
    

    def moment_x_from_tip(self,t_sp, h_st,t_st,t_sk,y):
        return self.weight_from_tip(t_sp, h_st,t_st,t_sk,y)*9.81*(self.span/2 - y)

#-----------Stress functions---------
    def bending_stress_x_from_tip(self,t_sp, h_st,t_st,t_sk,y):
        return self.moment_x_from_tip(t_sp, h_st,t_st,t_sk,y)/self.I_xx(t_sp,h_st,t_st,t_sk,y) * self.height(y)/2
    
    def N_xy(self,t_sp, h_st,t_st,t_sk,y):

        Vz=self.shear_z_from_tip(t_sp, h_st,t_st,t_sk,y)
        T =self.torque_from_tip(t_sp, h_st,t_st,t_sk,y)
        Ixx = self.I_xx()



        # Base region 1
        def qb1(self):
            return Vz * t_sk * (0.5 * self.height(y)) ** 2 * (np.cos(z) - 1) / Ixx
        I1 = qb1(pi / 2)

        # Base region 2
        qb2 = lambda z: -Vz[i] * t_sp * z ** 2 / (2 * Ixx)
        I2 = qb2(h)
        s2 = np.arange(0, h+ 0.1, 0.1)

        # Base region 3
        qb3 = lambda z: - Vz[i] * tarr[i] * (0.5 * h) * z / Ixx + I1 + I2
        I3 = qb3(0.6 * c)
        s3 = np.arange(0, 0.6*c+ 0.1, 0.1)

        # Base region 4
        qb4 = lambda z: -Vz[i] * t_sp * z ** 2 / (2 * Ixx)
        I4 = qb4(h)
        s4=np.arange(0, h+ 0.1, 0.1)

        # Base region 5
        qb5 = lambda z: -Vz[i] * tarr[i] / Ixx * (0.5 * h * z - 0.5 * 0.5 * h * z ** 2 / l_sk) + I3 + I4
        I5 = qb5(l_sk)

        # Base region 6
        qb6 = lambda z: Vz[i] * tarr[i] / Ixx * 0.5 * 0.5 * h / l_sk * z ** 2 + I5
        I6 = qb6(l_sk)

        # Base region 7
        qb7 = lambda z: -Vz[i] * t_sp * 0.5 * z ** 2 / Ixx
        I7 = qb7(-h)


        # Base region 8
        qb8 = lambda z: -Vz[i] * 0.5 * h * t_sp * z / Ixx + I6 - I7
        I8 = qb8(0.6 * c)

        # Base region 9
        qb9 = lambda z: -Vz[i] * 0.5 * t_sp * z ** 2 / Ixx
        I9 = qb9(-h)

        # Base region 10
        qb10 = lambda z: -Vz[i] * tarr[i] * (0.5 * h) ** 2 * (np.cos(z) - 1) / Ixx + I8 - I9

        #Torsion
        A1 = float(np.pi*h*c*0.15*0.5)
        A2 = float(h*0.6*c)
        A3 = float(h*0.25*c)

        T_A11 = 0.5 * A1 * self.perimiter_ellipse(h,0.15*c) * 0.5 * tarr[i]
        T_A12 = -A1 * h * t_sp
        T_A13 = 0
        T_A14 = -1/(0.5*self.shear_modulus)

        T_A21 = -A2 * h * t_sp
        T_A22 = A2 * h * t_sp * 2 + c*0.6*2*A2*tarr[i]
        T_A23 = -h*A2*t_sp
        T_A24 = -1/(0.5*self.shear_modulus)

        T_A31 = 0
        T_A32 = -A3 * h *t_sp
        T_A33 = A3 * h * t_sp + l_sk*A3*tarr[i]*2
        T_A34 = -1/(0.5*self.shear_modulus)

        T_A41 = 2*A1
        T_A42 = 2*A2
        T_A43 = 2*A3
        T_A44 = 0

        T_A = np.array([[T_A11, T_A12, T_A13, T_A14], [T_A21, T_A22, T_A23, T_A24], [T_A31, T_A32, T_A33, T_A34],[T_A41,T_A42,T_A43,T_A44]])
        T_B = np.array([0,0,0,T[i]])
        T_X = np.linalg.solve(T_A, T_B)



        # Redundant shear flow
        A11 = pi * (0.5 * h) / tarr[i] + h / t_sp
        A12 = -h / t_sp
        A21 = - h / t_sp
        A22 = 1.2 * c / tarr[i]
        A23 = -h / t_sp
        A32 = - h / t_sp
        A33 = 2 * l_sk / tarr[i] + h / t_sp



        B1 = 0.5 * h / tarr[i] * trapz([qb1(0),qb1(pi/2)], [0, pi / 2]) + trapz([qb2(0),qb2(0.5*h)], [0, 0.5 * h]) / t_sp - trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0])/ t_sp + trapz([qb10(-pi/2),qb10(0)], [-pi / 2, 0]) * 0.5 * h / tarr[i]
        B2 = trapz([qb2(0),qb2(0.5*h)], [0, 0.5 * h]) / t_sp + trapz([qb3(0),qb3(0.6*c)], [0, 0.6 * c]) / tarr[i] - trapz([qb7(-0.5*h),qb7(0)], [-0.5 * h, 0]) / t_sp + \
                trapz([qb4(0),qb4(0.5*h)], [0, 0.5 * h]) / t_sp + trapz([qb8(0),qb8(0.6*c)], [0, 0.6 * c]) / tarr[i] - trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0]) / t_sp
        B3 = trapz([qb5(0),qb5(l_sk)], [0, l_sk]) / tarr[i] + trapz([qb6(0),qb6(l_sk)], [0, l_sk]) / tarr[i] + trapz([qb4(0),qb4(0.5*h)], [0, 0.5 * h]) / t_sp - \
                trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0]) / t_sp

        A = np.array([[A11, A12, 0], [A21, A22, A23], [0, A32, A33]])
        B = -np.array([[B1], [B2], [B3]])
        X = np.linalg.solve(A, B)

        q01 = float(X[0])
        q02 = float(X[1])
        q03 = float(X[2])

        qT1 = float(T_X[0])
        qT2 = float(T_X[1])
        qT3 = float(T_X[1])

        # Compute final shear flow
        q2 = qb2(s2) - q01 - qT1 + q02 + qT2
        q3 = qb3(s3) + q02 + qT2
        q4 = qb4(s4) + q03 +qT3 - q02 - qT2

        max_region2 = max(q2)
        max_region3 = max(q3)
        max_region4 = max(q4)
        determine = max(max_region2, max_region3, max_region4)
        Nxy[i] = determine
        return Nxy

        


#-------Own formulas-------
'''


    def m(self,t_sp, h_st,t_st,t_sk):
        f = self.skin_interpolation(t_sp, h_st,t_st,t_sk)
        sta = self.get_y_points()
        rbw = self.rib_weight()

        f2 = interp1d(sta, f)

        rib_w = np.zeros(len(sta))
        moment = np.zeros(len(sta))

        for i in range(len(rib_w)):
            rib_w[i] = rbw(sta[i])
        for i in range(1, len(sta)):
            cursor = sta[i] * np.ones(len(sta))
            diff = np.subtract(cursor, sta)
            d = diff > 0
            diff = diff[d]
            rib_w = np.flip(rib_w)
            l = len(diff)
            rib_w = rib_w[0:l]
            produ = np.multiply(rib_w, diff)
            s = np.sum(produ)
            f3=trapz(f2(np.linspace(0,diff[0],10)),np.linspace(0,diff[0],10))
            moment[i] = 9.81 * f3 + 9.81 * s
        moment = np.flip(moment)
        return moment




    def m_eng(self,t_sp,h_st,t_st,t_sk):
        moment = self.m(t_sp, h_st,t_st,t_sk)
        x = self.get_y_points()
        f = interp1d(x, moment, kind='quadratic')

        x_engine = np.array([0.5 * self.span / 4, 0.5 * self.span / 2, 0.5 * 3 * self.span / 4])
        x_combi = np.concatenate((x, x_engine))
        x_sort = np.sort(x_combi)

        index1 = np.where(x_sort == 0.5 * 3 * self.span / 4)
        if len(index1[0]) == 1:
            index1 = int(index1[0])
        else:
            index1 = int(index1[0][0])
        y_new1 = f(x_sort[index1])

        index2 = np.where(x_sort == 0.5 * self.span / 2)
        if len(index2[0]) == 1:
            index2 = int(index2[0])
        else:
            index2 = int(index2[0][0])
        y_new2 = f(x_sort[index2])

        index3 = np.where(x_sort == 0.5 * self.span / 4)
        if len(index3[0]) == 1:
            index3 = int(index3[0])
        else:
            index3 = int(index3[0][0])
        y_new3 = f(x_sort[index3])

        y_engine = np.ndarray.flatten(np.array([y_new1, y_new2, y_new3]))
        y_combi = np.concatenate((moment, y_engine))
        y_sort = np.sort(y_combi)
        y_sort = np.flip(y_sort)

        for i in range(int(index1)):
            y_sort[i] = y_sort[i] + 9.81 * self.engine_weight * (0.5 * 3 * self.span / 4 - x_sort[i])
        for i in range(int(index2)):
            y_sort[i] = y_sort[i] + 9.81 * self.engine_weight * (0.5 * 2 * self.span / 4 - x_sort[i])
        for i in range(int(index3)):
            y_sort[i] = y_sort[i] + 9.81 * self.engine_weight * (0.5 * self.span / 4 - x_sort[i])

        return x_sort, y_sort





    def N_y(self, t_sp, h_st,t_st,t_sk):
        """ Check this function thoroughly

        """    
        sta = self.get_y_points()
        x_sort, moment = self.m_eng(t_sp,h_st,t_st,t_sk)
        h = self.height()

        Nx = np.zeros(len(sta))

        index1 = np.where(x_sort == 0.5 * 3 * self.span / 4)
        if len(index1[0]) == 1:
            index1 = int(index1[0])
        else:
            index1 = int(index1[0][0])

        index2 = np.where(x_sort == 0.5 * self.span / 2)
        if len(index2[0]) == 1:
            index2 = int(index2[0])
        else:
            index2 = int(index2[0][0])

        index3 = np.where(x_sort == 0.5 * self.span / 4)
        if len(index3[0]) == 1:
            index3 = int(index3[0])
        else:
            index3 = int(index3[0][0])

        moment = np.delete(moment, np.array([index1, index2, index3]))
        bend_stress=np.zeros(len(sta))
        for i in range(len(sta)):
            Ixx = self.I_xx(t_sp,h_st,t_st,t_sk)(sta[i])
            bend_stress[i] = moment[i] * 0.5 * h(sta[i]) / Ixx
            Nx[i] = bend_stress[i] * t_sk[i]
        return  Nx, bend_stress







    def shear_force(self,t_sp, h_st,t_st,t_sk):
        shear = self.shear_eng(t_sp, h_st, t_st, t_sk)[1]
        Vz = np.zeros(len(sta))

        sta = self.get_y_points()
        for i in range(len(sta)):
            Vz[i] = self.aero(sta[i])-shear[2 * i]
        return Vz

    def perimiter_ellipse(self,a,b):
        return float(np.pi *  ( 3*(a+b) - np.sqrt( (3*a + b) * (a + 3*b) ) )) #Ramanujans first approximation formula

    def torsion_sections(self,t_sk):
        wing = self.wing
        engine = self.engine
        ch = self.chord()
        sta = self.get_y_points()
        T = np.zeros(len(tarr))
        engine_weight = engine.mass_pertotalengine
        x_centre_wb = lambda x_w: wing.x_lemac + self.chord_root*0.25* + ch(x_w)*0.20
        for i in range(len(tarr)):
            if sta[i]< float(engine.y_rotor_loc[0]):
                T[i] = engine_weight * 9.81 * (x_centre_wb(engine.x_rotor_loc[0])-engine.x_rotor_loc[0]) + engine_weight * 9.81 * (x_centre_wb(engine.x_rotor_loc[2])-engine.x_rotor_loc[2])
            else:
                T[i] = engine_weight * 9.81 * (x_centre_wb(engine.x_rotor_loc[0])-engine.x_rotor_loc[0])
        #     print(sta[i],y_rotor_loc[0],x_centre_wb(engine.x_rotor_loc[0]))
        # print(f"\n\nT = {T}\n\n")
        return T

    def N_xy(self, t_sp, h_st,t_st,t_sk):
        engine = self.engine
        h1 = self.height()
        ch = self.chord()
        sta = self.get_y_points()
        Vz=self.shear_force(t_sp, h_st,t_st,t_sk)
        T =self.torsion_sections(t_sk)
        Nxy = np.zeros(len(tarr))

        for i in range(len(tarr)):
            Ixx1 = self.I_xx(t_sp,h_st,t_st,tarr[i])
            Ixx = Ixx1(sta[i])
            h = h1(sta[i])
            l_sk = sqrt(h ** 2 + (0.25 * self.chord_root) ** 2)
            c = ch(sta[i])

            # Base region 1
            qb1 = lambda z: Vz[i] * tarr[i] * (0.5 * h) ** 2 * (np.cos(z) - 1) / Ixx
            I1 = qb1(pi / 2)

            # Base region 2
            qb2 = lambda z: -Vz[i] * t_sp * z ** 2 / (2 * Ixx)
            I2 = qb2(h)
            s2 = np.arange(0, h+ 0.1, 0.1)

            # Base region 3
            qb3 = lambda z: - Vz[i] * tarr[i] * (0.5 * h) * z / Ixx + I1 + I2
            I3 = qb3(0.6 * c)
            s3 = np.arange(0, 0.6*c+ 0.1, 0.1)

            # Base region 4
            qb4 = lambda z: -Vz[i] * t_sp * z ** 2 / (2 * Ixx)
            I4 = qb4(h)
            s4=np.arange(0, h+ 0.1, 0.1)

            # Base region 5
            qb5 = lambda z: -Vz[i] * tarr[i] / Ixx * (0.5 * h * z - 0.5 * 0.5 * h * z ** 2 / l_sk) + I3 + I4
            I5 = qb5(l_sk)

            # Base region 6
            qb6 = lambda z: Vz[i] * tarr[i] / Ixx * 0.5 * 0.5 * h / l_sk * z ** 2 + I5
            I6 = qb6(l_sk)

            # Base region 7
            qb7 = lambda z: -Vz[i] * t_sp * 0.5 * z ** 2 / Ixx
            I7 = qb7(-h)


            # Base region 8
            qb8 = lambda z: -Vz[i] * 0.5 * h * t_sp * z / Ixx + I6 - I7
            I8 = qb8(0.6 * c)

            # Base region 9
            qb9 = lambda z: -Vz[i] * 0.5 * t_sp * z ** 2 / Ixx
            I9 = qb9(-h)

            # Base region 10
            qb10 = lambda z: -Vz[i] * tarr[i] * (0.5 * h) ** 2 * (np.cos(z) - 1) / Ixx + I8 - I9

            #Torsion
            A1 = float(np.pi*h*c*0.15*0.5)
            A2 = float(h*0.6*c)
            A3 = float(h*0.25*c)

            T_A11 = 0.5 * A1 * self.perimiter_ellipse(h,0.15*c) * 0.5 * tarr[i]
            T_A12 = -A1 * h * t_sp
            T_A13 = 0
            T_A14 = -1/(0.5*self.shear_modulus)

            T_A21 = -A2 * h * t_sp
            T_A22 = A2 * h * t_sp * 2 + c*0.6*2*A2*tarr[i]
            T_A23 = -h*A2*t_sp
            T_A24 = -1/(0.5*self.shear_modulus)

            T_A31 = 0
            T_A32 = -A3 * h *t_sp
            T_A33 = A3 * h * t_sp + l_sk*A3*tarr[i]*2
            T_A34 = -1/(0.5*self.shear_modulus)

            T_A41 = 2*A1
            T_A42 = 2*A2
            T_A43 = 2*A3
            T_A44 = 0

            T_A = np.array([[T_A11, T_A12, T_A13, T_A14], [T_A21, T_A22, T_A23, T_A24], [T_A31, T_A32, T_A33, T_A34],[T_A41,T_A42,T_A43,T_A44]])
            T_B = np.array([0,0,0,T[i]])
            T_X = np.linalg.solve(T_A, T_B)



            # Redundant shear flow
            A11 = pi * (0.5 * h) / tarr[i] + h / t_sp
            A12 = -h / t_sp
            A21 = - h / t_sp
            A22 = 1.2 * c / tarr[i]
            A23 = -h / t_sp
            A32 = - h / t_sp
            A33 = 2 * l_sk / tarr[i] + h / t_sp



            B1 = 0.5 * h / tarr[i] * trapz([qb1(0),qb1(pi/2)], [0, pi / 2]) + trapz([qb2(0),qb2(0.5*h)], [0, 0.5 * h]) / t_sp - trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0])/ t_sp + trapz([qb10(-pi/2),qb10(0)], [-pi / 2, 0]) * 0.5 * h / tarr[i]
            B2 = trapz([qb2(0),qb2(0.5*h)], [0, 0.5 * h]) / t_sp + trapz([qb3(0),qb3(0.6*c)], [0, 0.6 * c]) / tarr[i] - trapz([qb7(-0.5*h),qb7(0)], [-0.5 * h, 0]) / t_sp + \
                 trapz([qb4(0),qb4(0.5*h)], [0, 0.5 * h]) / t_sp + trapz([qb8(0),qb8(0.6*c)], [0, 0.6 * c]) / tarr[i] - trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0]) / t_sp
            B3 = trapz([qb5(0),qb5(l_sk)], [0, l_sk]) / tarr[i] + trapz([qb6(0),qb6(l_sk)], [0, l_sk]) / tarr[i] + trapz([qb4(0),qb4(0.5*h)], [0, 0.5 * h]) / t_sp - \
                 trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0]) / t_sp

            A = np.array([[A11, A12, 0], [A21, A22, A23], [0, A32, A33]])
            B = -np.array([[B1], [B2], [B3]])
            X = np.linalg.solve(A, B)

            q01 = float(X[0])
            q02 = float(X[1])
            q03 = float(X[2])

            qT1 = float(T_X[0])
            qT2 = float(T_X[1])
            qT3 = float(T_X[1])

            # Compute final shear flow
            q2 = qb2(s2) - q01 - qT1 + q02 + qT2
            q3 = qb3(s3) + q02 + qT2
            q4 = qb4(s4) + q03 +qT3 - q02 - qT2

            max_region2 = max(q2)
            max_region3 = max(q3)
            max_region4 = max(q4)
            determine = max(max_region2, max_region3, max_region4)
            Nxy[i] = determine
        return Nxy

    def local_buckling(self,t):#TODO
        buck = 4* pi ** 2 * self.E / (12 * (1 - self.poisson ** 2)) * (t / self.str_pitch) ** 2
        return buck


    def flange_buckling(self,t_st, w_st):#TODO
        buck = 2 * pi ** 2 * self.E / (12 * (1 - self.poisson ** 2)) * (t_st / w_st) ** 2
        return buck


    def web_buckling(self,t_st, h_st):#TODO
        buck = 4 * pi ** 2 * self.E / (12 * (1 - self.poisson ** 2)) * (t_st / h_st) ** 2
        return buck


    def global_buckling(self, h_st,t_st,t):#TODO
        # n = n_st(c_r, b_st)
        tsmr = (t * self.str_pitch + t_st * self.n_max * (h_st - t)) / self.str_pitch
        return 4 * pi ** 2 * self.E / (12 * (1 - self.poisson ** 2)) * (tsmr / self.str_pitch) ** 2


    def shear_buckling(self,t):#TODO
        buck = 5.35 * pi ** 2 * self.E / (12 * (1 - self.poisson)) * (t / self.str_pitch) ** 2
        return buck



    def buckling(self, t_sp, h_st,t_st,t_sk):#TODO
        Nxy = self.N_xy(t_sp, h_st,t_st,t_sk)
        Nx = self.N_x(t_sp, h_st,t_st,t_sk)[0]
        buck = np.zeros(len(tarr))
        for i in range(len(tarr)):
            Nx_crit = self.local_buckling(tarr[i])*tarr[i]
            Nxy_crit = self.shear_buckling(tarr[i])*tarr[i]
            buck[i] = Nx[i] / Nx_crit + (Nxy[i] / Nxy_crit) ** 2
        return buck
    




    def column_st(self, h_st,t_st,t_sk):#TODO
        #Lnew=new_L(b,L)
        Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 + t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
        i= pi ** 2 * self.E * Ist / (2*w_st* self.max_rib_pitch ** 2)
        return i


    def f_ult(self,h_st,t_st,t_sk):
        A_st = self.area_st(h_st,t_st)
        # n=n_st(c_r,b_st)
        c=self.chord()
        h=self.height()
        stations= self.get_y_points() #FIXME change this to an input 
        f_uts=np.zeros(len(tarr))
        for i in range(len(tarr)):
            A=self.n_max*A_st+0.6*c(stations[i])*tarr[i]
            f_uts[i]=self.sigma_uts*A
        return f_uts




    def buckling_constr(self, t_sp, h_st,t_st,t_sk):
        buck = self.buckling(t_sp, h_st,t_st,t_sk)#TODO
        vector = np.zeros(len(tarr))
        for i in range(len(tarr)):
            vector[i] = -1 * (buck[i] - 1)
        return vector


    def global_local(self, h_st,t_st,t_sk):
        diff = np.zeros(len(tarr))
        for i in range(len(tarr)):
            glob = self.global_buckling(h_st,t_st,tarr[i])
            loc = self.local_buckling(tarr[i])
            diff[i] = glob - loc #FIXEM glob
        #diff = self.global_buckling(h_st,t_st,tarr)  - self.local_buckling(tarr,b_st)

        return diff



    def local_column(self, h_st,t_st,t_sk):
        diff = np.zeros(len(tarr))
        for i in range(len(tarr)):
            col=self.column_st(h_st,t_st, tarr[i])
            loc = self.local_buckling(tarr[i])*tarr[i]
            diff[i] = col - loc
        return diff


    def flange_loc_loc(self, t_st,t_sk,h_st):
        w_st = h_st*0.5
        diff = np.zeros(len(tarr))
        flange = self.flange_buckling(t_st,w_st)
        for i in range(len(tarr)):
            loc = self.local_buckling(tarr[i])
            diff[i] = flange - loc
        return diff


    def web_flange(self, h_st,t_st,t_sk):
        diff = np.zeros(len(tarr))
        web = self.web_buckling(t_st, h_st)
        for i in range(len(tarr)):
            loc = self.local_buckling(tarr[i])
            diff[i] =web-loc
        return diff


    def von_Mises(self, t_sp, h_st,t_st,t_sk):
        # vm = np.zeros(len(tarr))
        Nxy=self.N_xy(t_sp, h_st,t_st,t_sk)
        bend_stress=self.N_x(t_sp, h_st,t_st,t_sk)[1] #
        tau_shear_arr = Nxy/tarr
        vm_lst = self.sigma_yield - np.sqrt(0.5 * (3 * tau_shear_arr ** 2+bend_stress**2))
        # for i in range(len(tarr)):
        #     tau_shear= Nxy[i] / tarr[i]
        #     vm[i]=sigma_yield-sqrt(0.5 * (3 * tau_shear ** 2+bend_stress[i]**2))
        return vm_lst



    def crippling(self, h_st,t_st,t_sk):
        crip= np.zeros(len(tarr))
        A = self.area_st(h_st, t_st)
        for i in range(len(tarr)):
            col = self.column_st( h_st,t_st,tarr[i])
            crip[i] = t_st * self.beta * self.sigma_yield* ((self.g * t_st ** 2 / A) * sqrt(self.E / self.sigma_yield)) ** self.m_crip - col
        return crip


    def post_buckling(self, t_sp, h_st,t_st,t_sk):
        f = self.f_ult(h_st,t_st,t_sk)
        ratio=2/(2+1.3*(1-1/self.pb))
        px= self.n_max*self.shear_force(t_sp, h_st,t_st,t_sk)
        diff=np.subtract(ratio*f,px)
        return diff


    # def compute_volume():
    #     pass


        # def __init__(self, **kwargs):
        #     super().__init__(**kwargs)


class Wingbox_optimization(om.ExplicitComponent):
    def __init__(self, wing, engine, material, aero, **kwargs):
        super().__init__(**kwargs)
        self.wing =  wing
        self.engine = engine
        self.material = material
        self.WingboxClass = Wingbox(wing,engine , material, aero)
        self.constr_lst = ["global_local", "post_buckling", "von_mises", "buckling_constr", "flange_loc_loc", "local_column", "crippling", "web_flange"]




    def setup(self):

        # Design variables
        self.add_input('tsp')
        self.add_input('hst')
        self.add_input('tst')
        self.add_input('tsk')

        # Constant inputs
        # self.add_input('b')
        # self.add_input('c_r')
        
        #Outputs used as constraints
        shape_outputs = self.WingboxClass.n_sections
        self.add_output('wing_weight')
        self.add_output('global_local',shape = (shape_outputs,))
        self.add_output('post_buckling',shape = (shape_outputs,))
        self.add_output('von_mises',shape = (shape_outputs,))
        self.add_output('buckling_constr',shape = (shape_outputs,))
        self.add_output('flange_loc_loc',shape = (shape_outputs,))
        self.add_output('local_column',shape = (shape_outputs,))
        self.add_output('crippling',shape = (shape_outputs,))
        self.add_output("web_flange",shape = (shape_outputs,))
        self.declare_partials('*', '*', method= 'fd')


    # def setup_partials(self):

    #     # Partial derivatives are done using finite difference
    #     self.declare_partials('*', '*', 'fd')

    def compute(self, inputs, outputs):

        # Design variables
        tsp = inputs['tsp'][0]
        hst = inputs['hst'][0]
        tst = inputs['tst'][0]
        tsk = inputs['tsk'][0]

        # Constants
        # span = inputs['b'][0]
        # chord_root = inputs['c_r'][0]


        weight = self.WingboxClass.wing_weight(tsp, hst, tst,tsk)
        constr = [
        self.WingboxClass.global_local( hst, tst, tsk),
        self.WingboxClass.post_buckling(tsp, hst,tst,tsk),
        self.WingboxClass.von_Mises(tsp, hst,tst,tsk),
        self.WingboxClass.buckling_constr(  tsp,  hst, tst,tsk),
        self.WingboxClass.flange_loc_loc(tst,tsk,hst),
        self.WingboxClass.local_column(hst,tst,tsk),
        self.WingboxClass.crippling(hst, tst, tsk), #ONLY
        self.WingboxClass.web_flange(hst,tst,tsk)
        ]
        #print(constr)





        outputs['wing_weight'] = weight
        outputs['global_local'][:] = constr[0]
        outputs['post_buckling'][:] = constr[1]
        outputs['von_mises'][:] = constr[2]
        outputs['buckling_constr'][:] = constr[3]
        outputs['flange_loc_loc'][:] = constr[4]
        outputs['local_column'][:] = constr[5]
        outputs['crippling'][:] = constr[6]
        outputs['web_flange'][:] = constr[7]

    

        print('===== Progress update =====')
        print(f"Current weight = {weight} [kg]")
        #print(f"The failing constraints were {str_lst[np.array(constr) < 0]}")



def WingboxOptimizer(x, wing, engine, material, aero):
    """ sets up optimziation procedure and runs the driver

    :param x: Initial estimate X = [tsp, trib, hst, tst, wst, tmax, tmin]
    :type x: nd.array
    :param wing: wing class from data structure
    :type wing: wing class
    :param engine: engine class from data structures
    :type engine: engine class
    """    

    OptClass = Wingbox_optimization(wing, engine, material, aero)


    prob = om.Problem()
    prob.model.add_subsystem('wingbox_design', OptClass)#, promotes_inputs=['AR1',
                                                                                        # 'AR2',

    # Initial values for the optimization 

    #Constants
    # prob.model.set_input_defaults('wingbox_design.b', wing.span)
    # prob.model.set_input_defaults('wingbox_design.c_r', wing.chord_root)
    # prob.model.set_input_defaults('wingbox_design.engine', engine)
    # prob.model.set_input_defaults('wingbox_design.wing', wing)

   


    prob.model.add_design_var('wingbox_design.tsp', lower = 0.001, upper= 0.1)
    prob.model.add_design_var('wingbox_design.hst', lower = 0.001 , upper= 0.4)
    prob.model.add_design_var('wingbox_design.tst', lower = 0.001, upper= 0.1)
    prob.model.add_design_var('wingbox_design.tsk', lower = 0.001, upper= 0.1)

    # Define constraints 
    prob.model.add_constraint('wingbox_design.global_local', lower=10.0e6)#Local skin buckling < Global skin buckling
    prob.model.add_constraint('wingbox_design.post_buckling', lower=1000)# Post buckling
    prob.model.add_constraint('wingbox_design.von_mises', lower=10.0e6) #Von mises
    prob.model.add_constraint('wingbox_design.buckling_constr', lower=0.)#combined compression and shear load buckling
    prob.model.add_constraint('wingbox_design.flange_loc_loc', lower=10.0e6) #Local sking buckling<stringer flange buckling 
    prob.model.add_constraint('wingbox_design.local_column', lower=10.0e6) #Local sking buckling<column stringer buckling
    prob.model.add_constraint('wingbox_design.crippling', lower=0.) # Crippling
    prob.model.add_constraint('wingbox_design.web_flange', lower=10.0e6)#Local skin buckling < stringer flange buckling

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.opt_settings['maxiter'] = 1000

    prob.model.add_objective('wingbox_design.wing_weight')

    prob.setup()

     # Initial estimate for the design variables
    prob.set_val('wingbox_design.tsp', x[0])
    prob.set_val('wingbox_design.hst', x[1])
    prob.set_val('wingbox_design.tst', x[2])
    prob.set_val('wingbox_design.tsk', x[3])

    #prob.model.list_inputs(True)

    prob.run_driver()
    #prob.check_partials(compact_print=True)
    #prob.check_totals()

    #prob.model.list_outputs()

    print(f"thickness spar = {prob.get_val('wingbox_design.tsp')*1000} [mm]")
    print(f"stringer height = {prob.get_val('wingbox_design.hst')*1000} [mm]")
    print(f"stringer thickness = {prob.get_val('wingbox_design.tst')*1000} [mm]")
    print(f"skin thickness = {prob.get_val('wingbox_design.tsk')*1000} [mm]")
    print(f"Wing weight= {prob.get_val('wingbox_design.wing_weight')} [kg]")


    output_lst =  np.array([
    prob.get_val('wingbox_design.tsp'),
    prob.get_val('wingbox_design.hst'),
    prob.get_val('wingbox_design.tst'),
    prob.get_val('wingbox_design.tsk')
    ])

    return output_lst

'''