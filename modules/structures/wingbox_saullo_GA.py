from math import *
import sys
import pathlib as pl
import os

import numpy as np
from scipy.integrate import trapz
from scipy.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as minimizeGA

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from input.data_structures.GeneralConstants import *
#$from input.data_structures.aero import Aero
from input.data_structures.engine import Engine
from input.data_structures.material import Material
from input.data_structures.wing import Wing
#$from modules.aero.avl_access import get_lift_distr

#------------ASSUMPTION----------
#Ribs carry no structural load. They only limit the buckling constraint on the stringers. So their thickness will be assumed to be 3 mm and their pitch will be optimized.
#Each stringer is assumed to be effective up until it's centerpoint is out of the center point of the spar.


class Wingbox():
    def __init__(self, wing, engine, material, aero, HOVER):
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
        #self.lift_func = get_lift_distr(wing, aero)
        self.engine = engine
        self.wing = wing
        self.t_rib = 4e-3 #[mm]
        self.x_lewing = wing.x_lewing
        self.thickness_to_chord = wing.thickness_to_chord
        self.hover_bool = HOVER


        #Wing
        self.taper = wing.taper
        self.n_max = n_max_req
        self.span = wing.span
        self.chord_root = wing.chord_root
        self.n_ribs = 5
        self.rib_pitch = (self.span/2)/(self.n_ribs-1)

        #Engine
        self.engine_weight = engine.mass_pertotalengine
        self.engine.dump()
        self.y_rotor_loc = engine.y_rotor_loc
        self.x_rotor_loc = engine.x_rotor_loc
        self.thrust_per_engine = engine.thrust_per_engine


        self.nacelle_w = engine.nacelle_width #TODO Check if it gets updated

        #STRINGERS
        self.n_str = 4
        self.str_array_root = np.linspace(0.15*self.chord_root,0.75*self.chord_root,self.n_str+2)

        self.y_mesh = 1000

        #GEOMETRY
        self.width = 0.6*self.chord_root
        self.str_pitch = 0.6*self.chord_root/(self.n_str+1) #THE PROGRAM ASSUMES THERE ARE TWO STRINGERS AT EACH END AS WELL

        #OPT related
        self.y = np.linspace(0, self.span/2, 10)


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

    def I_xx(self, x):#TODO implement dissappearing stringers
        t_sp, h_st, t_st, t_sk = x
        h = self.height(self.y)
        # nst = n_st(c_r, b_st)
        Ist = self.I_st_x(h_st, t_st)
        Isp = self.I_sp_x(t_sp, self.y)
        A = self.get_area_str(h_st,t_st)
        return 2 * (Ist + A * (0.5 * h) ** 2) * self.n_str + 2 * Isp + 2 * (0.6 * self.chord(self.y) * t_sk ** 3 / 12 + t_sk * 0.6 * self.chord_root * (0.5 * h) ** 2)

    def I_zz(self, x):#TODO implement dissappearing stringers
        y = self.y
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
        #return  self.lift_func(y)
        return -151.7143*9.81*y + 531*9.81

    def weight_from_tip(self, x):#TODO implement dissappearing stringers #TODO Include rib weights
        t_sp, h_st, t_st, t_sk = x
        y = self.y
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


    def total_weight(self, x):
        return self.weight_from_tip(x)[0]


    def shear_z_from_tip(self, x):
        if self.hover_bool:
            return -self.weight_from_tip(x)*9.81 + 2 * self.thrust_per_engine
        else:
            return -self.weight_from_tip(x)*9.81 + self.aero(self.y)

    def torque_from_tip(self,y):
        difference_array = np.absolute(y-self.y_rotor_loc[0])
        index = difference_array.argmin()
        if self.hover_bool == True:
            torque_array = np.zeros(index+1) + (self.thrust_per_engine- self.engine_weight*9.81)*(self.x_rotor_loc[0]-self.get_x_start_wb(y[index]))
            torque_array = np.append(torque_array,np.zeros(len(y)-len(torque_array)))
            return torque_array
        else:
            torque_array = np.zeros(index+1) + (self.engine_weight*9.81)*(self.x_rotor_loc[0]-self.get_x_start_wb(y[index]))
            torque_array = np.append(torque_array,np.zeros(len(y)-len(torque_array)))
            return torque_array


    def moment_x_from_tip(self, x):
        return self.weight_from_tip(x)*9.81*(self.span/2 - self.y)

#-----------Stress functions---------
    def bending_stress_x_from_tip(self, x):
        return self.moment_x_from_tip(x)/self.I_xx(x) * self.height(self.y)/2

    def shearflow_max_from_tip(self, x):
        t_sp, h_st, t_st, t_sk = x
        y = self.y
        Vz = self.shear_z_from_tip(x)
        T = self.torque_from_tip(y)
        Ixx = self.I_xx(x)
        height = self.height(y)
        chord = self.chord(y)
        Nxy = np.zeros(len(y))
        l_sk = self.l_sk(y)
        for i in range(len(y)):
            # Base region 1
            def qb1(z):
                return Vz[i] * t_sk * (0.5 * height[i]) ** 2 * (np.cos(z) - 1) / Ixx[i]
            I1 = qb1(pi / 2)

            # Base region 2
            def qb2(z):
                return -Vz[i] * t_sp * z ** 2 / (2 * Ixx[i])
            I2 = qb2(height[i])
            s2 = np.arange(0, height[i]+ 0.1, 0.1)

            # Base region 3
            def qb3(z):
                return - Vz[i] * t_sk * (0.5 * height[i]) * z / Ixx[i] + I1 + I2
            I3 = qb3(0.6 * chord[i])
            s3 = np.arange(0, 0.6*chord[i]+ 0.1, 0.1)

            # Base region 4
            def qb4(z):
                return -Vz[i] * t_sp * z ** 2 / (2 * Ixx[i])
            I4 = qb4(height[i])
            s4=np.arange(0, height[i]+ 0.1, 0.1)

            # Base region 5
            def qb5(z):
                return -Vz[i] * t_sk / Ixx[i] * (0.5 * height[i] * z - 0.5 * 0.5 * height[i] * z ** 2 / l_sk[i]) + I3 + I4
            I5 = qb5(l_sk[i])

            # Base region 6
            def qb6(z):
                return Vz[i] * t_sk / Ixx[i] * 0.5 * 0.5 * height[i] / l_sk[i] * z ** 2 + I5
            I6 = qb6(l_sk[i])

            # Base region 7
            def qb7(z):
                return -Vz[i] * t_sp * 0.5 * z ** 2 / Ixx[i]
            I7 = qb7(-height[i])


            # Base region 8
            def qb8(z):
                return -Vz[i] * 0.5 * height[i] * t_sp * z / Ixx[i] + I6 - I7
            I8 = qb8(0.6 * chord[i])

            # Base region 9
            def qb9(z):
                return -Vz[i] * 0.5 * t_sp * z ** 2 / Ixx[i]
            I9 = qb9(-height[i])

            # Base region 10
            def qb10(z):
                return -Vz[i] * t_sk * (0.5 * height[i]) ** 2 * (np.cos(z) - 1) / Ixx[i] + I8 - I9

            #Torsion
            A1 = float(np.pi*height[i]*chord[i]*0.15*0.5)
            A2 = float(height[i]*0.6*chord[i])
            A3 = float(height[i]*0.25*chord[i])

            T_A11 = 0.5 * A1 * self.perimiter_ellipse(height[i],0.15*chord[i]) * 0.5 * t_sk
            T_A12 = -A1 * height[i] * t_sp
            T_A13 = 0
            T_A14 = -1/(0.5*self.shear_modulus)

            T_A21 = -A2 * height[i] * t_sp
            T_A22 = A2 * height[i] * t_sp * 2 + chord[i]*0.6*2*A2*t_sk
            T_A23 = -height[i]*A2*t_sp
            T_A24 = -1/(0.5*self.shear_modulus)

            T_A31 = 0
            T_A32 = -A3 * height[i] *t_sp
            T_A33 = A3 * height[i] * t_sp + l_sk[i]*A3*t_sk*2
            T_A34 = -1/(0.5*self.shear_modulus)

            T_A41 = 2*A1
            T_A42 = 2*A2
            T_A43 = 2*A3
            T_A44 = 0

            T_A = np.array([[T_A11, T_A12, T_A13, T_A14], [T_A21, T_A22, T_A23, T_A24], [T_A31, T_A32, T_A33, T_A34],[T_A41,T_A42,T_A43,T_A44]])
            T_B = np.array([0,0,0,T[i]])
            T_X = np.linalg.solve(T_A, T_B)



            # Redundant shear flow
            A11 = pi * (0.5 * height[i]) / t_sk + height[i] / t_sp
            A12 = -height[i] / t_sp
            A21 = - height[i] / t_sp
            A22 = 1.2 * chord[i] / t_sk
            A23 = -height[i] / t_sp
            A32 = - height[i] / t_sp
            A33 = 2 * l_sk[i] / t_sk + height[i] / t_sp



            B1 = 0.5 * height[i] / t_sk * trapz([qb1(0),qb1(pi/2)], [0, pi / 2]) + trapz([qb2(0),qb2(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp - trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0])/ t_sp + trapz([qb10(-pi/2),qb10(0)], [-pi / 2, 0]) * 0.5 * height[i] / t_sk
            B2 = trapz([qb2(0),qb2(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp + trapz([qb3(0),qb3(0.6*chord[i])], [0, 0.6 * chord[i]]) / t_sk - trapz([qb7(-0.5*height[i]),qb7(0)], [-0.5 * height[i], 0]) / t_sp + \
                    trapz([qb4(0),qb4(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp + trapz([qb8(0),qb8(0.6*chord[i])], [0, 0.6 * chord[i]]) / t_sk - trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0]) / t_sp
            B3 = trapz([qb5(0),qb5(l_sk[i])], [0, l_sk[i]]) / t_sk + trapz([qb6(0),qb6(l_sk[i])], [0, l_sk[i]]) / t_sk + trapz([qb4(0),qb4(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp - \
                    trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0]) / t_sp

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

    def distrcompr_max_from_tip(self, x):
        t_sp, h_st, t_st, t_sk = x
        return self.bending_stress_x_from_tip(x) * t_sk


#-------Own formulas-------
    def local_buckling(self,t_sk):#TODO
        buck = 4* pi ** 2 * self.E / (12 * (1 - self.poisson ** 2)) * (t_sk / self.str_pitch) ** 2
        return buck


    def flange_buckling(self,t_st, w_st):#TODO
        buck = 2 * pi ** 2 * self.E / (12 * (1 - self.poisson ** 2)) * (t_st / w_st) ** 2
        return buck


    def web_buckling(self,t_st, h_st):#TODO
        buck = 4 * pi ** 2 * self.E / (12 * (1 - self.poisson ** 2)) * (t_st / h_st) ** 2
        return buck


    def global_buckling(self, h_st, t_st, t):#TODO
        # n = n_st(c_r, b_st)
        tsmr = (t * self.str_pitch + t_st * self.n_str * (h_st - t)) / self.str_pitch
        return 4 * pi ** 2 * self.E / (12 * (1 - self.poisson ** 2)) * (tsmr / self.str_pitch) ** 2


    def shear_buckling(self,t_sk):#TODO
        buck = 5.35 * pi ** 2 * self.E / (12 * (1 - self.poisson)) * (t_sk / self.str_pitch) ** 2
        return buck


    def buckling(self, x):#TODO
        t_sp, h_st, t_st, t_sk = x
        Nxy = self.shearflow_max_from_tip(x)
        Nx = self.distrcompr_max_from_tip(x)
        Nx_crit = self.local_buckling(t_sk)*t_sk
        Nxy_crit = self.shear_buckling(t_sk)*t_sk
        buck = Nx / Nx_crit + (Nxy / Nxy_crit) ** 2
        return buck


    def column_st(self, h_st, t_st, t_sk):#TODO
        w_st = 0.5*h_st
        #Lnew=new_L(b,L)
        Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 + t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
        i= pi ** 2 * self.E * Ist / (2*w_st* self.rib_pitch ** 2)#TODO IF HE FAILS REPLACE SPAN WITH RIB PITCH
        return i


    def f_ult(self, h_st,t_st,t_sk,y):
        A_st = self.get_area_str(h_st,t_st)
        # n=n_st(c_r,b_st)
        A=self.n_str*A_st+0.6*self.chord(y)*t_sk
        f_uts=self.sigma_uts*A
        return f_uts


    def buckling_constr(self, x):
        buck = self.buckling(x)
        return -1*(buck - 1)


    def global_local(self, x):
        t_sp, h_st, t_st, t_sk = x
        glob = self.global_buckling(h_st, t_st, t_sk)
        loc = self.local_buckling(t_sk)
        diff = glob - loc
        return diff


    def local_column(self, x):
        t_sp, h_st, t_st, t_sk = x
        col = self.column_st(h_st,t_st, t_sk)
        loc = self.local_buckling(t_sk)
        diff = col - loc
        return diff


    def flange_loc_loc(self, x):
        t_sp, h_st, t_st, t_sk = x
        w_st = h_st*0.5
        flange = self.flange_buckling(t_st,w_st)
        loc = self.local_buckling(t_sk)
        diff = flange - loc
        return diff


    def web_flange(self, x):
        t_sp, h_st, t_st, t_sk = x
        web = self.web_buckling(t_st, h_st)
        loc = self.local_buckling(t_sk)
        diff =web-loc
        return diff


    def von_Mises(self, x):
        y = self.y
        t_sp, h_st, t_st, t_sk = x
        Nxy=self.shearflow_max_from_tip(x)
        bend_stress=self.bending_stress_x_from_tip(x)
        tau_shear_arr = Nxy/t_sk
        vm_lst = self.sigma_yield - np.sqrt(0.5 * (3 * tau_shear_arr ** 2+bend_stress**2))
        return vm_lst


    def crippling(self, x):
        t_sp, h_st, t_st, t_sk = x
        A = self.get_area_str(h_st, t_st)
        col = self.column_st( h_st,t_st,t_sk)
        crip = t_st * self.beta * self.sigma_yield* ((self.g * t_st ** 2 / A) * sqrt(self.E / self.sigma_yield)) ** self.m_crip - col
        return crip


    def post_buckling(self, x):
        x = t_sp, h_st,t_st,t_sk,y
        f = self.f_ult(h_st,t_st,t_sk,y)
        ratio=2/(2+1.3*(1-1/self.pb))
        px= self.n_max*self.shear_z_from_tip(x)
        diff=np.subtract(ratio*f,px)
        return diff



if __name__ == '__main__':
    wing = Wing()
    wing.load()
    engine = Engine()
    engine.load()
    material = Material()
    material.load()
    #aero = Aero()
    #aero.load()

    wingbox = Wingbox(wing, engine, material, aero=None, HOVER=True)

    xlower = 5e-3, 1e-2, 1e-3, 5e-4
    xupper = 1e-1, 0.15, 1e-1, 1e-1

    bounds = np.vstack((xlower, xupper)).T

    tsp, hst, tst, tsk = 0.005, 0.1344, 0.01733, 5e-3
    X = [tsp, hst, tst, tsk]


    #NOTE GA optimizer to explore the design space
    objs = [wingbox.total_weight]

    constr_ieq = [
        lambda x: -wingbox.buckling_constr(x)[0],
        lambda x: -wingbox.local_column(x),
        lambda x: -wingbox.global_local(x),
        lambda x: -wingbox.von_Mises(x)[0],
        lambda x: -wingbox.flange_loc_loc(x),
        lambda x: -wingbox.crippling(x),
        lambda x: -wingbox.web_flange(x),
    ]

    problem = FunctionalProblem(len(X),
                                objs,
                                constr_ieq=constr_ieq,
                                xl=xlower,
                                xu=xupper,
                                )

    method = GA(pop_size=50, eliminate_duplicates=True)

    resGA = minimizeGA(problem, method, termination=('n_gen', 60), seed=1,
                     save_history=True, verbose=True)
    print('GA optimum variables', resGA.X)
    print('GA optimum weight', resGA.F)


    #NOTE final gradient descent to converget to a minimum point with SciPy minimize

    print()
    print('Final SciPy minimize optimization')
    options = dict(eps=1e-4, ftol=1e-3)
    constraints = [
        {'type': 'ineq', 'fun': lambda x: wingbox.buckling_constr(x)[0]},
        {'type': 'ineq', 'fun': lambda x: wingbox.local_column(x)},
        {'type': 'ineq', 'fun': lambda x: wingbox.global_local(x)},
        {'type': 'ineq', 'fun': lambda x: wingbox.von_Mises(x)[0]},
        {'type': 'ineq', 'fun': lambda x: wingbox.flange_loc_loc(x)},
        {'type': 'ineq', 'fun': lambda x: wingbox.crippling(x)},
        {'type': 'ineq', 'fun': lambda x: wingbox.web_flange(x)},
    ]
    resMin = minimize(wingbox.total_weight, resGA.X, method='SLSQP',
                   constraints=constraints, bounds=bounds, jac='3-point',
                   options=options)
    print(resMin)
