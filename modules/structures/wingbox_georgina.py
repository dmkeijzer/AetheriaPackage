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



class Wingbox():
    def __init__(self,wing,engine,material, aero):
        #Material
        self.poisson = material.poisson
        self.rho = material.rho
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

        #Wing
        self.taper = wing.taper
        self.n_max = n_max_req
        self.span = wing.span
        self.chord_root = wing.chord_root
       

        #Engine
        self.engine_weight = engine.mass_pertotalengine
        self.y_rotor_loc = engine.y_rotor_loc
        self.nacelle_w = engine.nacelle_width #TODO Check if it gets updated
        self.n_str = 10 
        #GEOMETRY
        self.width = 0.6*self.chord_root
        self.str_pitch = 0.6*self.chord_root/(self.n_str+2)
        
        #Set number of ribs in inboard and outboard section
        self.n_ribs_sec0 = 1 #Number of ribs inboard of inboard engine
        self.n_ribs_sec1 = 4 #Number of ribs inboard and outboard engines
        self.n_sections = self.n_ribs_sec0 + self.n_ribs_sec1 + 4
        #self.max_rib_pitch = np.max(np.diff(self.get_y_rib_loc()))
        self.max_rib_pitch = 1.05

    def aero(self, y):
        return  self.lift_func(y)

    def chord(self):
        c = lambda y: self.chord_root - self.chord_root * (1 - self.taper) * y * 2 / self.span
        c = lambda y: self.chord_root - self.chord_root * (1 - self.taper) * y * 2 / self.span
        return c

    #Determine rib positions in
    #  spanwise direction (y)
    def get_y_rib_loc(self):
        y_rib_0 = np.array([self.y_rotor_loc[0] - 0.5 * self.nacelle_w])
        y_rib_1 = np.array([self.y_rotor_loc[0] + 0.5 * self.nacelle_w])

        y_rib_2 = np.array([self.span/2])
        y_rib_sec0 = np.arange(0, y_rib_0, y_rib_0/(self.n_ribs_sec0 + 1))
        y_rib_sec1 = np.arange(y_rib_1,y_rib_2, (y_rib_2-y_rib_1)/(self.n_ribs_sec1 + 1))

        y_rib_loc = np.concatenate([y_rib_0,y_rib_2,y_rib_sec0,y_rib_sec1])

        y_rib_loc = np.sort(y_rib_loc)
        return y_rib_loc





    def height(self):
        c = self.chord()
        h = lambda Y: 0.17 * c(Y)
        return h


    def area_st(self, h_st,t_st,w_st):
        return t_st * (2 * w_st + h_st)




    def I_st(self, h_st,t_st,w_st):
        Ast = self.area_st(h_st, t_st, w_st)
        i = t_st * h_st ** 3 / 12 + w_st * t_st ** 3 / 12 + 2 * Ast * (0.5 * h_st) ** 2
        return i





    def w_sp(self):
        h = self.height()
        i = lambda z: 0.5 * h(z)
        return i




    def I_sp(self,t_sp):
        h = self.height()
        wsp = self.w_sp()
        i = lambda z: t_sp * (h(z) - 2 * t_sp) ** 3 / 12 + 2 * wsp(z) * t_sp ** 3 / 12 + 2 * t_sp * wsp(z) * (
                0.5 * h(z)) ** 2
        return i







    def I_xx(self,t_sp,h_st,t_st,w_st,tsk):
        h = self.height()
        # nst = n_st(c_r, b_st)
        Ist = self.I_st(h_st,t_st,w_st)
        Isp = self.I_sp(t_sp)
        A = self.area_st(h_st,t_st,w_st)
        i = lambda z: 2 * (Ist + A * (0.5 * h(z)) ** 2) * self.n_str + 2 * Isp(z) + 2 * (0.6 * self.chord_root * tsk ** 3 / 12 + tsk * 0.6 * self.chord_root * (0.5 * h(z)) ** 2)
        return i




    def t_arr(self,tmax, tmin):
        return np.linspace(tmax,tmin, self.n_ribs_sec0 + self.n_ribs_sec1 + 3)

        """ Replace function by our design variables, simplifies our process. List of thicknesses compatible with our sections. 
        # #TODO
        # - compatible with L

        # """    


    def rib_weight(self,t_rib):
        c = self.chord()
        h = self.height()
        w_rib = lambda z: 0.6 * c(z) * h(z) * t_rib * self.rho
        return w_rib



    def vol_func(self,z, th_sk, t_sp, h, c, A, nst):
        return self.rho * (2 * h(z) * t_sp + (pi * (3 * (0.5 * h(z) + 0.15 * c(z)) - sqrt((3 * 0.5 * h(z) + 0.15 * c(z)) * (0.5 * h(z) + 3 * 0.15 * c(z)))) + 2 * 0.6 * c(z) + sqrt(h(z) ** 2 / 4 + (0.25 * c(z)) ** 2)) *th_sk + A * 2 * nst)



    def panel_weight(self,t_sp, h_st,t_st,w_st,tmax,tmin):
        t_sk = self.t_arr(tmax,tmin)
        c = self.chord()
        h = self.height()
        stations = self.get_y_rib_loc()
        w = np.zeros(len(stations))
        A = self.area_st(h_st, t_st,w_st)
        #TODO check this still
        # vol_at_stations = np.vectorize(stations, np.resize(t_sk, np.size(stations)), t_sp, h, c, A, self.n_st)
        # w_alternative = cumulative_trapezoid(vol_at_stations, stations)
        # w_res = np.append(np.insert(np.diff(w_alternative), 0 , w_alternative[0]), 0)

        for i in range(len(t_sk)):
            vol = lambda z:  self.rho * (2 * h(z) * t_sp + (pi * (3 * (0.5 * h(z) + 0.15 * c(z)) - sqrt((3 * 0.5 * h(z) + 0.15 * c(z)) * (0.5 * h(z) + 3 * 0.15 * c(z)))) + 2 * 0.6 * c(z) + sqrt(h(z) ** 2 / 4 + (0.25 * c(z)) ** 2)) *t_sk[i] + A * 2 * self.n_str)
            w[i]=trapz([vol(stations[i]),vol(stations[i+1])],[stations[i],stations[i+1]])
        return w



    def wing_weight(self, t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        # b=abs(self.span)
        # c_r=abs(self.chord_root) #NOT USED

        stations = self.get_y_rib_loc()
        skin_weight = self.panel_weight(t_sp, h_st,t_st,w_st,tmax,tmin)
        cumsum = np.sum(skin_weight)
        rbw = self.rib_weight(t_rib)

        for i in stations:
            cumsum = cumsum + rbw(i)
        return cumsum





    def skin_interpolation(self,t_sp, h_st,t_st,w_st,tmax,tmin):
        skin_weight = self.panel_weight(t_sp, h_st,t_st,w_st,tmax,tmin)
        skin_weight = np.flip(skin_weight)
        skin_weight = np.cumsum(skin_weight)
        skin_weight = np.flip(skin_weight)
        return skin_weight





    def rib_interpolation(self, t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        f = self.skin_interpolation(t_sp, h_st,t_st,w_st,tmax,tmin)
        rbw = self.rib_weight(t_rib)
        sta = self.get_y_rib_loc()
        f2 = np.repeat(f, 2)
        sta2 = np.repeat(sta, 2)

        rib_w0 = np.zeros(len(sta))
        for i in range(len(rib_w0)):
            rib_w0[i] = rbw(sta[i])
        rib_w = np.flip(rib_w0)
        rib_w = np.cumsum(rib_w)
        rib_w = np.flip(rib_w)

        combined = np.add(f, rib_w)
        combined2 = np.subtract(combined, rib_w0)

        for i in range(len(sta)):
            f2[2 * i] = 9.81 * combined[i]
            f2[2 * i + 1] = 9.81 * combined2[i]
        return sta2, f2


    def shear_eng(self,t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        x,y = self.rib_interpolation( t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)
        f2 = interp1d(x, y)
        x_engine = np.array([self.y_rotor_loc[0],self.y_rotor_loc[2]])
        x_combi = np.concatenate((x, x_engine))
        x_sort = np.sort(x_combi)

        index1 = np.where(x_sort == self.y_rotor_loc[0])
        if len(index1[0]) == 1:
            index1 = int(index1[0])
        else:
            index1 = int(index1[0][0])
        y_new1 = f2(x_sort[index1]) + 9.81 * self.engine_weight

        index2 = np.where(x_sort == self.y_rotor_loc[2])
        if len(index2[0]) == 1:
            index2 = int(index2[0])
        else:
            index2 = int(index2[0][0])
        y_new2 = f2(x_sort[index2]) + 9.81 * self.engine_weight

        y_engine = np.ndarray.flatten(np.array([y_new1, y_new2]))
        y_combi = np.concatenate((y, y_engine))
        y_sort = np.sort(y_combi)
        y_sort = np.flip(y_sort)

        for i in range(int(index1)):
            y_sort[i] = y_sort[i] + 9.81 * self.engine_weight
        for i in range(int(index2)):
            y_sort[i] = y_sort[i] + 9.81 * self.engine_weight

        return x_sort, y_sort, index1, index2



    # def shear_eng(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    #     x = rib_interpolation(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[0]
    #     y = rib_interpolation(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[1]
    #     f2 = interp1d(x, y)
    #     x_engine = np.array([0.5 * b / 4, 0.5 * b / 2, 0.5 * 3 * b / 4])
    #     x_combi = np.concatenate((x, x_engine))
    #     x_sort = np.sort(x_combi)

    #     index1 = np.where(x_sort == 0.5 * 3 * b / 4)
    #     if len(index1[0]) == 1:
    #         index1 = int(index1[0])
    #     else:
    #         index1 = int(index1[0][0])
    #     y_new1 = f2(x_sort[index1]) + 9.81 * W_eng

    #     index2 = np.where(x_sort == 0.5 * b / 2)
    #     if len(index2[0]) == 1:
    #         index2 = int(index2[0])
    #     else:
    #         index2 = int(index2[0][0])
    #     y_new2 = f2(x_sort[index2]) + 9.81 * W_eng

    #     index3 = np.where(x_sort == 0.5 * b / 4)
    #     if len(index3[0]) == 1:
    #         index3 = int(index3[0])
    #     else:
    #         index3 = int(index3[0][0])
    #     y_new3 = f2(x_sort[index3]) + 9.81 * W_eng

    #     y_engine = np.ndarray.flatten(np.array([y_new1, y_new2, y_new3]))
    #     y_combi = np.concatenate((y, y_engine))
    #     y_sort = np.sort(y_combi)
    #     y_sort = np.flip(y_sort)

    #     for i in range(int(index1)):
    #         y_sort[i] = y_sort[i] + 9.81 * W_eng
    #     for i in range(int(index2)):
    #         y_sort[i] = y_sort[i] + 9.81 * W_eng
    #     for i in range(int(index3)):
    #         y_sort[i] = y_sort[i] + 9.81 * W_eng

    #     return x_sort, y_sort, index1, index2, index3



    def m(self,t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        f = self.skin_interpolation(t_sp, h_st,t_st,w_st,tmax,tmin)
        sta = self.get_y_rib_loc()
        rbw = self.rib_weight(t_rib)

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




    def m_eng(self,t_sp, t_rib,h_st,t_st,w_st,tmax,tmin):
        moment = self.m(t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)
        x = self.get_y_rib_loc()
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





    def N_x(self, t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        """ Check this function thoroughly

        """    
        sta = self.get_y_rib_loc()
        x_sort, moment = self.m_eng(t_sp, t_rib,h_st,t_st,w_st,tmax,tmin)
        h = self.height()
        tarr = self.t_arr(tmax,tmin)
        Nx = np.zeros(len(tarr))

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
        bend_stress=np.zeros(len(tarr))
        for i in range(len(tarr)):
            Ixx = self.I_xx(t_sp,h_st,t_st,w_st,tarr[i])(sta[i])
            bend_stress[i] = moment[i] * 0.5 * h(sta[i]) / Ixx
            Nx[i] = bend_stress[i] * tarr[i]
        return  Nx, bend_stress







    def shear_force(self,t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        shear = self.shear_eng(t_sp, t_rib, h_st, t_st, w_st,tmax,tmin)[1]
        tarr = self.t_arr(tmax,tmin)
        Vz = np.zeros(len(tarr))

        sta = self.get_y_rib_loc()
        for i in range(len(tarr)):
            Vz[i] = self.aero(sta[i])-shear[2 * i]
        return Vz

    def perimiter_ellipse(self,a,b):
        return float(np.pi *  ( 3*(a+b) - np.sqrt( (3*a + b) * (a + 3*b) ) )) #Ramanujans first approximation formula

    def torsion_sections(self,tmax,tmin):
        wing = self.wing
        engine = self.engine
        ch = self.chord()
        tarr = self.t_arr(tmax,tmin)
        sta = self.get_y_rib_loc()
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

    def N_xy(self, t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        engine = self.engine
        h1 = self.height()
        ch = self.chord()
        tarr = self.t_arr(tmax,tmin)
        sta = self.get_y_rib_loc()
        Vz=self.shear_force(t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)
        T =self.torsion_sections(tmax,tmin)
        Nxy = np.zeros(len(tarr))

        for i in range(len(tarr)):
            Ixx1 = self.I_xx(t_sp,h_st,t_st,w_st,tarr[i])
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



    def buckling(self, t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):#TODO
        Nxy = self.N_xy(t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)
        Nx = self.N_x(t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)[0]
        tarr = self.t_arr(tmax,tmin)
        buck = np.zeros(len(tarr))
        for i in range(len(tarr)):
            Nx_crit = self.local_buckling(tarr[i])*tarr[i]
            Nxy_crit = self.shear_buckling(tarr[i])*tarr[i]
            buck[i] = Nx[i] / Nx_crit + (Nxy[i] / Nxy_crit) ** 2
        return buck
    




    def column_st(self, h_st,t_st,w_st,t_sk):#TODO
        #Lnew=new_L(b,L)
        Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 + t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
        i= pi ** 2 * self.E * Ist / (2*w_st* self.max_rib_pitch ** 2)   
        return i


    def f_ult(self,h_st,t_st,w_st,tmax,tmin):
        A_st = self.area_st(h_st,t_st,w_st)
        # n=n_st(c_r,b_st)
        tarr=self.t_arr(tmax,tmin)
        c=self.chord()
        h=self.height()
        stations= self.get_y_rib_loc() #FIXME change this to an input 
        f_uts=np.zeros(len(tarr))
        for i in range(len(tarr)):
            A=self.n_max*A_st+0.6*c(stations[i])*tarr[i]
            f_uts[i]=self.sigma_uts*A
        return f_uts




    def buckling_constr(self, t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        buck = self.buckling(t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)#TODO
        tarr = self.t_arr(tmax,tmin)
        vector = np.zeros(len(tarr))
        for i in range(len(tarr)):
            vector[i] = -1 * (buck[i] - 1)
        return vector


    def global_local(self, h_st,t_st,tmax,tmin):
        tarr = self.t_arr(tmax,tmin)
        diff = np.zeros(len(tarr))
        for i in range(len(tarr)):
            glob = self.global_buckling(h_st,t_st,tarr[i])
            loc = self.local_buckling(tarr[i])
            diff[i] = glob - loc #FIXEM glob
        #diff = self.global_buckling(h_st,t_st,tarr)  - self.local_buckling(tarr,b_st)

        return diff



    def local_column(self, h_st,t_st,w_st,tmax,tmin):
        tarr = self.t_arr(tmax,tmin)
        diff = np.zeros(len(tarr))
        for i in range(len(tarr)):
            col=self.column_st(h_st,t_st,w_st, tarr[i])
            loc = self.local_buckling(tarr[i])*tarr[i]
            diff[i] = col - loc
        return diff


    def flange_loc_loc(self, t_st,w_st,tmax,tmin):
        tarr = self.t_arr(tmax,tmin)
        diff = np.zeros(len(tarr))
        flange = self.flange_buckling(t_st, w_st)
        for i in range(len(tarr)):
            loc = self.local_buckling(tarr[i])
            diff[i] = flange - loc
        return diff


    def web_flange(self, h_st,t_st,tmax,tmin):
        tarr = self.t_arr(tmax,tmin)
        diff = np.zeros(len(tarr))
        web = self.web_buckling(t_st, h_st)
        for i in range(len(tarr)):
            loc = self.local_buckling(tarr[i])
            diff[i] =web-loc
        return diff


    def von_Mises(self, t_sp, t_rib, h_st,t_st,w_st,tmax,tmin):
        # vm = np.zeros(len(tarr))
        tarr = self.t_arr(tmax,tmin)
        Nxy=self.N_xy(t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)
        bend_stress=self.N_x(t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)[1] #
        tau_shear_arr = Nxy/tarr
        vm_lst = self.sigma_yield - np.sqrt(0.5 * (3 * tau_shear_arr ** 2+bend_stress**2))
        # for i in range(len(tarr)):
        #     tau_shear= Nxy[i] / tarr[i]
        #     vm[i]=sigma_yield-sqrt(0.5 * (3 * tau_shear ** 2+bend_stress[i]**2))
        return vm_lst



    def crippling(self, h_st,t_st,w_st,tmax,tmin):
        tarr = self.t_arr(tmax,tmin)
        crip= np.zeros(len(tarr))
        A = self.area_st(h_st, t_st, w_st)
        for i in range(len(tarr)):
            col = self.column_st( h_st,t_st,w_st,tarr[i])
            crip[i] = t_st * self.beta * self.sigma_yield* ((self.g * t_st ** 2 / A) * sqrt(self.E / self.sigma_yield)) ** self.m_crip - col
        return crip


    def post_buckling(self, t_sp, t_rib, h_st,t_st,w_st, tmax,tmin):
        f = self.f_ult(h_st,t_st,w_st,tmax,tmin)
        ratio=2/(2+1.3*(1-1/self.pb))
        px= self.n_max*self.shear_force(t_sp, t_rib, h_st,t_st,w_st,tmax,tmin)
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




    def setup(self):

        # Design variables
        self.add_input('tsp')
        self.add_input('trib')
        self.add_input('hst')
        self.add_input('tst')
        self.add_input('wst')
        self.add_input('tmax')
        self.add_input("tmin")

        # Constant inputs
        # self.add_input('b')
        # self.add_input('c_r')
        
        #Outputs used as constraints
        self.add_output('wing_weight')
        self.add_output('global_local',shape = (8,))
        self.add_output('post_buckling',shape = (8,))
        self.add_output('von_mises',shape = (8,))
        self.add_output('buckling_constr',shape = (8,))
        self.add_output('flange_loc_loc',shape = (8,))
        self.add_output('local_column',shape = (8,))
        self.add_output('crippling',shape = (8,))
        self.add_output("web_flange",shape = (8,))
        self.declare_partials('*', '*', method= 'fd')


    # def setup_partials(self):

    #     # Partial derivatives are done using finite difference
    #     self.declare_partials('*', '*', 'fd')

    def compute(self, inputs, outputs):

        # Design variables
        tsp = inputs['tsp'][0]
        trib = inputs['trib'][0]
        hst = inputs['hst'][0]
        tst = inputs['tst'][0]
        wst = inputs['wst'][0]
        t_max = inputs['tmax'][0]
        t_min = inputs['tmin'][0]

        # Constants
        # span = inputs['b'][0]
        # chord_root = inputs['c_r'][0]


        weight = self.WingboxClass.wing_weight(tsp,trib, hst, tst,wst,t_max, t_min)
        print(f"tsp = {tsp}")
        constr = [
        self.WingboxClass.global_local( hst, tst,t_max, t_min),
        self.WingboxClass.post_buckling(tsp, trib, hst,tst,wst, t_max,t_min),
        self.WingboxClass.von_Mises(tsp, trib, hst,tst,wst,t_max,t_min),
        self.WingboxClass.buckling_constr(  tsp, trib,  hst, tst,wst,t_max, t_min),
        self.WingboxClass.flange_loc_loc(tst,wst,t_max, t_min),
        self.WingboxClass.local_column(hst,tst,wst,t_max, t_min),
        self.WingboxClass.crippling(hst, tst, wst, t_max, t_min), #ONLY
        self.WingboxClass.web_flange(hst,tst,t_max,t_min)
        ]
        print(constr)




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


    prob = om.Problem()
    prob.model.add_subsystem('wingbox_design', Wingbox_optimization(wing, engine, material, aero))#, promotes_inputs=['AR1',
                                                                                        # 'AR2',

    # Initial values for the optimization 

    #Constants
    # prob.model.set_input_defaults('wingbox_design.b', wing.span)
    # prob.model.set_input_defaults('wingbox_design.c_r', wing.chord_root)
    # prob.model.set_input_defaults('wingbox_design.engine', engine)
    # prob.model.set_input_defaults('wingbox_design.wing', wing)

   

    prob.model.add_design_var('wingbox_design.tsp', lower = 0.001, upper= 0.1)
    prob.model.add_design_var('wingbox_design.trib', lower = 0.001, upper= 0.1)
    prob.model.add_design_var('wingbox_design.hst', lower = 0.001 , upper= 0.4)
    prob.model.add_design_var('wingbox_design.tst', lower = 0.001, upper= 0.1)
    prob.model.add_design_var('wingbox_design.wst', lower = 0.001, upper= 0.4 )
    prob.model.add_design_var('wingbox_design.tmax', lower = 0.001, upper= 0.1)
    prob.model.add_design_var('wingbox_design.tmin', lower = 0.001, upper= 0.1)

    # # Define constraints 
    prob.model.add_constraint('wingbox_design.global_local')
    prob.model.add_constraint('wingbox_design.post_buckling')
    prob.model.add_constraint('wingbox_design.von_mises')
    prob.model.add_constraint('wingbox_design.buckling_constr')
    prob.model.add_constraint('wingbox_design.flange_loc_loc')
    prob.model.add_constraint('wingbox_design.local_column')
    prob.model.add_constraint('wingbox_design.crippling')
    prob.model.add_constraint('wingbox_design.web_flange')

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.opt_settings['maxiter'] = 1000

    prob.model.add_objective('wingbox_design.wing_weight')

    prob.setup()

     # Initial estimate for the design variables
    prob.set_val('wingbox_design.tsp', x[0])
    prob.set_val('wingbox_design.trib', x[1])
    prob.set_val('wingbox_design.hst', x[2])
    prob.set_val('wingbox_design.tst', x[3])
    prob.set_val('wingbox_design.wst', x[4])
    prob.set_val('wingbox_design.tmax', x[5])
    prob.set_val('wingbox_design.tmin', x[6])

    prob.model.list_inputs(True)

    prob.run_driver()
    # prob.check_partials()
    #prob.check_totals()

    #prob.model.list_outputs()

    print(f"thickness spar= {prob.get_val('wingbox_design.tsp')*1000} [mm]")
    print(f"thickness rib= {prob.get_val('wingbox_design.trib')*1000} [mm]")
    print(f"stringer height= {prob.get_val('wingbox_design.hst')*1000} [mm]")
    print(f"stringer thickness= {prob.get_val('wingbox_design.tst')*1000} [mm]")
    print(f"stringer width= {prob.get_val('wingbox_design.wst')*1000} [mm]")
    print(f"max skin thickness= {prob.get_val('wingbox_design.tmax')*1000} [mm]")
    print(f"min skin thickness= {prob.get_val('wingbox_design.tmin')*1000} [mm]")
    print(f"Wing weight= {prob.get_val('wingbox_design.wing_weight')} [kg]")


    output_lst =  np.array([
    prob.get_val('wingbox_design.tsp'),
    prob.get_val('wingbox_design.trib'),
    prob.get_val('wingbox_design.hst'),
    prob.get_val('wingbox_design.tst'),
    prob.get_val('wingbox_design.wst'),
    prob.get_val('wingbox_design.tmax'),
    prob.get_val('wingbox_design.tmin')
    ])

    return output_lst