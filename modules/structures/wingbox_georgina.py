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


#------- TO DO summary ----------------------------
#TODO The thick
taper = None
rho = None
W_eng = None
E = None
poisson = None
pb= None
beta= None
g= None
sigma_yield = None
m_crip = None
sigma_uts = None
n_max= None
y_rotor_loc = None

G = 26e9


class Wingbox():
    def __init__(self, engine, material, pitch_st) -> None:
        #Material
        self.n_st = n_st
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
        #Wing
        self.taper = taper
        self.n_max = n_max_req

        #Engine
        self.engine_weight = engine.mass_pertotalengine
        self.y_rotor_loc = engine.y_rotor_loc
        self.nacelle_w = engine.nacelle_w #TODO Check if it gets updated
        
        
        #Set number of ribs in inboard and outboard section
        self.n_ribs_sec0 = 1 #Number of ribs inboard of inboard engine
        self.n_ribs_sec1 = 4 #Number of ribs inboard and outboard engines

        #Set number of stringers in top and bottom
        self.n_str = 20

    #Determine rib positions in spanwise direction (y)
    def get_y_rib_loc(self):
        y_rib_0 = self.y_rotor_loc[0] - 0.5 * self.nacelle_w
        y_rib_1 = self.y_rotor_loc[0] + 0.5 * self.nacelle_w

        y_rib_3 = self.y_rotr_loc[1] - 0.5 * self.nacelle_w
        y_rib_2 = y_rib_3 - 0.15
        y_rib_sec0 = np.arange(0,y_rib_0,y_rib_0/self.n_ribs_sec0)
        y_rib_sec1 = np.arange(y_rib_1,y_rib_2, (y_rib_2-y_rib_1)/self.n_ribs_sec1)

        y_rib_loc = np.array([y_rib_0,y_rib_1,y_rib_2,y_rib_3],y_rib_sec0,y_rib_sec1)
        y_rib_loc = np.sort(y_rib_loc)
        return y_rib_loc




def chord(b, c_r):
    c = lambda y: c_r - c_r * (1 - taper) * y * 2 / b
    return c


def height(b, c_r):
    c = chord(b, c_r)
    h = lambda Y: 0.17 * c(Y)
    return h


def area_st(h_st,t_st,w_st):
    return t_st * (2 * w_st + h_st)




def I_st(h_st,t_st,w_st):
    Ast = area_st(h_st, t_st, w_st)
    i = t_st * h_st ** 3 / 12 + w_st * t_st ** 3 / 12 + 2 * Ast * (0.5 * h_st) ** 2
    return i





def w_sp(b, c_r):
    h = height(b, c_r)
    i = lambda z: 0.5 * h(z)
    return i




def I_sp(b, c_r,t_sp):
    h = height(b, c_r)
    wsp = w_sp(b, c_r)
    i = lambda z: t_sp * (h(z) - 2 * t_sp) ** 3 / 12 + 2 * wsp(z) * t_sp ** 3 / 12 + 2 * t_sp * wsp(z) * (
            0.5 * h(z)) ** 2
    return i




# def n_st(c_r, b_st):
#     return ceil(0.6 * c_r / b_st) + 1



# def n_ribs(b, L):
#     return ceil(0.5 * b / L) + 1



# def new_L(b, L):
#     """ FIx this function

#     """    
#     nr_sect = n_ribs(b, L) - 1
#     new_pitch = 0.5 * b / nr_sect
#     return new_pitch




# def new_bst(c_r, b_st):
#     """ Fix this function as well

#     """    
#     nr_sect = n_st(c_r, b_st) - 1
#     new_pitch = c_r / nr_sect
#     return new_pitch





def rib_coordinates(b, L):
    """ This functions will become an innput

    """    
    L_new = new_L(b, L)
    stations = np.arange(0, b / 2 + L_new, L_new)
    return stations






def I_xx(b,c_r,t_sp,b_st, h_st,t_st,w_st,t_sk):
    h = height(b, c_r)
    # nst = n_st(c_r, b_st)
    Ist = I_st(h_st,t_st,w_st)
    Isp = I_sp(b, c_r,t_sp)
    A = area_st(h_st,t_st,w_st)
    i = lambda z: 2 * (Ist + A * (0.5 * h(z)) ** 2) * nst + 2 * Isp(z) + 2 * (0.6 * c_r * t_sk ** 3 / 12 + t_sk * 0.6 * c_r * (0.5 * h(z)) ** 2)
    return i




# def t_arr(b, L,t):
#     """ Replace function by our design variables, simplifies our process. List of thicknesses compatible with our sections. 
#     #TODO
#     - compatible with L

#     """    
#     b=abs(b)
#     L=abs(L)
#     nr_ribs = n_ribs(b, L)
#     sections = np.zeros(nr_ribs - 1)

#     inte = int((len(sections)) // len(t))
#     mod = int((len(sections)) % len(t))
#     group = int(len(t) - mod)

#     arr = np.arange(inte * group, len(sections), inte + 1)

#     for i in range(group):
#         for j in range(inte):
#             sections[inte * i + j] = t[i]
#     for i in range(len(arr)):
#         cursor = arr[i]
#         for j in range(inte + 1):
#             sections[cursor + j] = t[group + i]
#     return sections



def rib_weight(b, c_r, t_rib):
    c = chord(b, c_r)
    h = height(b, c_r)
    w_rib = lambda z: 0.6 * c(z) * h(z) * t_rib * rho
    return w_rib



def vol_func(z, th_sk, t_sp, h, c, A, nst):
    return rho * (2 * h(z) * t_sp + (pi * (3 * (0.5 * h(z) + 0.15 * c(z)) - sqrt((3 * 0.5 * h(z) + 0.15 * c(z)) * (0.5 * h(z) + 3 * 0.15 * c(z)))) + 2 * 0.6 * c(z) + sqrt(h(z) ** 2 / 4 + (0.25 * c(z)) ** 2)) *th_sk + A * 2 * nst)

vol_func_vec = np.vectorize(vol_func)



def panel_weight(b, c_r,t_sp, L, b_st, h_st,t_st,w_st,t):
    t_sk = t_arr(b, L,t)
    c = chord(b, c_r)
    h = height(b, c_r)
    # nst = n_st(c_r, b_st)
    stations = rib_coordinates(b, L)
    w = np.zeros(len(stations))
    A = area_st(h_st, t_st,w_st)


    vol_at_stations = vol_func_vec(stations, np.resize(t_sk, np.size(stations)), t_sp, h, c, A, nst)
    w_alternative = cumulative_trapezoid(vol_at_stations, stations)
    w_res = np.append(np.insert(np.diff(w_alternative), 0 , w_alternative[0]), 0)
    


    # for i in range(len(t_sk)):
    #     vol = lambda z:  rho * (2 * h(z) * t_sp + (pi * (3 * (0.5 * h(z) + 0.15 * c(z)) - sqrt((3 * 0.5 * h(z) + 0.15 * c(z)) * (0.5 * h(z) + 3 * 0.15 * c(z)))) + 2 * 0.6 * c(z) + sqrt(h(z) ** 2 / 4 + (0.25 * c(z)) ** 2)) *t_sk[i] + A * 2 * nst)
    #     w[i]=trapz([vol(stations[i]),vol(stations[i+1])],[stations[i],stations[i+1]])
    return w_res



def wing_weight(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    b=abs(b)
    c_r=abs(c_r)
    t_sp=abs(t_sp)
    t_rib=abs(t_rib)
    L=abs(L)
    b_st=abs(b_st)
    h_st=abs(h_st)
    t_st=abs(t_st)
    w_st=abs(w_st)
    for i in range(len(t)):
        t[i]=abs(t[i])
    stations = rib_coordinates(b, L)
    skin_weight = panel_weight(b, c_r, t_sp, L, b_st, h_st,t_st,w_st,t)
    cumsum = np.sum(skin_weight)
    rbw = rib_weight(b, c_r, t_rib)

    for i in stations:
        cumsum = cumsum + rbw(i)
    return cumsum





def skin_interpolation(b, c_r, t_sp, L, b_st, h_st,t_st,w_st,t):
    skin_weight = panel_weight(b, c_r, t_sp, L, b_st, h_st,t_st,w_st,t)
    skin_weight = np.flip(skin_weight)
    skin_weight = np.cumsum(skin_weight)
    skin_weight = np.flip(skin_weight)
    return skin_weight





def rib_interpolation(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    f = skin_interpolation(b, c_r, t_sp, L, b_st, h_st,t_st,w_st,t)
    rbw = rib_weight(b, c_r, t_rib)
    sta = rib_coordinates(b, L)
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


def shear_eng(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    x,y = rib_interpolation(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    # y = rib_interpolation(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t, taper, rho)[1]
    f2 = interp1d(x, y)
    x_engine = np.array([y_rotor_loc[0],y_rotor_loc[2]])
    x_combi = np.concatenate((x, x_engine))
    x_sort = np.sort(x_combi)

    index1 = np.where(x_sort == y_rotor_loc[0])
    if len(index1[0]) == 1:
        index1 = int(index1[0])
    else:
        index1 = int(index1[0][0])
    y_new1 = f2(x_sort[index1]) + 9.81 * W_eng

    index2 = np.where(x_sort == y_rotor_loc[2])
    if len(index2[0]) == 1:
        index2 = int(index2[0])
    else:
        index2 = int(index2[0][0])
    y_new2 = f2(x_sort[index2]) + 9.81 * W_eng

    y_engine = np.ndarray.flatten(np.array([y_new1, y_new2]))
    y_combi = np.concatenate((y, y_engine))
    y_sort = np.sort(y_combi)
    y_sort = np.flip(y_sort)

    for i in range(int(index1)):
        y_sort[i] = y_sort[i] + 9.81 * W_eng
    for i in range(int(index2)):
        y_sort[i] = y_sort[i] + 9.81 * W_eng

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



def m(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    f = skin_interpolation(b, c_r, t_sp, L, b_st, h_st,t_st,w_st,t)
    sta = rib_coordinates(b, L)
    rbw = rib_weight(b, c_r, t_rib)

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




def m_eng(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    moment = m(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    x = rib_coordinates(b, L)
    f = interp1d(x, moment, kind='quadratic')

    x_engine = np.array([0.5 * b / 4, 0.5 * b / 2, 0.5 * 3 * b / 4])
    x_combi = np.concatenate((x, x_engine))
    x_sort = np.sort(x_combi)

    index1 = np.where(x_sort == 0.5 * 3 * b / 4)
    if len(index1[0]) == 1:
        index1 = int(index1[0])
    else:
        index1 = int(index1[0][0])
    y_new1 = f(x_sort[index1])

    index2 = np.where(x_sort == 0.5 * b / 2)
    if len(index2[0]) == 1:
        index2 = int(index2[0])
    else:
        index2 = int(index2[0][0])
    y_new2 = f(x_sort[index2])

    index3 = np.where(x_sort == 0.5 * b / 4)
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
        y_sort[i] = y_sort[i] + 9.81 * W_eng * (0.5 * 3 * b / 4 - x_sort[i])
    for i in range(int(index2)):
        y_sort[i] = y_sort[i] + 9.81 * W_eng * (0.5 * 2 * b / 4 - x_sort[i])
    for i in range(int(index3)):
        y_sort[i] = y_sort[i] + 9.81 * W_eng * (0.5 * b / 4 - x_sort[i])

    return x_sort, y_sort





def N_x(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    """ Check this function thoroughly

    """    
    sta = rib_coordinates(b, L)
    x_sort, moment = m_eng(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    # x_sort = m_eng(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[0]
    h = height(b, c_r)
    tarr = t_arr(b,L,t)
    Nx = np.zeros(len(tarr))

    index1 = np.where(x_sort == 0.5 * 3 * b / 4)
    if len(index1[0]) == 1:
        index1 = int(index1[0])
    else:
        index1 = int(index1[0][0])

    index2 = np.where(x_sort == 0.5 * b / 2)
    if len(index2[0]) == 1:
        index2 = int(index2[0])
    else:
        index2 = int(index2[0][0])

    index3 = np.where(x_sort == 0.5 * b / 4)
    if len(index3[0]) == 1:
        index3 = int(index3[0])
    else:
        index3 = int(index3[0][0])

    moment = np.delete(moment, np.array([index1, index2, index3]))
    bend_stress=np.zeros(len(tarr))
    for i in range(len(tarr)):
        Ixx = I_xx(b,c_r,t_sp,b_st, h_st,t_st,w_st,tarr[i])(sta[i])
        bend_stress[i] = moment[i] * 0.5 * h(sta[i]) / Ixx
        # Nx[i] = bend_stress[i] * tarr[i]
    return  bend_stress







def shear_force(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    shear = shear_eng(b, c_r, t_sp, t_rib, L, b_st, h_st, t_st, w_st,t)[1]
    tarr = t_arr(b, L,t)
    Vz = np.zeros(len(tarr))

    sta = rib_coordinates(b, L)
    aero= lambda y:-151.7143*9.81*y+531*9.81
    for i in range(len(tarr)):
        Vz[i] = aero(sta[i])-shear[2 * i]
    return Vz

# def perimiter_ellipse(a,b):
#     return float(np.pi *  ( 3*(a+b) - np.sqrt( (3*a + b) * (a + 3*b) ) )) #Ramanujans first approximation formula

# def torsion_sections(b,c_r,L,t,engine,wing):
#     ch = chord(b, c_r)
#     tarr = t_arr(b, L,t)
#     sta = rib_coordinates(b, L)
#     T = np.zeros(len(tarr))
#     engine_weight = engine.mass_pertotalengine
#     x_centre_wb = lambda x_w: wing.X_lemac + c_r*0.25* + ch(x_w)*0.20
#     for i in range(len(tarr)):
#         if sta[i]< float(engine.y_rotor_loc[0]):
#             T[i] = engine_weight * 9.81 * (x_centre_wb(engine.x_rotor_loc[0])-engine.x_rotor_loc[0]) + engine_weight * 9.81 * (x_centre_wb(engine.x_rotor_loc[2])-engine.x_rotor_loc[2])
#         else:
#             T[i] = engine_weight * 9.81 * (x_centre_wb(engine.x_rotor_loc[0])-engine.x_rotor_loc[0])
#     #     print(sta[i],y_rotor_loc[0],x_centre_wb(engine.x_rotor_loc[0]))
#     # print(f"\n\nT = {T}\n\n")
#     return T

# def N_xy(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t,Engine,Wing):
#     h1 = height(b, c_r)
#     ch = chord(b, c_r)
#     tarr = t_arr(b,L,t)
#     sta = rib_coordinates(b, L)
#     Vz=shear_force(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
#     T =torsion_sections(b,c_r,L,t,Engine,Wing)
#     Nxy = np.zeros(len(tarr))

#     for i in range(len(tarr)):
#         Ixx1 = I_xx(b,c_r,t_sp,b_st, h_st,t_st,w_st,tarr[i])
#         Ixx = Ixx1(sta[i])
#         h = h1(sta[i])
#         l_sk = sqrt(h ** 2 + (0.25 * c_r) ** 2)
#         c = ch(sta[i])

#         # Base region 1
#         qb1 = lambda z: Vz[i] * tarr[i] * (0.5 * h) ** 2 * (np.cos(z) - 1) / Ixx
#         I1 = qb1(pi / 2)

#         # Base region 2
#         qb2 = lambda z: -Vz[i] * t_sp * z ** 2 / (2 * Ixx)
#         I2 = qb2(h)
#         s2 = np.arange(0, h+ 0.1, 0.1)

#         # Base region 3
#         qb3 = lambda z: - Vz[i] * tarr[i] * (0.5 * h) * z / Ixx + I1 + I2
#         I3 = qb3(0.6 * c)
#         s3 = np.arange(0, 0.6*c+ 0.1, 0.1)

#         # Base region 4
#         qb4 = lambda z: -Vz[i] * t_sp * z ** 2 / (2 * Ixx)
#         I4 = qb4(h)
#         s4=np.arange(0, h+ 0.1, 0.1)

#         # Base region 5
#         qb5 = lambda z: -Vz[i] * tarr[i] / Ixx * (0.5 * h * z - 0.5 * 0.5 * h * z ** 2 / l_sk) + I3 + I4
#         I5 = qb5(l_sk)

#         # Base region 6
#         qb6 = lambda z: Vz[i] * tarr[i] / Ixx * 0.5 * 0.5 * h / l_sk * z ** 2 + I5
#         I6 = qb6(l_sk)

#         # Base region 7
#         qb7 = lambda z: -Vz[i] * t_sp * 0.5 * z ** 2 / Ixx
#         I7 = qb7(-h)


#         # Base region 8
#         qb8 = lambda z: -Vz[i] * 0.5 * h * t_sp * z / Ixx + I6 - I7
#         I8 = qb8(0.6 * c)

#         # Base region 9
#         qb9 = lambda z: -Vz[i] * 0.5 * t_sp * z ** 2 / Ixx
#         I9 = qb9(-h)

#         # Base region 10
#         qb10 = lambda z: -Vz[i] * tarr[i] * (0.5 * h) ** 2 * (np.cos(z) - 1) / Ixx + I8 - I9

#         #Torsion
#         A1 = float(np.pi*h*c*0.15*0.5)
#         A2 = float(h*0.6*c)
#         A3 = float(h*0.25*c)

#         T_A11 = 0.5 * A1 * perimiter_ellipse(h,0.15*c) * 0.5 * tarr[i]
#         T_A12 = -A1 * h * t_sp
#         T_A13 = 0
#         T_A14 = -1/(0.5*G)

#         T_A21 = -A2 * h * t_sp
#         T_A22 = A2 * h * t_sp * 2 + c*0.6*2*A2*tarr[i]
#         T_A23 = -h*A2*t_sp
#         T_A24 = -1/(0.5*G)

#         T_A31 = 0
#         T_A32 = -A3 * h *t_sp
#         T_A33 = A3 * h * t_sp + l_sk*A3*tarr[i]*2
#         T_A34 = -1/(0.5*G)

#         T_A41 = 2*A1
#         T_A42 = 2*A2
#         T_A43 = 2*A3
#         T_A44 = 0

#         T_A = np.array([[T_A11, T_A12, T_A13, T_A14], [T_A21, T_A22, T_A23, T_A24], [T_A31, T_A32, T_A33, T_A34],[T_A41,T_A42,T_A43,T_A44]])
#         T_B = np.array([0,0,0,T[i]])
#         T_X = np.linalg.solve(T_A, T_B)



#         # Redundant shear flow
#         A11 = pi * (0.5 * h) / tarr[i] + h / t_sp
#         A12 = -h / t_sp
#         A21 = - h / t_sp
#         A22 = 1.2 * c / tarr[i]
#         A23 = -h / t_sp
#         A32 = - h / t_sp
#         A33 = 2 * l_sk / tarr[i] + h / t_sp



#         B1 = 0.5 * h / tarr[i] * trapz([qb1(0),qb1(pi/2)], [0, pi / 2]) + trapz([qb2(0),qb2(0.5*h)], [0, 0.5 * h]) / t_sp - trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0])/ t_sp + trapz([qb10(-pi/2),qb10(0)], [-pi / 2, 0]) * 0.5 * h / tarr[i]
#         B2 = trapz([qb2(0),qb2(0.5*h)], [0, 0.5 * h]) / t_sp + trapz([qb3(0),qb3(0.6*c)], [0, 0.6 * c]) / tarr[i] - trapz([qb7(-0.5*h),qb7(0)], [-0.5 * h, 0]) / t_sp + \
#              trapz([qb4(0),qb4(0.5*h)], [0, 0.5 * h]) / t_sp + trapz([qb8(0),qb8(0.6*c)], [0, 0.6 * c]) / tarr[i] - trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0]) / t_sp
#         B3 = trapz([qb5(0),qb5(l_sk)], [0, l_sk]) / tarr[i] + trapz([qb6(0),qb6(l_sk)], [0, l_sk]) / tarr[i] + trapz([qb4(0),qb4(0.5*h)], [0, 0.5 * h]) / t_sp - \
#              trapz([qb9(-0.5*h),qb9(0)], [-0.5 * h, 0]) / t_sp

#         A = np.array([[A11, A12, 0], [A21, A22, A23], [0, A32, A33]])
#         B = -np.array([[B1], [B2], [B3]])
#         X = np.linalg.solve(A, B)

#         q01 = float(X[0])
#         q02 = float(X[1])
#         q03 = float(X[2])

#         qT1 = float(T_X[0])
#         qT2 = float(T_X[1])
#         qT3 = float(T_X[1])

#         # Compute final shear flow
#         q2 = qb2(s2) - q01 - qT1 + q02 + qT2
#         q3 = qb3(s3) + q02 + qT2
#         q4 = qb4(s4) + q03 +qT3 - q02 - qT2

#         max_region2 = max(q2)
#         max_region3 = max(q3)
#         max_region4 = max(q4)
#         determine = max(max_region2, max_region3, max_region4)
#         Nxy[i] = determine
#     return Nxy

def N_xy(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    h1 = height(b, c_r)
    ch = chord(b, c_r)
    tarr = t_arr(b,L,t)
    sta = rib_coordinates(b, L)
    Vz=shear_force(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    Nxy = np.zeros(len(tarr))

    for i in range(len(tarr)):
        Ixx1 = I_xx(b,c_r,t_sp,b_st, h_st,t_st,w_st,tarr[i])
        Ixx = Ixx1(sta[i])
        h = h1(sta[i])
        l_sk = sqrt(h ** 2 + (0.25 * c_r) ** 2)
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

        # Compute final shear flow
        q2 = qb2(s2) - q01 + q02
        q3 = qb3(s3) + q02
        q4 = qb4(s4) + q03 - q02

        max_region2 = max(q2)
        max_region3 = max(q3)
        max_region4 = max(q4)
        determine = max(max_region2, max_region3, max_region4)
        Nxy[i] = determine
    return Nxy


def local_buckling(c_r, b_st,t):
    bst = new_bst(c_r, b_st)
    buck = 4* pi ** 2 * E / (12 * (1 - poisson ** 2)) * (t / bst) ** 2
    return buck


def flange_buckling(t_st, w_st):
    buck = 2 * pi ** 2 * E / (12 * (1 - poisson ** 2)) * (t_st / w_st) ** 2
    return buck


def web_buckling(t_st, h_st):
    buck = 4 * pi ** 2 * E / (12 * (1 - poisson ** 2)) * (t_st / h_st) ** 2
    return buck


def global_buckling(c_r, b_st, h_st,t_st,t):
    # n = n_st(c_r, b_st)
    bst = new_bst(c_r, b_st)
    tsmr = (t * bst + t_st * n * (h_st - t)) / bst
    return 4 * pi ** 2 * E / (12 * (1 - poisson ** 2)) * (tsmr / bst) ** 2


def shear_buckling(c_r, b_st,t):
    bst = new_bst(c_r, b_st)
    buck = 5.35 * pi ** 2 * E / (12 * (1 - poisson)) * (t / bst) ** 2
    return buck



def buckling(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    Nxy = N_xy(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    Nx = N_x(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[0]
    tarr = t_arr(b, L,t)
    buck = np.zeros(len(tarr))
    for i in range(len(tarr)):
        Nx_crit = local_buckling(c_r, b_st,tarr[i])*tarr[i]
        Nxy_crit = shear_buckling(c_r, b_st,tarr[i])*tarr[i]
        buck[i] = Nx[i] / Nx_crit + (Nxy[i] / Nxy_crit) ** 2
    return buck




def column_st(b, L,h_st,t_st,w_st,t_sk):
    Lnew=new_L(b,L)
    Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 +t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
    i= pi ** 2 * E * Ist / (2*w_st* Lnew ** 2)
    return i


def f_ult(b,c_r,L,b_st,h_st,t_st,w_st,tarr):
    A_st = area_st(h_st,t_st,w_st)
    # n=n_st(c_r,b_st)
    tarr=t_arr(b,L,tarr)
    c=chord(b,c_r)
    h=height(b,c_r)
    stations=rib_coordinates(b,L) #FIXME change this to an input 
    f_uts=np.zeros(len(tarr))
    for i in range(len(tarr)):
        A=n*A_st+0.6*c(stations[i])*tarr
        f_uts[i]=sigma_uts*A
    return f_uts




def buckling_constr(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    buck = buckling(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    tarr = t_arr(b, L,t)
    vector = np.zeros(len(tarr))
    for i in range(len(tarr)):
        vector[i] = -1 * (buck[i] - 1)
    return vector[0]


def global_local(b, c_r, L, b_st, h_st,t_st,tarr):
    # for i in range(len(tarr)):
    #     glob = global_buckling(c_r, b_st, h_st,t_st,tarr[i])
    #     loc = local_buckling(c_r, b_st,tarr[i])
    #     diff[i] = glob - loc #FIXEM glob
    diff = global_buckling(c_r, b_st, h_st,t_st,tarr)  - local_buckling(c_r, b_st,tarr)

    return diff



def local_column(b, c_r, L, b_st, h_st,t_st,w_st,t):
    tarr = t_arr(b, L,t)
    diff = np.zeros(len(tarr))
    for i in range(len(tarr)):
        col=column_st(b, L, h_st,t_st,w_st, tarr[i])
        loc = local_buckling(c_r, b_st, tarr[i])*tarr[i]
        diff[i] = col - loc
    return diff[0]


def flange_loc_loc(b, c_r, L, b_st, t_st,w_st,t):
    tarr = t_arr(b, L,t)
    diff = np.zeros(len(tarr))
    flange = flange_buckling(t_st, w_st)
    for i in range(len(tarr)):
        loc = local_buckling(c_r, b_st, tarr[i])
        diff[i] = flange - loc
    return diff[0]


def web_flange(b,c_r, L,b_st, h_st,t_st,t):
    tarr = t_arr(b, L,t)
    diff = np.zeros(len(tarr))
    web = web_buckling(t_st, h_st)
    for i in range(len(tarr)):
        loc = local_buckling(c_r, b_st, tarr[i])
        diff[i] =web-loc
    return diff[0]


def von_Mises(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,tarr):
    # vm = np.zeros(len(tarr))
    Nxy=N_xy(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,tarr)
    bend_stress=N_x(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,tarr)[1] #
    tau_shear_arr = Nxy/tarr
    vm_lst = sigma_yield - np.sqrt(0.5 * (3 * tau_shear_arr ** 2+bend_stress**2))
    # for i in range(len(tarr)):
    #     tau_shear= Nxy[i] / tarr[i]
    #     vm[i]=sigma_yield-sqrt(0.5 * (3 * tau_shear ** 2+bend_stress[i]**2))
    return vm_lst[0]



def crippling(b,L, h_st,t_st,w_st,t):
    tarr = t_arr(b, L,t)
    crip= np.zeros(len(tarr))
    A = area_st(h_st, t_st, w_st)
    for i in range(len(tarr)):
        col = column_st(b, L,h_st,t_st,w_st,tarr[i])
        crip[i] = t_st* beta *sigma_yield* ((g * t_st ** 2 / A) * sqrt(E / sigma_yield)) ** m_crip - col
    return crip[0]


def post_buckling(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st, t):
    f=f_ult(b,c_r,L,b_st,h_st,t_st,w_st,t)
    ratio=2/(2+1.3*(1-1/pb))
    px= n_max*shear_force(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    diff=np.subtract(ratio*f,px)
    return diff[0]


def compute_volume():
    pass





class WingboxOptimizerDeprecated():
    """OUTDATED
    """    
    def __init__(self, x0, wing, engine,  max_iter= 500):
        """Initialisze the wingbox optimization

        :param x0:  [ tsp, trib, L, bst, hst, tst, wst, t]
        :type x0: list type
        :param wing: Wingclass from data structures
        :type wing: Wing class
        :param max_iter: The maximum amount of iterations that will be applied, defaults to 500
        :type max_iter: int, optional
        """        
        #enforce equal stringer size (this was an assumption made to simplify our process)
        # x0[6] = x0[4]
        self.x0 = np.array(x0)
        self.wing =  wing
        self.engine = engine
        self.max_iter = max_iter
        self.design_lst = []
        self.multiplier_lst = np.linspace(1,0,max_iter)
    
    def check_constraints(self,x):

        constr = [
        global_local(self.wing.span, self.wing.chord_root, x[2], x[3], x[4], x[5],[x[7]]),
        post_buckling(self.wing.span, self.wing.chord_root, x[0], x[1],  x[2], x[3], x[4], x[5], x[6], [x[7]]),
        von_Mises(self.wing.span, self.wing.chord_root, x[0], x[1], x[2], x[3], x[4], x[5],x[6],[x[7]], self.engine, self.wing),
        buckling_constr(self.wing.span, self.wing.chord_root, x[0], x[1], x[2], x[3], x[4], x[5],x[6],[x[7]], self.engine, self.wing),
        flange_loc_loc(self.wing.span, self.wing.chord_root, x[2], x[3],x[5],x[6],[x[7]]),
        local_column(self.wing.span, self.wing.chord_root, x[2], x[3],x[4],x[5],x[6],[x[7]]),
        crippling(self.wing.span,  x[2],  x[4], x[5], x[6], [x[7]]), #ONLY
        web_flange(self.wing.span, self.wing.chord_root, x[2], x[3], x[4], x[5], [x[7]])
        ]

        return np.array(constr) > 0
    
    def multiplier(self, iter):
        """Creates a multiplier coefficient based on your current iteration and whether you want to increase or decrease the value, becoming
        more refined the closer one gets to the maximum amount of iterations

        :param iter: current iteration
        :type iter: integer
        :return: multiplier
        :rtype: float 
        """

        return self.multiplier_lst[iter]
    
    def compute_weight(self, x):
        return wing_weight(self.wing.span, self.wing.chord_root,x[0],x[1], x[2], x[3], x[4], x[5],x[6],[x[7]])

    def edit_design(self,x, bool_array, iter):
        """Function edits the design based on the failure of design.

        :param bool_array: Output from constraints method
        :type bool_array: array with booleans only
        :return: _description_
        :rtype: _type_
        """        
        print(bool_array)
        print(x)
        #TODO increase whatever influences them see check constraints
        #TODO bit more thought into what to increment
                # :param x0:  [ tsp, trib, L, bst, hst, tst, wst, t]

        if bool_array.all():
            self.design_lst.append((x,iter))
            new_x = x + np.array([-5e-4,-1e-4, 1e-2, 5e-3, -5-3, -1e-4, -5e-3, -1e-3 ])*self.multiplier(iter)
        elif not bool_array[0]:   
            new_x = x + np.array([0,0, 0, -1e-3, 5e-4 , 5e-4, 5e-4, 0 ])*self.multiplier(iter)
        elif not bool_array[1]:   
            new_x = x + np.array([1e-3,0, 0, -1e-3, 5e-4 , 5e-4, 5e-4, 0 ])*self.multiplier(iter)
        elif not bool_array[2]:   
            new_x = x + np.array([1e-3,1e-3, -5e-3, -1e-3, 5e-4 , 1e-4, 1e-4, 1e-4 ])*self.multiplier(iter)
        elif not bool_array[3]:   
            new_x = x + np.array([1e-3,1e-3, -5e-3, -1e-3, 1e-3, 5e-4, 1e-3, 0])*self.multiplier(iter)
        elif not bool_array[4]:   
            new_x = x + np.array([0,0, 0, 0, 0, 1e-3, 1e-3, 0])*self.multiplier(iter)
        elif not bool_array[5]:   
            new_x = x + np.array([0,0, 0, 1e-3, 0, 0, 0, -5e-4])*self.multiplier(iter) #TODO increase spar and shiz as well
        elif not bool_array[6]:   
            new_x = x + np.array([0,0, 1e-2, 0, 0, 0, 0, 0])*self.multiplier(iter)
        elif not bool_array[7]:   
            new_x = x + np.array([0,0, 0, 0, 0, 1e-3, 0, 0])*self.multiplier(iter)
        else: 
            raise Exception("Error has been raised, code should not have reached this line.")

        return new_x


    def optimize(self):
        """Optimizes the wing weight in a rough fashion using our own crude optimizer
        Assumptions

        - stringer width and height are equal
        - Only wing torsion and lift forces are used
        - wingbox is symmetric

        :param x0: Initial estimate X = [tsp, trib, L, bst, hst, tst, wst, t]
        :type x0: array type
        :param wing: wing class from data structures
        :type wing: wing class
        """    

        x = self.x0

        for iter in range(self.max_iter):
            constr = self.check_constraints(x)
            new_x = self.edit_design(x, constr, iter)
            x = new_x

        weight_lst = [] 
        for design in self.design_lst:
            weight_lst.append(self.compute_weight(design[0]))
        weight_lst = np.array(weight_lst)
        idx = np.where(weight_lst == np.min(weight_lst))[0]
        optimum_design = self.design_lst[idx[0]][0]

        print(f"Iteration = {self.design_lst[idx[0]][1]}")
        print(f"thickness spar= {optimum_design[0]*1000} [mm]")
        print(f"thickness rib= {optimum_design[1]*1000} [mm]")
        print(f"rib pitch= {optimum_design[2]*1000} [mm]")
        print(f"stringer pitch= {optimum_design[3]*1000} [mm]")
        print(f"stringer height= {optimum_design[4]*1000} [mm]")
        print(f"stringer thcikness= {optimum_design[5]*1000} [mm]")
        print(f"stringer width= {optimum_design[6]*1000} [mm]")
        print(f"skin thickness= {optimum_design[7]*1000} [mm]")

        return  optimum_design, weight_lst[idx]


class Wingbox_optimization(om.ExplicitComponent):
    def __init__(self, wing, engine):
        super().__init__(self)
        self.wing =  wing
        self.engine = engine



    def setup(self):

        # Design variables
        self.add_input('tsp')
        self.add_input('trib')
        self.add_input('L')
        self.add_input('bst')
        self.add_input('hst')
        self.add_input('tst')
        self.add_input('wst')
        self.add_input('t')

        # Constant inputs
        self.add_input('b')
        self.add_input('c_r')
        
        #Outputs used as constraints
        self.add_output('wing_weight')
        self.add_output('global_local')
        self.add_output('post_buckling')
        self.add_output('von_mises')
        self.add_output('buckling_constr')
        self.add_output('flange_loc_loc')
        self.add_output('local_column')
        self.add_output('crippling')
        self.add_output("web_flange")


    def setup_partials(self):

        # Partial derivatives are done using finite difference
        self.declare_partials('*', '*', 'fd')

    def compute(self, inputs, outputs):

        # Design variables
        tsp = inputs['tsp'][0]
        trib = inputs['trib'][0]
        L = inputs['L'][0] #FIXME Use a fixed rib pitch
        bst = inputs['bst'][0] #FIXME 
        hst = inputs['hst'][0]
        tst = inputs['tst'][0]
        wst = inputs['wst'][0]
        t = inputs['t'][0]

        # Constants
        span = inputs['b'][0]
        chord_root = inputs['c_r'][0]

        weight = wing_weight(span, chord_root,tsp,trib, L, bst, hst, tst,wst,[t])

        constr = [
        global_local(span, chord_root, L, bst, hst, tst,[t]),
        post_buckling(span, chord_root, tsp, trib,  L, bst, hst, tst, wst, [t]),
        von_Mises(span, chord_root, tsp, trib, L, bst, hst, tst,wst,[t]),
        buckling_constr(span, chord_root, tsp, trib, L, bst, hst, tst,wst,[t]),
        flange_loc_loc(span, chord_root, L, bst,tst,wst,[t]),
        local_column(span, chord_root, L, bst,hst,tst,wst,[t]),
        crippling(span,  L,  hst, tst, wst, [t]), #ONLY
        web_flange(span, chord_root, L, bst, hst, tst, [t])
        ]




        outputs['wing_weight'] = weight
        outputs['global_local'] = constr[0]
        outputs['post_buckling'] = constr[1]
        outputs['von_mises'] = constr[2]
        outputs['buckling_constr'] = constr[3]
        outputs['flange_loc_loc'] = constr[4]
        outputs['local_column'] = constr[5]
        outputs['crippling'] = constr[6]
        outputs['web_flange'] = constr[7]

        str_lst =  np.array(["Global local", "Post buckling", "Von Mises", 
                    "Buckling", "Flange loc loc", "Local column",
                    "Crippling", "Web flange"])

        print('===== Progress update =====')
        print(f"Current weight = {weight} [kg]")
        print(f"The failing constraints were {str_lst[np.array(constr) < 0]}")



def WingboxOptimizer(x, wing, engine):
    """ sets up optimziation procedure and runs the driver

    :param x: Initial estimate X = [tsp, trib, L, bst, hst, tst, wst, t]
    :type x: nd.array
    :param wing: wing class from data structure
    :type wing: wing class
    :param engine: engine class from data structures
    :type engine: engine class
    """    


    prob = om.Problem()
    prob.model.add_subsystem('wingbox_design', Wingbox_optimization())#, promotes_inputs=['AR1',
                                                                                        # 'AR2',

    # Initial values for the optimization 

    #Constants
    prob.model.set_input_defaults('wingbox_design.b', wing.span)
    prob.model.set_input_defaults('wingbox_design.c_r', wing.chord_root)
    # prob.model.set_input_defaults('wingbox_design.engine', engine)
    # prob.model.set_input_defaults('wingbox_design.wing', wing)

    # Initial estimate for the design variables
    prob.model.set_input_defaults('wingbox_design.tsp', x[0])
    prob.model.set_input_defaults('wingbox_design.trib', x[1])
    prob.model.set_input_defaults('wingbox_design.L', x[2])
    prob.model.set_input_defaults('wingbox_design.bst', x[3])
    prob.model.set_input_defaults('wingbox_design.hst', x[4])
    prob.model.set_input_defaults('wingbox_design.tst', x[5])
    prob.model.set_input_defaults('wingbox_design.wst', x[6])
    prob.model.set_input_defaults('wingbox_design.t', x[7])


    # Define constraints 
    prob.model.add_constraint('wingbox_design.global_local', lower=0.)
    prob.model.add_constraint('wingbox_design.post_buckling', lower=0.)
    prob.model.add_constraint('wingbox_design.von_mises', lower=0.)
    prob.model.add_constraint('wingbox_design.buckling_constr', lower=0.)
    prob.model.add_constraint('wingbox_design.flange_loc_loc', lower=0.)
    prob.model.add_constraint('wingbox_design.local_column', lower=0.)
    prob.model.add_constraint('wingbox_design.crippling', lower=0.)
    prob.model.add_constraint('wingbox_design.web_flange', lower=0.)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.opt_settings['maxiter'] = 1000

    prob.model.add_design_var('wingbox_design.tsp', lower = 0., upper= 0.1)
    prob.model.add_design_var('wingbox_design.trib', lower = 0., upper= 0.1)
    prob.model.add_design_var('wingbox_design.L', lower = 0., upper=   1)
    prob.model.add_design_var('wingbox_design.bst',  lower = 0., upper= 0.5)
    prob.model.add_design_var('wingbox_design.hst', lower = 0. , upper= 0.4)
    prob.model.add_design_var('wingbox_design.tst', lower = 0., upper= 0.1)
    prob.model.add_design_var('wingbox_design.wst', lower = 0., upper= 0.4 )
    prob.model.add_design_var('wingbox_design.t', lower = 0., upper= 0.1)

    prob.model.add_objective('wingbox_design.wing_weight')

    prob.setup()
    prob.run_driver()

    print(f"thickness spar= {prob.get_val('wingbox_design.tsp')*1000} [mm]")
    print(f"thickness rib= {prob.get_val('wingbox_design.trib')*1000} [mm]")
    print(f"rib pitch= {prob.get_val('wingbox_design.L')*1000} [mm]")
    print(f"stringer pitch= {prob.get_val('wingbox_design.bst')*1000} [mm]")
    print(f"stringer height= {prob.get_val('wingbox_design.hst')*1000} [mm]")
    print(f"stringer thickness= {prob.get_val('wingbox_design.tst')*1000} [mm]")
    print(f"stringer width= {prob.get_val('wingbox_design.wst')*1000} [mm]")
    print(f"skin thickness= {prob.get_val('wingbox_design.t')*1000} [mm]")
    print(f"Wing weight= {prob.get_val('wingbox_design.wing_weight')} [kg]")


    output_lst =  np.array([
    prob.get_val('wingbox_design.tsp'),
    prob.get_val('wingbox_design.trib'),
    prob.get_val('wingbox_design.L'),
    prob.get_val('wingbox_design.bst'),
    prob.get_val('wingbox_design.hst'),
    prob.get_val('wingbox_design.tst'),
    prob.get_val('wingbox_design.wst'),
    prob.get_val('wingbox_design.t')
    ])

    return output_lst



