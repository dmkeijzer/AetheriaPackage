from math import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import pathlib as pl
import os

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




def n_st(c_r, b_st):
    return ceil(0.6 * c_r / b_st) + 1



def n_ribs(b, L):
    return ceil(0.5 * b / L) + 1



def new_L(b, L):
    nr_sect = n_ribs(b, L) - 1
    new_pitch = 0.5 * b / nr_sect
    return new_pitch




def new_bst(c_r, b_st):
    nr_sect = n_st(c_r, b_st) - 1
    new_pitch = c_r / nr_sect
    return new_pitch





def rib_coordinates(b, L):
    L_new = new_L(b, L)
    stations = np.arange(0, b / 2 + L_new, L_new)
    return stations






def I_xx(b,c_r,t_sp,b_st, h_st,t_st,w_st,t_sk):
    h = height(b, c_r)
    nst = n_st(c_r, b_st)
    Ist = I_st(h_st,t_st,w_st)
    Isp = I_sp(b, c_r,t_sp)
    A = area_st(h_st,t_st,w_st)
    i = lambda z: 2 * (Ist + A * (0.5 * h(z)) ** 2) * nst + 2 * Isp(z) + 2 * (0.6 * c_r * t_sk ** 3 / 12 + t_sk * 0.6 * c_r * (0.5 * h(z)) ** 2)
    return i




def t_arr(b, L,t):
    b=abs(b)
    L=abs(L)
    nr_ribs = n_ribs(b, L)
    sections = np.zeros(nr_ribs - 1)

    inte = int((len(sections)) // len(t))
    mod = int((len(sections)) % len(t))
    group = int(len(t) - mod)

    arr = np.arange(inte * group, len(sections), inte + 1)

    for i in range(group):
        for j in range(inte):
            sections[inte * i + j] = t[i]
    for i in range(len(arr)):
        cursor = arr[i]
        for j in range(inte + 1):
            sections[cursor + j] = t[group + i]
    return sections



def rib_weight(b, c_r, t_rib):
    c = chord(b, c_r)
    h = height(b, c_r)
    w_rib = lambda z: 0.6 * c(z) * h(z) * t_rib * rho
    return w_rib





def panel_weight(b, c_r,t_sp, L, b_st, h_st,t_st,w_st,t):
    t_sk = t_arr(b, L,t)
    c = chord(b, c_r)
    h = height(b, c_r)
    nst = n_st(c_r, b_st)
    stations = rib_coordinates(b, L)
    w = np.zeros(len(stations))
    A = area_st(h_st, t_st,w_st)



    for i in range(len(t_sk)):
        vol = lambda z:  rho * (2 * h(z) * t_sp + (pi * (3 * (0.5 * h(z) + 0.15 * c(z)) - sqrt((3 * 0.5 * h(z) + 0.15 * c(z)) * (0.5 * h(z) + 3 * 0.15 * c(z)))) + 2 * 0.6 * c(z) + sqrt(h(z) ** 2 / 4 + (0.25 * c(z)) ** 2)) *t_sk[i] + A * 2 * nst)
        w[i]=trapz([vol(stations[i]),vol(stations[i+1])],[stations[i],stations[i+1]])
    return w



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
    x = rib_interpolation(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[0]
    y = rib_interpolation(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[1]
    f2 = interp1d(x, y)
    x_engine = np.array([0.5 * b / 4, 0.5 * b / 2, 0.5 * 3 * b / 4])
    x_combi = np.concatenate((x, x_engine))
    x_sort = np.sort(x_combi)

    index1 = np.where(x_sort == 0.5 * 3 * b / 4)
    if len(index1[0]) == 1:
        index1 = int(index1[0])
    else:
        index1 = int(index1[0][0])
    y_new1 = f2(x_sort[index1]) + 9.81 * W_eng

    index2 = np.where(x_sort == 0.5 * b / 2)
    if len(index2[0]) == 1:
        index2 = int(index2[0])
    else:
        index2 = int(index2[0][0])
    y_new2 = f2(x_sort[index2]) + 9.81 * W_eng

    index3 = np.where(x_sort == 0.5 * b / 4)
    if len(index3[0]) == 1:
        index3 = int(index3[0])
    else:
        index3 = int(index3[0][0])
    y_new3 = f2(x_sort[index3]) + 9.81 * W_eng

    y_engine = np.ndarray.flatten(np.array([y_new1, y_new2, y_new3]))
    y_combi = np.concatenate((y, y_engine))
    y_sort = np.sort(y_combi)
    y_sort = np.flip(y_sort)

    for i in range(int(index1)):
        y_sort[i] = y_sort[i] + 9.81 * W_eng
    for i in range(int(index2)):
        y_sort[i] = y_sort[i] + 9.81 * W_eng
    for i in range(int(index3)):
        y_sort[i] = y_sort[i] + 9.81 * W_eng

    return x_sort, y_sort, index1, index2, index3



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
        d = [diff > 0]
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
    sta = rib_coordinates(b, L)
    moment = m_eng(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[1]
    x_sort = m_eng(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[0]
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
        Nx[i] = bend_stress[i] * tarr[i]
    return Nx, bend_stress







def shear_force(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    shear = shear_eng(b, c_r, t_sp, t_rib, L, b_st, h_st, t_st, w_st,t)[1]
    tarr = t_arr(b, L,t)
    Vz = np.zeros(len(tarr))

    sta = rib_coordinates(b, L)
    aero= lambda y:-151.7143*9.81*y+531*9.81
    for i in range(len(tarr)):
        Vz[i] = aero(sta[i])-shear[2 * i]
    return Vz

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
    n = n_st(c_r, b_st)
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


def f_ult(b,c_r,L,b_st,h_st,t_st,w_st,t):
    A_st = area_st(h_st,t_st,w_st)
    n=n_st(c_r,b_st)
    tarr=t_arr(b,L,t)
    c=chord(b,c_r)
    h=height(b,c_r)
    stations=rib_coordinates(b,L)
    f_uts=np.zeros(len(tarr))
    for i in range(len(tarr)):
        A=n*A_st+0.6*c(stations[i])*tarr[i]
        f_uts[i]=sigma_uts*A
    return f_uts




def buckling_constr(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    buck = buckling(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    tarr = t_arr(b, L,t)
    vector = np.zeros(len(tarr))
    for i in range(len(tarr)):
        vector[i] = -1 * (buck[i] - 1)
    return vector[0]


def global_local(b, c_r, L, b_st, h_st,t_st,t):
    tarr = t_arr(b, L,t)
    diff = np.zeros(len(tarr))
    for i in range(len(tarr)):
        glob = global_buckling(c_r, b_st, h_st,t_st,tarr[i])
        loc = local_buckling(c_r, b_st,tarr[i])
        diff[i] = glob - loc
    return diff[0]



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

def von_Mises(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t):
    tarr = t_arr(b, L,t)
    vm = np.zeros(len(tarr))
    Nxy=N_xy(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    bend_stress=N_x(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)[1]
    for i in range(len(tarr)):
        tau_shear= Nxy[i] / tarr[i]
        vm[i]=sigma_yield-sqrt(0.5 * (3 * tau_shear ** 2+bend_stress[i]**2))
    return vm[0]



def crippling(b,L, h_st,t_st,w_st,t):
    tarr = t_arr(b, L,t)
    crip= np.zeros(len(tarr))
    A = area_st(h_st, t_st, w_st)
    for i in range(len(tarr)):
        col = column_st(b, L,h_st,t_st,w_st,tarr[i])
        crip[i] = t_st* beta *sigma_yield* ((g * t_st ** 2 / A) * sqrt(E / sigma_yield)) ** m_crip-col
    return crip[0]


def post_buckling(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st, t):
    f=f_ult(b,c_r,L,b_st,h_st,t_st,w_st,t)
    ratio=2/(2+1.3*(1-1/pb))
    px= n_max*shear_force(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    diff=np.subtract(ratio*f,px)
    return diff[0]






def create_bounds(wing):
    """ Returns the bounds for the wingbox_optimization

    :param wing: Wing class from data structures
    :type wing: Wing class
    :return: tuple of tuple with 2 elements
    :rtype: tuple
    """    
    return ((wing.span - 1e-8, wing.span + 1e-8), (wing.chord_root - 1e-8, wing.chord_root + 1e-8), (0.001, 0.005), (0.001, 0.005), (0.007, 0.05), (0.001, 0.01),(0.001, 0.01),(0.001, 0.003),(0.004, 0.005),(0.001, 0.003))

def wingbox_optimization(x0, bounds):
    """ TODO note down assumption that we simulate the load on the wingtip for the engine

    :param x0: Initial estimate Design vector X = [b, cr, tsp, trib, L, bst, hst, tst, wst, t]
    :type x0: 
    :param bounds: Boundareies
    :type: tuple with tuples with 2 elements min and max
    :param material: The material class created in input/data_structures
    :type: Bespoke Material class
    :param wing: the material class created in input/data_structures
    :type: bespoke  wing class
    """    
    # fun = lambda x: wing_weight(x[0], x[1],x[2],x[3], x[4], x[5], x[6], x[7],x[8],[x[9]], material.rho, wing.taper)
    # cons = ({'type': 'ineq', 'fun': lambda x: global_local(x[0], x[1], x[4], x[5], x[6], x[7],[x[9]], material.E, material.poisson)},
    #         {'type': 'ineq', 'fun': lambda x: post_buckling(x[0], x[1], x[2], x[3],  x[4], x[5], x[6], x[7], x[8], [x[9]], const.n_max_req, material.sigma_uts, wing.taper, material.pb, engine.mass_pertotalengine, material.rho,engine.y_rotor_loc)}, #TODO N_max has to badded
    #         {'type': 'ineq', 'fun': lambda x: von_Mises(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],[x[9]], material.sigma_yield, wing.taper, material.rho, engine.mass_pertotalengine,engine.y_rotor_loc)}, # TODO Add sigma yield
    #         {'type': 'ineq', 'fun': lambda x: buckling_constr(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],[x[9]], wing.taper, material.rho, engine.mass_pertotalengine, material.E, material.poisson,engine.y_rotor_loc)},
    #         {'type': 'ineq', 'fun': lambda x: flange_loc_loc(x[0], x[1], x[4], x[5],x[7],x[8],[x[9]], material.E, material.poisson)},
    #         {'type': 'ineq', 'fun': lambda x: local_column(x[0], x[1], x[4], x[5],x[6],x[7],x[8],[x[9]], material.E, material.poisson)},
    #         {'type': 'ineq', 'fun': lambda x: crippling(x[0],  x[4],  x[6], x[7], x[8], [x[9]], material.beta, material.sigma_yield, material.E, material.m_crip, material.g)}, #TODO add beta, sigma yield, E, m_crip
    #         {'type': 'ineq', 'fun': lambda x: web_flange(x[0], x[1], x[4], x[5], x[6], x[7], [x[9]], material.E, material.poisson)})

    fun = lambda x: wing_weight(x[0], x[1],x[2],x[3], x[4], x[5], x[6], x[7],x[8],[x[9]])
    cons = ({'type': 'ineq', 'fun': lambda x: global_local(x[0], x[1], x[4], x[5], x[6], x[7],[x[9]])},
            {'type': 'ineq', 'fun': lambda x: post_buckling(x[0], x[1], x[2], x[3],  x[4], x[5], x[6], x[7], x[8], [x[9]])},
            {'type': 'ineq', 'fun': lambda x: von_Mises(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],[x[9]])},
            {'type': 'ineq', 'fun': lambda x: buckling_constr(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],[x[9]])},
            {'type': 'ineq', 'fun': lambda x: flange_loc_loc(x[0], x[1], x[4], x[5],x[7],x[8],[x[9]])},
            {'type': 'ineq', 'fun': lambda x: local_column(x[0], x[1], x[4], x[5],x[6],x[7],x[8],[x[9]])},
            {'type': 'ineq', 'fun': lambda x: crippling(x[0],  x[4],  x[6], x[7], x[8], [x[9]])},
            {'type': 'ineq', 'fun': lambda x: web_flange(x[0], x[1], x[4], x[5], x[6], x[7], [x[9]])})

    # bnds = ((5, 9), (1, 4), (0.001, 0.005), (0.001, 0.005), (0.007, 0.05), (0.001, 0.01),(0.001, 0.01),(0.001, 0.003),(0.004, 0.005),(0.001, 0.003))
    bnds = bounds
    rez = minimize(fun, x0, method='trust-constr',bounds=bnds, constraints=cons)
    return rez

