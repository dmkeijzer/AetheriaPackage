from math import *
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

taper = 0.45
rho = 2710
W_eng = 41.8

'''
Design vector X=[b,c_r,t,t_sp,t_rib,L,b_st,h_st]
    b-span
    c_-root chord
    t-vector with the variable skin thicknesses
    t_sp-thickness of the spar
    t_rib- thickness of the rib
    L-rib pitch
    b_st-stringer pitch
    h_st-stringer height
'''

'''
Compute chord and height of the cross-section as a function of y-span-wise location
'''


def chord(b, c_r):
    c = lambda y: c_r - c_r * (1 - taper) * y * 2 / b
    return c


def height(b, c_r):
    c = chord(b, c_r)
    h = lambda Y: 0.17 * c(Y)
    return h


'''
Stringer  geometry calculations
    - thickness and flange width from the global buckling curve interpolation requirements
    - stringer area and moment of inertia (without thin-wall assumption)
    - MOI of flange only for flange column buckling calculations
    - I used t_sk in the function definition to emphasize that the entry is a scalar and not the thickness vector
'''


def t_st(t_sk):
    return 0.79 * t_sk


def w_st(h_st):
    return 0.3 * h_st


def area_st(t_sk, h_st):
    tst = t_st(t_sk)
    wst = w_st(h_st)
    return tst * (2 * wst + h_st)


def I_st(t_sk, h_st):
    tst = t_st(t_sk)
    wst = w_st(h_st)
    i = tst * h_st ** 3 / 12 + (wst - tst) * tst ** 3 / 12 + 2 * tst * (wst - tst) * (0.5 * h_st) ** 2
    return i


def I_flange(t_sk, h_st):
    tst = t_st(t_sk)
    wst = w_st(h_st)
    return (wst - tst) * tst ** 3 / 12


# Spar

def w_sp(b, c_r):
    h = height(b, c_r)
    i = lambda z: 0.6 * h(z)
    return i


def I_sp(b, t_sp, c_r):
    h = height(b, c_r)
    wsp = w_sp(b, c_r)
    i = lambda z: t_sp * (h(z) - 2 * t_sp) ** 3 / 12 + 2 * wsp(z) * t_sp ** 3 / 12 + 2 * t_sp * wsp(z) * (
            0.5 * h(z)) ** 2
    return i


'''
Compute number of stringers and ribs to ensure constant pitch
    - first determine the number of stringers and ribs as given by the value of the variables
    - make the stringer and rib pitch constant
    - last function here stores the span-wise locations of the ribs
'''


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


''' Compute total MOI'''


def I_xx(b, t_sk, c_r, h_st, b_st):
    h = height(b, c_r)
    nst = n_st(c_r, b_st)
    Ist = I_st(t_sk, h_st)
    Isp = I_sp(b, t_sk, c_r)
    A = area_st(t_sk, h_st)
    i = lambda z: 2 * (Ist + A * (0.5 * h(z)) ** 2) * nst + 2 * Isp(z) + 2 * (
                0.6 * c_r * t_sk ** 3 / 12 + t_sk * 0.6 * c_r * (0.5 * h(z)) ** 2)
    return i

'''
Create array with thickness for each panel (section)-this function assigns a thickness to each panel from the design vector
'''


def t_arr(b, t, L):
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


'''
Weight calculation and weight distribution [kg]
    -rib weight as a function of span-wise position
    -array with panel weights (skin and stringers between two ribs)
'''

def rib_weight(b, c_r, t_rib):
    c = chord(b, c_r)
    h = height(b, c_r)
    w_rib = lambda z: 0.6 * c(z) * h(z) * t_rib * rho
    return w_rib

def panel_weight(b, c_r, t, t_sp, L, b_st, h_st):
    t_sk = t_arr(b, t, L)
    c = chord(b, c_r)
    h = height(b, c_r)
    nst = n_st(c_r, b_st)
    stations = rib_coordinates(b, L)

    w = np.zeros(len(stations))

    for i in range(len(t_sk)):
        A = area_st(t_sk[i], h_st)
        vol = lambda z: rho * (2 * h(z) * t_sp + (pi * (3 * (0.5 * h(z) + 0.15 * c(z)) - sqrt(
            (3 * 0.5 * h(z) + 0.15 * c(z)) * (0.5 * h(z) + 3 * 0.15 * c(z)))) + 2 * 0.6 * c(z) + sqrt(
            h(z) ** 2 / 4 + (0.25 * c(z)) ** 2)) *
                               t_sk[i] + A *2* nst)
        w[i] = quad(vol, stations[i], stations[i + 1])[0]
    return w

def wing_weight(b, c_r, t, t_sp, t_rib, L, b_st, h_st):
    stations = rib_coordinates(b, L)
    skin_weight = panel_weight(b, c_r, t, t_sp, L, b_st, h_st)
    cumsum = np.sum(skin_weight)
    rbw = rib_weight(b, c_r, t_rib)

    for i in stations:
        cumsum = cumsum + rbw(i)
    return cumsum


# Shear force due to panel weight only
def skin_interpolation(b, c_r, t, t_sp, L, b_st, h_st):
    skin_weight = panel_weight(b, c_r, t, t_sp, L, b_st, h_st)
    skin_weight = np.flip(skin_weight)
    skin_weight = np.cumsum(skin_weight)
    skin_weight = np.flip(skin_weight)
    return skin_weight


# Shear force due to panel and rib weight
def rib_interpolation(b, c_r, t, t_sp, t_rib, L, b_st, h_st):
    f = skin_interpolation(b, c_r, t, t_sp, L, b_st, h_st)
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

def shear_eng(b, c_r, t, t_sp, t_rib, L, b_st, h_st):
    x= rib_interpolation(b, c_r, t, t_sp, t_rib, L, b_st, h_st)[0]
    y= rib_interpolation(b, c_r, t, t_sp, t_rib, L, b_st, h_st)[1]
    f2 = interp1d(x, y)
    x_engine=np.array([0.5*b/4,0.5*b/2,0.5*3*b/4])
    x_combi=np.concatenate((x,x_engine))
    x_sort=np.sort(x_combi)

    index1 = np.where(x_sort == 0.5*3*b/4)
    if len(index1[0])==1:
        index1=int(index1[0])
    else:
        index1=int(index1[0][0])
    y_new1=f2(x_sort[index1])+9.81*W_eng

    index2 = np.where(x_sort == 0.5 *b / 2)
    if len(index2[0]) == 1:
        index2 = int(index2[0])
    else:
        index2 = int(index2[0][0])
    y_new2 = f2(x_sort[index2]) + 9.81*W_eng


    index3 = np.where(x_sort == 0.5 * b / 4)
    if len(index3[0]) == 1:
        index3 = int(index3[0])
    else:
        index3 = int(index3[0][0])
    y_new3 = f2(x_sort[index3]) + 9.81*W_eng



    y_engine = np.ndarray.flatten(np.array([y_new1,y_new2,y_new3]))
    y_combi = np.concatenate((y, y_engine))
    y_sort = np.sort(y_combi)
    y_sort=np.flip(y_sort)

    for i in range(int(index1)):
        y_sort[i]=y_sort[i]+9.81*W_eng
    for i in range(int(index2)):
        y_sort[i]=y_sort[i]+9.81*W_eng
    for i in range(int(index3)):
        y_sort[i]=y_sort[i]+9.81*W_eng

    import matplotlib.pyplot as plt
    plt.plot(x_sort,y_sort)
    plt.show()
    return x_sort,y_sort,index1,index2,index3

print(shear_eng(10, 1.2, np.array([0.002,0.001]), 0.005, 0.005,0.1, 0.05, 0.005))
