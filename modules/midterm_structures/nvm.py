# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import matplotlib.pyplot as plt
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from input.GeneralConstants import *

def pos(lst):
    return [x for x in lst if x > 0] or None

def dist_to_force(dist):
    F, f = [], 0
    for i in range(len(dist)):
        f += dist[i]
        F.append(f)
    F = F - F[-1]
    return F

def force_to_moment(F):
    M, m = [], 0
    for i in range(len(F)):
        m += F[i]*dy
        M.append(m)
    M = M - M[-1]
    return M

def c_span(c_r, c_t): return c_r + 2*(c_t-c_r)/b*span



dict_directory = str(list(pl.Path(__file__).parents)[2])+"\\input"          #determine file path
dict_name = ["J1_constants.json",  "L1_constants.json","W1_constants.json"] #define list with all the constants for each configuration

with open(dict_directory + "\\" + dict_name[0], "r") as jsonFile:
    data = json.loads(jsonFile.read())


dy = 0.001

b, mtom, ww, y_eng, cd,S, n_eng = data["b"], data["mtom"], data["wing_weight"], pos(data["y_rotor_loc"])[:-1], data["cd"], data["S"], len(data["y_rotor_loc"])
w_eng = (data["nacelle_weight"]+data["powertrain_weight"])/n_eng
c_r, c_t = data["c_root"], data["c_tip"]




span = np.arange(0, b/2 + dy, dy)
drag = 0.5*rho_cr*v_cr**2*cd*S

lift_dist = np.ones(len(span))*mtom*g0/b*dy
drag_dist = np.ones(len(span))*drag/b*dy
wing_dist = np.ones(len(span))*ww*g0/b*dy

indices = np.searchsorted(span, y_eng)
engine_dist = np.zeros_like(span)
engine_dist[indices] = w_eng*g0

thrust_dist = np.zeros_like(span)
thrust_dist[indices] = drag/n_eng

total_dist_z = - lift_dist + wing_dist + engine_dist
total_dist_x = thrust_dist - drag_dist

V_z = dist_to_force(total_dist_z)
V_x = dist_to_force(total_dist_x)

M_x = force_to_moment(V_z)
M_z = force_to_moment(V_x)

W_vz = dist_to_force(wing_dist)
E_vz = dist_to_force(engine_dist)
L_vz = dist_to_force(-lift_dist)

E_mx = force_to_moment(E_vz)

plt.plot(span, M_x/1000, color='blue')
plt.plot(span, V_z/1000, color='darkblue')
plt.plot(span, W_vz/1000,alpha=0.5)
plt.plot(span, E_vz/1000,alpha=0.5)
plt.plot(span, L_vz/1000,alpha=0.5)
plt.grid()
plt.show()

plt.plot(span, M_z/1000, color='red')
plt.plot(span, V_x/1000, color='darkred')
plt.grid()
plt.show





