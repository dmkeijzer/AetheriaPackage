# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import matplotlib.pyplot as plt
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from input.GeneralConstants import *

dy = 0.001

def pos(lst):
    return [x for x in lst if x > 0] or None

def dist_to_force(dist):
    F, f = [], 0
    for i in range(len(dist)):
        f += dist[i]
        F.append(f)
    F -= F[-1]
    return F

def force_to_moment(F):
    M, m = [], 0
    for i in range(len(F)):
        m += F[i]*dy
        M.append(m)
    M -= M[-1]
    return M

def c_span(c_r, c_t): return c_r + 2*(c_t-c_r)/b*span


def wing_root_cruise(dict_directory, dict_name, PRINT=False, ULTIMATE=False):
    with open(dict_directory + "\\" + dict_name, "r") as jsonFile:
        data = json.loads(jsonFile.read())

    mtom, cd, S, n_eng, cm, x_eng = data["mtom"],  data["cd"], data["S"], len(data["y_rotor_loc"]), data["cm"], data["x_rotor_loc"]
    w_eng = (data["nacelle_weight"]+data["powertrain_weight"])/n_eng

    if ULTIMATE:
        nult = data["n_ult"]
    else:
        nult = 1

    if data["name"] == "J1":
        b, c_r, c_t = data["b"], data["c_root"], data["c_tip"]
        x_eng, y_eng = pos(data["x_rotor_loc"])[:-1], pos(data["y_rotor_loc"])[:-1]
        ww = data["wing_weight"]
        W = mtom*g0
        d_i = 1
    if data["name"] == "L1":
        b, c_r, c_t = data["b2"], data["c_root2"], data["c_tip2"]
        x_eng, y_eng = pos(data["x_rotor_loc"])[6:18], pos(data["y_rotor_loc"])[6:18]
        ww = data["wing2_weight"]
        S1, S2 = data["S1"], data["S2"]
        W = mtom * S2/S *g0
        d_i = S2/S
    if data["name"] == "W1":
        b, c_r, c_t = data["b2"], data["c_root2"], data["c_tip2"]
        x_eng, y_eng = pos(data["x_rotor_loc"])[3:6], pos(data["y_rotor_loc"])[3:6]

        x_eng, y_eng = pos(data["x_rotor_loc"])[int(len(data["x_rotor_loc"])/4):int(len(data["x_rotor_loc"])/2)], pos(data["y_rotor_loc"])[3:6]
        ww = data["wing2_weight"]
        S1, S2 = data["S1"], data["S2"]
        W = mtom * S2/S *g0
        d_i = S2/S
    # spanwise coordinates
    span = np.arange(0, b/2 + dy, dy)

    # drag during cruise
    drag = 0.5*rho_cr*v_cr**2*cd*S*d_i

    # lift distribution during cruise
    lift_dist = np.ones(len(span))*W/b*dy*nult

    # drag distribution during cruise
    drag_dist = np.ones(len(span))*drag/b*dy

    # wing weight distribution
    wing_dist = np.ones(len(span))*ww*d_i*g0/b*dy

    # engine weight
    indices = np.searchsorted(span, y_eng)
    engine_dist = np.zeros_like(span)
    engine_dist[indices-1] = w_eng*g0

    torque_eng = np.zeros_like(span)
    if data["name"] == "J1":
        for i in range(len(y_eng)):
            torque_eng[0:indices[i]] += -w_eng*g0*(x_eng[2*i]-x_eng[2])

    # thrust per engine
    thrust_dist = np.zeros_like(span)
    thrust_dist[indices-1] = drag/n_eng

    # total force distribution in z and x direction
    total_dist_z =  (lift_dist - wing_dist - engine_dist)
    total_dist_x = thrust_dist - drag_dist

    # shear force along span in z and x direction
    V_z = dist_to_force(total_dist_z)
    V_x = dist_to_force(total_dist_x)
    F_m = cm*0.5*rho_cr*v_cr**2*(c_r + 2*(c_t-c_r)/b*span)**2*nult

    # moment along span around x and z axis
    M_x = force_to_moment(V_z)
    M_z = force_to_moment(V_x)
    T = force_to_moment(F_m) + torque_eng
    W_vz = dist_to_force(-wing_dist)
    E_vz = dist_to_force(-engine_dist)
    L_vz = dist_to_force(lift_dist)

    ### PLOTTING ###
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)

    # Plot Vz in the first subplot
    ax1.plot(span, V_z / 1000, color='darkblue', label='Vz')
    ax1.set_ylabel('Vz [kN]')
    ax1.legend()
    ax1.grid()

    # Plot Mx in the second subplot
    ax2.plot(span, M_x / 1000, color='blue', label='Mx')
    ax2.set_ylabel('Mx [kNm]')
    ax2.legend()
    ax2.grid()

    # Plot Vx in the third subplot
    ax3.plot(span, V_x / 1000, color='darkred', label='Vx')
    ax3.set_ylabel('Vx [kN]')
    ax3.legend()
    ax3.grid()

    # Plot Mz in the fourth subplot
    ax4.plot(span, M_z / 1000, color='red', label='Mz')
    ax4.set_ylabel('Mz [kNm]')
    ax4.legend()
    ax4.grid()

    # Plot T in the fifth subplot
    ax5.plot(span, T / 1000, color='green', label='T')
    ax5.set_xlabel('Span [m]')
    ax5.set_ylabel('T [kN]')
    ax5.legend()
    ax5.grid()

    # Set the title for the overall figure
    plt.suptitle(str(data["name"] + " Cruise"))

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.show()

    ### PRINTING ###
    if PRINT:
        print("For ", data["name"], " during cruise :")
        print("Vx at root: ", round(V_x[0]/1000,1), 'kN')
        print("Vz at root: ", round(V_z[0]/1000,1), 'kN')
        print("Mx at root: ", round(M_x[0]/1000,1), 'kNm')
        print("Mz at root: ", round(M_z[0]/1000,1), 'kNm')
        print("T at root: ", round(T[0]/1000,1), 'kNm')
        print("--------------------")

    data["Vx_cr"], data["Vz_cr"], data["Mx_cr"], data["Mz_cr"], data["T_cr"], = V_x[0], V_z[0], M_x[0], M_z[0], T[0]
    write_bool = int(input("Do you want to overwrite the current loading values? type 1 if you want to do this.")) == 1
    if write_bool==True:
        with open(dict_directory+"\\"+dict_name, "w") as jsonFile:
            json.dump(data, jsonFile,indent=2)
        print("Old files were overwritten.")

    return V_x, V_z, M_x, M_z, T

def wing_root_hover(dict_directory, dict_name, PRINT=False):
    with open(dict_directory + "\\" + dict_name, "r") as jsonFile:
        data = json.loads(jsonFile.read())

    mtom, cd, S, n_eng, cm, x_eng = data["mtom"],  data["cd"], data["S"], len(data["y_rotor_loc"]), data["cm"], data["x_rotor_loc"]
    w_eng = (data["nacelle_weight"]+data["powertrain_weight"])/n_eng

    if data["name"] == "J1":
        b, c_r, c_t = data["b"], data["c_root"], data["c_tip"]
        x_eng, y_eng = pos(data["x_rotor_loc"])[:-1], pos(data["y_rotor_loc"])[:-1]
        ww = data["wing_weight"]
        W = mtom*g0
        d_i = 1
    if data["name"] == "L1":
        b, c_r, c_t = data["b2"], data["c_root2"], data["c_tip2"]
        x_eng, y_eng = pos(data["x_rotor_loc"])[6:18], pos(data["y_rotor_loc"])[6:18]
        ww = data["wing2_weight"]
        S1, S2 = data["S1"], data["S2"]
        W = mtom * S2/S *g0
        d_i = S2/S
    if data["name"] == "W1":
        b, c_r, c_t = data["b2"], data["c_root2"], data["c_tip2"]
        x_eng, y_eng = pos(data["x_rotor_loc"])[3:6], pos(data["y_rotor_loc"])[3:6]

        x_eng, y_eng = pos(data["x_rotor_loc"])[int(len(data["x_rotor_loc"])/4):int(len(data["x_rotor_loc"])/2)], pos(data["y_rotor_loc"])[3:6]
        ww = data["wing2_weight"]
        S1, S2 = data["S1"], data["S2"]
        W = mtom * S2/S *g0
        d_i = S2/S
    # spanwise coordinates
    span = np.arange(0, b/2 + dy, dy)

    # drag during cruise
    drag = 0.5*rho_cr*v_cr**2*cd*S*d_i

    # lift distribution during cruise
    lift_dist = np.ones(len(span))*W/b*dy

    # drag distribution during cruise
    drag_dist = np.ones(len(span))*drag/b*dy

    # wing weight distribution
    wing_dist = np.ones(len(span))*ww*d_i*g0/b*dy

    # engine weight
    indices = np.searchsorted(span, y_eng)
    engine_dist = np.zeros_like(span)
    engine_dist[indices-1] = w_eng*g0

    torque_eng = np.zeros_like(span)
    if data["name"] == "J1":
        for i in range(len(y_eng)):
            torque_eng[0:indices[i]] += -w_eng*g0*(x_eng[2*i]-x_eng[2])

    # thrust per engine
    thrust_dist = np.zeros_like(span)
    thrust_dist[indices-1] = mtom*g0/n_eng

    # total force distribution in z and x direction
    total_dist_z =  (thrust_dist - wing_dist - engine_dist)
    #total_dist_x = thrust_dist - drag_dist

    # shear force along span in z and x direction
    V_z = dist_to_force(total_dist_z)
    #V_x = dist_to_force(total_dist_x)
    #F_m = cm*0.5*rho_cr*v_cr**2*(c_r + 2*(c_t-c_r)/b*span)**2*nult

    # moment along span around x and z axis
    M_x = force_to_moment(V_z)
    #M_z = force_to_moment(V_x)
    T = torque_eng
    W_vz = dist_to_force(-wing_dist)
    E_vz = dist_to_force(-engine_dist)
    T_vz = dist_to_force(thrust_dist)

    ### PLOTTING ###
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    # Plot Vz in the first subplot
    ax1.plot(span, V_z / 1000, color='darkblue', label='Vz')
    ax1.set_ylabel('Vz [kN]')
    ax1.legend()
    ax1.grid()

    # Plot Mx in the second subplot
    ax2.plot(span, M_x / 1000, color='blue', label='Mx')
    ax2.set_ylabel('Mx [kNm]')
    ax2.legend()
    ax2.grid()

    # Plot T in the third subplot
    ax3.plot(span, T / 1000, color='green', label='T')
    ax3.set_xlabel('Span [m]')
    ax3.set_ylabel('T [kN]')
    ax3.legend()
    ax3.grid()

    # Set the title for the overall figure
    plt.suptitle(str(data["name"] + " Hover"))

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.show()


    ### PRINTING ###
    if PRINT:
        print("For ", data["name"], " during hover :")
        print("Vz at root: ", round(V_z[0]/1000,1), 'kN')
        print("Mx at root: ", round(M_x[0]/1000,1), 'kNm')
        print("T at root: ", round(T[0]/1000,1), 'kNm')
        print("--------------------")

    data["Vz_vf"], data["Mx_vf"], data["T_vf"], = V_z[0], M_x[0], T[0]
    write_bool = int(input("Do you want to overwrite the current loading values? type 1 if you want to do this.")) == 1
    if write_bool==True:
        with open(dict_directory+"\\"+dict_name, "w") as jsonFile:
            json.dump(data, jsonFile,indent=2)
        print("Old files were overwritten.")

    return V_z, M_x, T