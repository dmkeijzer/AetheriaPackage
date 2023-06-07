import matplotlib.pyplot as plt
import os
import json
import sys
import numpy as np
import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import input.data_structures.GeneralConstants as const
from input.data_structures.aero import Aero
from input.data_structures.wing import Wing
from modules.avlwrapper import Geometry, Surface, Section, NacaAirfoil, Control, Point, Spacing, Session, Case, Parameter


Wingclass = Wing()
Wingclass.load()
Aeroclass = Aero()
Aeroclass.load()

dict_directory = "input/data_structures"
dict_name = "aetheria_constants.json"
with open(os.path.join(dict_directory, dict_name)) as f:
    data = json.load(f)

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")


def target_accelerations_new(vx, vy, y, y_tgt, vx_tgt, max_ax, max_ay, max_vy):

    # Limit altitude
    vy_tgt = np.maximum(np.minimum(-0.5 * (y - y_tgt), max_vy), -max_vy)

    # Slow down when approaching 15 m while going too fast in horizontal direction
    if 15 + (np.abs(vy) / ay_target_descend) > y > y_tgt and abs(vx) > 0.25:
        vy_tgt = 0

    # Keep horizontal velocity zero when flying low
    if y < 10:
        vx_tgt_1 = 0
    else:
        vx_tgt_1 = vx_tgt

    # Limit speed
    ax_tgt = np.minimum(np.maximum(-0.5 * (vx - vx_tgt_1), -max_ax), max_ax)
    ay_tgt = np.minimum(np.maximum(-0.5 * (vy - vy_tgt), -max_ay), max_ay)

    return ax_tgt, ay_tgt


def numerical_simulation(vx_start, y_start, mass, g0, S, CL_max, alpha_stall, CD):
    print('this is still running')
    # Initialization
    vx = 0
    vy = 2.
    t = 0
    x = 0
    y = y_start
    ay = 0
    ax = 0
    alpha_T = 0.5*np.pi  # starting propeller angle in rad
    dt = 0.1
    v_stall = 40
    vy0 = 2
    #gamma = np.arctan2(climb_gradient,1)

    t_end = 300

    # Check whether the aircraft needs to climb or descend
    # if y_start > y_tgt:
    # max_vy = roc
    # max_ax = ax_target_descend
    # max_ay = ay_target_descend

    # else:
    #     max_vy = const.roc
    #     max_ax = ax_target_climb
    #     max_ay = ay_target_climb

    # Lists to store everything
    y_lst = []
    x_lst = []
    vy_lst = []
    vx_lst = []
    alpha_T_lst = []
    T_lst = []
    t_lst = []
    ax_lst = []
    D_lst = []
    rho_lst = []

    # Preliminary calculations
    running = True
    while running:

        t += dt
        alpha_T = 0.5*np.pi - (t/t_end)*(0.5*np.pi-alpha_stall)
        rho = 1.220  # PLACEHOLDER

        # ======== Actual Simulation ========

        #lift and drag
        CL = CL_max

        L = 0.5*rho*vx*vx*S*CL
        D = 0.5*rho*vx*vx*S*CD

        ay = ( 0.125*v_stall-vy0) / t_end
        # thrust
        T = (mass*g0 - L*np.cos(alpha_stall) + mass*ay)/np.sin(alpha_T)
        V = np.sqrt(vx**2 + vy**2)
        # ax
        ax = (T * np.cos(alpha_T) - L*np.sin(alpha_stall) - D)/mass

        vx = vx + ax*dt
        vy = vy + ay*dt

        x = x + vx*dt
        y = y + vy*dt

        # append lists
        y_lst.append(y)
        x_lst.append(x)
        vy_lst.append(vy)
        vx_lst.append(vx)
        alpha_T_lst.append(alpha_T*180/np.pi)
        T_lst.append(T*V/1000)
        t_lst.append(t)
        ax_lst.append(ax)
        D_lst.append(D)
        rho_lst.append(rho)

        if t > t_end:
            running = False
    plt.plot(t_lst, T_lst)
    #plt.axis('equal')
    plt.show()

    return y_lst, x_lst, vy_lst, vx_lst, t_lst, alpha_T_lst, T_lst


print(np.max(numerical_simulation(vx_start=0, y_start=30.5,
      mass=data["mtom"], g0=const.g0, S=data['S'], CL_max=data["cLmax_flaps20"], alpha_stall=0.1, CD=data["cd"])[6]))
