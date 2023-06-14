import matplotlib.pyplot as plt
import os
import json
import sys
import numpy as np
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
from modules.avlwrapper import Geometry, Surface, Section, NacaAirfoil, Control, Point, Spacing, Session, Case, \
    Parameter
from input.data_structures.wing import Wing
from input.data_structures.aero import Aero
import input.data_structures.GeneralConstants as const

Wingclass = Wing()
Wingclass.load()
Aeroclass = Aero()
Aeroclass.load()


def numerical_simulation(y_start, mass, g0, S, CL_climb, alpha_climb, CD_climb, Adisk, lod_climb, eff_climb, v_stall):
    # print('this is still running')
    # Initialization
    vx = 0
    vy = 2.
    t = 0
    x = 0
    y = y_start
    dt = 0.01
    E = 0

    # Chosen parameters
    vy0 = 2
    
    # Choose transition time
    t_end = 30

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
    P_lst = []
    acc_lst = []
    L_lst = []

    # Preliminary calculations
    running = True
    while running:

        t += dt
        alpha_T = 0.5 * np.pi - (t / t_end) * (0.5 * np.pi - alpha_climb)
        rho = 1.220

        # ======== Actual Simulation ========

        # Lift and drag
        L = 0.5 * rho * vx * vx * S * CL_climb
        D = 0.5 * rho * vx * vx * S * CD_climb

        vy_end = 0.125 * v_stall  # achieves final climb gradient of 12.5%
        ay = (vy_end - vy0) / t_end

        # Thrust and power
        T = (mass * g0 - L * np.cos(alpha_climb) + mass * ay) / np.sin(alpha_T)
        V = np.sqrt(vx ** 2 + vy ** 2)

        P_hover = T * vy + 1.2 * T * \
            (-(vy / 2) + np.sqrt((vy ** 2 / 4) + (T / (2 * rho * Adisk))))
        P_climb = mass * g0 * \
            (np.sqrt(2 * mass * g0 * (S / rho)) * (1 / lod_climb) + vy) / eff_climb

        Ptot = (P_hover *np.sin(alpha_T)+ P_climb * np.cos(alpha_T)) / 1

        # Acceleration, velocity and position updates
        ax = (T * np.cos(alpha_T) - L * np.sin(alpha_climb) - D) / mass

        vx = vx + ax * dt
        vy = vy + ay * dt

        x = x + vx * dt
        y = y + vy * dt

        # Energy integrand per time step
        E += Ptot * dt

        # Longitudinal acceleration in g-force
        acc_g = (ax)/ g0

        # Append lists for all parameters
        t_lst.append(t)
        ax_lst.append(ax)
        y_lst.append(y)
        x_lst.append(x)
        vy_lst.append(vy)
        vx_lst.append(vx)
        alpha_T_lst.append(alpha_T * 180 / np.pi)
        T_lst.append(T)
        D_lst.append(D)
        P_lst.append(Ptot / 1000)
        acc_lst.append(acc_g)
        L_lst.append(L)

        if t > t_end:
            running = False

    acc_lst = np.array(acc_lst)

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot data on each subplot
    axs[0, 0].plot(t_lst, P_lst, color='blue')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Power [kW]')
    axs[0, 0].grid()

    axs[0, 1].plot(x_lst, y_lst, color='red')
    axs[0, 1].set_ylim(0,200)
    axs[0, 1].set_xlabel('X-position [m]')
    axs[0, 1].set_ylabel('Y-position [m]')
    axs[0, 1].grid()

    axs[1, 0].plot(t_lst, L_lst, color='green')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Lift [N]')
    axs[1, 0].grid()

    axs[1, 1].plot(t_lst, acc_lst, color='orange')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Longitudinal acceleration [g]')
    axs[1, 1].grid()
    
    # Adjust spacing between subplots
    fig.tight_layout()
    
    # Display the figure
    #plt.show()

    return E, y_lst, t_lst, x_lst, V


# print(np.max(numerical_simulation(y_start=30.5, mass=data["mtom"], g0=const.g0, S=data['S'], CL_climb=data['cl_climb_clean'],
#                                   alpha_climb=data['alpha_climb_clean'], CD_climb=data["cdi_climb_clean"] + data["cd0"],
#                                   Adisk=data["diskarea"], lod_climb=data['ld_climb'], eff_climb=data['eff'],
#                                   v_stall=data['v_stall'])[6]))


def numerical_simulation_landing(vx_start, descend_slope, mass, g0, S, CL, alpha, CD, Adisk):
    # print('this is still running')
    # Initialization
    vx = vx_start
    vy = vx_start*descend_slope
    vy_start = vx_start*descend_slope
    vy_end = -2
    y_start = 72
    y_end = 15
    t = 0
    x = 0
    y = y_start
    dt = 0.01
    E = 0

    # Chosen parameters
    vy0 = vx * descend_slope

    # Choose transition time
    t_end = 100

    # Lists to store everything
    y_lst = []
    x_lst = []
    vy_lst = []
    vx_lst = []
    alpha_T_lst = []
    T_lst = []
    t_lst = []
    ax_lst = []
    L_lst = []
    P_lst = []
    acc_lst = []
    acc_y_lst = []
    ay_lst = []
    V_lst =[]
    level_out = False

    # Preliminary calculations
    running = True
    while running:

        t += dt

        rho = 1.220
        # ======== Actual Simulation ========

        # lift and drag
        L = 0.5 * rho * vx * vx * S * CL
        D = 0.5 * rho * vx * vx * S * CD

        # thrust and power
        alpha_T = alpha + 0.5 * np.pi

        if vy > 0:
            level_out = True

        if t <5:
            phase_1 = True
            phase_2 = False
        elif level_out == True:
           phase_1 = False
           phase_2 = False
        else:
            phase_1 = False
            phase_2 = True

        if phase_1 == True:  # phase 1 gliding and turn propellers
            ay = (L * np.cos(alpha) - mass*g0)/mass
            T = 0
            ax = (-L * np.sin(alpha) - D) / mass
            vy_level_out = vy

        elif phase_2 == True:  # leveling out
            t_level_out = 5
            alpha_T = np.pi*0.5
            ay = - vy_level_out / t_level_out
            T = (-L * np.cos(alpha) + mass * g0 + mass * ay) / np.sin(alpha_T)
            ax = (T * np.cos(alpha_T) - L * np.sin(alpha) - D) / mass
            y_level_out = y

        elif phase_1 == False and phase_2 == False and y>y_end:  # descending and vx -> 0
            ay = (vy_end ** 2) / (2 * (y_end - y_level_out))
            T = (-L * np.cos(alpha) + mass * g0 + mass * ay) / np.sin(alpha_T)
            ax = (T * np.cos(alpha_T) - L * np.sin(alpha) - D) / mass

        elif y < y_end:
            vy = -2
            vx = 0
            alpha_T = np.pi*0.5
            T = mass * g0
            ax = 0
            ay = 0
            
        P_hover = T * vy + 1.2 * T * \
            (-(vy / 2) + np.sqrt((vy ** 2 / 4) + (T / (2 * rho * Adisk))))

        Ptot = P_hover

        vx = vx + ax * dt
        vy = vy + ay * dt

        x = x + vx * dt
        y = y + vy * dt

        E += Ptot * dt

        acc_g = ax / g0
        acc_y = (ay+g0)/g0
        # Append lists of parameters
        t_lst.append(t)
        ay_lst.append(ay)
        ax_lst.append(ax)
        y_lst.append(y)
        x_lst.append(x)
        vy_lst.append(vy)
        vx_lst.append(vx)
        alpha_T_lst.append(alpha_T * 180 / np.pi)
        T_lst.append(T)
        L_lst.append(L)
        P_lst.append(Ptot/1000)
        acc_lst.append(acc_g)
        acc_y_lst.append(acc_y)
        V_lst.append(np.sqrt(vx**2+vy**2))

        if t > t_end or y < 0:
            running = False
    acc_lst = np.array(acc_lst)

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot data on each subplot
    axs[0, 0].plot(t_lst, P_lst, color='blue')
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Power [kW]')
    axs[0, 0].grid()

    axs[0, 1].plot(x_lst, y_lst, color='red')
    axs[0, 1].set_ylim(0,200)
    axs[0, 1].set_xlabel('X-position [m]')
    axs[0, 1].set_ylabel('Y-position [m]')
    axs[0, 1].grid()

    axs[1, 0].plot(t_lst, L_lst, color='green')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Lift [N]')
    axs[1, 0].grid()

    axs[1, 1].plot(t_lst, acc_lst, color='orange')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Longitudinal acceleration [g]')
    axs[1, 1].grid()

    # Adjust spacing between subplots
    fig.tight_layout()

    # Display the figure
    # plt.show()

    return E, y_lst, t_lst, x_lst, P_lst # Energy, y, t, x, P


print()