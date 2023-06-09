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




def numerical_simulation(l_x_1, l_x_2, l_x_3, l_y_1, l_y_2, l_y_3, T_max, y_start, mass, g0, S, CL_climb, alpha_climb, CD_climb, Adisk, lod_climb, eff_climb, v_stall):
    # print('this is still running')
    # Initialization
    vx = 0
    vy = 2.
    V = np.sqrt(vx**2 + vy**2)
    # v_climb = np.sqrt((1767.4453125*2)/(rho*CL_climb))
    t = 0
    x = 0
    y = y_start
    gamma_climb = np.arctan2(0.125, 1)
    theta_climb = gamma_climb + alpha_climb
    alpha_T = np.pi/2 
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
    T1_lst = []
    T2_lst = []
    T3_lst = []
    maxT = []
    # Preliminary calculations
    running = True
    while running:

        t += dt
        rho = 1.220

        alpha_T = 0.5 * np.pi - (t / t_end) * (0.5 * np.pi - alpha_climb)
        #alpha_T = (1 - t/t_end)*alpha_T

        #if alpha_T < theta_climb:
         #   alpha_T = theta_climb

        # ======== Actual Simulation ========

        # Lift and drag
        L = 0.5 * rho * vx**2 * S * CL_climb
        D = 0.5 * rho * vx**2 * S * CD_climb

        vy_end = 0.125 * v_stall  # achieves final climb gradient of 12.5%
        ay = (vy_end - vy0) / t_end
        
        # Thrust and power
        T = (mass * g0 - L * np.cos(alpha_climb) + mass * ay) / np.sin(alpha_T)
        Ttot = T
        # T2 = T*0.5
        # T3 = ((Ttot - T2)*np.sin(alpha_T)*l_x_1 - (Ttot - T2)*np.cos(alpha_T)*l_y_1 + T2*np.sin(alpha_T)*l_x_2 - T2*np.cos(alpha_T)*l_y_2) / (np.sin(alpha_T)*(l_x_1+l_x_3) + np.cos(alpha_T)*(l_y_3 - l_y_1))
        # T1 = Ttot - T2 - T3
        # if (T1<0) or (T3<0) or V>40:
        #     print("Thrust is negative!!!")
        #     print("T1", T1, T3)
        
            # break
        #T = (mass*ay + mass*g0 + D*np.sin(gamma_climb) - L*np.cos(theta_climb)) / np.sin(theta_climb + alpha_T)
        V = np.sqrt(vx ** 2 + vy ** 2)

        P_hover = T * vy + 1.2 * T * \
            (-(vy / 2) + np.sqrt((vy ** 2 / 4) + (T / (2 * rho * Adisk))))
        P_climb = mass * g0 * \
            (np.sqrt(2 * mass * g0 * (S / rho)) * (1 / lod_climb) + vy) / eff_climb

        Ptot = (P_hover *np.sin(alpha_T)+ P_climb * np.cos(alpha_T)) / 1

        # Acceleration, velocity and position updates
        ax = (T * np.cos(alpha_T) - L * np.sin(alpha_climb) - D) / mass
        #ax = (0.5*T*np.cos(theta_climb + alpha_T) - L*np.sin(theta_climb) - D*np.cos(gamma_climb)) / mass

        vx += ax * dt
        vy += ay * dt

        x += vx * dt
        y += vy * dt

        # if V>v_stall:
        #     #print('transition complete')
        #     break

        # Energy integrand per time step
        E += Ptot * dt

        # Longitudinal acceleration in g-force
        acc_g = ax/g0

        # Append lists for all parameters
        # T1_lst.append(T1/2)
        # T2_lst.append(T2/2)
        # T3_lst.append(T3/2)
        t_lst.append(t)
        ax_lst.append(ax)
        y_lst.append(y)
        x_lst.append(x)
        vy_lst.append(vy)
        vx_lst.append(vx)
        alpha_T_lst.append(alpha_T * 180 / np.pi)
        T_lst.append(T)
        D_lst.append(D)
        P_lst.append(Ptot/1000)
        acc_lst.append(acc_g)
        L_lst.append(L)
        maxT.append(10500)
        
        if t > t_end:
            running = False

    acc_lst = np.array(acc_lst)


    # Create figure for maximum thurst
    

    # Plot data
    # plt.plot(t_lst, T1_lst, label="Inboard propellers", linewidth = 2.5)
    # plt.plot(t_lst, T2_lst, label="Outboard propellers", linewidth = 2.5)
    # plt.plot(t_lst, T3_lst, label="V-tail propellers", linewidth = 2.5)
    # plt.plot(t_lst, maxT, label ='Maximum Thrust', linewidth = 2.5)
    
    # marker_interval = 100
    # plt.plot(t_lst[::marker_interval], T1_lst[::marker_interval], label="Inboard propellers",marker='o', color='blue')
    # plt.plot(t_lst[::marker_interval], T2_lst[::marker_interval], label="Outboard propellers",marker='s', color='orange')
    # plt.plot(t_lst[::marker_interval], T3_lst[::marker_interval], label="V-tail propellers",marker='^', color='green')
    # plt.plot(t_lst[::marker_interval], maxT[::marker_interval], label="Maximum Thrust",marker='x', color='red')

    # plt.xlabel("Time [s]", fontsize = 15)
    # plt.ylabel("Thrust [N]", fontsize = 15)
    # # plt.title("Front Propeller Thrust")
    # plt.grid()
    # plt.legend(fontsize = 15)


    # plt.tight_layout(pad = 2.0)
    # plt.show()
    # Create a figure and subplots
    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # # # Plot data on each subplot
    # axs[0, 0].plot(t_lst, P_lst, color='blue')
    # axs[0, 0].set_xlabel('Time [s]')
    # axs[0, 0].set_ylabel('Power [kW]')
    # axs[0,0].set_title('Power vs Time')
    # axs[0,0].set_ylim(0,900)
    # axs[0, 0].grid()

    # axs[0, 1].plot(x_lst, y_lst, color='red')
    # axs[0, 1].set_ylim(0,150)
    # axs[0, 1].set_xlabel('X-position [m]')
    # axs[0, 1].set_ylabel('Y-position [m]')
    # axs[0,1].set_title("Y vs X position")
    # axs[0, 1].grid()

    # axs[1, 0].plot(t_lst, L_lst, color='green')
    # axs[1, 0].set_xlabel('Time [s]')
    # axs[1, 0].set_ylabel('Lift [N]')
    # axs[1, 0].set_title('Lift vs Time')
    # axs[1, 0].grid()

    # axs[1, 1].plot(t_lst, alpha_T_lst, color='orange')
    # axs[1, 1].set_xlabel('Time [s]')
    # axs[1, 1].set_ylabel('Propeller angle [deg]')
    # axs[1,1].set_title('Propeller angle vs time')
    # axs[1,1].set_ylim(0,100)
    # axs[1, 1].grid()

    # # Adjust spacing between subplots
    # fig.tight_layout(pad = 2.0)
    
    # # Display the figure
    # plt.show()
 
    return E, y_lst, t_lst, x_lst, V, P_lst


#print((numerical_simulation(y_start=30.5, mass=2158.35754, g0=const.g0, S=11.975442, CL_climb=0.592,
#                                  alpha_climb=0.0873, CD_climb=0.0238580,
#                                   Adisk=17.986312, lod_climb=220.449, eff_climb=0.9,
 #                                  v_stall=45)[6]))


def numerical_simulation_landing(vx_start, descend_slope, mass, g0, S, CL, alpha, CD, Adisk):
    # print('this is still running')
    # Initialization
    vx = vx_start
    vy = vx_start*descend_slope
    vy_start = vx_start*descend_slope
    vy_end = -2
    y_start = 73
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

        if t < 5:
            phase_1 = True
            phase_2 = False
        elif level_out == True:
            phase_1 = False
            phase_2 = False
        else:
            phase_1 = False
            phase_2 = True

        if phase_1 == True:  # phase 1 gliding and turn propellers
            gamma = np.arctan2(vy,vx)
            ay = (L * np.cos(alpha+gamma) - mass*g0 - D*np.sin(gamma))/mass
            T = 0
            ax = (-L * np.sin(alpha+gamma) - D*np.cos(gamma)) / mass
            vy_level_out = vy
            # print(gamma*180/np.pi)

        elif phase_2 == True: # leveling out
            gamma = np.arctan2(vy,vx)
            t_level_out = 5
            alpha_T = np.pi*0.5
            ay = - vy_level_out / t_level_out
            T = (-L * np.cos(alpha+gamma) + mass * g0 + mass * ay - D*np.sin(gamma)) / np.sin(alpha_T+gamma)
            if T<0:
                T=0
            ax = (T * np.cos(alpha_T) - L * np.sin(alpha+gamma) - D*np.cos(gamma)) / mass
            y_level_out = y
            # print(gamma*180/np.pi)

        elif phase_1 == False and phase_2 == False and y>y_end:  # descending and vx -> 0
            ay = (vy_end ** 2) / (2 * (y_end - y_level_out))
            T = (-L * np.cos(alpha) + mass * g0 + mass * ay) / np.sin(alpha_T)
            if T<0:
                T=0
            ax = (T * np.cos(alpha_T) - L * np.sin(alpha) - D) / mass

        if y < y_end or vx<0 or vx == 0:
            vy = -2
            vx = 0
            alpha_T = np.pi*0.5
            T = mass * g0
            ax = 0
            ay = 0
        try:
            P_hover = T * vy + 1.2 * T * \
                (-(vy / 2) + np.sqrt((vy ** 2 / 4) + (T / (2 * rho * Adisk))))
        except RuntimeWarning:
            print(T)
            P_hover = 0

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
     
    # # Create a figure and subplots
    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # # Plot data on each subplot
    # axs[0, 0].plot(t_lst, P_lst, color='blue')
    # axs[0, 0].set_xlabel('Time [s]')
    # axs[0, 0].set_ylabel('Power [kW]')
    # axs[0,0].set_title('Power vs Time')
    # axs[0, 0].grid()

    # axs[0, 1].plot(x_lst, y_lst, color='red')
    # axs[0, 1].set_ylim(0,150)
    # axs[0, 1].set_xlabel('X-position [m]')
    # axs[0, 1].set_ylabel('Y-position [m]')
    # axs[0,1].set_title("Y vs X position")
    # axs[0, 1].grid()

    # axs[1, 0].plot(t_lst, L_lst, color='green')
    # axs[1, 0].set_xlabel('Time [s]')
    # axs[1, 0].set_ylabel('Lift [N]')
    # axs[1, 0].set_title('Lift vs Time')
    # axs[1, 0].grid()

    # axs[1, 1].plot(t_lst, acc_lst, color='orange')
    # axs[1, 1].set_xlabel('Time [s]')
    # axs[1, 1].set_ylabel('Longitudinal acceleration [g]')
    # axs[1,1].set_title('Acceleration vs Time')
    # axs[1, 1].grid()


    # # Adjust spacing between subplots
    # fig.tight_layout(pad =2.0)

    # # Display the figure
    # plt.show()

    return E, y_lst, t_lst, x_lst, P_lst  # Energy, y, t, x, P


#print(numerical_simulation_landing(vx_start=data['v_stall_flaps20'], descend_slope=-0.04, mass=data["mtom"], g0=const.g0,
 #                                   S=data['S'], CL=data['cl_descent_trans_flaps20'], alpha=data['alpha_descent_trans_flaps20'],
 #                                   CD=data["cdi_descent_trans_flaps20"]+data['cd0'], Adisk=data["diskarea"])[0])
