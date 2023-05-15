import matplotlib.pyplot as plt
import numpy as np
from Flight_performance.Flight_performance_final import mission, evtol_performance
from Aero_tools import ISA, speeds
from Final_optimization import constants_final as const
#import sys, os
import matplotlib

from Preliminary_Lift.main_aero import Cl_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag



plt.rcParams["figure.figsize"] = (5, 4)

class sensitivity_analysis:
    """
    This class performs sensitivity analysis for different flight parameters.

    TODO:
        - Add optimum speeds
    """
    def __init__(self, MTOM, CLmax, S, comp_drag, battery_capacity, EOM, loiter_time, A_disk, P_max, V_cr, h_cr):

        self.MTOM = MTOM
        self.h_cruise_opt = h_cr
        self.V_cruise = V_cr
        self.CL_max = CLmax
        self.S = S
        self.Drag = comp_drag
        self.bat_cap = battery_capacity
        self.EOM = EOM
        self.loiter_time = loiter_time
        self.A_disk = A_disk
        self.P_max = P_max


        self.path = '../Flight_performance/Figures/'

    def cruising_altitude(self):

        # Range of cruising altitudes
        altitude = np.linspace(300, 3000, 6)

        # Arrays to store values
        dist  = np.zeros(np.size(altitude))
        time  = np.zeros(np.size(altitude))

        for i, h in enumerate(altitude):
            print(i)
            V = speeds(altitude = h, m = self.MTOM, CLmax=self.CL_max, S = self.S, componentdrag_object = self.Drag)

            performance = evtol_performance(cruising_alt = h,  cruise_speed = self.V_cruise, S = self.S,
                                            CL_max = self.CL_max, mass = self.MTOM, battery_capacity = self.bat_cap,
                                            EOM = self.EOM, loiter_time = self.loiter_time, A_disk = self.A_disk,
                                            P_max = self.P_max, CL_alpha_curve=Cl_alpha_curve, CD_a_w=CD_a_w,
                                            CD_a_f=CD_a_f, alpha_lst=alpha_lst, Drag=Drag)

            dist[i], time[i] = performance.range(cruising_altitude = h, cruise_speed = self.V_cruise, mass = self.MTOM)

        # Plot results
        plt.plot(altitude, dist/1000)
        plt.xlabel('Cruising altitude [m]')
        plt.ylabel('Range [km]')
        plt.grid()
        plt.tight_layout(pad=0.05)
        plt.savefig(self.path + 'energy_sensitivity_altitude' +'.pdf')
        plt.show()

        plt.plot(altitude, dist/time)
        plt.xlabel('Cruising altitude [m]')
        plt.ylabel('Block speed [m/s]')
        plt.grid()
        plt.tight_layout(pad=0.05)
        plt.savefig(self.path + 'time_sensitivity_altitude' + '.pdf')
        plt.show()

    def cruise_speed(self, plotting = True):

        # Get the stall speed
        V = speeds(self.h_cruise_opt, self.MTOM, CLmax=self.CL_max, S = self.S, componentdrag_object = self.Drag)
        V_stall = V.stall()

        # Range of cruise speeds
        cruise_speeds = np.linspace(V_stall, 100, 20)

        # Arrays to store values
        dist = np.zeros(np.size(cruise_speeds))
        time = np.zeros(np.size(cruise_speeds))

        for i, V_cr in enumerate(cruise_speeds):
            print(i)
            performance = evtol_performance(cruising_alt = self.h_cruise_opt,  cruise_speed = V_cr, S = self.S,
                                            CL_max = self.CL_max, mass = self.MTOM, battery_capacity = self.bat_cap,
                                            EOM = self.EOM, loiter_time = self.loiter_time, A_disk = self.A_disk,
                                            P_max = self.P_max, CL_alpha_curve = Cl_alpha_curve, CD_a_w = CD_a_w,
                                            CD_a_f = CD_a_f, alpha_lst = alpha_lst, Drag = Drag)

            dist[i], time[i] = performance.range(cruising_altitude=self.h_cruise_opt,
                                                 cruise_speed = V_cr, mass = self.MTOM)

        # Find the optimal speed, for testing purposes
        idx     = np.argmax(dist)
        V_opt   = cruise_speeds[idx]

        if plotting:
            # Plot results
            plt.plot(cruise_speeds, dist / 1000)
            #plt.vlines(V.cruise()[0], 0, 300)
            plt.xlabel('Cruise speed [m/s]')
            plt.ylabel('Range [km]')
            plt.grid()
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'energy_sensitivity_speed' + '.pdf')
            plt.show()

            plt.plot(cruise_speeds, dist / time)
            plt.xlabel('Cruise speed [m/s]')
            plt.ylabel('Block speed [m/s]')
            plt.grid()
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'time_sensitivity_speed' + '.pdf')
            plt.show()

        return V_opt

    def wind(self, test_wind = 72, testing = False, plotting = True):

        # Different wind speeds (tail- and headwinds)
        winds = np.arange(-10, 12, 2)

        # Arrays to store values
        dist = np.zeros(np.size(winds))
        time = np.zeros(np.size(winds))
        for i, wind in enumerate(winds):

            # Speed objects
            V = speeds(altitude = self.h_cruise_opt, m = self.MTOM, CLmax=self.CL_max,
                       S = self.S, componentdrag_object = self.Drag)

            performance = evtol_performance(cruising_alt = self.h_cruise_opt,  cruise_speed = self.V_cruise, S = self.S,
                                            CL_max = self.CL_max, mass = self.MTOM, battery_capacity = self.bat_cap,
                                            EOM = self.EOM, loiter_time = self.loiter_time, A_disk = self.A_disk,
                                            P_max = self.P_max, CL_alpha_curve = Cl_alpha_curve, CD_a_w = CD_a_w,
                                            CD_a_f = CD_a_f, alpha_lst = alpha_lst, Drag = Drag)

            dist[i], time[i] = performance.range(cruising_altitude=self.h_cruise_opt,
                                                 cruise_speed=self.V_cruise, mass=self.MTOM, wind_speed = wind)

        # Range with a headwind as high as the wind speed
        if testing:
            test_dist, _ = performance.range(cruising_altitude=self.h_cruise_opt,
                                             cruise_speed = 60, mass=self.MTOM, wind_speed = test_wind)
            return test_dist

        if plotting:
            # Plot results
            plt.plot(winds, dist / 1000)
            plt.xlabel('Wind [m/s]')
            plt.ylabel('Range [km]')
            plt.grid()
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'energy_sensitivity_wind' + '.pdf')
            plt.show()

            plt.plot(winds, dist / time)
            plt.xlabel('Wind [m/s]')
            plt.ylabel('Block speed [m/s]')
            plt.grid()
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'time_sensitivity_wind' + '.pdf')
            plt.show()


# Data
mass = 2800
cruising_alt = 1000
cruise_speed = 72
CL_max = 1.5856
wing_surface = 19.82
EOM = mass - (const.m_pax*4 + const.m_cargo_tot)
A_disk = 8
P_max  = 1.81e6
V_cr = 72.1
h_cr = 1000


sensitivity = sensitivity_analysis(MTOM = mass, CLmax = CL_max, S = wing_surface, comp_drag=Drag,
                                   battery_capacity=1084e6, EOM=EOM, loiter_time= 15*60,
                                   A_disk = A_disk, P_max = P_max, V_cr = V_cr, h_cr=h_cr)

sensitivity.cruising_altitude()
sensitivity.cruise_speed()
sensitivity.wind()
