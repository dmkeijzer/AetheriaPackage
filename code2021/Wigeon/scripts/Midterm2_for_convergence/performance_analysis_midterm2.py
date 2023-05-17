import numpy as np
import matplotlib.pyplot as plt
from Aero_tools import ISA
import json
# from Transition_simulation import transition_EOM
from Airfoil_analysis_midterm2 import airfoil_stats

class initial_sizing:
    def __init__(self, h, path, drag_polar, V_stall, V_max, n_turn, ROC, V_cr, ROC_hover, MTOW, CD_vertical, eff_prop,
                 eff_hover, disk_load):

        # Make the drag polar object an attribute
        self.drag_polar = drag_polar

        # self.path = path
        # datafile = open(self.path, "r")
        #
        # # Read data from json file
        # self.data = json.load(datafile)
        # datafile.close()

        # Extracting aerodynamic data
        # aero        = self.data["Aerodynamics"]
        self.CLmax,_,_,_,_,_,_,_,_  = airfoil_stats()
        self.CD_vertical = CD_vertical    # Drag polar for the fuselage at 90 degrees

        # Atmospherics
        atm_flight  = ISA(h)    # atmospheric conditions during flight
        atm_LTO     = ISA(0)    # atmospheric conditions at landing and take-off (assumed sea-level)
        self.rho_flight = atm_flight.density()
        self.rho_LTO    = atm_LTO.density()

        # Requirements, and others
        # reqs        = self.data["Requirements"]
        self.Vs     = V_stall
        self.Vmax   = V_max
        self.ROC    = ROC
        self.nmax   = n_turn
        self.Vcr    = V_cr

        self.ROC_ho = ROC_hover

        # Propulsion constants
        self.eff_prop   = eff_prop    # [-] Propeller efficiency during normal flight
        self.eff_hover  = eff_hover   # [-] Propeller efficiency during hover
        self.TA         = disk_load   # [kg/m^2]  Disk loading for ducted fans

        # Structures data
        self.MTOW       = MTOW

        # # Preparing matplotlib
        # fig = plt.figure()
        # self.ax1 = fig.add_subplot(111)
        # self.ax2 = self.ax1.twiny()

        # ======= Drag polar calculations =======
        # Range of CL values, ignoring negative values as they are not relevant
        CL  = np.linspace(0.0001, self.CLmax, 1000)
        CD  = drag_polar.CD(CL)     # Get the drag coefficient from the aerodynamics department

        # Get the index of the minimum CD/CL
        idx         = np.argmin(CD/CL)

        # index of CL for maximum climb
        idx_climb   = np.argmax((CL**3)/(CD**2))

        # CL for minimum CD/CL
        self.CL_range   = drag_polar.CL_des()

        # Get the CL
        self.CL_climb   = CL[idx_climb]

        # Testing whether the CL is indeed the optimum one
        dist  = CL/CD
        climb = (CL**3)/(CD**2)

        test_dist   = np.array([dist[idx-1], dist[idx], dist[idx+1]])
        test_climb  = np.array([climb[idx_climb-1], climb[idx_climb], climb[idx_climb+1]])

        # Checking the result
        if np.argmax(test_dist) != 1 or np.argmax(test_climb) != 1:
            print("Optimal CL values incorrect")

    def stall(self, V_stall):
        # Stall speed (assuming low altitude)
        return 0.5*self.rho_LTO*(V_stall**2)*self.CLmax

    def max_speed(self, WS):

        CL = 2*WS/(self.rho_flight*self.Vmax*self.Vmax)

        # Max speed (Reduction of power with altitude was neglected, as no combustion is involved (revise this for H2))
        return self.eff_prop*((self.drag_polar.CD(CL) *
               0.5*self.rho_flight*self.Vmax*self.Vmax*self.Vmax/WS)**-1)

    def climb(self, WS, climb_rate):

        # Optimal speed
        V = np.sqrt(2*WS/(self.rho_flight*self.CL_climb))

        # Climb performance
        return ((climb_rate + (self.drag_polar.CD(self.CL_climb)) *
                0.5*self.rho_flight*(V**3)/WS)**-1)*self.eff_prop

    def turn(self, WS, V, n_turn):

        CL_turn = 2*n_turn*WS/(self.rho_flight*V*V)

        if any(CL_turn > self.CLmax):
            # print("Check")
            CL_turn = np.where(CL_turn < self.CLmax, CL_turn, np.nan)

        # print(CL_turn)
        # Turning performance (at high altitude and max speed)
        return self.eff_prop*(((self.drag_polar.CD(CL_turn)) * 0.5*self.rho_flight*V*V*V/WS)**-1)

    def vertical_flight(self, WS, ROC_hover):
        # Most of the equations used here come from:
        # Comprehensive preliminary sizing/resizing method for a fixed wing – VTOL electric UAV

        # Thrust-to-weight required to allow for a sufficient rate of climb, plus a safety factor for acceleration
        TWR = 1.2*(1 + self.CD_vertical * 0.5*self.rho_flight*ROC_hover**2/WS)

        # Obtaining Power to weight based on the TWR required
        return (TWR*np.sqrt(self.TA/(2*self.rho_LTO))/self.eff_hover)**-1

    def wing_loading(self):

        # Range of wing loadings to plot the power loadings
        self.WS = np.arange(100, 4000, 1)

        # Produce lines for the WS, WP diagram
        self.WS_stall           = self.stall(self.Vs)
        self.WP_speed           = self.max_speed(self.WS)
        self.WP_climb           = self.climb(self.WS, self.ROC)
        self.WP_turn_max_speed  = self.turn(self.WS, self.Vmax, self.nmax)
        # self.WP_turn_cruise     = self.turn(self.WS, self.Vcr, self.nmax)
        self.WP_hov             = self.vertical_flight(self.WS, self.ROC_ho)

        # # Plot wing and thrust loading diagrams
        # self.ax1.plot(self.WS, self.WP_speed, label = 'Maximum speed')
        # self.ax1.plot(self.WS, self.WP_climb, label = 'Rate of climb')
        # self.ax1.plot(self.WS, self.WP_turn_max_speed,  label = 'Turn performance @ $V_{max}$')
        # # self.ax1.plot(self.WS, self.WP_turn_cruise,     label='Turn performance @ $V_{cr}$')
        # # print(self.WP_hov)
        # self.ax1.plot(self.WS, self.WP_hov, label = 'Vertical flight requirements')
        # self.ax1.plot(np.ones(np.size(self.WS))*self.WS_stall, np.linspace(0, 0.1, np.size(self.WS)), label = 'Stall speed')

    def design_point(self):
        """
        Identifies the design space, and finds the optimal design point. Since not all hover engines work during cruise,
        this is done twice, once including hover, and once without. It is assumed that the wing loading at stall will be
        limiting, the most critical power loading corresponding to this is then found. It should be checked in the plot
        whether this indeed corresponds to the most optimal design point.
        """

        # Border of the design space
        # crit = np.minimum(np.minimum(np.minimum(np.minimum(self.WP_speed, self.WP_turn_cruise), self.WP_turn_max_speed),
        #                   self.WP_climb), self.WP_hov)[self.WS < self.WS_stall]
        #
        # crit_cruise = np.minimum(np.minimum(np.minimum(self.WP_speed, self.WP_turn_cruise), self.WP_turn_max_speed),
        #                   self.WP_climb)[self.WS < self.WS_stall]

        crit = np.minimum(np.minimum(np.minimum(self.WP_speed, self.WP_turn_max_speed),
                          self.WP_climb), self.WP_hov)[self.WS < self.WS_stall]

        crit_cruise = np.minimum(np.minimum(self.WP_speed, self.WP_turn_max_speed),
                          self.WP_climb)[self.WS < self.WS_stall]


        # Range of wing loadings in the design space
        ws_crit = self.WS[self.WS < self.WS_stall]

        # Select design point.
        # !!! Check manually in the plot !!!
        # print(ws_crit)
        self.des_WS         = ws_crit[-1]
        self.des_WP         = crit[-1]          # Design power loading including hover
        self.des_WP_cruise  = crit_cruise[-1]   # Design power loading considering only cruise

        # Write the design point to the data file
        # FP = self.data["Flight performance"]

        # FP["WS"]           = float(self.des_WS)
        # FP["WP hover"]     = float(self.des_WP)
        # FP["WP cruise"]    = float(self.des_WP_cruise)

        # datfile = open(self.path, "w")
        # json.dump(self.data, datfile)
        # datfile.close()
        #
        # Plot design space and point
        # self.ax1.fill_between(ws_crit, np.zeros(np.size(crit)), crit, facecolor='green', alpha=0.2)
        # self.ax1.fill_between(ws_crit, np.zeros(np.size(crit_cruise)), crit_cruise, facecolor='limegreen', alpha=0.2)
        # #self.ax1.plot(self.des_WS, self.des_WP, 'D', label='Design point')
        # #self.ax1.legend()
        # self.ax1.set_ylim(0, 0.25)
        # self.ax1.set_ylabel("Power loading [N/W]")
        # self.ax1.set_xlabel("Wing loading [N/m^2]")
        # plt.tight_layout()

        # # Save the figure
        # path = '../Flight_performance/Figures/wing_power_loading.pdf'
        # plt.savefig(path)
        #
        # plt.show()

        # Print results
        # print()
        # print('====== Design point ====== ')
        # print('Wing loading         :  ', np.round(self.des_WS, 4), '  N/m^2')
        # print('Power loading        : ', np.round(self.des_WP, 4), 'N/W')
        # print('Power loading cruise : ', np.round(self.des_WP_cruise, 4), 'N/W')

        return self.des_WS

    def sizing(self):

        # Run wing and power loading
        self.wing_loading()
        self.design_point()

        # Sizing the wing
        S  = self.MTOW/self.des_WS

        # Total power needed
        P_tot       = self.MTOW/self.des_WP         # Including hover
        P_cruise    = self.MTOW/self.des_WP_cruise  # Cruise only

        # Store results to the data file
        # FP = self.data["Flight performance"]
        # FP["S"]         = S
        # FP["P tot"]     = P_tot
        # FP["P cruise"]  = P_cruise
        # datfile = open(self.path, "w")
        # json.dump(self.data, datfile)
        # datfile.close()

        # print()
        # print('====== Initial sizing ======')
        # print('Wing area:                      ', np.round(S, 2), 'm^2')
        # print('Total shaft power (incl. hover):', np.round(P_tot, 0), 'W')
        # print('Total shaft power (cruise only):', np.round(P_cruise, 0), 'W')

        return self.des_WS, self.des_WP

    def testing(self):

        # Just some random inputs for testing
        WS_test     = 500
        V_test      = 100

        # If n = 1, the max speed line should equal that of the turning requirement
        WP_turn  = self.turn(WS_test, V_test, 1)
        WP_speed = self.max_speed(WS_test)

        if abs(WP_turn - WP_speed) > 1e-3:
            print("Turning or speed equations implemented incorrectly")
            print('diff', WP_turn, WP_speed)
        # Testing the climb requirement
        descent = self.climb(WS_test, -10)
        climb   = self.climb(WS_test, 10)

        if descent > climb:
            print("Climb equation implemented incorrectly")

        # Hover test
        descent_ho  = self.vertical_flight(WS_test, -10)
        ascent_ho   = self.vertical_flight(WS_test, 10)

        if descent_ho > ascent_ho:
            print("Hover implemented incorrectly")


# class mission_analysis:
#     def __init__(self, path, h_cruise, m_pl, ac_energy, concept, save_data = False):
#
#         # Import aircraft data
#         self.path       = path
#         datafile        = open(self.path, "r")
#         self.data       = json.load(datafile)
#         datafile.close()
#
#         self.concept = concept
#
#         # Structural data
#         self.struc  = self.data["Structures"]
#         self.EOW    = self.struc["EOW"]
#         self.MTOW   = self.struc["MTOW"]
#
#         # Flight performance data
#         self.FP             = self.data["Flight performance"]
#         self.S              = self.FP["S"]
#         self.WS             = self.FP["WS"]
#
#         self.WP_ho          = self.FP["WP hover"]
#         self.WP_cr          = self.FP["WP cruise"]
#         self.t_to           = self.FP["t_TO"]
#         self.t_la           = self.FP["t_land"]
#
#         # Requirements
#         self.req    = self.data["Requirements"]
#         self.ROC    = self.req["ROC"]
#         self.ROC_ho = self.req["ROC_hover"]
#         self.ROD_ho = self.req["ROD_hover"]
#         self.t_loit = self.req["Loiter time"]
#         self.V_st   = self.req["V_stall"]
#         self.t_hov_loit = self.req["Hover loiter"]
#
#         # Aerodynamic data
#         self.aero   = self.data["Aerodynamics"]
#         self.CLmax  = self.aero["CLmax_front"]
#         self.A      = self.aero["AR"]
#         self.e      = self.aero["e"]
#         self.CLmin  = self.aero["CLforCDmin"]
#         self.CDmin  = self.aero["CDmin"]
#         self.StotSw = self.aero["Stot/Sw"]
#         self.k = 1 / (np.pi * self.A * self.e)
#
#         # Propulsion data
#         self.prop       = self.data["Propulsion"]
#         self.TA         = self.prop["TA"]
#         self.hover_eff  = self.prop["eff_hover"]
#         self.cruise_eff = self.prop["eff_cruise"]
#
#         # Atmospheric properties
#         ISA_cr  = ISA(h_cruise)
#
#         # Atmospheric properties at climb, assuming the average height is half the cruise altitude
#         h_climb = h_cruise/2
#         ISA_cl  = ISA(h_climb)
#
#         # Sea level atmosphere
#         ISA_sl  = ISA(0)
#
#         # Densities
#         self.rho_cr  = ISA_cr.density()
#         self.rho_cl  = ISA_cl.density()
#         self.rho_sl  = ISA_sl.density()
#
#         # Other parameters
#         self.W              = self.EOW + (m_pl*9.81)    # [N] Aircraft weight
#         self.h_cruise       = h_cruise      # [m] Cruising altitude
#         self.h_climb        = h_climb       # [m] Average climbing altitude
#         self.capacity       = ac_energy     # [J] Aircraft energy
#         self.h_tr           = 40            # [m] Transition altitude
#         self.save           = save_data     # Boolean to see if data has to be saved or not
#
#         # Warn if the aircraft is overweight
#         if np.any(self.W > self.MTOW):
#             print()
#             print("Aircraft is too heavy, reduce payload weight")
#             print()
#
#         # ======= Drag polar calculations =======
#         # Range of CL values, ignoring negative values as they are not relevant
#         CL = np.linspace(0.0001, self.CLmax, 10000)
#         CD = self.CDmin + (((CL - self.CLmin) ** 2) * self.k)
#
#         # Get the index of the minimum CD/CL
#         idx = np.argmin(CD / CL)
#
#         # index of CL for maximum climb
#         idx_climb = np.argmax((CL ** 3) / (CD ** 2))
#
#         # CL for minimum CD/CL
#         self.CL_range = CL[idx]
#
#         # Get the CL
#         self.CL_climb = CL[idx_climb]
#
#         # Descent angle
#         self.gamma_descent = np.arctan(CL[idx]/CD[idx])#np.radians(self.FP["Gamma descent"])
#
#         # Testing whether the CL is indeed the optimum one
#         dist = CL / CD
#         climb = (CL ** 3) / (CD ** 2)
#
#         test_dist = np.array([dist[idx - 1], dist[idx], dist[idx + 1]])
#         test_climb = np.array([climb[idx_climb - 1], climb[idx_climb], climb[idx_climb + 1]])
#
#         # Checking the result
#         if np.argmax(test_dist) != 1 or np.argmax(test_climb) != 1:
#             print("Optimal CL values incorrect")
#
#     def optimal_speeds(self, save = False):
#
#         # Optimal climb speed
#         V_climb     = np.sqrt(2*self.W/(self.rho_cl*self.S*self.CL_climb))
#
#         # Optimal cruise speed
#         V_cruise    = np.sqrt(2*self.W/(self.rho_cr*self.S*self.CL_range))
#
#         # If the input is not an array, store the optimal speed
#         if save:
#
#             # Store cruise speed
#             FP = self.data["Flight performance"]
#             FP["V_cruise"] = V_cruise
#             datfile = open(self.path, "w")
#             json.dump(self.data, datfile)
#             datfile.close()
#             print()
#             print("Optimal cruise speed (", np.round(V_cruise), "m/s) stored")
#
#         return V_climb, V_cruise
#
#     def hover_power(self, ROC, W):
#
#         TWR = 1.2 * (1 + ((ROC ** 2) * self.rho_sl * self.StotSw / self.WS))
#         T   = TWR*self.MTOW
#
#         A_prop = T/self.TA
#
#         P_hov = T + 1.2*T*(-ROC/2 + np.sqrt(ROC**2/4 + T/(2*self.rho_sl*A_prop)))
#
#         if np.any(P_hov < 0):
#             print("hover power implemented incorrectly")
#
#         if np.any(P_hov > self.MTOW/self.WP_ho):
#             print("More power is needed than is available")
#
#         return P_hov
#
#     def cruise_power(self, W):
#
#         # Drag-to-lift during cruise (optimal conditions)
#         CDCL_cr = (self.CDmin + (((self.CL_range - self.CLmin) ** 2) * self.k))/self.CL_range
#
#         _, V_cruise = self.optimal_speeds()
#
#         P_cr    = W*V_cruise*CDCL_cr/self.cruise_eff
#
#         if np.any(P_cr < 0):
#             print("Cruise power implemented incorrectly")
#
#         return P_cr
#
#     def climb_power(self, ROC_climb, V_climb, W):
#
#         # Climb CLCD
#         CD_cl = self.CDmin + (((self.CL_climb - self.CLmin) ** 2) * self.k)
#         CLCD_cl = self.CL_climb / CD_cl
#
#         # Climb power, assuming climb efficiency is the same as cruise efficiency
#         P_cl   = (W*V_climb*(CLCD_cl**-1) + W*ROC_climb)/self.cruise_eff
#
#         if np.any(P_cl < 0):
#             print("Climb power implemented incorrectly")
#
#         if np.any(P_cl > self.MTOW/self.WP_cr):
#             print("More power is used than is available")
#             #print(P_cl, self.MTOW/self.WP_cr)
#
#         return P_cl
#
#     def descent_power(self, gamma_descend, V_desc, W):
#
#         # Descent CLCD (same as climb)
#         CD_cl = self.CDmin + (((self.CL_climb - self.CLmin) ** 2) * self.k)
#         CLCD_cl = self.CL_climb / CD_cl
#
#         P_des   = np.maximum(W*V_desc*((CLCD_cl**-1) - np.sin(gamma_descend))/self.cruise_eff, 0)
#
#         if np.any(P_des < 0):
#             print("Descent power implemented incorrectly or descent angle too steep to maintain speed (use brakes/HLD)")
#
#         return P_des
#
#     def climb_energy(self, ROC):
#
#         # Optimal speed
#         V_climb, _ = self.optimal_speeds()
#
#         # Climb power
#         P_climb    = self.climb_power(self.ROC, V_climb, self.W)
#
#         # Time needed to climb
#         t_climb   = (self.h_cruise - self.h_tr)/ROC
#
#         # Energy spent
#         E_climb   = P_climb*t_climb
#
#         return E_climb
#
#     def descent_energy(self):
#
#         V_desc, _   = self.optimal_speeds()
#
#         # Power needed
#         P_desc      = self.descent_power(self.gamma_descent, V_desc, self.W)
#
#         # Time needed to descent
#         t_desc = (self.h_cruise - self.h_tr) / (V_desc*np.sin(self.gamma_descent))
#
#         # Energy spent
#         E_desc = P_desc * t_desc
#
#         return E_desc
#
#     def to_hover_energy(self):
#
#         # Get hover power
#         P_hover_to  = self.hover_power(self.ROC_ho, self.W)
#
#         E_hover_to  = P_hover_to*self.t_to
#
#         return E_hover_to
#
#     def la_hover_energy(self):
#
#         # Get hover power
#         P_hover_la  = self.hover_power(self.ROD_ho, self.W)
#
#         E_hover_la  = P_hover_la*self.t_la
#
#         return E_hover_la
#
#     def hover_loiter_energy(self):
#
#         P_hover_loit = self.hover_power(0, self.W)
#
#         E_hover_loit = P_hover_loit*self.t_hov_loit
#
#         return E_hover_loit
#
#     def loiter_energy(self):
#
#         # Speed for minimum power is the same as maximum climb rate
#         V_loit, _ = self.optimal_speeds()
#
#         # Power needed
#         P_loit = self.climb_power(0, V_loit, self.W)
#
#         # Energy spent
#         E_desc = P_loit * self.t_loit
#
#         return E_desc
#
#     def cruise_energy(self, mission_range):
#
#         # Cruise power
#         P_cr    = self.cruise_power(self.W)
#
#         V_cl, V_cr = self.optimal_speeds(save = self.save)
#
#         # Calculate the total distance spent in cruise (Assuming the transition distance is negligible)
#         t_climb = (self.h_cruise - self.h_tr)/self.ROC
#         d_climb = np.sqrt((V_cl**2) - (self.ROC**2))*t_climb
#         d_desc  = (self.h_cruise - self.h_tr)/np.tan(self.gamma_descent)
#
#         d_cruise = mission_range - d_desc - d_climb
#
#         # Check if climb and descent don't take more space than the mission
#         if np.any(d_cruise < 0):
#             print("Cruise distance needed for climb and descent longer than mission time, reduce cruise altitude")
#
#         # Time spent in cruise (Assuming the aircraft flies at optimal speed)
#         t_cruise = d_cruise/V_cr
#
#         E_cruise = P_cr*t_cruise
#
#         return E_cruise
#
#     def transition_energy(self):
#
#         # Since the transition energy is calculated by a numerical time-stepping simulation, a loop is made to find the
#         # energy associated with each weight
#
#         W_lst = []
#         if type(self.W) == np.ndarray:
#             for weight in self.W:
#                 # Call the transition class
#                 trans   = transition_EOM(weight, self.path)
#
#                 # Transition energy
#                 plotting = False
#                 if weight > 18500:
#                     plotting = False
#                 E_trans = trans.simulate(plotting = plotting)
#
#                 W_lst.append(E_trans)
#
#
#             E_transition = np.array(W_lst)
#             plt.plot(self.W, E_transition)
#             plt.show()
#
#         else:
#             # Call the transition class
#             trans = transition_EOM(self.W, self.path)
#
#             # Transition energy
#             E_transition = trans.simulate(plotting=True)
#             print(E_transition)
#
#         return E_transition
#
#     def range(self):
#
#         E_hover_to  = self.to_hover_energy()
#         E_hover_la  = self.la_hover_energy()
#         E_climb     = self.climb_energy(self.ROC)
#         E_descent   = self.descent_energy()
#         E_loiter    = self.loiter_energy()
#         E_hov_loit  = self.hover_loiter_energy()
#
#         # Transition energy, assuming the take-off and landing transition take as much energy
#         E_trans     = self.transition_energy()
#
#         # Based on the energy available to the aircraft, estimate the amount left for cruise
#         E_cruise    = self.capacity - E_hover_to - E_hover_la - E_climb - E_descent - E_loiter - 2*E_trans - E_hov_loit
#
#         # Cruise power
#         P_cruise    = self.cruise_power(self.W)
#
#         # Cruising time
#         t_cruise    = E_cruise/P_cruise
#
#         # Cruise distance
#         _, V_cruise = self.optimal_speeds()
#         d_cruise    = V_cruise*t_cruise
#
#         return d_cruise
#
#     def climb_performance(self, h, V, dedicated_hover = False):
#
#         isa = ISA(h)
#         rho = isa.density()
#
#         # Required power
#         CL  = 2*self.W/(rho*V*V*self.S)
#         CD  = self.CDmin + (((CL - self.CLmin) ** 2) * self.k)
#         P_r = CD*0.5*rho*V*V*V*self.S
#
#         #print(dedicated_hover, (self.WP_ho*dedicated_hover + self.WP_cr * (not dedicated_hover)) ** -1, self.WP_cr**-1)
#         #print(not dedicated_hover)
#         # Rate of climb (depending on whether there are dedicated hover engines, power loading is chosen)
#         RC  = np.minimum(((((self.WP_cr*dedicated_hover + self.WP_ho * (not dedicated_hover)) ** -1)
#                                         * self.cruise_eff) - P_r/self.W)/V, 1)*V
#
#         return RC#np.degrees(np.arcsin((self.WP_ho**-1 - (P_r/self.W))/V))#RC
#
#     def climb_perf_chart(self):
#
#         # Range of altitudes
#         h = np.arange(300, 3000, 800)
#
#         for alt in h:
#             # Calculate the density to reduce the stall speed with altitude
#             rho = ISA(alt).density()
#             V = np.arange(self.V_st*np.sqrt(self.rho_sl/rho), 200, 0.1)
#
#             RC = self.climb_performance(alt, V)
#             print()
#             print('Maximum climb rate: ', max(RC))
#             print()
#             label = 'height: ' + str(alt) + ' [m]'
#             plt.plot(V, RC, label = label)
#
#         plt.xlabel("Speed [m/s]")
#         plt.ylabel("Rate of climb [m/s]")
#         plt.grid()
#         plt.legend()
#         plt.tight_layout()
#         path = 'C:/Users/Egon Beyne/Desktop/DSE/Plots/climb_perf_' + str(self.concept) + '.pdf'
#         plt.savefig(path)
#         plt.show()
#
#     def total_energy(self, mission_range, pie = False):
#
#         # Energy needed in each flight phase (Ignoring transition for now, as this is juts used for comparison)
#         E_hover_to  = self.to_hover_energy()
#         E_hover_la  = self.la_hover_energy()
#         E_climb     = self.climb_energy(self.ROC)
#         E_descent   = self.descent_energy()
#         E_loiter    = self.loiter_energy()
#         E_cruise    = self.cruise_energy(mission_range)
#         E_trans     = 2*self.transition_energy()
#         E_hov_loit  = self.hover_loiter_energy()
#
#         # Pie chart with all the energies
#         if pie:
#             labels    = ['Take-off', 'Climb', 'Cruise', 'Descent', 'Land', 'Loiter', 'Transition', 'Hover loiter']
#             fractions = [E_hover_to, E_climb, E_cruise, E_descent, E_hover_la, E_loiter, E_trans, E_hov_loit]
#
#             plt.pie(fractions, labels = labels, autopct='%1.1f%%')#, startangle= 75)
#             #plt.legend(loc = 'best')
#             plt.tight_layout()
#             path = 'C:/Users/Egon Beyne/Desktop/DSE/Plots/Energy_breakdown_' + str(self.concept) + '.pdf'
#             plt.savefig(path, bbox_inches = 'tight')
#             plt.show()
#
#         # Total energy needed for the mission
#         E_tot   = E_hover_to + E_hover_la + E_climb + E_descent + E_loiter + E_cruise + E_trans + E_hov_loit
#
#         return E_tot
