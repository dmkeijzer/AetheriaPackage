# uthor: Damien & Can & Lucas


import numpy as np
import sys
import pathlib as pl
import os
from warnings import warn
import scipy.optimize as optimize

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.flight_perf.EnergyPower import *
from modules.flight_perf.transition_model import *
import input.GeneralConstants as const
from modules.misc_tools.ISA_tool  import ISA
from input.data_structures import *

class MissionClass:
    """
    This class simulates the take-off and landing af a tilt-wing eVTOL. Hover, transition and horizontal flight are
    all considered together, no distinction is made between the phases up to and after cruise.
    """
    def __init__(self, aero, aircraft, wing ,engine, power):

        """
        :param mass:            [kg]    Aircraft mass
        :param cruising_alt:    [m]     Cruising altitude
        :param cruise_speed:    [m/s]   Cruise speed
        :param wing_surface:    [m^2]   Wing surface
        :param t_loiter:        [s]     Time in loiter (cruise config)
        :param rotational_rate: [deg/s] Maximum rotational rate of the wing actuators
        :param roc:             [m/s]   Maximum rate-of-climb
        :param rod:             [m/s]   Maximum rate-of-descend
        :param mission_dist:    [m]     Mission distance
        """
        # Assign data strutes as attributes
        self.aero = aero
        self.aircraft = aircraft
        self.wing = wing
        self.engine = engine
        self.power = power


        # Temporary placeholders, REMOVE BEFORE RUNNING OPTIMIZATION
        self.mission_dist = const.mission_dist
        self.t_loiter = const.t_loiter
        self.m = aircraft.MTOM
        self.S = wing.surface
        self.max_rot = np.radians(const.max_rotation)
        self.CL_max = aero.cL_max
        self.A_disk =  engine.total_disk_area
        self.P_max =  aircraft.hoverPower

        # Design variables
        self.ax_target_climb = 0.5 * const.g0  # These are actually maximal values
        self.ay_target_climb = 0.2 * const.g0

        self.ax_target_descend = 0.5 * const.g0
        self.ay_target_descend = 0.2 * const.g0

        self.roc = const.roc_cr
        self.rod = const.rod_cr

        self.h_cruise = const.h_cruise
        self.v_cruise = const.v_cr

        plt.rcParams.update({'font.size': 16})



    def max_thrust(self, rho, V):

        def fprime1(T,V,rho): 
            eff = const.prop_eff
            return (0.4*V + 1.2*(V**2/4 + T/(2*self.A_disk*rho))**0.5 + 0.3*T/(self.A_disk*rho*(V**2/4 + T/(2*self.A_disk*rho))**0.5))/eff
        
        def fprime2(T,V,rho):
            eff = const.prop_eff
            return (1.2/(V**2 + 2*T/(self.A_disk*rho))**0.5 - 0.6*T/(self.A_disk*rho*(V**2 + 2*T/(self.A_disk*rho))**1.5))/(self.A_disk*eff*rho)


        def thrust_to_power_max(T, V, rho):
            return self.thrust_to_power(T, V, rho)[1] - self.P_max

        if isinstance(V, np.ndarray):
            Tlst = []
            for v in V:
                T_max  = optimize.newton(thrust_to_power_max, fprime= fprime1, fprime2= fprime2, x0=20000, args=(v, rho), maxiter=1000)
                Tlst.append(T_max)

            return np.array(Tlst)
        else:

           return optimize.newton(thrust_to_power_max, fprime=fprime1, fprime2= fprime2, x0=20000, args=(V, rho), maxiter=100000)


    def aero_coefficients(self, angle_of_attack):
        """
        Calculates the lift and drag coefficients of the aircraft for a given angle of attack.

        :param angle_of_attack: angle of attack experienced by the wing [rad]
        :return: CL and CD
        """


        # Get the CL of the wings at the angle of attack
        CL = self.aero.cL_alpha*angle_of_attack

        # Drag, assuming the fuselage is parallel to te incoming flow
        CD =  self.aero.cd0_cruise +  CL**2*(1/(np.pi*self.wing.aspect_ratio*self.aero.e))

        return CL, CD

    def thrust_to_power(self, T, V, rho):
        """
        This function calculates the available power associated with a certain thrust level. Note that this power
        still needs to be converted to brake power later on. This will be implemented when more is known about this
        from the power and propulsion department

        :param T: Thrust provided by the engines [N]
        :param V: Airspeed [m/s]
        :return: P_a: available power
        """

        P_a = T * V + 1.2 * T * (-V / 2 + np.sqrt(V ** 2 / 4 + T / (2 * rho * self.A_disk)))

        # Interpolate between efficiencies
        eff =  const.prop_eff

        P_r = P_a/eff

        return P_a, P_r

    def target_accelerations_new(self, vx, vy, y, y_tgt, vx_tgt, max_ax, max_ay, max_vy):

        # Limit altitude
        vy_tgt = np.maximum(np.minimum(-0.5 * (y - y_tgt), max_vy), -max_vy)

        # Slow down when approaching 15 m while going too fast in horizontal direction
        if const.transition_height_baseline + (np.abs(vy) / self.ay_target_descend) > y > y_tgt and abs(vx) > 0.25:
            vy_tgt = 0

        # Keep horizontal velocity zero when flying low
        if y < const.transition_height_baseline - 80:
            vx_tgt_1 = 0
        else:
            vx_tgt_1 = vx_tgt

        # Limit speed
        ax_tgt = np.minimum(np.maximum(-0.5 * (vx - vx_tgt_1), -max_ax), max_ax)
        ay_tgt = np.minimum(np.maximum(-0.5 * (vy - vy_tgt), -max_ay), max_ay)

        return ax_tgt, ay_tgt

    def numerical_simulation(self, vx_start, y_start, th_start, y_tgt, vx_tgt):
        # print('this is still running')
        # Initialization
        vx = float(vx_start)
        vy = 0.
        t = 0
        x = 0
        y = y_start
        th = th_start
        T = 5000
        dt = const.time_step

        # Check whether the aircraft needs to climb or descend
        if y_start > y_tgt:
            max_vy = self.rod
            max_ax = self.ax_target_descend
            max_ay = self.ay_target_descend

        else:
            max_vy = self.roc
            max_ax = self.ax_target_climb
            max_ay = self.ay_target_climb

        # Lists to store everything
        y_lst = []
        x_lst = []
        vy_lst = []
        vx_lst = []
        th_lst = []
        T_lst = []
        t_lst = []
        ax_lst = []
        ay_lst = []
        rho_lst = []

        # Preliminary calculations
        running = True
        while running:

            t += dt

            # ======== Actual Simulation ========

            rho = ISA(y).density()
            V = np.sqrt(vx ** 2 + vy ** 2)
            gamma = np.arctan(vy / vx)
            alpha = th - gamma

            # Get the aerodynamic coefficients
            CL, CD = self.aero_coefficients(alpha)

            # Make aerodynamic forces dimensional
            L = 0.5 * rho * V * V * self.S * CL
            D = 0.5 * rho * V * V * self.S * CD

            # Get the target accelerations
            ax_tgt, ay_tgt = self.target_accelerations_new(vx, vy, y, y_tgt, vx_tgt,
                                                           max_ax, max_ay, max_vy)

            # If a constraint on rotational speed is added, calculate the limits in rotation
            th_min, th_max = th - self.max_rot * dt, th + self.max_rot * dt
            T_min, T_max = T - 200, T + 200  # TODO: Sanity check

            # Calculate the accelerations
            ax = float((-D * np.cos(gamma) - L * np.sin(gamma) + T * np.cos(th)) / self.m)
            ay = float((L * np.cos(gamma) - self.m * const.g0 - D * np.sin(gamma) + T * np.sin(th)) / self.m)

            # Prevent going underground
            if y <= 0:
                vy = 0

            # Solve for the thrust and wing angle, using the target acceleration values
            th = np.arctan2((self.m * ay_tgt + self.m * const.g0 - L * np.cos(gamma) + D * np.sin(gamma)),
                            (self.m * ax_tgt + D * np.cos(gamma) + L * np.sin(gamma)))

            th = np.maximum(np.minimum(th, th_max), th_min)

            # Thrust can be calculated in two ways, result should be very close
            T = (self.m * ay_tgt + self.m * const.g0 - L * np.cos(gamma) + D * np.sin(gamma)) / np.sin(th)
            #T = (self.m*ax_tgt + D*np.cos(gamma) + L*np.sin(gamma))/np.cos(th)

            # Apply maximum and minimum bounds on thrust, based on maximum power, and on rate of change of thrust
            T = float(np.minimum(np.maximum(np.minimum(np.maximum(T, T_min), T_max), 0), self.max_thrust(rho,V*np.cos(alpha))))

            # Perform numerical integration
            vx += float(ax) * dt
            vy += float(ay) * dt

            x += vx * dt
            y += vy * dt

            # Store everything
            y_lst.append(y)
            x_lst.append(x)
            vy_lst.append(vy)
            vx_lst.append(vx)
            th_lst.append(th)
            T_lst.append(T)
            t_lst.append(t)
            ax_lst.append(ax)
            ay_lst.append(ay)
            rho_lst.append(rho)

            # Check if end conditions are satisfied

            if abs(vx - vx_tgt) < 0.8 and abs(y - y_tgt) < 0.5 and abs(vy) < 0.5 and t >= 5 or t > 600:
                running = False

                # if t > 600:
                #     print("Take-off takes longer than 10 minutes")

        # Convert everything to arrays
        y_arr = np.array(y_lst)
        x_arr = np.array(x_lst)
        vy_arr = np.array(vy_lst)
        vx_arr = np.array(vx_lst)
        th_arr = np.array(th_lst)
        T_arr = np.array(T_lst)
        t_arr = np.array(t_lst)
        ax_arr = np.array(ax_lst)
        ay_arr = np.array(ay_lst)
        rho_arr = np.array(rho_lst)
        V_arr = np.sqrt(vx_arr ** 2 + vy_arr ** 2)

        # ======= Get Required outputs =======

        # Get the available power
        P_a, P_r = self.thrust_to_power(T_arr, V_arr*np.cos(th_arr - np.tan(vy_arr/vx_arr)), rho_arr)

        # TODO: IMPLEMENT
        P_tot   = P_r #+ self.P_systems + self.P_peak

        distance = x_lst[-1]
        energy = np.sum(P_tot * dt)
        time = t

        max_power  = np.max(P_tot)
        max_thrust = np.max(T_arr)

        return distance, energy, time, max_power, max_thrust

    def power_cruise_config(self, altitude, speed, mass):

        # Density at cruising altitude
        rho = ISA(altitude).density()

        # Lift coefficient during cruise, based on the cruise speed (can be non-optimal)
        CL_cruise = 2 * mass * const.g0 / (speed * speed * self.S * rho)

        # Drag coefficient
        CD_cruise = self.aero.cd_cruise
        eff_cruise = const.prop_eff

        D_cruise = CD_cruise*0.5*rho*speed*speed*self.S
        #print('cruise_drag', CD_cruise)
        P = self.thrust_to_power(D_cruise, speed, rho)[1]

        return P, D_cruise

    def total_energy(self, simplified = False):

        #P_cruise, D_cruise = self.power_cruise_config(self.h_cruise, self.v_cruise, self.m)

        # print('wrong if statement')
        # Get the energy and distance needed to reach cruise
        d_climb, E_climb, t_climb, P_m_to, T_m_to = self.numerical_simulation(vx_start=0.001, y_start=0,
                                                                                th_start=np.pi / 2, y_tgt=self.h_cruise,
                                                                                vx_tgt=self.v_cruise)

        # Get the energy and distance needed to descend
        d_desc, E_desc, t_desc, P_m_la, T_m_la = self.numerical_simulation(vx_start=self.v_cruise,
                                                                            y_start=self.h_cruise,
                                                                            th_start = np.radians(5), y_tgt=0, vx_tgt=0)

        # Distance spent in cruise
        d_cruise = self.mission_dist  - d_desc - d_climb

        # Time spent cruising
        t_cruise = d_cruise / self.v_cruise

        # Get the brake power used in cruise
        P_cruise, D_cruise = self.power_cruise_config(self.h_cruise, self.v_cruise, self.m)  # + self.P_systems
        self.aircraft.cruisePower = P_cruise

        # Loiter power
        V_loit = np.sqrt(2*self.aircraft.MTOM*const.g0/(const.rho_sl*self.wing.surface*self.aero.cl_climb_clean))
        P_loiter, _ = self.power_cruise_config(altitude=self.h_cruise, speed=V_loit, mass=self.m)  # + self.P_systems

        # Cruise energy
        E_cruise = P_cruise * t_cruise

        # Loiter energy
        E_loiter = P_loiter * self.t_loiter
        # Get the total energy consumption
        E_tot = E_cruise + E_climb + E_desc + E_loiter

        # Mission time
        t_tot = t_climb + t_desc + t_cruise + self.t_loiter

        # Pie chart
     #    labels = ['Take-off', 'Cruise', 'Landing', 'Loiter']
        Energy = [E_climb, E_cruise, E_desc, E_loiter]
        Time = [t_climb, t_cruise, t_desc, self.t_loiter]

        return E_tot, t_tot, max(P_m_to, P_m_la), max(T_m_to, T_m_la), t_cruise + self.t_loiter, Energy

def get_energy_power_perf(WingClass, EngineClass, AeroClass, PerformanceClass):
    """ Computes relevant performance metrics from the mission

    :param WingClass: _description_
    :type WingClass: _type_
    :param EngineClass: _description_
    :type EngineClass: _type_
    :param AeroClass: _description_
    :type AeroClass: _type_
    :param PerformanceClass: _description_
    :type PerformanceClass: _type_
    :return: _description_
    :rtype: _type_
    """    


    
    atm = ISA(const.h_cruise)
    rho_cr = atm.density()

    #==========================Energy calculation ================================= 
    warn("All functions using flap setting and different flight configuration should be called using a function. Theya are currently \
         Not being updated. Thus a lot of these constats are completely not applicable")



    #----------------------- Take-off-----------------------
    P_takeoff = powertakeoff(PerformanceClass.MTOM, const.g0, const.roc_hvr, EngineClass.total_disk_area, const.rho_sl)

    E_to = P_takeoff * const.t_takeoff
    #-----------------------Transition to climb-----------------------
    transition_simulation = numerical_simulation(l_x_1=3.7057, l_x_2=1.70572142*0.75, l_x_3=4.5, l_y_1=0.5, l_y_2=0.5, l_y_3=0.789+0.5, T_max=10500, y_start=30.5, mass=PerformanceClass.MTOM, g0=const.g0, S=WingClass.surface, CL_climb=AeroClass.cl_climb_clean,
                                alpha_climb=AeroClass.alpha_climb_clean, CD_climb=AeroClass.cdi_climb_clean + AeroClass.cd0_cruise,
                                Adisk=EngineClass.total_disk_area, lod_climb=AeroClass.ld_climb, eff_climb=const.prop_eff , v_stall=const.v_stall)

    E_trans_ver2hor = transition_simulation[0]
    transition_power_max = np.max(transition_simulation[0])
    final_trans_distance = transition_simulation[3][-1]
    final_trans_altitude = transition_simulation[1][-1]
    t_trans_climb = transition_simulation[2][-1]

    #----------------------- Horizontal Climb --------------------------------------------------------------------
    average_h_climb = (const.h_cruise  - final_trans_altitude)/2
    rho_climb = ISA(average_h_climb).density()
    v_climb = const.roc_cr/np.sin(const.climb_gradient)
    v_aft= v_exhaust(PerformanceClass.MTOM, const.g0, rho_climb, EngineClass.total_disk_area, v_climb)
    PerformanceClass.prop_eff = propeff(v_aft, v_climb)
    climb_power_var = powerclimb(PerformanceClass.MTOM, const.g0, WingClass.surface, rho_climb, AeroClass.ld_climb, PerformanceClass.prop_eff, const.roc_cr)
    t_climb = (const.h_cruise  - final_trans_altitude) / const.roc_cr
    E_climb = climb_power_var * t_climb

    #----------------------- Transition (from horizontal to vertical)-----------------------
    warn("Transition horizontal to vertical broken gives insane high values don' know what todo with it")
    transition_simulation_landing = numerical_simulation_landing(vx_start=const.v_stall_flaps20, descend_slope= const.descent_slope, mass=PerformanceClass.MTOM, g0=const.g0,
                                S=WingClass.surface , CL=const.cl_descent_trans_flaps20, alpha=const.alpha_descent_trans_flaps20,
                                CD=const.cdi_descent_trans_flaps20 + AeroClass.cd0_cruise, Adisk=EngineClass.total_disk_area)
    E_trans_hor2ver = transition_simulation_landing[0]
    transition_power_max_landing = np.max(transition_simulation_landing[4])
    final_trans_distance_landing = transition_simulation_landing[3][-1]
    final_trans_altitude_landing = transition_simulation_landing[1][-1]  
    t_trans_landing = transition_simulation_landing[2][-1]


    # ----------------------- Horizontal Descent-----------------------
    P_desc = powerdescend(PerformanceClass.MTOM, const.g0, WingClass.surface, rho_climb, AeroClass.ld_climb, PerformanceClass.prop_eff, const.rod_cr)
    t_desc = (const.h_cruise - final_trans_altitude_landing)/const.rod_cr # Equal descend as ascend
    E_desc = P_desc* t_desc
    d_desc = (const.h_cruise - final_trans_altitude_landing)/const.descent_slope
    v_descend = const.rod_cr/const.descent_slope

    #-----------------------------Cruise-----------------------
    P_cr = powercruise(PerformanceClass.MTOM, const.g0, const.v_cr, AeroClass.ld_cruise, PerformanceClass.prop_eff)
    d_climb = final_trans_distance + (const.h_cruise  - final_trans_altitude)/np.tan(const.climb_gradient) #check if G is correct
    d_cruise = const.mission_dist - d_desc - d_climb - final_trans_distance - final_trans_distance_landing
    t_cr = (const.mission_dist - d_desc - d_climb - final_trans_distance - final_trans_distance_landing)/const.v_cr
    E_cr = P_cr * t_cr # used the new cruise power 

    #----------------------- Loiter cruise-----------------------
    P_loit_cr = powerloiter(PerformanceClass.MTOM, const.g0, WingClass.surface, const.rho_cr, AeroClass.ld_climb,PerformanceClass.prop_eff)
    E_loit_hor = P_loit_cr * const.t_loiter

    #----------------------- Loiter vertically-----------------------
    P_loit_land = hoverstuffopen(PerformanceClass.MTOM*const.g0, const.rho_sl,EngineClass.total_disk_area, PerformanceClass.TW_max)[1]
    E_loit_vert = P_loit_land * 30 # 30 sec for hovering vertically
    # print('t', 30)


    #---------------------------- TOTAL ENERGY CONSUMPTION ----------------------------
    E_total = E_to + E_trans_ver2hor + E_climb + E_cr + E_desc + E_loit_hor + E_loit_vert + E_trans_hor2ver 

    #---------------------------- Writing to JSON and printing result  ----------------------------
    PerformanceClass.mission_energy = E_total
    PerformanceClass.hoverPower = transition_power_max
    # print('Pto',P_takeoff)
    PerformanceClass.climbPower = climb_power_var


    PerformanceClass.takeoff_energy = E_to
    PerformanceClass.trans2hor_energy = E_trans_ver2hor
    PerformanceClass.climb_energy = E_climb
    PerformanceClass.cruise_energy = E_cr
    PerformanceClass.descend_energy = E_desc
    PerformanceClass.hor_loiter_energy = E_loit_hor
    PerformanceClass.trans2ver_energy = E_trans_hor2ver
    PerformanceClass.ver_loiter_energy = E_loit_vert

    PerformanceClass.mission_energy = E_total
    PerformanceClass.cruisePower = P_cr
    PerformanceClass.hoverPower = transition_power_max
    PerformanceClass.climbPower = climb_power_var

    return WingClass, EngineClass, AeroClass, PerformanceClass

def get_performance_updated(aero, aircraft, wing, engine, power):

     mission = MissionClass(aero, aircraft, wing, engine, power)
     aircraft.mission_energy, aircraft.mission_time, aircraft.hoverPower, aircraft.max_thrust, Ellipsis, energy_distribution = mission.total_energy()
     aircraft.climb_energy, aircraft.cruise_energy,aircraft.descend_energy, aircraft.hor_loiter_energy = energy_distribution

