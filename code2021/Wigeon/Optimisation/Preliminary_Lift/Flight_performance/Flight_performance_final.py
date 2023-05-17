import numpy as np
import matplotlib.pyplot as plt
from Aero_tools import ISA, speeds
import scipy.interpolate as interpolate
import sys
from constants import g, eff_hover, eff_prop
import scipy.optimize as optimize
# TODO: Remove this import in the integrated program, make sure aerodynamics is called first and the variables have the
# same names
#from Preliminary_Lift.main_aero import Cl_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag
plt.rcParams["figure.figsize"] = (5, 4)

class mission:
    """
    This class simulates the take-off and landing af a tilt-wing eVTOL. Hover, transition and horizontal flight are
    all considered together, no distinction is made between the phases up to and after cruise.

    !!!!! THIS FILE STILL CONTAINS PLACEHOLDER VALUES TO TEST ITS FUNCTIONALITY. REMOVE BEFORE OPTIMIZING DESIGNS !!!!!
    TODO:
        - Add energy estimation
        - Add aerodynamic data  (Lift and drag curves)
        - Add propulsion data   (Thrust-to-power, max thrust)
        - Cruise performance
        - Descend performance
        - Add optimum speeds part
        - Efficiencies
    """

    def __init__(self, mass, cruising_alt, cruise_speed, CL_max, wing_surface, A_disk, P_max,
                 Cl_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag,
                 t_loiter=15 * 60, rotational_rate=5, roc=5, rod=4, mission_dist=300e3, plotting=False):

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

        # Temporary placeholders, REMOVE BEFORE RUNNING OPTIMIZATION
        self.mission_dist = mission_dist
        self.m = mass
        self.S = wing_surface
        self.max_rot = np.radians(rotational_rate)
        self.CL_max = CL_max
        self.A_disk = A_disk
        self.P_max = P_max

        # Design variables
        self.ax_target_climb = 0.5 * g  # These are actually maximal values
        self.ay_target_climb = 0.2 * g

        self.ax_target_descend = 0.5 * g
        self.ay_target_descend = 0.2 * g

        self.roc = roc
        self.rod = rod

        self.h_cruise = cruising_alt
        self.v_cruise = cruise_speed
        self.t_loiter = t_loiter

        plt.rcParams.update({'font.size': 16})
        self.path = '../Flight_performance/Figures/'
        self.plotting = plotting

        self.Cl_alpha_curve = Cl_alpha_curve
        self.CD_a_w = CD_a_w
        self.CD_a_f = CD_a_f
        self.alpha_lst = alpha_lst
        self.Drag = Drag

    def max_thrust(self, rho, V):

        def thrust_to_power_max(T, V, rho):
            return self.thrust_to_power(T, V, rho)[1] - self.P_max

        if isinstance(V, np.ndarray):
            Tlst = []
            for v in V:
                T_max  = optimize.newton(thrust_to_power_max, x0=20000, args=(v, rho), maxiter=1000)
                Tlst.append(T_max)

            return np.array(Tlst)
        else:

            optimize.newton(thrust_to_power_max, x0=20000, args=(V, rho), maxiter=100000)

            return optimize.newton(thrust_to_power_max, x0=20000, args=(V, rho), maxiter=100000)

    def aero_coefficients(self, angle_of_attack):
        """
        Calculates the lift and drag coefficients of the aircraft for a given angle of attack.

        :param angle_of_attack: angle of attack experienced by the wing [rad]
        :return: CL and CD
        """

        alpha = np.degrees(angle_of_attack)

        alpha = np.maximum(np.minimum(88.8, alpha), 0)

        # Interpolate CL, CD vs alpha
        CL_alpha = interpolate.interp1d(self.alpha_lst, self.Cl_alpha_curve)
        CD_alpha = interpolate.interp1d(self.alpha_lst, self.CD_a_w)
        CD_f = interpolate.interp1d(self.alpha_lst, self.CD_a_f)(0)

        # Get the CL of the wings at the angle of attack
        CL = CL_alpha(alpha)

        # Drag, assuming the fuselage is parallel to te incoming flow
        CD = CD_alpha(alpha) + CD_f

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
        eff = eff_hover + V*(eff_prop - eff_hover)/self.v_cruise

        P_r = P_a/eff

        return P_a, P_r

    def target_accelerations_new(self, vx, vy, y, y_tgt, vx_tgt, max_ax, max_ay, max_vy):

        # Limit altitude
        vy_tgt = np.maximum(np.minimum(-0.5 * (y - y_tgt), max_vy), -max_vy)

        # Slow down when approaching 15 m while going too fast in horizontal direction
        if 15 + (np.abs(vy) / self.ay_target_descend) > y > y_tgt and abs(vx) > 0.25:
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
        dt = 0.01

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
            ay = float((L * np.cos(gamma) - self.m * g - D * np.sin(gamma) + T * np.sin(th)) / self.m)

            # Prevent going underground
            if y <= 0:
                vy = 0

            # Solve for the thrust and wing angle, using the target acceleration values
            th = np.arctan2((self.m * ay_tgt + self.m * g - L * np.cos(gamma) + D * np.sin(gamma)),
                            (self.m * ax_tgt + D * np.cos(gamma) + L * np.sin(gamma)))

            th = np.maximum(np.minimum(th, th_max), th_min)

            # Thrust can be calculated in two ways, result should be very close
            T = (self.m * ay_tgt + self.m * g - L * np.cos(gamma) + D * np.sin(gamma)) / np.sin(th)
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

        # Add to total energy

        if self.plotting:
            fig, ax1 = plt.subplots()
            ax1.plot(t_arr, np.degrees(th_arr), c='orange', label = 'Wing angle')
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Wing angle [deg]")

            ax2 = ax1.twinx()
            ax2.plot(t_arr, T_arr, label = 'Thrust')
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Thrust [N]")

            ax1.grid()
            fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)#loc = 'upper center')
            fig.tight_layout(pad=0.05)
            plt.savefig(self.path + 'inputs_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '.pdf')
            plt.show()

            #plt.subplot(221)
            plt.plot(t_arr, V_arr)
            plt.xlabel("Time [s]")
            plt.ylabel("Speed [m/s]")
            plt.grid()
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_V.pdf')
            plt.show()

            #plt.subplot(222)
            plt.plot(x_arr, y_arr)
            plt.xlabel("Distance [m]")
            plt.ylabel("Altitude [m]")
            plt.grid()
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_profile.pdf')
            plt.show()

            #plt.subplot(223)
            plt.plot(t_arr, vy_arr)
            plt.xlabel("Time [s]")
            plt.ylabel("$v_y$ [m/s]")
            plt.grid()
            plt.tight_layout(pad=0.8)
            plt.savefig(self.path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_vy.pdf')
            plt.show()

            plt.plot(t_arr, P_tot/1e3)
            plt.xlabel("Time [s]")
            plt.ylabel("Power [kW]")
            plt.grid()
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_P.pdf')
            plt.show()

            #plt.subplot(224)
            plt.plot(t_arr, np.sqrt((ay_arr + g) ** 2 + ax_arr ** 2) / g)
            plt.xlabel("Time [s]")
            plt.ylabel("accelerations [g]")
            plt.grid()

            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_g.pdf')
            plt.show()

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
        CL_cruise = 2 * mass * g / (speed * speed * self.S * rho)

        # Drag coefficient
        CD_cruise = self.Drag.CD(CL_cruise)
        eff_cruise = eff_hover + speed*(eff_prop - eff_hover)/self.v_cruise

        D_cruise = CD_cruise*0.5*rho*speed*speed*self.S
        #print('cruise_drag', CD_cruise)
        P = self.thrust_to_power(D_cruise, speed, rho)[1]

        return P, D_cruise

    def total_energy(self, simplified = False):

        #P_cruise, D_cruise = self.power_cruise_config(self.h_cruise, self.v_cruise, self.m)

        # Distance spent in cruise
        d_cruise = self.mission_dist  # - d_desc - d_climb

        # Time spent cruising
        t_cruise = d_cruise / self.v_cruise

        # Get the brake power used in cruise
        P_cruise, D_cruise = self.power_cruise_config(self.h_cruise, self.v_cruise, self.m)  # + self.P_systems

        V = speeds(altitude=self.h_cruise, m=self.m, CLmax=self.CL_max, S=self.S, componentdrag_object=self.Drag)

        # Loiter power
        V_loit = V.climb()
        P_loiter, _ = self.power_cruise_config(altitude=self.h_cruise, speed=V_loit, mass=self.m)  # + self.P_systems

        # Cruise energy
        E_cruise = P_cruise * t_cruise

        # Loiter energy
        E_loiter = P_loiter * self.t_loiter

        if simplified:
            d_climb = 13000
            E_climb = 0.111*(E_cruise + E_loiter)
            t_climb = 200
            T_m_to  = 1.5*9.81*self.m
            P_m_to  = 1.7e6

            d_desc  = 20000
            E_desc = 0.037*(E_cruise + E_loiter)
            t_desc = 300
            T_m_la  =1
            P_m_la = 1

        else:
            # print('wrong if statement')
            # Get the energy and distance needed to reach cruise
            d_climb, E_climb, t_climb, P_m_to, T_m_to = self.numerical_simulation(vx_start=0.001, y_start=0,
                                                                                  th_start=np.pi / 2, y_tgt=self.h_cruise,
                                                                                  vx_tgt=self.v_cruise)

            # Get the energy and distance needed to descend
            d_desc, E_desc, t_desc, P_m_la, T_m_la = self.numerical_simulation(vx_start=self.v_cruise,
                                                                               y_start=self.h_cruise,
                                                                               th_start = np.radians(5), y_tgt=0, vx_tgt=0)

        # Get the total energy consumption
        E_tot = E_cruise + E_climb + E_desc + E_loiter

        # Mission time
        t_tot = t_climb + t_desc + t_cruise + self.t_loiter

        # Pie chart
        labels = ['Take-off', 'Cruise', 'Landing', 'Loiter']
        Energy = [E_climb, E_cruise, E_desc, E_loiter]
        Time = [t_climb, t_cruise, t_desc, self.t_loiter]

        if self.plotting:

            plt.pie(Energy, labels=labels, autopct='%1.1f%%')
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'energy_breakdown.pdf')
            plt.show()


            plt.pie(Time, labels=labels, autopct='%1.1f%%')
            plt.tight_layout(pad=0.05)
            plt.savefig(self.path + 'time_breakdown.pdf')

            plt.show()

        return E_tot, t_tot, max(P_m_to, P_m_la), max(T_m_to, T_m_la), t_cruise + self.t_loiter


class evtol_performance:
    """
    This script calculates different performance characteristics for a tilt-wing evtol.
    """
    def __init__(self, cruising_alt, cruise_speed, S, CL_max, mass, battery_capacity, EOM, loiter_time, A_disk, P_max,
                 CL_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag):

        # Change this when datafile is final
        self.S      = S
        self.CL_max = CL_max
        self.m      = mass
        self.W      = self.m*g
        self.bat_E  = battery_capacity
        self.v_cruise = cruise_speed
        self.h_cruise = cruising_alt
        self.EOM    = EOM
        self.t_loiter = loiter_time
        self.A_disk = A_disk
        self.P_max  = P_max

        self.CL_alpha_curve = CL_alpha_curve
        self.alpha_lst = alpha_lst
        self.CD_a_w = CD_a_w
        self.CD_a_f = CD_a_f
        self.Drag = Drag

        CD_f_alpha = interpolate.interp1d(alpha_lst, CD_a_f)
        CD_w_alpha = interpolate.interp1d(alpha_lst, CD_a_w)
        self.CD_vert = CD_f_alpha(88.9) + CD_w_alpha(88.9)

        plt.rcParams.update({'font.size': 16})
        self.path = '../Flight_performance/Figures/'

    def thrust_to_power(self, T, V, rho):
        """
        This function calculates the available power associated with a certain thrust level. Note that this power
        still needs to be converted to brake power later on. This will be implemented when more is known about this
        from the power and propulsion department

        :param T: Thrust provided by the engines [N]
        :param V: Airspeed [m/s]
        :return: P_a: available power
        """

        P_a = T * V + 1.2 * T * ((-V / 2) + np.sqrt((V**2)/4 + (T / (2 * rho * self.A_disk))))

        # Interpolate between efficiencies
        eff = eff_hover + V*(eff_prop - eff_hover)/self.v_cruise

        P_r = P_a/eff
        ## print(P_a, P_r)
        return P_r - self.P_max

    def max_thrust(self, rho, V):

        if isinstance(V, np.ndarray):
            Tlst = []
            for v in V:
                T_max  = optimize.newton(self.thrust_to_power, x0=20000, args=(v, rho), maxiter=100000)
                Tlst.append(T_max)

            return np.array(Tlst)
        else:
            return optimize.newton(self.thrust_to_power, x0=20000, args=(V, rho), maxiter=100000)

    def quickly_checking_something(self):

        h = np.arange(0, 1000, 10)
        rhos = ISA(h).density()

        V = 30
        for rho in rhos:
            T_max = self.max_thrust(rho, V)

            plt.plot(rho, T_max, '+')
        plt.title('thrust')
        plt.show()


    def climb_performance(self, testing = False):

        # Altitudes to consider climb performance at
        altitudes   = np.array([self.h_cruise, 3000])

        for h in altitudes:

            # Density at altitude
            rho     = ISA(h).density()

            # Stall speed
            V_stall = np.sqrt(2*self.W/(rho*self.S*self.CL_max))

            # Range of speeds
            V   = np.linspace(V_stall, 220, 100)

            # Lift coefficient
            CL  = 2*self.W/(rho*V*V*self.S)

            err = 1
            while np.any(err > 0.1):
                # Drag coefficient
                CD  = self.Drag.CD(CL)

                # Drag
                D   = CD*0.5*rho*V*V*self.S

                # Maximum available thrust
                T   = self.max_thrust(rho, V)

                gamma = np.arcsin(np.minimum((T - D)/self.W, 1))

                # Climb rate, setting a hard limit on climbs more than 90 degrees
                RC = np.minimum((T - D)/self.W, 1)*V

                CLnew = 2*self.W*np.cos(gamma)/(rho*V*V*self.S)

                err = abs(CL - CLnew)

                CL = CLnew

            # Plot results
            plt.plot(V, RC, label = 'Altitude: ' + str(h))

            if h == self.h_cruise:

                idx = np.argmax(np.abs(RC))

            # If running a test, return the speed for which the rate of climb is closest to zero
            if h == self.h_cruise and testing:

                idx = np.argmin(np.abs(RC))

                return V[idx]

        plt.grid()
        plt.xlabel('Speed [m/s]')
        plt.ylabel('Rate-of-climb [m/s]')
        plt.legend()
        plt.tight_layout(pad=0.05)
        plt.savefig(self.path + 'Climb_performance_cruise' + '.pdf')
        plt.show()

        return V[idx]

    def vertical_equilibrium(self, altitude, m, testing = False, test_thrust = 0):#rate_of_climb, altitude, m, testing = False, test_thrust = None):

        # Density at altitude
        rho     = ISA(altitude).density()

        # thrust = self.max_thrust(rho, rate_of_climb)
        # if testing:
        #     thrust = test_thrust

        RC_init     = 20
        if testing:
            RC_init = -5
        err = 1
        N_it = 0
        # Iterate until the rate-of-climb converges
        while abs(err) > 0.1:

            T       = self.max_thrust(rho, RC_init)

            if testing:
                T = test_thrust

            RC  = np.sqrt(abs(T - m*g)/(self.CD_vert*0.5*rho*self.S))*np.sign(T - m*g)

            err     = abs(RC) - abs(RC_init)
            N_it += 1

            if N_it > 500:
                print('vertical_climb does not converge')
                break

            RC_init = RC

        return RC

    def vertical_climb(self, testing = False):

        # Range of altitudes and masses
        masses    = np.linspace(self.EOM, self.m, 4)
        altitudes = np.arange(0, 3000, 100)
        RC  = np.zeros(np.size(altitudes))

        for j, m in enumerate(masses):
            for i, h in enumerate(altitudes):

                # Solve the equation for the rate of climb
                RC[i] = self.vertical_equilibrium(h, m)  # optimize.root_scalar(self.vertical_equilibrium, x0 = 5, args = (h, m, testing))#fsolve(self.vertical_equilibrium, 5, args = (h, m, testing))

            # Plot the results
            plt.plot(altitudes, RC, label = 'mass: ' + str(m))

        plt.xlabel('Altitude [m]')
        plt.ylabel('Rate-of-climb [m/s]')
        plt.grid()
        plt.legend()
        plt.tight_layout(pad=0.05)
        plt.savefig(self.path + 'Climb_performance_vertical' + '.pdf')
        plt.show()

    def range(self, cruising_altitude, cruise_speed, mass, wind_speed = 0, loiter = False):

        # Call mission class
        energy = mission(mass=mass, cruising_alt=cruising_altitude, cruise_speed=cruise_speed, CL_max = self.CL_max,
                         wing_surface = self.S, A_disk = self.A_disk, P_max = self.P_max,
                         Cl_alpha_curve = self.CL_alpha_curve, CD_a_w = self.CD_a_w, CD_a_f = self.CD_a_f,
                         alpha_lst = self.alpha_lst, Drag = self.Drag,plotting = False)

        # Get the distances and energy needed for take-off and landing
        d_la, E_la, t_la,_,_ = energy.numerical_simulation(vx_start=cruise_speed, y_start=cruising_altitude,
                                                       th_start=np.radians(5), y_tgt=0, vx_tgt=0)

        d_to, E_to, t_to,_,_ = energy.numerical_simulation(vx_start=0.001, y_start=0, th_start=np.pi / 2,
                                                       y_tgt=cruising_altitude, vx_tgt=cruise_speed)

        # Power needed for cruise
        P_cr,_ = energy.power_cruise_config(altitude = cruising_altitude, speed = cruise_speed, mass = mass)

        V = speeds(altitude = cruising_altitude, m = mass, CLmax = self.CL_max, S = self.S, componentdrag_object=self.Drag)

        # Power needed for loiter
        P_lt,_ = energy.power_cruise_config(altitude = cruising_altitude, speed = V.climb(), mass = mass)

        # Energy needed for loiter
        E_lt = P_lt*self.t_loiter

        # Find the remaining energy for cruise
        E_cr = self.bat_E - E_la - E_to - E_lt*loiter

        if np.any(E_cr <= 0):
            print("No energy left for cruise")

        # Time in cruise
        t_cr = E_cr / P_cr

        # Cruising distance
        d_cr = (cruise_speed - wind_speed) * t_cr

        # Mission time
        t_tot = t_to + t_la + t_cr + self.t_loiter*loiter

        return d_cr, t_tot

    def payload_range(self):

        # Range of payload masses
        payload_mass = np.linspace(0, self.m - self.EOM, 5)

        # Total mass
        mass = payload_mass + self.EOM

        d_cr = np.zeros(np.size(payload_mass))
        t_cr = np.zeros(np.size(payload_mass))

        # Loop through all masses
        for i, m in enumerate(mass):

            # Get the range
            d_cr[i], t_cr[i] = self.range(cruising_altitude=self.h_cruise, cruise_speed = self.v_cruise, mass = m,
                                          loiter = True)

        # Plot results
        plt.plot(d_cr/1000, payload_mass)
        plt.xlabel('Range [km]')
        plt.ylabel('Payload mass [kg]')
        plt.grid()
        plt.tight_layout(pad=0.05)
        plt.savefig(self.path + 'Payload_range' + '.pdf')
        plt.show()

    def power_polar(self, h):

        rho = ISA(h).density()

        speed = speeds(h, self.m, self.CL_max, self.S, self.Drag)

        V   = np.linspace(speed.stall(), 200, 200)

        # Calculate the drag
        CL  = 2*self.W/(rho*V*V*self.S)
        CD  = self.Drag.CD(CL)

        D   = CD*0.5*rho*V*V*self.S

        # Brake power required
        P_br = self.thrust_to_power(D, V, rho) + self.P_max

        plt.plot(V, P_br)
        plt.plot(V, D*V)
        plt.plot(V, P_br - D*V)
        plt.grid()
        plt.show()



