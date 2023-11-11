from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.constants import g
import AetheriaPackage.GeneralConstants as const
from AetheriaPackage.ISA_tool import ISA
import os

def powerloading_climbrate(eff, ROC, WS,rho,CD0,e,A):
    k = 1/(e*A*np.pi)
    CLCDratio = 3/(4*k)* np.sqrt(3*CD0/k)
    return (ROC+np.sqrt(2*WS/rho)*(1/CLCDratio))**(-1) * eff

def powerloading_turningloadfactor(rho,V,WS,eff,A,e,loadfactor,CD0):
    k = 1/(e*A*np.pi)
    n = loadfactor
    
    WP = (CD0*0.5*rho*V*V*V/WS + WS*n*n*k/(0.5*rho*V))**-1 *eff

    return WP

def powerloading_thrustloading(WS,rho,ROC,StotS):
    return 1.2*(1+(1/WS)*rho*ROC**2*StotS)
    #return 1.2*(1+np.ones(np.shape(WS)))*1.3

def powerloading_verticalflight(MTOM,TW,A_tot,rho,eff,ducted_bool):
    W = MTOM *const.g0
    T = TW * W
    
    if ducted_bool==True:
        return (0.5*np.sqrt((T*T*T)/(W*W*rho*A_tot)))**(-1)*eff
    else:
        return (np.sqrt((T*T*T)/(2*W*W*rho*A_tot)))**(-1)*eff    
       
def powerloading_climbgradient(e,A,CD0,WS,rho,eff,G):
    CL = np.sqrt(np.pi*e*A*CD0)
    CD = 2*CD0
    WP = (np.sqrt(WS*2/(rho*CL))*(G+CD/CL))**(-1) * eff
    return WP

def wingloading_stall(CLmax,V_stall,rho):
    return CLmax*0.5*rho*V_stall*V_stall

def get_wing_power_loading(perf_par, wing, engine, aero, cont_factor=1.1):
    """ Returns the wing loading and thrust of the weight based on performance parameters

    :param perf_par: performance parameter class from data structues
    :type perf_par: PerformanceParameters class
    :param wing:  wing class from data structues
    :type wing:  Wing class
    """    

    #Check if it"s lilium or not to define the variable that will say to vertical_flight what formula to use.
    WS_range = np.arange(1,4000,1)
    #data["WS"],data["TW"],data["WP_cruise"],data["WP_hover"] = plot_wing_power_loading_graphs(data["eff"], data["StotS"], data["diskloading"], data["name"],WS_range,i)


    #CALCULATE ALL THE VALUES FOR THE GRAPHS
    TW_range = powerloading_thrustloading(WS_range,const.rho_sl,const.roc_hvr, perf_par.Stots)  
    #if data["name"] == "J1":   
    #    TW_range = TW_range*1.3     #Added 30% extra thrust to maintain stability
    CLIMBRATE = cont_factor*powerloading_climbrate(perf_par.prop_eff,const.roc_cr, WS_range,const.rho_cr,aero.cd0_cruise ,aero.e,wing.aspect_ratio)
    TURN_VCRUISE = cont_factor*powerloading_turningloadfactor(const.rho_cr,const.v_cr,WS_range, perf_par.prop_eff ,wing.aspect_ratio,aero.e, perf_par.turn_loadfactor,aero.cd0_cruise)
    TURN_VMAX = cont_factor*powerloading_turningloadfactor(const.rho_cr,perf_par.v_max, WS_range, perf_par.prop_eff ,wing.aspect_ratio ,aero.e ,perf_par.turn_loadfactor,aero.cd0_cruise)
    VERTICALFLIGHT = cont_factor*powerloading_verticalflight(perf_par.MTOM ,TW_range, engine.total_disk_area ,const.rho_sl,perf_par.prop_eff ,False)
    STALLSPEED = wingloading_stall(aero.cL_max ,perf_par.v_stall, const.rho_sl)
    CLIMBGRADIENT = cont_factor*powerloading_climbgradient(aero.e ,wing.aspect_ratio ,aero.cd0_cruise,WS_range,const.rho_sl,perf_par.prop_eff ,const.climb_gradient)

    #DETERMINE LOWEST
    lowest_area_y_novf = []
    lowest_area_y = []
    lowest_area_x = np.arange(0,int(STALLSPEED),1)
    for i in lowest_area_x:
        lowest_area_y.append(min(CLIMBRATE[i],TURN_VCRUISE[i],TURN_VMAX[i],CLIMBGRADIENT[i],VERTICALFLIGHT[i]))
        lowest_area_y_novf.append(min(CLIMBRATE[i],TURN_VCRUISE[i],TURN_VMAX[i],CLIMBGRADIENT[i]))
        
    #DETERMINE LIMITING FACTORS
    margin = 0.95
    perf_par.wing_loading_cruise = STALLSPEED*margin
    perf_par.TW_max = powerloading_thrustloading(perf_par.wing_loading_cruise,const.rho_sl,const.roc_hvr, perf_par.Stots)
    WP_cruise = lowest_area_y_novf[-1]*margin
    WP_hover = lowest_area_y[-1]*margin
    aero.cL_cruise = 2/(const.rho_cr*const.v_cr**2)*perf_par.wing_loading_cruise
    return perf_par, wing, engine, aero

def v_exhaust(MTOM, g0, rho, atot, vinf):
    Vcr = np.sqrt(2 * (MTOM * g0) / (rho * atot) + vinf ** 2)
    return Vcr


def vhover(MTOM, g0, rho, atot):
    Vhov = np.sqrt(2 * (MTOM * g0) / (rho * atot))
    return Vhov


def propeff(vcr, vinf):
    prop = 2 / (1 + vcr / vinf)
    return prop


def hoverstuffopen(T, rho, atot, toverw):
    Phopen = T ** 1.5 / np.sqrt(2 * rho * atot)
    PhopenMAX = (T * toverw) ** 1.5 / (2 * np.sqrt(rho * atot))
    energyhoveropen = 90 / 3600 * Phopen * 1.3
    energyhoveropenMAX = 90 / 3600 * PhopenMAX * 1.3
    return Phopen, PhopenMAX, energyhoveropen, energyhoveropenMAX


def hoverstuffduct(T, rho, atot, toverw):
    Phduct = T ** 1.5 / (2 * np.sqrt(rho * atot))
    PhductMAX = (T * toverw) ** 1.5 / (2 * np.sqrt(rho * atot))
    energyhoverduct = 90 / 3600 * Phduct * 1.3
    energyhoverductMAX = 90 / 3600 * PhductMAX * 1.3
    return Phduct, PhductMAX, energyhoverduct, energyhoverductMAX


def powertakeoff(MTOM, g0, roc_hvr, diskloading, rho):
    P_to = MTOM * g0 * roc_hvr + 1.2 * MTOM * g0 * (
                (-roc_hvr / 2) + np.sqrt((roc_hvr ** 2 / 4) + (MTOM * g0 / (2 * rho * diskloading))))
    return P_to


def powercruise(MTOM, g0, v_cr, lift_over_drag, propeff):
    powercruise = MTOM * g0 * v_cr / (lift_over_drag * propeff)
    return powercruise


def powerclimb(MTOM, g0, S, rho, lod_climb, prop_eff, ROC):
    climb_power = MTOM * g0 * (np.sqrt(2 * MTOM * g0 * (1 / lod_climb) / (S * rho)) + ROC) / prop_eff
    return climb_power


def powerloiter(MTOM, g0, S, rho, lod_climb, prop_eff):
    loiter_power = MTOM * g0 * (np.sqrt(2 * MTOM * g0 * (1 / lod_climb) / (S * rho))) / prop_eff
    return loiter_power


def powerdescend(MTOM, g0, S, rho, lod_climb, prop_eff, ROD):
    climb_power = MTOM * g0 * (np.sqrt(2 * MTOM * g0 * (1 / lod_climb) / (S * rho)) - ROD) / prop_eff
    return climb_power

class MissionClass:
    """
    This class simulates the take-off and landing af a tilt-wing eVTOL. Hover, transition and horizontal flight are
    all considered together, no distinction is made between the phases up to and after cruise.
    """
    def __init__(self, aero, aircraft, wing ,engine, power, plot=False):

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
        self.plot = plot

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

    def get_cd(self, CL):
        """
        Calculates the lift and drag coefficients of the aircraft for a given angle of attack.

        :param angle_of_attack: angle of attack experienced by the wing [rad]
        :return: CL and CD
        """



        CD =  self.aero.cd0_cruise +  CL**2*(1/(np.pi*self.wing.aspect_ratio*self.aero.e))

        return  CD

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
        if const.transition_height + (np.abs(vy) / self.ay_target_descend) > y > y_tgt and abs(vx) > 0.25:
            vy_tgt = 0

        # Keep horizontal velocity zero when flying low
        if y < const.transition_height - 30:
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
            warn("Currently maximum thrust is not being checked for")
            # T = float(np.minimum(np.maximum(np.minimum(np.maximum(T, T_min), T_max), 0), self.max_thrust(rho,V*np.cos(alpha))))

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

        if self.plot:
            plt.cla()

            fig, axs = plt.subplots(2,2, sharex= False)
            fig.set_size_inches(9,9)

            axs[0,0].plot(x_arr, y_arr)
            axs[0,0].set_xlabel("X position")
            axs[0,0].set_ylabel("Y position")
            axs[0,0].grid()

            axs[0,1].plot(t_arr, vx_arr, label= "Vx")
            axs[0,1].plot(t_arr, vy_arr, label= "Vy")
            axs[0,1].set_xlabel("time [s]")
            axs[0,1].set_ylabel("Velocities ")
            axs[0,1].legend()
            axs[0,1].grid()

            axs[1,0].plot(t_arr, P_tot/1000, label= "Power")
            axs[1,0].set_xlabel("time [s]")
            axs[1,0].set_ylabel("Power [kW]")
            axs[1,0].legend()
            axs[1,0].grid()

            axs[1,1].plot(t_arr, ax_arr, label= "ax")
            axs[1,1].plot(t_arr, ay_arr, label= "ay")
            axs[1,1].set_xlabel("time [s]")
            axs[1,1].set_ylabel("Acceleration [m*s^-2]")
            axs[1,1].legend()
            axs[1,1].grid()

            fig.tight_layout()
            plt.savefig(os.path.join(os.path.expanduser("~"),"Downloads", f"performance_{y_start}_{y_tgt}_plot.pdf"), bbox_inches="tight")

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

        #-----------------------Common phases----------------------------------------------
        wing_loading = self.aircraft.MTOM/self.wing.surface*g

        #-----------------------Hover phases----------------------------------------------
        t_hover = 30/const.roc_hvr
        thrust_hvr = self.aircraft.MTOM*g*1.2
        power_hvr = self.thrust_to_power(thrust_hvr, 0, const.rho_sl)[1]
        E_hover= 2*t_hover*self.thrust_to_power(self.aircraft.MTOM*g*1.2, 0, const.rho_sl)[1]

        #--------------- Climb phase ----------------------------------------------------
        q_inf_climb = 0.5*const.rho_sl*(const.v_climb)**2
        cl_climb =  wing_loading * np.cos(const.climb_angle)/q_inf_climb
        cd_climb = self.get_cd(cl_climb)
        Lift_climb = cl_climb*q_inf_climb*self.wing.surface
        Drag_climb = cd_climb*q_inf_climb*self.wing.surface

        t_climb = self.h_cruise/const.roc_cr
        T_m_to = self.aircraft.MTOM*g*np.sin(const.climb_angle) +  Drag_climb
        P_m_to = self.thrust_to_power(T_m_to, const.v_climb, const.rho_sl)[1]
        E_climb = P_m_to*t_climb
        d_climb = t_climb*const.v_climb*np.cos(const.climb_angle)

        #--------------- descent phase ----------------------------------------------------

        # cl_descent, cd_descent = self.aero_coefficients(alpha_descent)
        v_descent =  np.sqrt((2*np.cos(self.aircraft.glide_slope))/(const.rho_cr*self.aero.cl_ld_max)*wing_loading)

        q_inf_desc = 0.5*const.rho_sl*(v_descent)**2

        t_desc = self.h_cruise/(v_descent*np.sin(self.aircraft.glide_slope))
        E_desc = 0
        d_desc = t_climb*v_descent*np.cos(self.aircraft.glide_slope)

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
        E_tot = E_cruise + E_climb + E_desc + E_loiter + E_hover

        # Mission time
        t_tot = t_climb + t_desc + t_cruise + self.t_loiter

        # Pie chart
     #    labels = ['Take-off', 'Cruise', 'Landing', 'Loiter']
        Energy = [E_climb, E_cruise, E_desc, E_loiter, E_hover]
        Time = [t_climb, t_cruise, t_desc, self.t_loiter, t_hover*2]

        return E_tot, t_tot, power_hvr, thrust_hvr, t_cruise + self.t_loiter, Energy

def get_energy_power_perf(WingClass, EngineClass, AeroClass, PerformanceClass):
    raise DeprecationWarning(f"This function should not be called, has unstable behaviour!")
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

def get_performance_updated(aero, aircraft, wing, engine, power, plot= False):

     mission = MissionClass(aero, aircraft, wing, engine, power, plot)
     aircraft.mission_energy, aircraft.mission_time, aircraft.hoverPower, aircraft.max_thrust, Ellipsis, energy_distribution = mission.total_energy()
     aircraft.climb_energy, aircraft.cruise_energy,aircraft.descend_energy, aircraft.hor_loiter_energy, aircraft.hover_energy  = energy_distribution


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
            phase_2 = True
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

