import numpy as np


def max_thrust(rho, V):

    def fprime1(T,V,rho): 
        eff = eff_hover + V*(eff_prop - eff_hover)/v_cruise
        return (0.4*V + 1.2*(V**2/4 + T/(2*A_disk*rho))**0.5 + 0.3*T/(A_disk*rho(V**2/4 + T/(2*A_disk*rho))**0.5))/eff
    
    def fprime2(T,V,rho):
        eff = eff_hover + V*(eff_prop - eff_hover)/v_cruise
        return (1.2/(V**2 + 2*T/(A_disk*rho))**0.5 - 0.6*T/(A_disk*rho(V**2 + 2*T/(A_disk*rho))**1.5))/(A_disk*eff*rho)


    def thrust_to_power_max(T, V, rho):
        return thrust_to_power(T, V, rho)[1] - P_max

    if isinstance(V, np.ndarray):
        Tlst = []
        for v in V:
            T_max  = optimize.newton(thrust_to_power_max, fprime= fprime1, fprime2= fprime2, x0=20000, args=(v, rho), maxiter=1000)
            Tlst.append(T_max)

        return np.array(Tlst)
    else:

        return optimize.newton(thrust_to_power_max, fprime=fprime1, fprime2= fprime2, x0=20000, args=(V, rho), maxiter=100000)

def aero_coefficients(angle_of_attack):
    """
    Calculates the lift and drag coefficients of the aircraft for a given angle of attack.

    :param angle_of_attack: angle of attack experienced by the wing [rad]
    :return: CL and CD
    """

    alpha = np.degrees(angle_of_attack)

    alpha = np.maximum(np.minimum(88.8, alpha), 0)

    # Interpolate CL, CD vs alpha
    CL_alpha = interpolate.interp1d(alpha_lst, Cl_alpha_curve)
    CD_alpha = interpolate.interp1d(alpha_lst, CD_a_w)
    CD_f = interpolate.interp1d(alpha_lst, CD_a_f)(0)

    # Get the CL of the wings at the angle of attack
    CL = CL_alpha(alpha)

    # Drag, assuming the fuselage is parallel to te incoming flow
    CD = CD_alpha(alpha) + CD_f

    return CL, CD

def thrust_to_power(T, V, rho):
    """
    This function calculates the available power associated with a certain thrust level. Note that this power
    still needs to be converted to brake power later on. This will be implemented when more is known about this
    from the power and propulsion department

    :param T: Thrust provided by the engines [N]
    :param V: Airspeed [m/s]
    :return: P_a: available power
    """

    P_a = T * V + 1.2 * T * (-V / 2 + np.sqrt(V ** 2 / 4 + T / (2 * rho * A_disk)))

    # Interpolate between efficiencies
    eff = eff_hover + V*(eff_prop - eff_hover)/v_cruise

    P_r = P_a/eff

    return P_a, P_r

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

def numerical_simulation(vx_start, y_start, th_start, y_tgt, vx_tgt):
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
        max_vy = const.rod
        max_ax = ax_target_descend
        max_ay = ay_target_descend

    else:
        max_vy = const.roc
        max_ax = ax_target_climb
        max_ay = ay_target_climb

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
        CL, CD = aero_coefficients(alpha)

        # Make aerodynamic forces dimensional
        L = 0.5 * rho * V * V * S * CL
        D = 0.5 * rho * V * V * S * CD

        # Get the target accelerations
        ax_tgt, ay_tgt = target_accelerations_new(vx, vy, y, y_tgt, vx_tgt,
                                                        max_ax, max_ay, max_vy)

        # If a constraint on rotational speed is added, calculate the limits in rotation
        th_min, th_max = th - max_rot * dt, th + max_rot * dt
        T_min, T_max = T - 200, T + 200  # TODO: Sanity check

        # Calculate the accelerations
        ax = float((-D * np.cos(gamma) - L * np.sin(gamma) + T * np.cos(th)) / m)
        ay = float((L * np.cos(gamma) - m * g - D * np.sin(gamma) + T * np.sin(th)) / m)

        # Prevent going underground
        if y <= 0:
            vy = 0

        # Solve for the thrust and wing angle, using the target acceleration values
        th = np.arctan2((m * ay_tgt + m * g - L * np.cos(gamma) + D * np.sin(gamma)),
                        (m * ax_tgt + D * np.cos(gamma) + L * np.sin(gamma)))

        th = np.maximum(np.minimum(th, th_max), th_min)

        # Thrust can be calculated in two ways, result should be very close
        T = (m * ay_tgt + m * g - L * np.cos(gamma) + D * np.sin(gamma)) / np.sin(th)
        #T = (m*ax_tgt + D*np.cos(gamma) + L*np.sin(gamma))/np.cos(th)

        # Apply maximum and minimum bounds on thrust, based on maximum power, and on rate of change of thrust
        T = float(np.minimum(np.maximum(np.minimum(np.maximum(T, T_min), T_max), 0), max_thrust(rho,V*np.cos(alpha))))

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
    P_a, P_r = thrust_to_power(T_arr, V_arr*np.cos(th_arr - np.tan(vy_arr/vx_arr)), rho_arr)

    # TODO: IMPLEMENT
    P_tot   = P_r #+ P_systems + P_peak

    # Add to total energy

    if plotting:
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
        plt.savefig(path + 'inputs_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '.pdf')
        plt.show()

        #plt.subplot(221)
        plt.plot(t_arr, V_arr)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [m/s]")
        plt.grid()
        plt.tight_layout(pad=0.05)
        plt.savefig(path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_V.pdf')
        plt.show()

        #plt.subplot(222)
        plt.plot(x_arr, y_arr)
        plt.xlabel("Distance [m]")
        plt.ylabel("Altitude [m]")
        plt.grid()
        plt.tight_layout(pad=0.05)
        plt.savefig(path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_profile.pdf')
        plt.show()

        #plt.subplot(223)
        plt.plot(t_arr, vy_arr)
        plt.xlabel("Time [s]")
        plt.ylabel("$v_y$ [m/s]")
        plt.grid()
        plt.tight_layout(pad=0.8)
        plt.savefig(path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_vy.pdf')
        plt.show()

        plt.plot(t_arr, P_tot/1e3)
        plt.xlabel("Time [s]")
        plt.ylabel("Power [kW]")
        plt.grid()
        plt.tight_layout(pad=0.05)
        plt.savefig(path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_P.pdf')
        plt.show()

        #plt.subplot(224)
        plt.plot(t_arr, np.sqrt((ay_arr + g) ** 2 + ax_arr ** 2) / g)
        plt.xlabel("Time [s]")
        plt.ylabel("accelerations [g]")
        plt.grid()

        plt.tight_layout(pad=0.05)
        plt.savefig(path + 'transition_' + 'climb' * (y_tgt > 10) + 'descend' * (y_tgt < 10) + '_g.pdf')
        plt.show()

    distance = x_lst[-1]
    energy = np.sum(P_tot * dt)
    time = t

    max_power  = np.max(P_tot)
    max_thrust = np.max(T_arr)

    return distance, energy, time, max_power, max_thrust

numerical_simulation(vx_start=0, y_start=0, th_start=, y_tgt, vx_tgt)