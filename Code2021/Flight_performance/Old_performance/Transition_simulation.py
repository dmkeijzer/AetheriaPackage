import numpy as np
import matplotlib.pyplot as plt
import json
from Aero_tools import ISA
import scipy.optimize as optimize


class transition_EOM:
    def __init__(self, W, path):

        self.path = path
        datafile = open(self.path, "r")

        # Read data from json file
        self.data = json.load(datafile)
        datafile.close()

        self.k = 1.2 # See Tilt-wing evtol takeoff optimization
        self.rho = ISA(0).density()
        self.CL = 1.0
        self.W  = W


        self.m = self.W/9.81

        # Aerodynamic data
        aero = self.data["Aerodynamics"]
        self.CDmin  = aero["CDmin"]
        self.CLmin  = aero["CLforCDmin"]
        self.CLmax  = aero["CLmax_front"]
        self.A      = aero["AR"]
        self.e      = aero["e"]
        self.StotSw = aero["Stot/Sw"]

        FP          = self.data["Flight performance"]
        struc       = self.data["Structures"]
        prop        = self.data["Propulsion"]
        req         = self.data["Requirements"]
        self.MTOW   = struc["MTOW"]
        self.ROC_hover = req["ROC_hover"]
        self.WS     = FP["WS"]

        TWR = 1.2 * (1 + ((self.ROC_hover ** 2) * self.rho * self.StotSw / self.WS))
        self.T_max  = TWR*self.MTOW
        self.TA     = prop["TA"]
        self.A_prop = self.T_max/self.TA

        #FP = self.data["Flight performance"]
        self.S = FP["S"]
        self.P_max = FP["P tot"]

    def disk_power(self, T, P_in, a_T, V):

        V_perp = V*np.cos(a_T)

        # Calculate the power provided by the rotating cruise engines
        P = T*V_perp + self.k*T*(-V_perp/2 + np.sqrt(V_perp**2/4 + T/(2*self.rho*self.A_prop))) - P_in

        return P

    def max_thrust(self, motor_AoA, speed):

        init = np.minimum(self.P_max/(speed+0.0001), self.T_max)

        # Find the thrust for a certain input power and speed
        T = optimize.newton(self.disk_power, init, args = (self.P_max, motor_AoA, speed, ))
        T_bis = optimize.bisect(self.disk_power, 0, 40000, args = (self.P_max, motor_AoA, speed, ))

        return T

    def horizontal_equilibrium(self, motor_angle, speed, thrust):

        # Drag coefficient
        CD  = self.CDmin + (((self.CL - self.CLmin) ** 2) * self.k)

        # Drag force
        D   = 0.5*self.rho*(speed**2)*self.S*CD

        # Acceleration in x (positive in direction of V)
        a_x = (-D + thrust*np.cos(motor_angle))/self.m

        return a_x

    def vertical_equilibrium(self, motor_angle, speed, thrust):

        # Lift force
        L = 0.5*self.rho*speed*speed*self.S*self.CL

        # Acceleration in y (vertical direction, positive upwards)
        a_y = (L + np.sin(motor_angle)*thrust - self.W)/self.m

        return a_y

    def thrust_vertical(self, motor_angle, speed):

        # Lift force
        L       = 0.5 * self.rho * speed * speed * self.S * self.CL

        thrust  = (self.W - L)/np.sin(motor_angle)

        return thrust

    def thrust_horizontal(self, motor_angle, speed):

        # Drag coefficient
        CD = self.CDmin + (((self.CL - self.CLmin) ** 2) * self.k)

        # Drag force
        D       = 0.5 * self.rho * speed * speed * self.S * CD

        thrust  = D/np.cos(motor_angle)

        return thrust

    def simulate(self, plotting = False):

        # Initial values
        t   = 0.
        dt  = 0.1
        vx  = 0.     # Horizontal speed
        vy  = 0.     # Vertical speed
        x   = 0.#np.array([[0], [0]], dtype = 'float64')     # horizontal position
        y   = 0.     # Vertical position
        V   = 0.
        i_T = np.pi/2
        a_T = i_T
        a = 0
        # List to store state variables
        V_lst   = []
        t_lst   = []
        x_lst   = []
        y_lst   = []
        i_lst   = []
        E_lst   = []

        # Main loop
        running = True
        while running:

            t   += dt

            # Maximum thrust
            T_max   = self.max_thrust(a_T, V)

            # Engine thrust setting for vertical equilibrium, or to maintain speed
            T_vert  = self.thrust_vertical(i_T, V)
            T_hor   = self.thrust_horizontal(i_T, V)
            T_req   = np.maximum(T_vert, T_hor)


            # In the beginning op transition the motors are rotated with a constant speed, correct for future rotation
            if T_req < T_max:
                i_T -= 2*np.pi*dt/180

                # Recalculate the thrust for the rotated engines
                T_vert = self.thrust_vertical(i_T, V)
                T_hor = self.thrust_horizontal(i_T, V)
                T   = max(T_vert, T_hor)

            # If the required thrust is higher than the maximum thrust, wait for the speed to increase
            else:
                T   = T_max
                i_T -= 0.5*np.pi*dt/180

            # Power used
            P   = self.disk_power(T, 0, a_T, V)

            i_T     = max(i_T, 0)

            # Numerical integration
            ax = self.horizontal_equilibrium(i_T, V, T)
            ay = self.vertical_equilibrium(i_T, V, T)

            vx += ax*dt
            vy += ay*dt

            x += vx*dt
            y += vy*dt

            V       = np.sqrt(vx**2  + vy**2)

            alpha   = np.arctan2(vy, vx)
            a_T     = i_T + alpha

            # Store everything
            V_lst.append(V)
            t_lst.append(t)
            x_lst.append(x)
            y_lst.append(y)
            i_lst.append(i_T)
            E_lst.append(P*dt)

            # Stop the simulation if the engines are rotated enough and climb begins
            if t > 100 or i_T < 0.5e-1:
                running = False

        if plotting:
            plt.subplot(311)
            plt.plot(x_lst, y_lst)
            plt.xlabel("x-position [m]")
            plt.ylabel("y [m]")
            plt.grid()

            plt.subplot(312)
            plt.plot(t_lst, V_lst)
            plt.xlabel("Time [s]")
            plt.ylabel("V [m/s]")
            plt.grid()

            plt.subplot(313)
            plt.plot(np.array(t_lst), np.array(i_lst)*180/np.pi)
            plt.xlabel("Time [s]")
            plt.ylabel("Motor angle [deg]")
            plt.grid()

            print("end speed", V_lst[-1])
            plt.tight_layout()
            plt.show()

        # Return the total energy
        return sum(E_lst)
#
#
# trans = transition_EOM(18000, "../data/inputs_config_1.json")
# trans.simulate(plotting = True)
