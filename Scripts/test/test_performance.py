import unittest
import numpy as np
from Aero_tools import ISA
from Flight_performance.Flight_performance_final import evtol_performance, mission, Drag
import scipy.optimize as optimize
from constants import g, eff_hover, eff_prop


class testPerformance(unittest.TestCase):
    def test_max_speed(self):


        P_max = 1.5e6
        S        = 10
        W        = 2000
        A_disk  = 8

        rho     = ISA(300).density()

        # Finding V is iterative
        V = 100
        err = 10

        while err > 0.5:

            V_init  = V
            CL      = 2*W/(rho*V*V*S)
            CD      = Drag.CD(CL)

            eff = eff_hover + V*(eff_prop - eff_hover)/60

            # Compare the maximum speed from the climb rate plot to an analytical solution
            V       = (2*eff*P_max/(rho*S*CD))**(1/3)

            err = abs(V - V_init)

        V_max = V

        performance = evtol_performance(300, 60, S = S, CL_max = 1.5, mass = 2000, battery_capacity = 500e6,
                                        EOM = 1500, loiter_time = 30*60, A_disk = A_disk, P_max = P_max)

        V_max_RC    = performance.climb_performance(testing = True)

        self.assertAlmostEqual(V_max, V_max_RC, delta = 5)

    def test_vertical_climb(self):

        mass = 2000
        A_disk = 8
        P_max = 1.5e6

        performance = evtol_performance(cruising_alt=300, cruise_speed=60, S = 14, CL_max = 1.5, mass = mass,
                                        battery_capacity = 500e6, EOM = 1500, loiter_time = 30*60, A_disk = A_disk,
                                        P_max = P_max)

        rho    = ISA(0).density()

        # Prediction using terminal velocity
        V_fall  = -np.sqrt(2*performance.W/(performance.CD_vert*rho*performance.S))

        # Prediction using actual model
        V_mod   = performance.vertical_equilibrium(300, testing = True, test_thrust = 0, m = mass)

        self.assertAlmostEqual(V_fall, V_mod, delta = 2)

        # Test what happens when thrust equals weight
        RC_hover = performance.vertical_equilibrium(300, mass, testing = True, test_thrust = mass*g)

        self.assertAlmostEqual(RC_hover, 0)

    def test_max_thrust(self):

        # Check the numerical inverse of the T vs P relationship
        perf = evtol_performance(cruising_alt = 300, cruise_speed = 60, CL_max=1.5, A_disk=10, S = 14,
                                 battery_capacity=4e6*1.2, mass = 3000, EOM = 3000 - 475, loiter_time = 30*60,
                                 P_max = 3000000)

        T_max = perf.max_thrust(1.225, 50)

        P_m   = perf.thrust_to_power(T_max, 50, 1.225)

        self.assertAlmostEqual(P_m, 0)

    def test_range_and_energy(self):

        # Initial constants
        mass = 2000
        m_PL = 500
        h_cr = 400
        v_cr = 60
        CL_max = 1.5
        S = 14
        A_disk = 8
        P_max = 1.6e6
        target_range = 300e3

        miss = mission(mass=mass, cruising_alt=h_cr, cruise_speed=v_cr, CL_max=CL_max, wing_surface=S, P_max = P_max,
                       A_disk=A_disk, t_loiter=30 * 60, rotational_rate=5, mission_dist=target_range)

        E_tot, t_tot = miss.total_energy()

        perf = evtol_performance(cruising_alt=h_cr, cruise_speed=v_cr, S=S, CL_max=CL_max, mass=mass,
                                 battery_capacity = E_tot, EOM=mass-m_PL, loiter_time=30 * 60, A_disk=A_disk,
                                 P_max=P_max)

        R, t = perf.range(cruising_altitude = h_cr, cruise_speed = v_cr, mass = mass, loiter = True)

        # Compare the calculated range with the target range
        self.assertAlmostEqual(target_range, R)

        # Compare the total times
        self.assertAlmostEqual(t_tot, t)


