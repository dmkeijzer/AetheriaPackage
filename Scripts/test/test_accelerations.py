import unittest
import numpy as np

from Flight_performance.Flight_performance_final import mission


class testAccelerations(unittest.TestCase):
    def test_zero_conditions(self):

        # Introduce a mission where the target conditions are similar as the initial conditions
        standstill      = mission(mass = 2000, cruising_alt = 305, cruise_speed = 60)
        dist, E, time   = standstill.numerical_simulation(vx_start = 1e-9, y_start = 0,
                                                          th_start = np.pi/2, y_tgt = 0, vx_tgt = 0, plotting = False)

        # Energy used and distance covered should equal zero, time should be around 5
        self.assertAlmostEqual(dist, 0, places = 5)
        self.assertAlmostEqual(time, 5, places = 1)

    def test_straight_up(self):

        # Mission where the target velocity in x is zero, but the altitude 300m, no distance should be covered
        vertical        = mission(mass = 2000, cruising_alt = 305, cruise_speed = 60)
        dist, E, time   = vertical.numerical_simulation(vx_start = 1e-9, y_start = 0,
                                                        th_start = np.pi/2, y_tgt = 300, vx_tgt = 0, plotting = False)

        # Check if no distance is covered
        self.assertAlmostEqual(dist, 0, places = 5)

    def test_forward(self):

        # Mission where the change in altitude is zero
        cruise          = mission(mass = 2000, cruising_alt = 305, cruise_speed = 60)
        dist_up, E_up, time_up  = cruise.numerical_simulation(vx_start = 60, y_start = 300, th_start = np.radians(1),
                                                              y_tgt = 300+1e-9, vx_tgt = 60, plotting = True)

        dist_dn, E_dn, time_dn  = cruise.numerical_simulation(vx_start=60, y_start=300, th_start=np.radians(1),
                                                              y_tgt=300-1e-9, vx_tgt=60, plotting=False)

        # Test whether the climb and descend part of the model agree, inputting a small difference in final altitude
        self.assertAlmostEqual(E_up, E_dn, delta = 30)
        self.assertAlmostEqual(dist_up, dist_dn, places = 1)

        # Same mission, but using the cruise part of the program
        P_cruise = cruise.power_cruise_config(altitude = 305, speed = 60, mass = 2000)
        E_cruise = P_cruise*time_dn

        # Check whether the energies match
        #self.assertAlmostEqual(E, E_cruise)







