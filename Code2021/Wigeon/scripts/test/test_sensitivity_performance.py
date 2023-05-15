import unittest
import numpy as np
from Flight_performance.Sensitivity_analysis import sensitivity_analysis
from Aero_tools import speeds


class Test_sensitivity_analysis(unittest.TestCase):
    def test_optimal_speed(self):

        # Get the optimal cruise speed from the aero tools
        V           = speeds(305, m = 2000)
        V_opt_at    = V.cruise()

        # Get the optimal speed from the sensitivity analysis
        sensitivity = sensitivity_analysis(2000)

        V_opt_sa    = sensitivity.cruise_speed(plotting = False)

        # Test whether they agree (only to within +-2.5 m/s, due to high step size in the sensitivity analysis)
        self.assertAlmostEqual(V_opt_at, V_opt_sa, delta = 2.5)

    def test_wind(self):

        sensitivity = sensitivity_analysis(2000)

        # Get the range when the headwind equals the airspeed, should equal zero
        range_hw = sensitivity.wind(test_wind = 60, testing = True, plotting = False)

        # No wind
        range_nw = sensitivity.wind(test_wind = 0, testing = True, plotting = False)

        # Range with a tailwind equal to the cruise speed, range should double
        range_tw = sensitivity.wind(test_wind = -60, testing = True, plotting = False)

        self.assertAlmostEqual(range_hw/1000, 0, delta = 5)
        self.assertAlmostEqual(range_tw/1000, 2*range_nw/1000, 5)
