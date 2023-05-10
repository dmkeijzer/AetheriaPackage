import unittest

import PropandPower.engine_sizing_positioning as esp


class testPropulsiob(unittest.TestCase):
    def test_eng_size(self):
        # Example variables:
        span = 9
        fus_width = 2
        clearance_fus = 0.2
        clearance_prop = 0.2
        N_prop = 12
        MTOM = 1000

        # Try sizing the radius with those variables and see if it is correct
        sample_eng_size = esp.PropSizing(span, fus_width, N_prop, clearance_fus, clearance_prop, MTOM, xi_0=0.25)

        # Calculated by hand, maybe not the best way of testing

        # Check if the radius on these conditions is 0.58 metres
        self.assertAlmostEqual(sample_eng_size.radius(), 0.58, delta=0.01)

        # Check if area in these conditions is
        area_prop = 1
        self.assertAlmostEqual(sample_eng_size.disk_loading(), 84.1088, places=2)

