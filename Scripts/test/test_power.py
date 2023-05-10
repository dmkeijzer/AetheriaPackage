import unittest
import PropandPower.battery as bat
import PropandPower.redundancy_battery_config as red


class testpower(unittest.TestCase):
    def test_bat_size(self):
        # Example variables
        sp_en_den = 40
        vol_en_den = 17
        tot_energy = 250
        cost = 0.5
        DoD = 0.9
        P_den = 19
        P_max = 1
        safety = 1.5
        EOL_C = 0.85

        P_max_crazy = 13245
        # Size batteries with values and check if it's correct
        sample_bat_size = bat.Battery(sp_en_den, vol_en_den, tot_energy, cost, DoD, P_den, P_max, safety, EOL_C)
        sample_bat_size_crazypower = bat.Battery(sp_en_den, vol_en_den, tot_energy, cost, DoD, P_den, P_max_crazy,
                                                 safety, EOL_C)

        # Test battery mass test
        self.assertAlmostEqual(sample_bat_size.mass(), 12.255, places=2)  # Size for energy
        self.assertAlmostEqual(sample_bat_size_crazypower.mass(), 1366.873, places=2)  # Size for energy

        # Test volume
        self.assertAlmostEqual(sample_bat_size.volume(), 0.028835, places=5)

        # Test price
        self.assertAlmostEqual(sample_bat_size.price(), 0.245098, places=3)

    def test_bat_red(self):
        # Example variables
        # Characteristics from the plane
        V_motor = 134  # V
        E_tot = 12345  # kWh
        per_mot = 0.81

        # Cell characteristics
        V_cell = 2.35  # V
        C_cell = 137.312  # Ah

        # inputs
        n_mot = 8  # Number of motors in aircraft
        n_bat_mot = 3  # Number of batteries per motor

        # Run sample calc
        sample_bat_red = red.redundancy_power(V_motor, E_tot, V_cell, C_cell, n_mot, n_bat_mot, per_mot)

        # Individual tests
        self.assertEqual(sample_bat_red.N_cells_mot(), 30989)
        self.assertEqual(sample_bat_red.N_cells_misc(), 7269)
        self.assertEqual(sample_bat_red.N_cells_tot(), 38258)
        self.assertEqual(sample_bat_red.N_ser(), 58)
        self.assertEqual(sample_bat_red.N_par(), 535)
        self.assertEqual(sample_bat_red.N_par_new(), 552)
        self.assertEqual(sample_bat_red.N_cells_mot_new(), 32016)
        self.assertEqual(sample_bat_red.N_cells_new(), 39285)
        self.assertAlmostEqual(sample_bat_red.increase_mot(), 3.314079189, places=5)
        self.assertEqual(sample_bat_red.increase()[0], 1027)
        self.assertAlmostEqual(sample_bat_red.increase()[1], 2.684406, places=3)
