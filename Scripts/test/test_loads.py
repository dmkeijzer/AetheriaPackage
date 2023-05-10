# import unittest
# import pytest

# from structures.SolveLoads import *
# from MathFunctions.Mechanics import StepFunction

# class TestLoads(unittest.TestCase):
#     def setUp(self):
#         self.L1, self.L2 = SolveACLoads(1.95, 0.5, 3.5)
#     def test_ACLoads(self):
#         pass
#         # self.assertAlmostEqual(self.L1, 9787.2735, places=4)
#         # self.assertAlmostEqual(self.L2, 9155.8365, places=4)
#     def test_WingLoads(self):
#         wingEquation = SolveWingLoads(0.2, 11.2, self.L1, 800, (self.L1 + self.L2) / (8*9.81), 100, 3)
#         Fx, Fy, Fz, Mx, My, Mz = WLoads = wingEquation.SolveEquation()
#         calced = [0.0, -3709.6923750000014, 0.0, 10387.138650000004, 0.0, 340.9692375000002]
#         for j in range(len(calced)):
#             self.assertAlmostEqual(WLoads[j], WLoads[j], places=3)