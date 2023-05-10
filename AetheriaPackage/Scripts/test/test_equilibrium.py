import unittest
import pytest

from structures.Equilibrium import PointLoad, Moment, RunningLoad, EquilibriumEquation
from structures.MathFunctions import StepFunction

test_data = [[float(d) for d in dat.split(' ')] for dat in '''47.88 -1.064
49.05 -1.090
50.22 -1.116
51.39 -1.142
52.56 -1.168
53.73 -1.194
54.90 -1.220
56.07 -1.246
57.24 -1.272
58.36 -1.297'''.split('\n')]
# https://www.matec-conferences.org/articles/matecconf/abs/2019/04/matecconf_eaaic2018_06004/matecconf_eaaic2018_06004.html

L, w, h = 100, 10, 15
E = 200e9
I = w * h ** 3 / 12

class TestEquilibrium(unittest.TestCase):
    def test_PointLoad(self):
        pl = PointLoad([1, 0, 0], [0, 1, 0])
        self.assertEqual(pl.force()[0], 1)
        self.assertEqual(pl.moment()[2], -1)
    
    def test_Moment(self):
        mom = Moment([0, 0, 1])
        self.assertTrue(all(i == 0 for i in mom.force()))
        self.assertListEqual(list(mom.moment()), [0, 0, 1])
    
    def test_RunningLoad(self):
        q = RunningLoad([[1]*5, [2]*5], range(5), 0)
        self.assertListEqual(list(q.force()), [0, 4, 8])
        self.assertListEqual(list(q.moment()), [0, -16, 8])

    def test_equilibrium(self):
        load1 = PointLoad([1, 0, 0], [0, 1, 0])
        load2 = PointLoad([1, 0, 0], [-1, 0, 0])
        load3 = PointLoad([0, 1, 0], [0, 1, 0])

        F1 = PointLoad([0, 1, 0], [0, 1, 0])
        F2 = PointLoad([1, 0, 0], [-1, 0, 0])
        F3 = PointLoad([0, -1, 0], [1, 0, 0])

        Eql = EquilibriumEquation(kloads=[load1, load2, load3], ukloads=[F1, F2, F3])
        Eql.SetupEquation()
        self.assertListEqual(list(Eql.SolveEquation()), [-2, -2, -1])

    def test_beam_deflection(self):
        for distload, deflection in test_data:
            q = RunningLoad([[-distload, -distload], [0, 0]], [0, L], axis=0)
            Fixedx = PointLoad([1, 0, 0], [0, 0, 0])
            Fixedy = PointLoad([0, 1, 0], [0, 0, 0])
            eqn = EquilibriumEquation(kloads=[q], ukloads=[Fixedx, Fixedy, Moment([0, 0, 1])])
            eqn.SetupEquation()
            Ax, Ay, Ma = eqn.SolveEquation()
            self.assertAlmostEqual(Ax, 0)
            self.assertAlmostEqual(Ay, distload * L) # Verifying Reaction Loads
            self.assertAlmostEqual(Ma, distload * L * L / 2)

            V = StepFunction([[Ay, 0, 0], [q(0)[0], 0, 1]])
            M = -V.integral(-Ma)
            self.assertAlmostEqual(V(L), 0)
            self.assertAlmostEqual(M(L), 0)
            d = -M.integral().integral()*1e3 / (E * I)
            self.assertAlmostEqual(d(L), deflection*1e-3, places=6)
