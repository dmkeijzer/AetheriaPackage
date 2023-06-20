import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
# sys.path.append(os.path.join(list(pl.Path(__file__).parents)[2], "modules","powersizing"))

import input.data_structures.GeneralConstants as const
from scipy.optimize import minimize
from input.data_structures.engine import Engine


class PylonSizing():
    def __init__(self, engine, L):
        self.mass_eng = engine.mass_pertotalengine
        self.L = L
        self.Tmax =  2.5*2200*9.81/6
        self.moment = self.Tmax*L

    def I_xx(self, x): return np.pi/4 *  ((x[0] + x[1])**4 - x[0]**4)

    def get_area(self, x):
        return np.pi*((x[1] + x[0])**2 - x[0]**2)

    def weight_func(self, x):
        return np.pi*((x[1] + x[0])**2 - x[0]**2)*const.rho_composite*self.L


    def get_stress(self, x):
        return (self.moment*(x[1] + x[0]))/self.I_xx(x)

    # def r2_larger_than_r1(self, x):
    #     # print(f"r2>r1 = {x[1] - x[0]}")
    #     return x[1] - x[0]

    def column_buckling_constraint(self, x):
        # print(f"r1, r2 = {x[0], x[1]}")
        # print(f"column buckling = {(np.pi**2*const.E_alu*self.I_xx(x))/(self.L**2*self.get_area(x))- self.get_stress(x)}")
        return (np.pi**2*const.E_composite*self.I_xx(x))/(self.L**2*self.get_area(x)) - self.get_stress(x)

    def von_mises_constraint(self, x):
        # print(f"Von Mises = {const.sigma_yield -1/np.sqrt(2)*self.get_stress(x)} ")
        return const.sigma_yield - 1/np.sqrt(2)*self.get_stress(x)

    def eigenfreq_constraint(self, x):
        # print(f"Eigenfrequency = {1/(2*np.pi)*np.sqrt((3*const.E_alu*self.I_xx(x))/(self.L**3*self.mass_eng))}")
        print(f"Ixx = {self.I_xx(x)}")
        return 1/(2*np.pi)*np.sqrt((3*const.E_composite*self.I_xx(x))/(self.L**3*self.mass_eng)) - const.eigenfrequency_lim_pylon


    def  optimize_pylon_sizing(self, x0):

        cons = (
            {'type': 'ineq', 'fun': self.column_buckling_constraint },
                {'type': 'ineq', 'fun': self.von_mises_constraint }
                # {'type': 'ineq', 'fun': self.eigenfreq_constraint}
                )
        bnds = ((0.095, 0.1), (0.001,0.2))

        res = minimize(self.weight_func, x0, method='SLSQP', bounds=bnds, constraints=cons)

        return res

if __name__ == "__main__":
    engine = Engine()
    engine.load()
    L = 2
    Pylon = PylonSizing(engine, L)
    x0 = (0.095,0.0093)
    print(Pylon.weight_func(x0)*2)
    print(Pylon.eigenfreq_constraint(x0))
    print(Pylon.von_mises_constraint(x0))
    print(Pylon.column_buckling_constraint(x0))
    # res = Pylon.optimize_pylon_sizing(x0)
    # print(res.x)
    # print(res.success)
