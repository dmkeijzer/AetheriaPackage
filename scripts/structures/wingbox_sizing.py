import sys
import os
import pathlib as pl
from scipy.optimize import minimize

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from modules.structures.wingbox_georgina *


x0=np.array([7, 1.5, 0.003, 0.003, 0.12, 0.07, 0.003,0.003,0.004,0.0022])

fun = lambda x: wing_weight(x[0], x[1],x[2],x[3], x[4], x[5], x[6], x[7],x[8],[x[9]])
cons = ({'type': 'ineq', 'fun': lambda x: global_local(x[0], x[1], x[4], x[5], x[6], x[7],[x[9]])},
        {'type': 'ineq', 'fun': lambda x: post_buckling(x[0], x[1], x[2], x[3],  x[4], x[5], x[6], x[7], x[8], [x[9]])},
        {'type': 'ineq', 'fun': lambda x: von_Mises(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],[x[9]])},
        {'type': 'ineq', 'fun': lambda x: buckling_constr(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],[x[9]])},
        {'type': 'ineq', 'fun': lambda x: flange_loc_loc(x[0], x[1], x[4], x[5],x[7],x[8],[x[9]])},
        {'type': 'ineq', 'fun': lambda x: local_column(x[0], x[1], x[4], x[5],x[6],x[7],x[8],[x[9]])},
        {'type': 'ineq', 'fun': lambda x: crippling(x[0],  x[4],  x[6], x[7], x[8], [x[9]])},
        {'type': 'ineq', 'fun': lambda x: web_flange(x[0], x[1], x[4], x[5], x[6], x[7], [x[9]])})


bnds = ((5, 9), (1, 4), (0.001, 0.005), (0.001, 0.005), (0.007, 0.05), (0.001, 0.01),(0.001, 0.01),(0.001, 0.003),(0.004, 0.005),(0.001, 0.003))
rez = minimize(fun, x0, method='trust-constr',bounds=bnds, constraints=cons)
print(rez)