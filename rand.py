
from math import log10
import numpy as np
import sys
import os



path = r"C:\Users\damie\OneDrive\Desktop\Damien\DSE\AetheriaPackage\modules\structures"


sys.path.append(path)

import wingbox_georgina as wb

wb.taper = 0.45
wb.rho = 2710
wb.W_eng = 41.8
wb.E = 70e9
wb.poisson = 0.3
wb.pb=2.5
wb.beta=1.42
wb.g=5
wb.sigma_yield = 430e6
wb.m_crip = 0.85
wb.sigma_uts = 640e6
wb.n_max=2.5
wb.W_eng = 80
wb.y_rotor_loc =  [
            2.3,
            -2.3,
            5.0,
            -5.0,
            2.3,
            -2.3
      ]

print(sys.executable)

x=np.array([10.1,1.7, 0.003, 0.003, 0.12, 0.07, 0.003,0.003,0.004,0.0022])    # :param x0: Initial estimate Design vector X = [b, cr, tsp, trib, L, bst, hst, tst, wst, t]

@profile
def func():
    print(wb.wing_weight(x[0], x[1],x[2],x[3], x[4], x[5], x[6], x[7],x[8],[x[9]]))
    print(wb.global_local(x[0], x[1], x[4], x[5], x[6], x[7],[x[9]]))
    print(wb.post_buckling(x[0], x[1], x[2], x[3],  x[4], x[5], x[6], x[7], x[8], [x[9]]))
    print(wb.von_Mises(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],[x[9]]))
    print(wb.buckling_constr(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8],[x[9]]))
    print(wb.flange_loc_loc(x[0], x[1], x[4], x[5],x[7],x[8],[x[9]]))
    print(wb.local_column(x[0], x[1], x[4], x[5],x[6],x[7],x[8],[x[9]]))
    print(wb.crippling(x[0],  x[4],  x[6], x[7], x[8], [x[9]]))
    print(wb.web_flange(x[0], x[1], x[4], x[5], x[6], x[7], [x[9]]))

func()


