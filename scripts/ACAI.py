from AetheriaPackage.sim_contr import acai, pylon_calc
import numpy as np


r = 0.275
r1  = np.sqrt(3)*r/2
Bf = np.array([[1,1,1,1,1,1],
                [0, -r1,-r1,0,r1,r1],
                [r, r/2, -r/2, -r, -r/2, r/2],
                [-0.1,0.1,-0.1,0.1,-0.1,0.1]])
fcmin = np.zeros((6,1))
fcmax=6.125*np.ones((6,1))
Tg = np.array([[1.535*9.8],
                [0],
                [0],
                [0]])
res = acai(Bf, fcmin, fcmax, Tg)
print(res)
assert np.isclose(res, 1.486052554907109)
## From OG ACAI paper
