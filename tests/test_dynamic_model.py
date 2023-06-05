import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from modules.stab_ctrl.aetheria_stability_derivatives import longitudinal_derivatives, lateral_derivatives, eigval_finder_sym, eigval_finder_asymm


@pytest.fixture
def example_values():
    return {
        "symm": {"Iyy":12080, "m": 2500, "c":1.2, "long_stab_dervs": longitudinal_derivatives(0.04 , 0.6, 2500*9.8 ,1.2,12, 2500, 1.2, 3, 0.1, 0.02, 0.1, 0.95, np.radians(3), 80, Cmafuse=None, Cmqfuse=0, CLa=3.7, CLah=1.6, depsda=0.11,
                             CDa=None, Vh=None, Vfuse=7, cla=None, A=7, clah=None,
                             Ah=4, b=10, k=None, Sh=3)},
        "asymm":{"Ixx": 10440, "Izz": 21720, "Ixz": 500, "m": 2500, "b":10, "CL":0.6, "lat_stab_dervs": lateral_derivatives(0.08,2500,1.2, 2, 3, 12, 10, np.radians(2), 0.4, 0.1, CLav=1.8, Vv=None, CLa=3.7, clav=None,
                        Av=3, cla=None, A=7, Cn_beta_dot=None,CY_beta_dot=None)}
    }

def find_eigval_symm():
    mydict = longitudinal_derivatives(0.04, 0.6, 2500 * 9.8, 1.2, 12, 2500, 1.2, 3, 0.1, 0.02, 0.1, 0.95, np.radians(3), 80,
                            Cmafuse=None, Cmqfuse=0, CLa=3.7, CLah=1.6, depsda=0.11,
                            CDa=None, Vh=None, Vfuse=7, cla=None, A=7, clah=None,
                            Ah=4, b=10, k=None, Sh=3)
    c=1.2
    v = 80
    muc = mydict["muc"]
    Czadot = mydict["Cz_adot"]
    Cmadot = mydict["Cm_adot"]
    Cxu = mydict["Cxu"]
    Cxa = mydict["Cxa"]
    Cz0 = mydict["Cz0"]
    Czu = mydict["Czu"]
    Cza = mydict["Cza"]
    Cx0 = mydict["Cx0"]
    Czq = mydict["Czq"]
    Cmu = mydict["Cmu"]
    Cma = mydict["Cma"]
    Cmq = mydict["Cmq"]
    KY2 = 12080 / (2500*1.2**2)
    P = np.array([[-2*muc*c/v, 0, 0, 0],
                  [0, (Czadot -2*muc)*c/v, 0, 0],
                  [0,0,-c/v,0],
                  [0,Cmadot*c/v, 0, -2*muc*KY2*c/v]])
    Q = np.array([[-Cxu, -Cxa, -Cz0, 0],
                  [-Czu, -Cza, Cx0, -(Czq+2*muc)],
                  [0,0,0,-1],
                  [-Cmu, -Cma, 0, -Cmq]])
    A = np.linalg.inv(P)@Q
    return np.linalg.eigvals(A)

def find_eigval_asymm():
    mydict = lateral_derivatives(0.08,2500,1.2, 2, 3, 12, 10, np.radians(2), 0.4, 0.1, CLav=1.8, Vv=None, CLa=3.7, clav=None,
                        Av=3, cla=None, A=7, Cn_beta_dot=None,CY_beta_dot=None)
    b=10
    v=80
    Cyb = mydict["Cyb"]
    Cyp = mydict["Cyp"]
    Cyr = mydict["Cyr"]
    Cy_bdot = mydict["Cy_beta_dot"]
    Clb = mydict["Clb"]
    Clp = mydict["Clp"]
    Clr = mydict["Clr"]
    Cnb = mydict["Cnb"]
    Cnp = mydict["Cnp"]
    Cnr = mydict["Cnr"]
    Cn_bdot = mydict["Cn_beta_dot"]
    mub = mydict["mub"]
    CL = 0.6
    KXZ = 0.002
    KX2 = 10440/(2500*10**2)
    KZ2 = 21720/(2500*10**2)
    P = np.array([[(Cy_bdot - 2*mub)*b/v, 0,0, 0],
                  [0,-b/(2*v),0,0],
                  [0,0, -4*mub*KX2*b/v, 4*mub*KXZ*b/v],
                  [Cn_bdot*b/v, 0, 4*mub*KXZ*b/v, -4*mub*KZ2*b/v]])
    Q = np.array([[-Cyb, -CL, -Cyp, -(Cyr-4*mub)],
                  [0,0,-1,0],
                  [-Clb, 0, -Clp, -Clr],
                  [-Cnb, 0, -Cnp, -Cnr]])
    A = np.linalg.inv(P)@Q
    return np.linalg.eigvals(A)

def test_dynamic_model(example_values):
    eigvalslong = eigval_finder_sym(**example_values["symm"]).sort()
    eigvalslat = eigval_finder_asymm(**example_values["asymm"]).sort()
    eigvalslong_ver = find_eigval_symm().sort()
    eigvalslat_ver = find_eigval_asymm().sort()
    for i in range(len(eigvalslong)):
        assert np.isclose(eigvalslong[i], eigvalslong_ver[i])
    for i in range(len(eigvalslat)):
        assert (np.isclose(eigvalslat[i], eigvalslat_ver[i]))
