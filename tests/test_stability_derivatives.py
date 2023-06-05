import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from scripts.stab_ctrl.aetheria_stability_derivatives import *

@pytest.fixture
def example_values():
    return {
        "airfoil2wing": {"cla": 4, "A": 10},
        "dwk": {"lh": 3, "b": 10},
        "dw": {"k": 3, "CLa": 3.7 ,"A": 10},
        "Cma_fuse": {"Vfuse": 7, "S": 12, "c": 1.2},
        "Cnb_fuse": {"Vfuse": 7, "S": 12, "b": 10},
        "CDa": {"CL0": 0.2, "CLa": 3.7, "A": 10},
        "Cxa": {"CL0": 0.2, "CDa": 0.05},
        "Cza": {"CLa": 3.7, "CD0": 0.002},
        "Vh": {"Sh": 3, "lh": 3, "S": 12, "c":1.2},
        "Czq": {"CLah": 1.6, "Vh":0.62},
        "Cma": {"CLa": 3.7, "lcg": 0.1, "c": 1.2, "CLah":1.6, "Vh":0.62, "depsda":0.11, "Cmafuse":0.97},
        "Cmq": {"CLah": 1.6, "Vh": 0.62, "lh":3, "c":1.2, "Cmqfuse": 0},
        "Vv": {"Sv": 1.3, "lv": 3, "S": 12, "b": 10},
        "Cyb": {"Sv": 1.3, "S": 12, "CLav": 1.8},
        "Cyr": {"Vv": 0.033, "CLav": 1.8},
        "Clb": {"CLa": 3.7, "dihedral": np.radians(5), "taper": 0.4},
        "Clp": {"CLa": 3.7, "taper": 0.4},
        "Clr": {"CL0": 0.2},
        #"Cnb": {"CLav": 1.8, "Vv": 0.033, "Cnbfuse": -0.12},
        "Cnp": {"CL0": 0.2},
        "Cnr": {"CLav": 1.8, "Vv": 0.033, "lv": 3, "b": 10},
        "czadot": {"CLah":1.6, "Sh": 3, "S": 12, "Vh_V2": 0.95, "depsda": 0.11, "lh": 3, "c":1.2},
        "cmadot": {"CLah":1.6, "Sh": 3, "S": 12, "Vh_V2": 0.95, "depsda": 0.11, "lh": 3, "c":1.2},
        "muc": {"m": 2500, "rho": 1.2, "S": 12, "c": 1.2},
        "mub": {"m": 2500, "rho": 1.2, "S": 12, "b": 10},
        "Cx0": {"W": 2500*9.8, "theta_0":15* np.pi/180, "rho": 1.2, "V": 80, "S": 12},
        "Cz0": {"W": 2500 * 9.8, "theta_0": 15*np.pi / 180, "rho":1.2, "V": 80, "S": 12}
    }

def test_mass_calculation(example_values):
    af2wing = airfoil_to_wing_CLa(**example_values["airfoil2wing"])
    dwk = downwash_k(**example_values["dwk"])
    dw = downwash(**example_values["dw"])
    Cmafuse = Cma_fuse(**example_values["Cma_fuse"])
    Cnbfuse = Cnb_fuse(**example_values["Cnb_fuse"])
    CDa = CDacalc(**example_values["CDa"])
    Cxatest = Cxa(**example_values["Cxa"])
    Cxqtest = Cxq()
    Czatest = Cza(**example_values["Cza"])
    Vh = Vhcalc(**example_values["Vh"])
    Czqtest = Czq(**example_values["Czq"])
    Cmatest = Cma(**example_values["Cma"])
    Cmqtest = Cmq(**example_values["Cmq"])
    Vv = Vvcalc(**example_values["Vv"])
    Cybtest = Cyb(**example_values["Cyb"])
    Cyrtest = Cyr(**example_values["Cyr"])
    Cyptest = Cyp()
    Clbtest = Clb(**example_values["Clb"])
    Clptest = Clp(**example_values["Clp"])
    Clrtest = Clr(**example_values["Clr"])
    #Cnbtest = Cnb(**example_values["Cnb"])
    Cnptest = Cnp(**example_values["Cnp"])
    Cnrtest = Cnr(**example_values["Cnr"])
    Czadot = CZ_adot(**example_values["czadot"])
    Cmadot = Cm_adot(**example_values["cmadot"])
    muctest = muc(**example_values["muc"])
    mubtest = mub(**example_values["mub"])
    Cz0test = Cz0(**example_values["Cz0"])
    Cx0test = Cx0(**example_values["Cx0"])

    assert np.isclose(af2wing, 3.54822585)
    assert np.isclose(dwk, 2.97411154)
    assert np.isclose(dw, 0.353323974)
    assert np.isclose(Cmafuse, 0.972222222)
    assert np.isclose(Cnbfuse, -0.116666667)
    assert np.isclose(CDa, 0.0471098632)
    assert np.isclose(Cxatest, 0.15)
    assert np.isclose(Cxqtest, 0)
    assert np.isclose(Czatest, -3.702)
    assert np.isclose(Vh, 0.625)
    assert np.isclose(Czqtest, -1.984)
    assert np.isclose(Cmatest, 0.395453333)
    assert np.isclose(Cmqtest, -4.96)
    assert np.isclose(Vv, 0.0325)
    assert np.isclose(Cybtest, -0.195)
    assert np.isclose(Cyrtest, -0.1188)
    assert np.isclose(Cyptest, 0)
    assert np.isclose(Clbtest, -0.0691898382)
    assert np.isclose(Clptest, -0.48452381)
    assert np.isclose(Clrtest, 0.05)
    #assert np.isclose(Cnbtest, -0.0606)
    assert np.isclose(Cnptest, -0.025)
    assert np.isclose(Cnrtest, -0.03564)
    assert np.isclose(Czadot, -0.1045)
    assert np.isclose(Cmadot, -0.26125)
    assert np.isclose(muctest, 144.675926)
    assert np.isclose(mubtest, 17.3611111)
    assert np.isclose(Cz0test, -0.513567334)
    assert np.isclose(Cx0test, 0.137609952)
