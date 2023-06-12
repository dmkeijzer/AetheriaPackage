import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from modules.aero.clean_class2drag  import Reynolds, Mach_cruise, FF_fus, FF_wing, S_wet_fus, CD_upsweep, CD_base, C_fe_fus, C_fe_wing, CD_fus, CD_wing, CD0, CDi, CD, lift_over_drag, Oswald_eff, Oswald_eff_tandem

@pytest.fixture
def example_values():
    return {
        "reynolds": {"rho_cruise": 1.0, "V_cruise": 100, "mac": 5, "mu": 0.1, "k": 0.1},
        "mach_cruise": {"V_cruise": 100, "gamma": 1.4, "R": 300, "T_cruise": 270},
        "ff_fus": {"l": 10, "d": 2},
        "ff_wing": {"toc": 0.3, "xcm": 0.3, "M": 0.3, "sweep_m": 0},
        "s_wet_fus": {"d": 2, "l1": 3, "l2": 4, "l3": 3},
        "cd_upsweep": {"u": 0.1, "d": 2, "S_wet_fus": 10},
        "cd_base": {"M": 0.3, "A_base": 10, "S_wet_fus": 10},
        "c_fe_fus": {"frac_lam_fus": 0.1, "Reynolds": 1500000, "M": 0.3},
        "c_fe_wing": {"frac_lam_wing": 0.1, "Reynolds": 15000000, "M": 0.3},
        "cd_fus": {"C_fe_fus": 0.1, "FF_fus": 1, "S_wet_fus": 10},
        "cd_wing": {"name": "J1", "c_fe_wing": 0.1, "FF_wing": 1, "S_wet_wing": 10, "S": 10},
        "cd0": {"S": 10, "cd_fus": 0.01, "cd_wing": 0.01, "cd_upsweep": 0.01, "cd_base": 0.01},
        "cdi": {"CL": 1.0, "A": 8, "e": 0.8},
        "cd": {"cd0": 0.02, "cdi": 0.2},
        "lift_over_drag": {"CL_output": 1.0, "CD_output": 0.15},
        "oswald_eff": {"A": 8},
        "oswald_eff_tandem": {"b1": 10, "b2": 5, "h": 0.6}
    }

def test_drag2_calculation(example_values):
    reynolds = Reynolds(**example_values["reynolds"])
    mach_cruise = Mach_cruise(**example_values["mach_cruise"])
    ff_fus = FF_fus(**example_values["ff_fus"])
    ff_wing = FF_wing(**example_values["ff_wing"])
    s_wet_fus = S_wet_fus(**example_values["s_wet_fus"])
    cd_upsweep = CD_upsweep(**example_values["cd_upsweep"])
    cd_base = CD_base(**example_values["cd_base"])
    c_fe_fus = C_fe_fus(**example_values["c_fe_fus"])
    c_fe_wing1 = C_fe_wing(**example_values["c_fe_wing"])
    cd_fus = CD_fus(**example_values["cd_fus"])
    cd_wing = CD_wing(**example_values["cd_wing"])
    cd0 = CD0(**example_values["cd0"])
    cdi = CDi(**example_values["cdi"])
    cd = CD(**example_values["cd"])
    lift_over_drag = lift_over_drag(**example_values["lift_over_drag"])
    oswald_eff = Oswald_eff(**example_values["oswald_eff"])
    oswald_eff_tandem = Oswald_eff_tandem(**example_values["oswald_eff_tandem"])


    assert np.isclose(reynolds.mass, 573.014)
    assert np.isclose(mach_cruise, 0.3)
    assert np.isclose(ff_fus.ff, 0.2)
    assert np.isclose(ff_wing.ff, 0.3)
    assert np.isclose(s_wet_fus.s_wet, 150)
    assert np.isclose(cd_upsweep.cd, 0.2)
    assert np.isclose(cd_base.cd, 0.3)
    assert np.isclose(c_fe_fus.c_fe, 0.1)
    assert np.isclose(c_fe_wing1.c_fe, 0.2)
    assert np.isclose(cd_fus.cd, 0.05)
    assert np.isclose(cd_wing.cd, 0.04)
    assert np.isclose(cd0.cd0, 0.1)
    assert np.isclose(cdi.cdi, 0.16)
    assert np.isclose(cd.cd, 0.22)
    assert np.isclose(lift_over_drag.L_over_D, 15)
    assert np.isclose(oswald_eff.e, 0.84)
    assert np.isclose(oswald_eff_tandem.e, 0.83)