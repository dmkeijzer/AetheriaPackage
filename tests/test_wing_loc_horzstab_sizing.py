import pytest
import os
import sys
import pathlib as pl
import numpy as np

#sys.path.append(str(list(pl.Path(__file__).parents)[1]))
#os.chdir(str(list(pl.Path(__file__).parents)[1]))

from scripts.stab_ctrl.wing_loc_horzstab_sizing import stabcg, ctrlcg, CLaAhcalc, x_ac_fus_1calc, x_ac_fus_2calc, betacalc, CLahcalc, stab_formula_coefs, CLh_approach_estimate, cmac_fuselage_contr, ctrl_formula_coefs

def example_values():
    return {
        "stabcg": {"ShS": None, "x_ac": None, "CLah": None, "CLaAh": None, "depsda": None, "lh": None, "c": None, "VhV2": None},
        "ctrlcg": {"ShS": None, "x_ac": None, "Cmac": None, "CLAh": None, "CLh": None,"lh": None, "c": None, "VhV2": None},
        "CLaAh": {"CLaw": None, "b_f": None, "b": None, "S": None, "c_root": None},
        "x_ac_fus1": {"b_f": None, "h_f": None, "l_fn": None, "CLaAh": None, "S": None, "MAC": None},
        "x_ac_fus2": {"b_f": None, "S": None, "b": None, "Lambdac4": None, "taper": None, "MAC": None},
        "beta": {"M": None},
        "CLah": {"A_h": None, "beta": None, "eta": None, "Lambdah2": None},
        "stabformula": {"CLah": None, "CLaAh": None, "depsda": None, "l_h": None, "MAC": None, "Vh_V_2": None, "x_ac_stab_bar": None, "SM": None},
        "CLh": {"A_h", None},
        "cmac_fus": {"b_f": None, "l_f": None, "h_f": None, "CL0_approach": None, "S": None, "MAC": None, "CLaAh": None},
        "ctrlformula": {"CLh_approach": None, "CLAh_approach": None, "l_h": None, "MAC": None, "Vh_V_2": None, "Cm_ac": None, "x_ac_stab_bar": None},
    }
def test_wing_loc_horzstab_sizing(example_values):
    stabcgtest = stabcg(**example_values["stabcg"])
    ctrlcgtest = ctrlcg(**example_values["ctrlcg"])
    CLaAh = CLaAhcalc(**example_values["CLaAh"])
    x_ac_fus_1 = x_ac_fus_1calc(**example_values["x_ac_fus1"])
    x_ac_fus_2 = x_ac_fus_2calc(**example_values["x_ac_fus2"])
    beta = betacalc(**example_values["beta"])
    CLah = CLahcalc(**example_values["CLah"])
    stabm, stabq = stab_formula_coefs(**example_values["stabformula"])
    CLh = CLh_approach_estimate(**example_values["CLh"])
    cmac_fus = cmac_fuselage_contr(**example_values["cmac_fus"])
    ctrlm, ctrlq = ctrl_formula_coefs(**example_values["ctrlformula"])

    assert np.isclose(stabcgtest, 0)
    assert np.isclose(ctrlcgtest, 0)
    assert np.isclose(CLaAh, 0)
    assert np.isclose(x_ac_fus_1, 0)
    assert np.isclose(x_ac_fus_2, 0)
    assert np.isclose(beta, 0)
    assert np.isclose(CLah, 0)
    assert np.isclose(stabm, 0)
    assert np.isclose(stabq, 0)
    assert np.isclose(CLh, 0)
    assert np.isclose(cmac_fus, 0)
    assert np.isclose(ctrlm, 0)
    assert np.isclose(ctrlq, 0)
