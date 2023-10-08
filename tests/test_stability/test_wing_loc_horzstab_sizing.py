import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from modules.stab_ctrl.wing_loc_horzstab_sizing import stabcg, ctrlcg, CLaAhcalc, x_ac_fus_1calc, x_ac_fus_2calc, betacalc, CLahcalc, stab_formula_coefs, CLh_approach_estimate, cmac_fuselage_contr, ctrl_formula_coefs, wing_location_horizontalstab_size

@pytest.fixture
def example_values():
    return {
        "stabcg": {"ShS": 0.15, "x_ac": 0.24, "CLah": 1.6, "CLaAh":3.7, "depsda": 0.11, "lh": 3, "c": 1.2, "VhV2": 0.95, "SM": 0.05},
        "ctrlcg": {"ShS": 0.15, "x_ac": 0.24, "Cmac": -0.4, "CLAh": 1.7, "CLh": -0.3,"lh": 3, "c": 1.2, "VhV2": 0.95},
        "CLaAh": {"CLaw": 3, "b_f": 0.9, "b": 10, "S": 12, "c_root": 1.2},
        "x_ac_fus1": {"b_f": 0.9, "h_f": 1.1, "l_fn": 2.8, "CLaAh":3.7, "S":12, "MAC":1.2},
        "x_ac_fus2": {"b_f": 0.9, "S":12, "b": 10, "Lambdac4": np.radians(2), "taper":0.4, "MAC":1.2},
        "beta": {"M": 0.23},
        "CLah": {"A_h": 5, "beta": 0.92, "eta": 0.95, "Lambdah2":np.radians(15)},
        "stabformula": {"CLah": 1.6, "CLaAh": 3.7, "depsda":0.11, "l_h":3, "MAC": 1.2, "Vh_V_2":0.95, "x_ac_stab_bar":0.24, "SM": 0.05},
        "CLh": {"A_h": 5},
        "cmac_fus": {"b_f": 0.9, "l_f": 7, "h_f":1.1, "CL0_approach":0.6, "S":12, "MAC":1.2, "CLaAh":3.7},
        "ctrlformula": {"CLh_approach": -0.5, "CLAh_approach": 1.9, "l_h":3, "MAC":1.2, "Vh_V_2": 0.95, "Cm_ac":-0.4, "x_ac_stab_bar": 0.24},
    }

def test_wing_loc_functions(example_values):
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

    assert np.isclose(stabcgtest, 0.327108108)
    assert np.isclose(ctrlcgtest, 0.412426471)
    assert np.isclose(CLaAh, 3.36428375)
    assert np.isclose(x_ac_fus_1, -0.0936486486)
    assert np.isclose(x_ac_fus_2, 0.00389402422)
    assert np.isclose(beta, 0.973190629)
    assert np.isclose(CLah, 4.23088125)
    assert np.isclose(stabm, 1.0940272)
    assert np.isclose(stabq, 0.207865168)
    assert np.isclose(CLh, -0.598491581)
    assert np.isclose(cmac_fus, -0.0748648959)
    assert np.isclose(ctrlm, -1.6)
    assert np.isclose(ctrlq, 0.720842105)


def test_wing_location_hor_sizing(wing, fuselage, aero, veetail, aircraft, power , engine, stability):
    wing_location_horizontalstab_size(wing, fuselage, aero, veetail, aircraft, power, engine, stability, 6)


