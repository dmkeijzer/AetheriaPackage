import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from modules.flight_perf.EnergyPower import v_exhaust, vhover, propeff, hoverstuffduct, hoverstuffopen, powerclimb, powercruise, powerdescend, powerloiter

@pytest.fixture
def example_values():
    return {
        "v_exhaust": {"MTOM": 5000, "g0": 9.81, "rho": 1.225, "atot": 10, "vinf": 83},
        "vhover": {"MTOM": 5000, "g0": 9.81, "rho": 1.225, "atot": 10},
        "propeff": {"vcr": 83, "vinf": 80},
        "hoverstuffopen": {"T": 290, "rho": 1.225, "atot": 10, "toverw": 10},
        "hoverstuffduct": {"T": 290, "rho": 1.225, "atot": 10, "toverw": 10},
        "powercruise": {"MTOM": 5000, "g0": 9.81, "v_cr": 83, "lift_over_drag": 15, "propeff": 0.6},
        "powerclimb": {"MTOM": 5000, "g0": 9.81, "S": 15, "rho": 1.225, "lod_climb": 100, "prop_eff": 0.6, "ROC": 5},
        "powerloiter": {"MTOM": 5000, "g0": 9.81, "S": 15, "rho": 1.225, "lod_climb": 100, "prop_eff": 0.6},
        "powerdescend": {"MTOM": 5000, "g0": 9.81, "S": 15, "rho": 1.225, "lod_climb": 100, "prop_eff": 0.6, "ROD": 3},

    }

def test_mission_energy(example_values):
    v_exhaust_test = v_exhaust(**example_values["v_exhaust"])
    vhover_test = vhover(**example_values["vhover"])
    propeff_test = propeff(**example_values["propeff"])
    hoverstuffopen_test = hoverstuffopen(**example_values["hoverstuffopen"])
    hoverstuffduct_test = hoverstuffduct(**example_values["hoverstuffduct"])
    powercruise_test = powercruise(**example_values["powercruise"])
    powerclimb_test = powerclimb(**example_values["powerclimb"])
    powerloiter_test = powerloiter(**example_values["powerloiter"])
    powerdescend_test = powerdescend(**example_values["powerdescend"])

    assert np.isclose(v_exhaust_test, 100)
    assert np.isclose(vhover_test, 70.71067811865475)
    assert np.isclose(propeff_test, 0.5)
    assert np.isclose(hoverstuffopen_test[0], 1573.1640228666924)
    assert np.isclose(hoverstuffopen_test[1], 4719.492068600077)
    assert np.isclose(hoverstuffopen_test[2], 11.166666666666666)
    assert np.isclose(hoverstuffopen_test[3], 33.50000000000001)
    assert np.isclose(hoverstuffduct_test[0], 785.2652335617312)
    assert np.isclose(hoverstuffduct_test[1], 2355.7957006851936)
    assert np.isclose(hoverstuffduct_test[2], 5.583333333333334)
    assert np.isclose(hoverstuffduct_test[3], 16.75)
    assert np.isclose(powercruise_test, 5595.409401435346)
    assert np.isclose(powerclimb_test, 5746.325711037199)
    assert np.isclose(powerloiter_test, 5992.201383912708)
    assert np.isclose(powerdescend_test, 6016.151426588486)