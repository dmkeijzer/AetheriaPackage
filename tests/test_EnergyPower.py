import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

import numpy as np
from modules.midterm_prop_flight_perf.EnergyPower import (
    v_exhaust,
    vhover,
    propeff,
    hoverstuffopen,
    hoverstuffduct,
    powercruise,
    powerclimb,
    powerloiter,
    powerdescend,
)

@pytest.fixture
def example_values():
    return {
        "v_exhaust": {
            "MTOM": 5000,
            "g0": 9.81,
            "rho": 1.225,
            "atot": 100,
            "vinf": 10
        },
        "vhover": {
            "MTOM": 5000,
            "g0": 9.81,
            "rho": 1.225,
            "atot": 100
        },
        "propeff": {
            "vcr": 110,
            "vinf": 100
        },
        "hoverstuffopen": {
            "T": 10000,
            "rho": 1.225,
            "atot": 100,
            "toverw": 0.5
        },
        "hoverstuffduct": {
            "T": 10000,
            "rho": 1.225,
            "atot": 100,
            "toverw": 0.5
        },
        "powercruise": {
            "MTOM": 5000,
            "g0": 9.81,
            "v_cr": 100,
            "lift_over_drag": 20,
            "propeff": 0.8
        },
        "powerclimb": {
            "MTOM": 5000,
            "g0": 9.81,
            "S": 50,
            "rho": 1.225,
            "lod_climb": 10,
            "prop_eff": 0.8,
            "ROC": 5
        },
        "powerloiter": {
            "MTOM": 5000,
            "g0": 9.81,
            "S": 50,
            "rho": 1.225,
            "lod_climb": 10,
            "prop_eff": 0.8
        },
        "powerdescend": {
            "MTOM": 5000,
            "g0": 9.81,
            "S": 50,
            "rho": 1.225,
            "lod_climb": 10,
            "prop_eff": 0.8,
            "ROD": 5
        }
    }

def test_v_exhaust(example_values):
    inputs = example_values["v_exhaust"]
    assert np.isclose(v_exhaust(**inputs), 10)

def test_vhover(example_values):
    inputs = example_values["vhover"]
    assert np.isclose(vhover(**inputs), 10)

def test_propeff(example_values):
    inputs = example_values["propeff"]
    assert np.isclose(propeff(**inputs), 20)

def test_hoverstuffopen(example_values):
    inputs = example_values["hoverstuffopen"]
    assert np.isclose(hoverstuffopen(**inputs), np.array([10, 10, 10, 10])).all()

def test_hoverstuffduct(example_values):
    inputs = example_values["hoverstuffduct"]
    assert np.isclose(hoverstuffduct(**inputs), np.array([10, 10 ,10 ,10])).all()

def test_powercruise(example_values):
    inputs = example_values["powercruise"]
    assert np.isclose(powercruise(**inputs), 10)

def test_powerclimb(example_values):
    inputs = example_values["powerclimb"]
    assert np.isclose(powerclimb(**inputs), 10)

def test_powerloiter(example_values):
    inputs = example_values["powerloiter"]
    assert np.isclose(powerloiter(**inputs), 10)

def test_powerdescend(example_values):
    inputs = example_values["powerdescend"]
    assert np.isclose(powerdescend(**inputs), 10)
