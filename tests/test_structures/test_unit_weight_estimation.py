import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from modules.structures.ClassIIWeightEstimation import Wing, Fuselage, LandingGear, Powertrain, HorizontalTail, Nacelle, H2System, Miscallenous

@pytest.fixture
def example_values():
    return {
        "wing": {"mtom": 5000, "S": 100, "n_ult": 3, "A": 8},
        "fuselage1": {"identifier": "J1", "mtom": 5000, "max_per": 20, "lf": 10, "npax": 50},
        "fuselage2": {"identifier": "L1", "mtom": 5000, "max_per": 20, "lf": 10, "npax": 50},
        "landing_gear": {"mtom": 5000},
        "powertrain": {"p_max": 10000, "p_dense": 10},
        "horizontal_tail": {"w_to": 5000, "S_h": 50, "A_h": 6, "t_r_h": 2},
        "nacelle": {"p_to": 1e6},
        "miscallenous": {"mtom": 5000, "oew": 3000, "npax": 50},
    }

def test_mass_calculation(example_values):
    wing = Wing(**example_values["wing"])
    fuselage1 = Fuselage(**example_values["fuselage1"])
    fuselage2 = Fuselage(**example_values["fuselage2"])
    landing_gear = LandingGear(**example_values["landing_gear"])
    powertrain = Powertrain(**example_values["powertrain"])
    horizontal_tail = HorizontalTail(**example_values["horizontal_tail"])
    nacelle = Nacelle(**example_values["nacelle"])
    miscallenous = Miscallenous(**example_values["miscallenous"])

    assert np.isclose(wing.mass, 573.014)
    assert np.isclose(fuselage1.mass, 339.022323)
    assert np.isclose(fuselage2.mass, 419.163791)
    assert np.isclose(landing_gear.mass, 202.812107)
    assert np.isclose(powertrain.mass, 1000)
    assert np.isclose(horizontal_tail.mass, 50.760872)
    assert np.isclose(nacelle.mass, 145.984168)
    assert np.isclose(miscallenous.mass,1891.746)

