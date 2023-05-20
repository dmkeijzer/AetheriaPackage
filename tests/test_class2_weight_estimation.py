import pytest
import os
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from modules.midterm_structures.ClassIIWeightEstimation import Wing, Fuselage, LandingGear, Powertrain, HorizontalTail, Nacelle, H2System, Miscallenous

@pytest.fixture
def example_values():
    return {
        "wing": {"mtom": 5000, "S": 100, "n_ult": 3, "A": 8},
        "fuselage": {"identifier": "J1", "mtom": 5000, "max_per": 20, "lf": 10, "npax": 50},
        "landing_gear": {"mtom": 5000},
        "powertrain": {"p_max": 10000, "p_dense": 10},
        "horizontal_tail": {"w_to": 5000, "S_h": 50, "A_h": 6, "t_r_h": 2},
        "nacelle": {"w_to": 5000},
        "h2system": {"energy": 10000, "cruisePower": 5000, "hoverPower": 3000},
        "miscallenous": {"mtom": 5000, "oew": 3000, "npax": 50},
    }

def test_mass_calculation(example_values):
    wing = Wing(**example_values["wing"])
    fuselage = Fuselage(**example_values["fuselage"])
    landing_gear = LandingGear(**example_values["landing_gear"])
    powertrain = Powertrain(**example_values["powertrain"])
    horizontal_tail = HorizontalTail(**example_values["horizontal_tail"])
    nacelle = Nacelle(**example_values["nacelle"])
    h2system = H2System(**example_values["h2system"])
    miscallenous = Miscallenous(**example_values["miscallenous"])

    assert wing.mass == 10
    assert fuselage.mass == 10
    assert landing_gear.mass == 10
    assert powertrain.mass == 10
    assert horizontal_tail.mass == 10
    assert nacelle.mass == 10
    assert h2system.mass == 10
    assert miscallenous.mass == 10

# test_mass_calculation(example_values)
