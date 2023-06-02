import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

import input.data_structures.GeneralConstants  as const
from modules.aero.prop_wing_interaction  import *
from input.data_structures.ISA_tool import ISA

@pytest.fixture
def example_values():
    return {
        "airfoil2wing": {"cla": 4, "A": 10},
        "C_T": {"T": 1500, "rho": 1.220, "V_0": 80, "S_W": 10},
        "V_delta": {"C_T": 0.03, "S_W": 10, "n_e": 4, "D": 2, "V_0": 80},
        "D_star": {"D": 2, "V_0": }
    }

def test_mass_calculation(example_values):
    af2wing = airfoil_to_wing_CLa(**example_values["airfoil2wing"])


    assert np.isclose(af2wing, 3.54822585)
    assert np.isclose(dwk, 2.97411154)
