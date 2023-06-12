import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from scripts.stab_ctrl.Aileron_Sizing import size_aileron

@pytest.fixture
def example_values():
    return {
        "size_aileron": {"b": 10, "V": 100, "aileron_max": 0.35, "roll_rate" :0.35, "Cla": 6 , "Cd0": 0.1, "c_r": 1, "taper": 0.4, "CLa": 5,"ca_c_ratio": 0.2, "step": 0.01,"S": 10}}


def test_aileron_sizing(example_values):
    size_aileron = size_aileron(**example_values["size_aileron"])


    assert np.isclose(size_aileron, 2.92)

