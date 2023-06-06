import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from modules.structures.geometry_forms import i_xx_solid,i_zz_solid,j_y_solid,i_xx_thinwalled,i_zz_thinwalled,enclosed_area_thinwalled,area_thinwalled
@pytest.fixture
def example_values():
    return {
        "i_xx_solid": {"width": 50,
                       "height": 20},
        "i_zz_solid": {"width": 50,
                       "height": 20},
        "j_y_solid": {"diameter": 20},
        "i_xx_thinwalled": {"width": 50,
                       "height": 20,
                        "thickness":0.1},
        "i_zz_thinwalled": {"width": 50,
                            "height": 20,
                            "thickness": 0.1},
        "enclosed_area_thinwalled": {"width": 10,
                            "height": 8,
                            "thickness": 0.1},
        "area_thinwalled": {"width": 10,
                             "height": 8,
                             "thickness": 0.1}
    }


def test_i_xx_solid(example_values):
    inputs = example_values['i_xx_solid']
    assert np.isclose(i_xx_solid(**inputs), 33333.333)

def test_i_zz_solid(example_values):
    inputs = example_values['i_zz_solid']
    assert np.isclose(i_zz_solid(**inputs), 208333.333)

def test_j_y_solid(example_values):
    inputs = example_values['j_y_solid']
    assert np.isclose(j_y_solid(**inputs), 15707.963)

def test_i_xx_thinwalled(example_values):
    inputs = example_values['i_xx_thinwalled']
    assert np.isclose(i_xx_thinwalled(**inputs),666.667)

def test_i_zz_thinwalled(example_values):
    inputs = example_values["i_zz_thinwalled"]
    assert np.isclose(i_zz_thinwalled(**inputs),1666.667)


def test_enclosed_area_thinwalled(example_values):
    inputs = example_values["enclosed_area_thinwalled"]
    assert np.isclose(enclosed_area_thinwalled(**inputs),78.21)

def test_area_thinwalled(example_values):
    inputs = example_values["area_thinwalled"]
    assert np.isclose(area_thinwalled(**inputs),3.56)