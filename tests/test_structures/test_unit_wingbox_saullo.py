import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from scripts.structures.wingbox_saullo import Wingbox
from input.data_structures.wing import Wing
from input.data_structures.engine import Engine
from input.data_structures.material import Material
from input.data_structures.aero import Aero

WingClass = Wing()
EngineClass = Engine()
MaterialClass = Material()
AeroClass = Aero()
WingClass.load()
EngineClass.load()
MaterialClass.load()
AeroClass.load()

wb = Wingbox(WingClass, EngineClass, MaterialClass, AeroClass, True)

@pytest.fixture
def example_values_geometry():
    return {
        "perimiter_ellipse": {"a": 3, "b": 5},
        "chord": {"y" : np.array([1,2])},
        "height": {"y" : np.array([1,2])},
        "l_sk" : {"y" : 2.0},
        "get_w_str" : {"h_str": 2.0}
    }

def test_geometry_calculation(example_values_geometry):
    perimiter_ellipse_ = wb.perimiter_ellipse(**example_values_geometry["perimiter_ellipse"])
    assert np.isclose(perimiter_ellipse_, 25.527)

    chord_ = wb.chord(**example_values_geometry["chord"])
    assert np.isclose(chord_[0],1.50164)
    assert np.isclose(chord_[1],1.297558)

    height_ = wb.height(**example_values_geometry["height"])
    assert np.isclose(height_[0],0.1801968)
    assert np.isclose(height_[1],0.155707)

    l_sk_ = wb.l_sk(**example_values_geometry["l_sk"])
    assert np.isclose(l_sk_, 0.3598238)

    get_w_str_ = wb.get_w_str(**example_values_geometry["get_w_str"])
    assert np.isclose(get_w_str_, 1.6)
