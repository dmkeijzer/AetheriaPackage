import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from modules.structures.stress_forms import bending_stress, normal_stress, torsion_thinwalled_closed, torsion_circular_section, shear_thin_walled_rectangular_section,critical_buckling_stress,wohlers_curve,paris_law

@pytest.fixture()
def example_values():
    return {
        "bending_stress": {
            "moment_x": 10000,
            "moment_z": 20000,
            "i_xx": 1,
            "i_zz": 2,
            "i_xz": 0.5,
            "width": 3,
            "height": 2
        },
        "normal_stress": {
            "force": 100,
            "area": 20
        },
        "torsion_thinwalled_closed": {
            "torque": 100,
            "thickness": 0.1,
            "enclosed_area": 10
        },
        "torsion_circular_section": {
            "torque": 50,
            "dist": 2,
            "j_z": 5
        },
        "shear_thin_walled_rectangular_section": {
            "width": 2,
            "height": 5,
            "thickness": 0.1,
            "i_xx": 2,
            "i_zz": 1,
            "Vx": 3000,
            "Vz": 4000
        },
        "critical_buckling_stress": {
            "C": 2,
            "t": 3,
            "b": 4
        },
        "wohlers_curve": {
            "C": 10,
            "m": 2,
            "S": 5
        },
        "paris_law": {
            "C": 0.5,
            "beta": 0.2,
            "load": 100,
            "m": 2,
            "a_f": 4,
            "a_0": 2
        }
    }

def test_bending_stress(example_values):
    inputs = example_values['bending_stress']
    assert np.isclose(bending_stress(**inputs), np.array([18571.43,-18571.43])).all()

def test_normal_stress(example_values):
    inputs = example_values['normal_stress']
    assert np.isclose(normal_stress(**inputs), 5.0)

def test_torsion_thinwalled_closed(example_values):
    inputs = example_values['torsion_thinwalled_closed']
    assert np.isclose(torsion_thinwalled_closed(**inputs), 50)

def test_torsion_circular_section(example_values):
    inputs = example_values['torsion_circular_section']
    assert np.isclose(torsion_circular_section(**inputs), 20.0)

def test_shear_thin_walled_rectangular_section(example_values):
    inputs = example_values['shear_thin_walled_rectangular_section']
    assert np.isclose(shear_thin_walled_rectangular_section(**inputs), (0.3125, 0.625, 0.3125, 0.625))X

def test_critical_buckling_stress(example_values):
    inputs = example_values['critical_buckling_stress']
    assert np.isclose(critical_buckling_stress(**inputs), 4.840634296972383)

def test_wohlers_curve(example_values):
    inputs = example_values['wohlers_curve']
    assert np.isclose(wohlers_curve(**inputs), 0.4)

def test_paris_law(example_values):
    inputs = example_values['paris_law']
    assert np.isclose(paris_law(**inputs), -1.830504064271163)