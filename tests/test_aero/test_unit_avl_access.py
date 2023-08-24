
import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))


from modules.aero.avl_access import get_lift_distr, get_strip_array, get_tail_lift_distr
from tests.setup.fixtures import wing, aero, aircraft, engine, config



def test_get_lift_distr(wing, aero):
    lift_func, results = get_lift_distr(wing, aero, plot=False, test= True)
    span_points = np.linspace(0, wing.span/2, 300)

    assert np.isclose(results["Cruise"]["Totals"]["CLtot"], aero.cL_cruise)
    assert results["Cruise"]["Totals"]["Alpha"] > 0.3  # Assert that angle of attach has a reasonable value
    assert (np.diff(np.vectorize(lift_func)(span_points)) < 0).all() # Assert that the lift only decreases towards the tip

def test_get_tail_lift_distr(wing, aero):
    lift_func, results = get_tail_lift_distr(wing, Tail, aero, plot=False, test=True)
    span_points = np.linspace(0, wing.span/2, 300)



def test_get_strip_forces(wing, aero):
    y_le_arr, cl_strip_arr= get_strip_array( wing, aero, plot= False)

    assert np.max(y_le_arr) < wing.span/2 and np.min(y_le_arr) > 0  # Make sure all coordinates are within bounds
    assert np.where(cl_strip_arr == np.max(cl_strip_arr))[0][0] > 1  # Assert maximum lift coefficient is not at the root
    assert (cl_strip_arr < aero.cL_cruise + 0.1).all() # Assert reasonalbe values for lift coefficients

