
import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))


from modules.aero.avl_access import get_lift_distr
from input.data_structures.wing import Wing
from input.data_structures.aero import Aero


WingClass = Wing()
AeroClass = Aero()

WingClass.chord_root = 1.7
WingClass.chord_tip = 0.7
WingClass.span = 10
WingClass.surface = (1.7 + 0.7)/2*10
WingClass.chord_mac = 1.22
WingClass.sweep_LE = np.radians(3)


AeroClass.cL_cruise = 0.43

def test_get_lift_distr():
    lift_func, results = get_lift_distr(WingClass, AeroClass, plot= False, test= True)
    span_points = np.linspace(0, WingClass.span/2, 300)

    assert np.isclose(results["Cruise"]["Totals"]["CLtot"], AeroClass.cL_cruise)
    assert results["Cruise"]["Totals"]["Alpha"] > 0.3  # Assert that angle of attach has a reasonable value
    assert (np.diff(np.vectorize(lift_func)(span_points)) < 0).all() # Assert that the lift only decreases towards the tip



