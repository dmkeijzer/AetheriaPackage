import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from  input.data_structures import *
from modules.preliminary_sizing.wing_power_loading_functions import get_wing_power_loading

PerfClass = PerformanceParameters()
WingClass = Wing()
EngClass = Engine()
AeroClass = Aero()

PerfClass.load()
WingClass.load()
EngClass.load()
AeroClass.load()

test_val = 1

AeroClass.cL_cruise = test_val
PerfClass.wing_loading_cruise = test_val


def test_get_wing_power_loading():
    get_wing_power_loading(PerfClass, WingClass, EngClass, AeroClass)
    assert not np.isclose(AeroClass.cL_cruise, test_val)
    assert not np.isclose(PerfClass.wing_loading_cruise, test_val)

test_get_wing_power_loading()