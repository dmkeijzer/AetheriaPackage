import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from  input.data_structures import *
from modules.preliminary_sizing.wing_power_loading import get_wing_power_loading

PerfClass = PerformanceParameters()
PerfClass.load()

def test_get_wing_power_loading():
    get_wing_power_loading(PerfClass)

test_get_wing_power_loading()
