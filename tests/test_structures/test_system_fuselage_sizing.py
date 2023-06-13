import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.structures.fuselage_length import get_fuselage_sizing
from input.data_structures.fuselage import Fuselage
from input.data_structures.fuellCell import FuelCell
from input.data_structures.performanceparameters import PerformanceParameters
from input.data_structures.hydrogenTank import HydrogenTank

FuseClass = Fuselage()
PerfClass = PerformanceParameters()
pstack = FuelCell()
TankClass = HydrogenTank()

FuseClass.load()
PerfClass.load()
TankClass.load()

test_val = 1
test_len = len(FuseClass.__annotations__)

FuseClass.bc = test_val
FuseClass.hc = test_val
FuseClass.bf = test_val
FuseClass.hf = test_val
FuseClass.length_tail = test_val
FuseClass.height_fuselage_inner = test_val
FuseClass.height_fuselage_outer = test_val

def test_get_fuselage_sizing():
    get_fuselage_sizing(TankClass, pstack, PerfClass, FuseClass)
    assert test_len == len(FuseClass.__annotations__)
    assert not np.isclose(FuseClass.bc, test_val)
    assert not np.isclose(FuseClass.hc, test_val)
    assert not np.isclose(FuseClass.bf, test_val)
    assert not np.isclose(FuseClass.hf, test_val)
    assert not np.isclose(FuseClass.length_tail, test_val)
    assert not np.isclose(FuseClass.height_fuselage_inner, test_val)
    assert not np.isclose(FuseClass.height_fuselage_outer, test_val)

