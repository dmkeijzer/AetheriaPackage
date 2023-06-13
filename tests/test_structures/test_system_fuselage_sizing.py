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

FuseClass.

def test_get_fuselage_sizing():
    get_fuselage_sizing(TankClass, pstack, PerfClass, FuseClass)
    print(FuseClass)

test_get_fuselage_sizing()
