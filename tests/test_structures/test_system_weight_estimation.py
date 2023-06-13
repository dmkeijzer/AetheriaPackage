import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from  input.data_structures import *
from modules.structures.ClassIIWeightEstimation import get_weight_vtol

PerfClass = PerformanceParameters()
WingClass = Wing()
EngClass = Engine()
VtailClass  = VeeTail()
FuseClass = Fuselage()

PerfClass.load()
WingClass.load()
EngClass.load()
VtailClass.load()
FuseClass.load()


test_lengths = [len(i.__annotations__) for i in [PerfClass, WingClass, EngClass, VtailClass, FuseClass]]
test_val = 1

WingClass.wing_weight = test_val
VtailClass.vtail_weight = test_val
FuseClass.fuselage_weight = test_val
EngClass.mass_pernacelle = test_val
PerfClass.OEM =  test_val
PerfClass.MTOM

def test_get_weight_vtol():
    perf_par, wing, vtail, fuselage, engine = get_weight_vtol(PerfClass, FuseClass, WingClass, EngClass, VtailClass)
    assert test_lengths == [len(i.__annotations__) for i in [perf_par,wing ,engine,vtail,fuselage]]
    assert not np.isclose(fuselage.fuselage_weight, test_val)
    assert not np.isclose(wing.wing_weight, test_val)
    assert not np.isclose(vtail.vtail_weight, test_val)
    assert not np.isclose(engine.mass_pernacelle, test_val)
    assert not np.isclose(perf_par.OEM, test_val)
    assert not np.isclose(perf_par.MTOM, test_val)
