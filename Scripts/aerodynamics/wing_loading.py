import os
import json
import sys
import numpy as np
import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
from modules.avlwrapper import Geometry, Surface, Section, NacaAirfoil, Control, Point, Spacing, Session, Case, Parameter

from input.data_structures.wing import Wing
from input.data_structures.aero import Aero
import input.data_structures.GeneralConstants as const
from modules.aero.func_wing_loading import get_lift_distr


dict_directory = "input/data_structures"
dict_name = "aetheria_constants.json"
with open(os.path.join(dict_directory, dict_name)) as f:
    data = json.load(f)

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

if __name__ == '__main__':


    WingClass = Wing()
    AeroClass = Aero()

    WingClass.load()
    AeroClass.load()

    res = get_lift_distr(WingClass, AeroClass, plot= True)

