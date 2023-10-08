import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))


from modules.stab_ctrl.vee_tail_rudder_elevator_sizing import get_control_surface_to_tail_chord_ratio, get_K

def test_control_surface_to_tail_chord_ratio(wing, fuselage, veetail, aero):
    output = get_control_surface_to_tail_chord_ratio(wing, fuselage, veetail, aero, 0.8, 5 )
    print(output)


def test_get_k():
    output = get_K(0.5,12)
    print(output)

