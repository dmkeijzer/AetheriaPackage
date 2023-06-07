import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from scripts.stab_ctrl.vee_tail_rudder_elevator_sizing import get_K, get_c_control_surface_to_c_vee_ratio, get_tail_dihedral_and_area, get_control_surface_to_tail_chord_ratio

@pytest.fixture
def example_values():
    return {
        "get_K": {"taper_h": 0.5, "AR_h": 10},
        "get_c_control_surface_to_c_vee_ratio": {"tau": 0.2},
        "get_tail_dihedral_and_area": {"S_hor": 3, "Cn_beta_req": 0.08, "Fuselage_volume": 10, "S": 10, "b": 10,"l_v":5,"AR_h":4,"taper_h":0.4},
        "get_control_surface_to_tail_chord_ratio": {"b"=10, "Fuselage_volume":10,"S_hor": 3,"downwash_angle_landing": 0.14, "aoa_landing": 0.2, "CL_h": -0.5, "CL_a_h":5, "V_tail_to_V_ratio": 0.9, "l_v": 5, "S": 10,"c": 1,"taper_h": 0.4, "AR_h": 4}
    }


def test_wing_loc_horzstab_sizing(example_values):
    get_K = get_K(**example_values["get_K"])
    get_c_control_surface_to_c_vee_ratio = get_c_control_surface_to_c_vee_ratio(**example_values["get_c_control_surface_to_c_vee_ratio"])
    get_tail_dihedral_and_area = get_tail_dihedral_and_area(**example_values["get_tail_dihedral_and_area"])
    get_control_surface_to_tail_chord_ratio = get_control_surface_to_tail_chord_ratio(**example_values["get_control_surface_to_tail_chord_ratio"])


    assert np.isclose(get_K, 0.77)
    assert np.isclose(get_c_control_surface_to_c_vee_ratio,0.1)
    assert np.isclose(get_tail_dihedral_and_area, (0.5760531654909002, 4.265714322061353) )
    assert np.isclose(get_control_surface_to_tail_chord_ratio, )
