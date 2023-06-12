
#import input.data_structures.GeneralConstants as const
import pytest
import os
import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from input.data_structures.ISA_tool import ISA
from modules.aero.prop_wing_interaction import *

@pytest.fixture
def example_values():
    return {
        "C_T": {"T": 1500, "rho": 1.220, "V_0": 80, "S_W": 10},
        "V_delta": {"C_T": 0.0384, "S_W": 10, "n_e": 4, "D": 2, "V_0": 80},
        "D_star": {"D": 2, "V_0": 80, "V_delta": 10},
        "A_s_eff": {"b_W": 10, "S_W": 10, "n_e": 4, "D": 2, "V_0": 80, "V_delta": 10},
        "CL_eff": {"mach": 0.2, "A_s_eff": 5, "sweep_half": 0},
        "alpha_s": {"CL_wing": 0.4, "CL_alpha_s_eff": 0.3, "alpha_0": 1, "V_0": 80, "V_delta": 10, "delta_alpha_zero_f": 0},
        "angle_of_attack": {"CL_wing": 0.4, "CL_alpha_s_eff": 0.3, "alpha_0": 1, "V_0": 80, "V_delta": 10, "delta_alpha_zero_f": 0},
        "i_cs": {"CL_wing": 0.4, "CL_alpha_s_eff": 0.3, "alpha_0": 1, "V_0": 80, "V_delta": 10, "delta_alpha_zero_f": 0},
        "sin_epsilon": {"CL_alpha_s_eff": 0.3, "alpha_s": 0.1, "A_s_eff": 6, "CL_wing": 0.4, "A_w": 10},
        "sin_epsilon_s": {"CL_alpha_s_eff": 0.3, "alpha_s": 0.1, "A_s_eff": 6, "CL_wing": 0.4, "A_w": 10},
        "CL_ws": {"S_W": 10, "b_W": 10, "n_e": 4, "D_star": 1.5, "sin_epsilon": 0.05, "V_0": 80, "V_delta": 10, "sin_epsilon_s": 0.05, "CL_wing": 0.4},
        "prop_lift_thrust": {"T": 1500, "rho": 1.220, "V_0": 80, "S_W": 10, "angle_of_attack": 0.05}
    }


def test_mass_calculation(example_values):
    thrust_coef = C_T(**example_values["C_T"])
    V_delta_test = V_delta(**example_values["V_delta"])
    D_star_test = D_star(**example_values["D_star"])
    effective_aspect = A_s_eff(**example_values["A_s_eff"])[0]
    CL_effective = CL_effective_alpha(**example_values["CL_eff"])
    angle_of_attack = alpha_s(**example_values["angle_of_attack"])[1]
    i_cs = alpha_s(**example_values["i_cs"])[2]
    alpha_s_test = alpha_s(**example_values["alpha_s"])[0]
    sin_eps = sin_epsilon_angles(**example_values["sin_epsilon"])[0]
    sin_eps_s = sin_epsilon_angles(**example_values["sin_epsilon_s"])[1]
    CL_ws_test = CL_ws(**example_values["CL_ws"])[1]
    prop_l_thrust = prop_lift_thrust(**example_values["prop_lift_thrust"])

    assert np.isclose(thrust_coef, 0.038422)
    assert np.isclose(V_delta_test, 1.21311)
    assert np.isclose(D_star_test, 1.94365)
    assert np.isclose(effective_aspect, 4.7232)
    assert np.isclose(CL_effective, 4.1715)
    assert np.isclose(angle_of_attack, 0.0174533)
    assert np.isclose(i_cs, 2.3333333)
    assert np.isclose(alpha_s_test, 1.34976)
    assert np.isclose(sin_eps, 0.02546479)
    assert np.isclose(sin_eps_s, 0.0031777796)
    assert np.isclose(CL_ws_test, 0.804174088)
    assert np.isclose(prop_l_thrust, 0.0019203)
