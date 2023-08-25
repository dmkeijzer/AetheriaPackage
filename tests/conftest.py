import pytest
import sys
import pathlib as pl
from configparser import ConfigParser
import os

sys.path.append(str(list(pl.Path(__file__).parents)[1]))
os.chdir(str(list(pl.Path(__file__).parents)[1]))

from input.data_structures import *

@pytest.fixture
def config():
    config = ConfigParser()
    config.read(r"tests/setup/config_test.cfg")
    return config


@pytest.fixture
def wing(config):
    return Wing.load(config["test_settings"]["dataset"])

@pytest.fixture
def aircraft(config):
    return AircraftParameters.load(config["test_settings"]["dataset"])


@pytest.fixture
def aero(config):
    return Aero.load(config["test_settings"]["dataset"])

@pytest.fixture
def engine(config):
    return Engine.load(config["test_settings"]["dataset"])

def test_fixtures( wing, aircraft, aero, engine):
    print(wing)
    print(aircraft)
    print(aero)
    print(engine)



@pytest.fixture
def values_drag():
    return {
        "reynolds": {"rho_cruise": 1.0, "V_cruise": 100, "mac": 5, "mu": 0.1, "k": 0.1},
        "mach_cruise": {"V_cruise": 100, "gamma": 1.4, "R": 300, "T_cruise": 270},
        "ff_fus": {"l": 10, "d": 2},
        "ff_wing": {"toc": 0.3, "xcm": 0.3, "M": 0.3, "sweep_m": 0},
        "s_wet_fus": {"d": 2, "l1": 3, "l2": 4, "l3": 3},
        "cd_upsweep": {"u": 0.1, "d": 2, "s_wet_fus": 10},
        "cd_base": {"M": 0.3, "A_base": 10, "s_wet_fus": 10},
        "c_fe_fus": {"frac_lam_fus": 0.1, "reynolds": 1500000, "M": 0.3},
        "c_fe_wing": {"frac_lam_wing": 0.1, "reynolds": 15000000, "M": 0.3},
        "cd_fus": {"c_fe_fus": 0.1, "ff_fus": 1, "s_wet_fus": 10},
        "cd_wing": {"name": "J1", "c_fe_wing": 0.1, "ff_wing": 1, "s_wet_wing": 10, "S": 10},
        "cd0": {"S": 10, "cd_fus": 0.01, "cd_wing": 0.01, "cd_upsweep": 0.01, "cd_base": 0.01},
        "cdi": {"CL": 1.0, "A": 8, "e": 0.8},
        "cd": {"cd0": 0.02, "cdi": 0.2},
        "lift_over_drag": {"CL_output": 1.0, "CD_output": 0.15},
        "oswald_eff": {"A": 8},
        "oswald_eff_tandem": {"b1": 10, "b2": 5, "h": 0.6}
    }

@pytest.fixture
def values_slipstream():
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
