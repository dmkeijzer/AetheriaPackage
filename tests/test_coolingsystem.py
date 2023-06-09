
import os
import sys
import pathlib as pl
import numpy as np


sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.cooling.coolingsystem import CoolingsystemPerformance ,RadiatorPerformance

def test_mass_flow():
    # Test case 1: Positive values
    heat = 1000  # W
    delta_temperature = 10  # K
    heat_capacity = 1000  # J/(kg K)
    expected_result = 0.1  # kg/s
    assert RadiatorPerformance.mass_flow(heat, delta_temperature, heat_capacity) == expected_result

    # Test case 2: Zero heat
    heat = 0  # W
    delta_temperature = 10  # K
    heat_capacity = 1000  # J/(kg K)
    expected_result = 0.0  # kg/s
    assert RadiatorPerformance.mass_flow(heat, delta_temperature, heat_capacity) == expected_result

    # Test case 3: Zero delta temperature
    heat = 1000  # W
    delta_temperature = 0  # K
    heat_capacity = 1000  # J/(kg K)
    # Division by zero should raise a ZeroDivisionError
    try:
        RadiatorPerformance.mass_flow(heat, delta_temperature, heat_capacity)
        assert False, "Expected ZeroDivisionError"
    except ZeroDivisionError:
        pass