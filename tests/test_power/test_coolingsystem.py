
import os
import sys
import pathlib as pl
import numpy as np
import json
import pytest

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))
from input.data_structures import Radiator, Fluid
from modules.cooling.coolingsystem import CoolingsystemPerformance ,RadiatorPerformance


def test_mass_flow():

    # Test case 1: Positive values
    heat = 1000  # W
    delta_temperature = 10  # K
    heat_capacity = 1000  # J/(kg K)
    expected_result = 0.1  # kg/s
    assert CoolingsystemPerformance.mass_flow(heat, delta_temperature, heat_capacity) == expected_result

    # Test case 2: Zero heat
    heat = 0  # W
    delta_temperature = 10  # K
    heat_capacity = 1000  # J/(kg K)
    expected_result = 0.0  # kg/s
    assert CoolingsystemPerformance.mass_flow(heat, delta_temperature, heat_capacity) == expected_result

    # Test case 3: Zero delta temperature
    heat = 1000  # W
    delta_temperature = 0  # K
    heat_capacity = 1000  # J/(kg K)
    # Division by zero should raise a ZeroDivisionError
    try:
        CoolingsystemPerformance.mass_flow(heat, delta_temperature, heat_capacity)
        assert False, "Expected ZeroDivisionError"
    except ZeroDivisionError:
        pass
    
def test_exchange_effectiveness():
        # Test case 1: Cr = 0.5, NTU = 1
        result = CoolingsystemPerformance.exchange_effectiveness(0.5, 1)
        assert np.isclose(result, 0.54476,rtol = 1e-2)
        
        # Test case 2: Cr = 1, NTU = 2
        result = CoolingsystemPerformance.exchange_effectiveness(2, 2)
        assert np.isclose(result, 0.43083,rtol = 1e-2)

def test_power_fan_massflow():
    # Test case 1: massflow = 1.5 kg/s, density = 2 kg/m^3, fan_area = 0.2 m^2
    result = CoolingsystemPerformance.power_fan_massflow(1.5, 2, 0.2)
    expected_result = 5.27343
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: massflow = 0.8 kg/s, density = 1.2 kg/m^3, fan_area = 0.1 m^2
    result = CoolingsystemPerformance.power_fan_massflow(0.8, 1.2, 0.1)
    expected_result = 8.8888
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_power_fan_airspeed():
    # Test case 1: airspeed = 10 m/s, fan_area = 0.5 m^2, density = 1.2 kg/m^3
    result = CoolingsystemPerformance.power_fan_airspeed(10, 0.5, 1.2)
    expected_result = 300.
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: airspeed = 5 m/s, fan_area = 0.3 m^2, density = 1.5 kg/m^3
    result = CoolingsystemPerformance.power_fan_airspeed(5, 0.3, 1.5)
    expected_result = 28.125
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_max_heat_transfer():
    # Test case 1: c_min = 100 W/K, T_hot_in = 400 K, T_cold_in = 300 K
    result = CoolingsystemPerformance.max_heat_transfer(100, 400, 300)
    expected_result = 10000
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: c_min = 50 W/K, T_hot_in = 350 K, T_cold_in = 250 K
    result = CoolingsystemPerformance.max_heat_transfer(50, 350, 250)
    expected_result = 5000
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_number_transfer_units():
    # Test case 1: overall_heat_transfer_capacity = 5000 W/K, cmin = 100 W/K
    result = CoolingsystemPerformance.number_transfer_units(5000, 100)
    expected_result = 50
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: overall_heat_transfer_capacity = 10000 W/K, cmin = 200 W/K
    result = CoolingsystemPerformance.number_transfer_units(10000, 200)
    expected_result = 50
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_calculate_heat_expelled():
    # Test case 1: c_hot = 200 W/K, c_cold = 300 W/K, T_hot_in = 400 K, T_cold_in = 300 K, overall_heat_transfer_capacity = 5000 W/K
    result, epsilon = CoolingsystemPerformance.calculate_heat_expelled(200, 300, 400, 300, 5000)
    expected_result = 19047.6
    expected_epsilon =0.95
    
    assert np.isclose(result, expected_result, rtol=1e-2)
    assert np.isclose(epsilon, expected_epsilon, rtol=1e-2)

    # Test case 2: c_hot = 150 W/K, c_cold = 250 W/K, T_hot_in = 350 K, T_cold_in = 250 K, overall_heat_transfer_capacity = 7500 W/K
    result, epsilon = CoolingsystemPerformance.calculate_heat_expelled(150, 250, 350, 250, 7500)
    expected_result = 14708.60
    expected_epsilon = 0.980
    assert np.isclose(result, expected_result, rtol=1e-2)
    assert np.isclose(epsilon, expected_epsilon, rtol=1e-2)

def set_up_radiator():
    radiator = Radiator(W_HX=1,H_HX=2, Z_HX=1, 
                        h_tube=0.005, t_tube=0.0001, t_channel=0.0002,
                        s_fin=0.001, l_fin = 0.002,h_fin = 0.002, t_fin=0.0001 )
    radiator = RadiatorPerformance.hx_geometry(radiator)
    return radiator

def test_fin_efficiency():
    # Test case 1: h_c_cold = 1000 W/(m^2 K), thermal_conductivity_fin = 50 W/(m K), t_fin = 0.1 m, h_fin = 0.02 m
    radiator = set_up_radiator()
    radiator.t_fin = 0.1
    radiator.h_fin = 0.02
    result = RadiatorPerformance.fin_efficiency(1000, 50, radiator)
    expected_result = 0.694
  
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: h_c_cold = 800 W/(m^2 K), thermal_conductivity_fin = 60 W/(m K), t_fin = 0.08 m, h_fin = 0.015 m
    radiator.t_fin = 0.08
    radiator.h_fin = 0.015
    result = RadiatorPerformance.fin_efficiency(800, 60, radiator)
    expected_result = 0.807
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_surface_efficiency():
    # Test case 1
    radiator = set_up_radiator()
    eta_fin = 0.7
    expected_result = 0.99
    result = RadiatorPerformance.surface_efficiency(radiator, eta_fin)
    assert np.isclose(result, expected_result, rtol=1e-1)

    # Test case 2
    radiator = set_up_radiator()
    eta_fin = 0.8
    expected_result = 0.99
    result = RadiatorPerformance.surface_efficiency(radiator, eta_fin)
    assert np.isclose(result, expected_result, rtol=1e-1)

def test_hx_thermal_resistance():
    # Test case 1: A_hot = 0.8 m^2, A_cold = 1.2 m^2, h_c_cold = 1000 W/(m^2 K), h_c_hot = 1200 W/(m^2 K), eta_surface = 0.9
    radiator = set_up_radiator()
    h_c_cold = 1000
    h_c_hot = 1200
    eta_surface = 0.9
    expected_result = 1.4e-6

    result = RadiatorPerformance.hx_thermal_resistance(radiator, h_c_cold, h_c_hot, eta_surface)
    assert np.isclose(result, expected_result, rtol=1e-1)

    # Test case 2: A_hot = 1.2 m^2, A_cold = 1.8 m^2, h_c_cold = 800 W/(m^2 K), h_c_hot = 900 W/(m^2 K), eta_surface = 0.8
    radiator = set_up_radiator()
    h_c_cold = 800
    h_c_hot = 900
    eta_surface = 0.8
    expected_result = 1.9e-6

    result = RadiatorPerformance.hx_thermal_resistance(radiator, h_c_cold, h_c_hot, eta_surface)
    assert np.isclose(result, expected_result, rtol=1e-1)

def test_colburn_factor():
    # Test case 1: Reynolds = 1000, alpha = 0.2, delta = 0.5, gamma = 0.8
    reynolds = 1000
    alpha = 0.2
    delta = 0.5
    gamma = 0.8
    expected_result = 0.01868

    result = RadiatorPerformance.colburn_factor(reynolds, alpha, delta, gamma)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: Reynolds = 2000, alpha = 0.5, delta = 0.3, gamma = 1.2
    reynolds = 2000
    alpha = 0.5
    delta = 0.3
    gamma = 1.2
    expected_result = 0.0102372

    result = RadiatorPerformance.colburn_factor(reynolds, alpha, delta, gamma)
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_hydraulic_diameter_HX():
    # Test case 1: width = 0.1 m, height = 0.2 m
    width = 0.1
    height = 0.2
    expected_result = 0.133333

    result = RadiatorPerformance.hydralic_diameter_HX(width, height)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: width = 0.05 m, height = 0.15 m
    width = 0.05
    height = 0.15
    expected_result = 0.075

    result = RadiatorPerformance.hydralic_diameter_HX(width, height)
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_reynolds_HX():
    # Test case 1: mass_flux = 0.5 kg/s, hydraulic_diameter = 0.1 m, viscosity = 1e-3
    mass_flux = 0.5
    hydraulic_diameter = 0.1
    viscosity = 1e-3
    expected_result = 50.0

    result = RadiatorPerformance.reynolds_HX(mass_flux, hydraulic_diameter, viscosity)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: mass_flux = 0.8 kg/s, hydraulic_diameter = 0.05 m, viscosity = 2e-3
    mass_flux = 0.8
    hydraulic_diameter = 0.05
    viscosity = 2e-3
    expected_result = 20

    result = RadiatorPerformance.reynolds_HX(mass_flux, hydraulic_diameter, viscosity)
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_prandtl_heat():
    # Test case 1: heat_capacity = 1000 J/(kg K), viscosity = 1e-3 kg/(m s), thermal_conductivity = 0.5 W/(m K)
    heat_capacity = 1000
    viscosity = 1e-3
    thermal_conductivity = 0.5
    expected_result = 2.0

    result = RadiatorPerformance.prandtl_heat(heat_capacity, viscosity, thermal_conductivity)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: heat_capacity = 800 J/(kg K), viscosity = 2e-3 kg/(m s), thermal_conductivity = 0.8 W/(m K)
    heat_capacity = 800
    viscosity = 2e-3
    thermal_conductivity = 0.8
    expected_result = 2.0

    result = RadiatorPerformance.prandtl_heat(heat_capacity, viscosity, thermal_conductivity)
    assert np.isclose(result, expected_result, rtol=1e-2)


def test_heat_capacity_cold():
    # Test case 1: colburn = 0.2, mass_flux = 0.5 kg/s, c_p = 1000 J/(kg K), prandtl = 0.7
    colburn = 0.2
    mass_flux = 0.5
    c_p = 1000
    prandtl = 0.7
    expected_result = 126.84

    result = RadiatorPerformance.heat_capacity_cold(colburn, mass_flux, c_p, prandtl)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: colburn = 0.4, mass_flux = 0.8 kg/s, c_p = 800 J/(kg K), prandtl = 0.9
    colburn = 0.4
    mass_flux = 0.8
    c_p = 800
    prandtl = 0.9
    expected_result = 274.628

    result = RadiatorPerformance.heat_capacity_cold(colburn, mass_flux, c_p, prandtl)
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_calculate_flam():
    # Test case 1: Re = 1000, AR_channel = 0.5
    Re = 1000
    AR_channel = 0.5
    expected_result = 0.015628

    result = RadiatorPerformance.calculate_flam(Re, AR_channel)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: Re = 2000, AR_channel = 0.8
    Re = 2000
    AR_channel = 0.8
    expected_result = 0.00719

    result = RadiatorPerformance.calculate_flam(Re, AR_channel)
    assert np.isclose(result, expected_result, rtol=1e-2)


def test_heat_capacity_hot():
    # Test case 1: Re = 5000, Pr = 0.7, f = 0.03, Dh = 0.05, k = 0.5, pipe_length = 2
    Re = 5000
    Pr = 0.7
    f = 0.03
    Dh = 0.05
    k = 0.5
    pipe_length = 2
    expected_result = 626.089
    result = RadiatorPerformance.heat_capacity_hot(Re, Pr, f, Dh, k, pipe_length)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2: Re = 15000, Pr = 1.0, f = 0.05, Dh = 0.08, k = 0.8
    Re = 15000
    Pr = 1.0
    f = 0.05
    Dh = 0.08
    k = 0.8
    expected_result = 3750.0

    result = RadiatorPerformance.heat_capacity_hot(Re, Pr, f, Dh, k)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 3: Re = 2000, Pr = 0.5, f = 0.02, Dh = 0.02, k = 0.3
    Re = 2000
    Pr = 0.5
    f = 0.02
    Dh = 0.02
    k = 0.3
    expected_result = 117.288
    pipe_length= 2
    result = RadiatorPerformance.heat_capacity_hot(Re, Pr, f, Dh, k,pipe_length)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 4: Re = 6000000, Pr = 1.2, f = 0.06, Dh = 0.1, k = 0.7
    Re = 6000000
    Pr = 1.2
    f = 0.06
    Dh = 0.1
    k = 0.7

    with pytest.raises(ValueError):
        RadiatorPerformance.heat_capacity_hot(Re, Pr, f, Dh, k)

def test_pressure_drop():
    # Test case 1
    friction = 0.03
    mass_flux = 0.1
    length = 1.0
    hydraulic_diameter = 0.05
    density = 1000.0

    expected_result = 1.2e-05

    result = RadiatorPerformance.pressure_drop(friction, mass_flux, length, hydraulic_diameter, density)
    assert np.isclose(result, expected_result, rtol=1e-2)

    # Test case 2
    friction = 0.02
    mass_flux = 0.2
    length = 1
    hydraulic_diameter = 0.08
    density = 800.0

    expected_result = 2.5e-05

    result = RadiatorPerformance.pressure_drop(friction, mass_flux, length, hydraulic_diameter, density)
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_mass_flux():
    # Test case
    mass_flow = 5.0
    A_crossectional = 2.0
    expected_result = 2.5
    result = RadiatorPerformance.mass_flux(mass_flow, A_crossectional)
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_mass_radiator():
    # Test case 1
    HX = set_up_radiator()
    density_material = 8000.0

    expected_result = 3842.1

    result = RadiatorPerformance.mass_radiator(HX, density_material)
    assert np.isclose(result, expected_result, rtol=1e-2)

def test_cooling_radiator():
    HX = set_up_radiator()
    coolant = Fluid(viscosity = 0.355e-3, thermal_conductivity = 0.65, heat_capacity = 4184, density=997)
    air = Fluid(heat_capacity=1005, density= 1.225,viscosity=18e-6, thermal_conductivity=25.87e-3)


    mass_flow_cold = 0.5  # kg/s
    mass_flow_hot = 0.3  # kg/s

    R_tot, delta_pressure = RadiatorPerformance.cooling_radiator(HX, mass_flow_cold, mass_flow_hot, air, coolant)

    # Test thermal resistance calculation
    expected_R_tot = 1.0169e-05  # Replace with the expected valu
    assert np.isclose(R_tot, expected_R_tot, rtol=1e-6)

    # Test pressure drop calculation
    expected_delta_pressure = 9596.563 
    assert np.isclose(delta_pressure, expected_delta_pressure, rtol=1e-6)
