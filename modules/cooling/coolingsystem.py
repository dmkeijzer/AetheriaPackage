import os
import sys
import pathlib as pl
import numpy as np


sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

#from input.data_structures import Battery, FuelCell
from input.data_structures.fluid import Fluid
from input.data_structures.radiator import Radiator

class Coolingsystem:
    """This is wrapper class for all the functions depended on the """

    def mass_flow(heat, delta_temperature: float, heat_capacity: float ) -> float:
        return heat / (delta_temperature * heat_capacity)

    def exchange_effectiveness(Cr: float, NTU: float) -> float: 
        """
        :param: Cr[-]: c_min / c_max 
        :param: NTU [-]: Number of transfer units
        :return: epsilon[-]: exchange effectiveness
        """
        return 1 - np.exp( (1/Cr )*  NTU ** (0.22) * ( np.exp(-1 * Cr* NTU ** (0.78)) -1 ) )

    def power_fan_massflow(massflow: float, density: float, fan_area: float) -> float:
        """
        Calculate the fan power required to obtain specific massflow

        :param: massflow[kg/s] 
        :param: density[kg/(m^3)]
        :param: fan_area[m^2]
        :return: fanpower[W]
        """
        return 0.25 * (massflow * massflow * massflow) / (density * density * fan_area * fan_area) # 1/4 * massflow^3 / (rho^2 * area^2)

    def power_fan_airspeed(airspeed: float, fan_area: float, density:float):
        """
        Calculate the fan power required to obtain specific airspeed
        
        :param: airspeed[m/s] 
        :param: density[kg/(m^3)]
        :param: fan_area[m^2]
        :return: fanpower[W]
        """

        return 0.5* density * fan_area * airspeed * airspeed * airspeed

    def max_heat_transfer(c_min: float, T_hot_in:float, T_cold_in: float) -> float:
        """
        Calculate the maximum heat transfer rate.

        :param c_min: Minimum heat capacity rate [W/k]
        :param T_hot_in: Inlet temperature of the hot fluid [K]
        :param T_cold_in: Inlet temperature of the cold fluid [K]
        :return: Maximum heat transfer rate [W]
        """
        return c_min * (T_hot_in - T_cold_in)

    def number_transfer_units(overall_heat_transfer_capacity:float , cmin: float ):
        """
        Calculate the NTU

        :param overall_heat_transfer_capacity: [W/K]
        :param cmin:  Minimum heat capacity rate [W/K]
        :return: NTU [-]
        """
        return overall_heat_transfer_capacity  / cmin


    def calculate_heat_expelled(c_hot: float, c_cold: float, T_hot_in: float, T_cold_in: float,overall_heat_transfer_capacity : float):
        """
        Calculate the heat expelled by a heat exchanger.

        :param c_hot: heat capacity rate of the hot fluid [W/k]
        :param c_cold: heat capacity rate of the cold fluid [W/k]
        :param T_hot_in: Inlet temperature of the hot fluid [K]
        :param T_cold_in: Inlet temperature of the cold fluid [K]
        :param overall_heat_transfer_capacity: [W / k]
        :return: Heat expelled by the heat exchanger [W]
        """
        
        #determine which side is min and which side is the max
        c_min = min(c_hot,c_cold)
        c_max = max(c_cold, c_hot)
        
        #determine the max heat transfer out
        Qmax = Coolingsystem.max_heat_transfer(c_min= c_min , T_hot_in= T_hot_in, T_cold_in= T_cold_in)

        #determine heat exchanger effectivenes
        cr = c_min/c_max
        NTU = Coolingsystem.number_transfer_units(overall_heat_transfer_capacity,c_min)
        epsilon = Coolingsystem.exchange_effectiveness(cr, NTU)

        #determine heat expelled by the radiator
        Q_expelled = epsilon * Qmax
        return Q_expelled



def hx_geometry(radiator : Radiator, Z_HX: float, H_HX: float, W_HX: float) -> Radiator:
    """calculate surface areas of the radiator 
        Function based on the geometry specified in A. Scoccimarro thesis on preliminary thermal management sizing
    """
    W_channel = radiator.h_tube - 2* radiator.t_tube
    H_channel = radiator.h_tube - 2* radiator.t_tube

    radiator.s_fin = radiator.t_fin / radiator.HX_gamma
    radiator.l_fin = radiator.t_fin / radiator.HX_delta
    radiator.h_fin = radiator.t_fin / (radiator.HX_alpha * radiator.HX_gamma) 

    radiator.N_channel = Z_HX / (W_channel + radiator.t_channel) *H_HX / (radiator.h_fin + radiator.h_tube)

    radiator.A_hot = 2 * (W_channel * H_channel) * W_HX * radiator.N_channel

    radiator.N_fin = W_HX / (radiator.s_fin + radiator.t_fin) * H_HX / (radiator.h_fin * radiator.h_tube)
    radiator.A_fin = 2 * radiator.h_fin * radiator.l_fin + 2 * radiator.h_fin * radiator.t_fin + radiator.s_fin * radiator.t_fin

    A_primary = 2 * radiator.s_fin * radiator.l_fin

    radiator.A_cold = radiator.N_fin * (radiator.A_fin + A_primary)
    radiator.A_fs_cross = radiator.s_fin * radiator.h_fin * radiator.N_fin
    radiator.A_cross_hot = W_channel * H_channel * radiator.N_channel

    return radiator


def fin_efficiency(h_c_cold: float, thermal_conductivity_fin: float, radiator) -> float:
    """
    calculate fin efficiency to be used in the surface efficiency

    :param: h_c_cold[W/(m^2 K)]: Heat transfer coefficient cold side (air)
    :param: thermal_conductivity_fin[W/(m K)]: 
    :param: radiator
    :return: eta_fin[-]: fin efficiency
    """
    ml = np.sqrt(2* h_c_cold / (thermal_conductivity_fin * radiator.t_fin)) * (radiator.h_fin + radiator.t_fin)/2
    eta_fin = np.tanh(ml)/ ml
    return eta_fin

def surface_efficiency(radiator: Radiator, eta_fin: float ) -> float:
    """ function that calculates surface efficiency for thermal resistance"""
    return 1- (radiator.A_fin/ radiator.A_cold * (1 - eta_fin))  

def hx_thermal_resistance(radiator: Radiator, h_c_cold: float, h_c_hot: float, eta_surface) -> Radiator:
    return 1/(h_c_hot * radiator.A_hot) + 1 / (radiator.A_cold * h_c_cold * eta_surface)

def colburn_factor(radiator: Radiator, reynolds: float) -> float:
    j = 0.6522 * reynolds**(-0.5403) * radiator.HX_alpha**(-0.1541) * radiator.HX_delta**0.1499 * radiator.HX_gamma**(-0.0678)  *  (1 + 5.269e-5 * reynolds**1.34 * radiator.HX_alpha**0.504 * radiator.HX_delta**0.456 * radiator.HX_gamma**(-1.055))**0.1
    return j

def hydralic_diameter_HX(A_thermal: float , A_crosssection: float, depth: float ) -> float:
    return 4 * A_crosssection / A_thermal * depth

def reynolds_HX(mass_flux: float, hydraulic_diameter: float, viscosity) -> float:
    return mass_flux * hydraulic_diameter / viscosity

def prandtl_heat(heat_capacity: float, viscosity: float, thermal_conductivity: float) -> float:
    return heat_capacity * viscosity / thermal_conductivity

def heat_capacity_cold(colburn : float, mass_flux: float, c_p: float, prandtl: float) -> float:
    return colburn * mass_flux * c_p / (prandtl ** (2/3))

def calculate_flam(Re, AR_channel):
    flam = 24 * (1 - 1.3553 * AR_channel + 1.9467 * AR_channel**2 - 1.7012 * AR_channel**3 + 0.9564 * AR_channel**4 - 0.2537 * AR_channel**5) * Re**(-1)
    return flam

def heat_capacity_hot(Re: float, Pr: float, f: float, Dh: float, k: float) -> float:
    if Re > 3e3 and Re < 1e4:
        hc = ((Re - 1000) * Pr * (f / 2) * (k / Dh)) / (1 + 12.7 * (Pr**(2/3) - 1) * (f / 2)**0.5)
        return hc
    elif Re > 1e4 and Re < 5e6:
        hc = (Re * Pr * (f / 2) * (k / Dh)) / (1 + 12.7 * (Pr**(2/3) - 1) * (f / 2)**0.5)
        return hc
    else: 
        raise ValueError("Re is not in range to calculate heat transfer coefficient hot")

def mass_flux(mass_flow: float, A_crossectional: float) -> float:
    return mass_flow / A_crossectional