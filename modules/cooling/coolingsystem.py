import os
import sys
import pathlib as pl
import numpy as np


sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

#from input.data_structures import Battery, FuelCell
from input.data_structures.fluid import Fluid

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

    def number_transfer_units(heat_transfer_coefficient: float, exchange_area: float, cmin: float ):
        """
        Calculate the NTU

        :param heat_transfer_coefficient: [W / (m^2 k)]
        :param exchange_area: Surface area of the heat exchanger [m^2]
        :param cmin:  Minimum heat capacity rate [W/k]
        :return: NTU [-]
        """
        return heat_transfer_coefficient * exchange_area  / cmin


    def calculate_heat_expelled(c_hot: float, c_cold: float, T_hot_in: float, T_cold_in: float, heat_transfer_coefficient_exchanger: float, Area_heat_exchanger):
        """
        Calculate the heat expelled by a heat exchanger.

        :param c_hot: heat capacity rate of the hot fluid [W/k]
        :param c_cold: heat capacity rate of the cold fluid [W/k]
        :param T_hot_in: Inlet temperature of the hot fluid [K]
        :param T_cold_in: Inlet temperature of the cold fluid [K]
        :param heat_transfer_coefficient_exchanger: Heat transfer coefficient of the heat exchanger [W / (m^2 k)]
        :param Area_heat_exchanger: Surface area of the heat exchanger [m^2]
        :return: Heat expelled by the heat exchanger [W]
        """
        
        #determine which side is min and which side is the max
        c_min = min(c_hot,c_cold)
        c_max = max(c_cold, c_hot)
        
        #determine the max heat transfer out
        Qmax = Coolingsystem.max_heat_transfer(c_min= c_min , T_hot_in= T_hot_in, T_cold_in= T_cold_in)

        #determine heat exchanger effectivenes
        cr = c_min/c_max
        NTU = Coolingsystem.number_transfer_units(heat_transfer_coefficient_exchanger, Area_heat_exchanger,c_min)
        epsilon = Coolingsystem.exchange_effectiveness(cr, NTU)

        #determine heat expelled by the radiator
        Q_expelled = epsilon * Qmax
        return Q_expelled

