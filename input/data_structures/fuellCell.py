from dataclasses import dataclass
import GeneralConstants as constants
import sys 
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[0]))



@dataclass
class FuelCell:
    """
    This class is to estimate the parameters of a Fuel Cell.

    :param maxpower: max power from FC [KW]
    :param mass: mass [kg]
    :param Cost: Cost of the fuel cell [US$] (not loaded)
    :param Efficiency: Efficiency of the fuel cell s
    """
    maxpower = 125 #W
    mass = 42 #kg
    efficiency = .55
    length = 0.582 #m
    width = 0.43 #m
    depth = 0.156 #m

    def heat(self, power):
        """
        :param: power[kW]: electric power generated by the fuel cell
        :return: heat generated by the fuel cell
        """
        return (1 - self.efficiency)/ self.efficiency * power

    @property
    def volume(self):
        return self.length * self.width * self.depth #volume in m^3
    
    @property
    def price(self):
        return self.maxpower * 75 # 75 dollars per kW

