from dataclasses import dataclass
import GeneralConstants as constants
import sys 
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[0]))



@dataclass
class FuelCell:
    """
    This class is to estimate the parameters of a Fuel Cell.

    :param PowerDensity: Power density [kW/kg]
    :param powerRequired: Required power output of the fuel cell [kW] (not loaded)
    :param VolumeDensity: Volumetric energy density [kWh/l]
    :param Cost: Cost of the fuel cell [US$] (not loaded)
    :param Efficiency: Efficiency of the fuel cell 
    """
    powerDensity: float = None
    powerRequired: float = None
    volumeDensity: float  = None
    cost: float = None
    efficiency: float = None

    def load(self):
        self.powerDensity = constants.PowerDensityFuellCell
        self.volumeDensity = constants.VolumeDensityFuellCell
        self.efficiency = constants.effiencyFuellCell

    def mass(self):
        """
        :return: Mass of the battery
        """
        return self.powerRequired / self.PowerDensity

    def volume(self):
        """
        :return: Volume of the fuell cell [m^3]
        """
        return self.powerRequired /self.VolumeDensity  * 0.001

    def price(self):
        """
        :return: Approx cost of the battery [US$]
        """
        return self.powerRequired * self.Cost

